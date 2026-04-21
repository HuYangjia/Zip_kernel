# V9 — Submatrix-Level Mixed-Precision Quantization (子矩阵级混合精度量化)

> **Document Type**: AI Instruction Document (用于 AI 编码的技术规格文档)
> **Target Model**: Qwen3-4B-Instruct-2507
> **Date**: 2026-04-17
> **Status**: Draft — 待讨论确认后进入实施阶段
> **Predecessor**: V7 Smooth + Tail Absorb (固定 RANK 列 INT8 补偿)

---

## 0. Background & Motivation (背景与动机)

### 0.1 Problem Statement

在 V7/V8 实验中，我们使用 **固定的 tail_rank 列** 作为 INT8 高精度补偿区域。实验结果表明：

- rank 参数对 PPL 影响极小（r16/r64/r128 差异仅 0.01-0.08）
- 固定列位置无法精确覆盖 outlier 分布区域
- Outlier 在权重矩阵中的分布是 **非均匀的**，集中在特定的行列交叉区域

**核心问题**：固定 RANK 列的 INT8 补偿是 **列粒度** 的，无法针对 outlier 的 **局部聚集特性** 进行精准补偿。

### 0.2 Proposed Solution

将权重矩阵 W ∈ R^{X_out × Y_in} 分割为 **子矩阵网格**，以子矩阵为粒度进行混合精度分配：

- 给定子矩阵尺寸 (brow, bcol)，将 W 分割为 (X_out/brow) × (Y_in/bcol) 个子矩阵块
- 给定 5% 的高精度预算，选择量化误差最大的 5% 子矩阵使用 INT8 量化
- 其余 95% 子矩阵使用 INT4 量化
- 推理时使用 **完整 INT4 矩阵 + 稀疏 INT8 修正矩阵** 的双矩阵计算方案

### 0.3 Relationship to Existing Code

本方案基于现有代码库构建：

- **基础 GPTQ 流程**: `qwen3_gptq.py` 中的 `qwen3_sequential()` 函数
- **GPTQ 核心类**: `gptq/gptq.py` 中的 `GPTQ` 类
- **量化器**: `gptq/quant.py` 中的 `Quantizer` 类
- **SmoothQuant 预处理**: `qwen3_smooth.py`
- **Tail Absorb 参考**: `gptq_tail_absorb.py` 中的 `GPTQTailAbsorb` 类

---

## 1. Mathematical Formulation (数学定义)

### 1.1 Weight Matrix Partitioning

Given weight matrix W ∈ R^{d_out × d_in} for a Linear layer:

```
Block dimensions: brow (block row size), bcol (block column size)
Grid dimensions: nrow = ceil(d_out / brow), ncol = ceil(d_in / bcol)
Total blocks: N_total = nrow × ncol
```

Block (i, j) covers rows [i*brow : min((i+1)*brow, d_out)] and cols [j*bcol : min((j+1)*bcol, d_in)]:

```
W_block[i, j] = W[i*brow:(i+1)*brow, j*bcol:(j+1)*bcol]
```

### 1.2 Budget Allocation

```
budget_ratio = 0.05  (5% of total blocks)
N_high = max(1, round(N_total * budget_ratio))
N_low = N_total - N_high
```

### 1.3 Block Selection Metric — Sensitivity Score

For each block (i, j), compute a **sensitivity score** S[i, j] that estimates the quantization error impact. Three candidate metrics (to be evaluated):

#### Metric A: Weight Magnitude (最简单，无需校准数据)

```python
S[i, j] = torch.norm(W_block[i, j], p='fro')  # Frobenius norm
```

#### Metric B: Quantization Error (需要一次量化模拟)

```python
W_q4 = quantize_to_int4(W_block[i, j])
S[i, j] = torch.norm(W_block[i, j] - W_q4, p='fro')
```

#### Metric C: Hessian-Weighted Error (最精确，需要 Hessian 对角线)

```python
W_q4 = quantize_to_int4(W_block[i, j])
E = W_block[i, j] - W_q4
H_diag_block = H_diag[j*bcol:(j+1)*bcol]  # Hessian diagonal for these columns
S[i, j] = torch.sum(E ** 2 * H_diag_block.unsqueeze(0))
```

**Recommended starting point**: Metric B (quantization error). It directly measures what we want to minimize, and is cheap to compute (one INT4 quantize-dequantize pass per block).

### 1.4 Block Selection Algorithm

```python
# Compute sensitivity scores for all blocks
scores = torch.zeros(nrow, ncol)
for i in range(nrow):
    for j in range(ncol):
        block = W[i*brow:(i+1)*brow, j*bcol:(j+1)*bcol]
        block_q4 = fake_quantize_int4(block)
        scores[i, j] = torch.norm(block - block_q4, p='fro')

# Select top-k blocks for INT8
flat_scores = scores.flatten()
_, topk_indices = torch.topk(flat_scores, k=N_high)
high_precision_mask = torch.zeros(nrow * ncol, dtype=torch.bool)
high_precision_mask[topk_indices] = True
high_precision_mask = high_precision_mask.reshape(nrow, ncol)
```

---

## 2. Quantization Strategy (量化策略)

### 2.1 Per-Block Quantization with Shared Scales

**Key design decision**: How to share quantization scales across blocks.

#### Option 1: Per-Block Independent Scale (每个子矩阵独立 scale)

```
For each block (i, j):
  if high_precision_mask[i, j]:
    scale[i, j], zero[i, j] = compute_int8_scale(W_block[i, j])
  else:
    scale[i, j], zero[i, j] = compute_int4_scale(W_block[i, j])
```

- **Pros**: Maximum flexibility, each block has optimal scale
- **Cons**: Large scale storage overhead (one scale per block per row)
- **Scale tensor shape**: (nrow, ncol, brow) for per-row-per-block scales

#### Option 2: Per-Group Scale (与现有 GPTQ groupsize 对齐) — **RECOMMENDED**

```
groupsize = 128 (existing GPTQ default)
bcol should be a multiple of groupsize, OR groupsize should be a multiple of bcol

For INT4 blocks:
  Use standard GPTQ per-group scale (groupsize=128 along input dimension)
  Scale shape: (d_out, d_in // groupsize) — same as standard GPTQ

For INT8 blocks:
  Option A: Per-column symmetric scale (same as current Tail Absorb)
    scale_int8[col] = max(|W[:, col]|) / 127
  Option B: Per-block symmetric scale
    scale_int8[i, j] = max(|W_block[i, j]|) / 127
```

**Recommended**: Use existing GPTQ per-group scale for INT4 blocks. For INT8 blocks, use per-block symmetric scale (Option B), which gives a single scale per block.

#### Option 3: Row-wise Scale with Bit-width Mask

```
For each row r:
  For INT4 groups: standard GPTQ group scale
  For INT8 blocks: per-block scale
  
Metadata: high_precision_mask (nrow × ncol boolean tensor) + per-block INT8 scales
```

### 2.2 Integration with GPTQ Error Propagation

**Critical**: The GPTQ algorithm processes columns left-to-right. Submatrix-level mixed precision must be compatible with this column-wise processing.

#### Approach: Column-wise Processing with Block-aware Bit-width

During GPTQ's `fasterquant` column iteration:

```python
for col_idx in range(d_in):
    block_col = col_idx // bcol  # which block column
    
    for row_start in range(0, d_out, brow):
        block_row = row_start // brow
        row_end = min(row_start + brow, d_out)
        
        if high_precision_mask[block_row, block_col]:
            # INT8 quantization for this row segment
            q[row_start:row_end] = int8_fakequant(w[row_start:row_end])
        else:
            # INT4 quantization for this row segment (standard GPTQ)
            q[row_start:row_end] = int4_quantize(w[row_start:row_end], scale, zero, maxq)
    
    # Error propagation (standard GPTQ Hessian compensation)
    err = (w - q) / d
    W[:, col_idx+1:] -= err.unsqueeze(1) @ Hinv[col_idx, col_idx+1:].unsqueeze(0)
```

**Important**: Error propagation remains **full-matrix** (not block-local). Only the quantization bit-width decision is block-level.

### 2.3 Quantization Pipeline

```
FP16 原始权重
  → SmoothQuant(alpha=1)
  → [Phase 1] Compute block sensitivity scores (one INT4 fake-quant pass)
  → [Phase 2] Select top-5% blocks as INT8
  → [Phase 3] GPTQ with block-aware mixed precision
  → [Output] W_int4 (full matrix) + W_delta_int8 (sparse correction blocks)
```

---

## 3. Inference Computation (推理计算方案)

### 3.1 Decomposition: Dense INT4 + Sparse INT8 Correction

After quantization, the weight matrix is decomposed as:

```
W_quantized = W_int4_full + W_correction_sparse
```

Where:
- `W_int4_full`: Complete d_out × d_in matrix, ALL elements quantized to INT4
  - For INT8 blocks: the INT4 approximation (lower precision)
- `W_correction_sparse`: Sparse matrix, only non-zero in INT8 block positions
  - `W_correction_sparse[block] = W_int8[block] - W_int4[block]`
  - This is the **precision gain** from using INT8 instead of INT4

### 3.2 Matrix Multiplication

```
Y = W_quantized @ X
  = W_int4_full @ X + W_correction_sparse @ X
  = Y_base + Y_correction
```

- `Y_base = W_int4_full @ X`: Standard dense INT4 GEMM (can use existing INT4 kernels)
- `Y_correction = W_correction_sparse @ X`: Sparse block GEMM (only 5% blocks are non-zero)

### 3.3 Sparse Block GEMM Implementation Options

#### Option A: Block-Sparse GEMM (推荐)

```python
# Precompute: store correction blocks in BSR (Block Sparse Row) format
# correction_values: (N_high, brow, bcol) — the non-zero blocks
# correction_indices: (N_high, 2) — (block_row, block_col) positions

Y_correction = torch.zeros(d_out, seq_len)
for idx in range(N_high):
    br, bc = correction_indices[idx]
    row_slice = slice(br * brow, (br + 1) * brow)
    col_slice = slice(bc * bcol, (bc + 1) * bcol)
    Y_correction[row_slice] += correction_values[idx] @ X[col_slice]
```

#### Option B: Gather-Scatter with Dense Sub-GEMM

```python
# Gather relevant X columns for each block column group
# Perform batched small GEMM
# Scatter-add results to Y
```

#### Option C: Custom CUDA Kernel (最高性能)

Use Triton or CUTLASS to implement a fused block-sparse GEMM kernel.

### 3.4 Overhead Analysis

```
Budget = 5% of blocks
Dense INT4 GEMM: O(d_out × d_in × seq_len) — full cost
Sparse correction GEMM: O(0.05 × d_out × d_in × seq_len) — 5% of full cost

Total overhead ≈ 5% additional compute
Memory overhead: 5% × (8-4)/4 = 5% additional weight storage (INT8 correction blocks)
```

**Practical considerations**:
- Block-sparse GEMM has lower arithmetic intensity than dense GEMM
- Memory access patterns for sparse blocks may not be optimal
- For small budget (5%), the overhead should be manageable
- PyTorch's `torch.sparse_bsr_tensor` supports BSR format since PyTorch 2.0

---

## 4. Data Structures & Storage Format (数据结构)

### 4.1 Quantized Weight Representation

```python
@dataclass
class SubmatrixMixedPrecisionWeight:
    """Per-layer quantized weight storage."""
    
    # --- Dense INT4 base matrix ---
    w_int4: torch.Tensor          # (d_out, d_in), dtype=int8 (packed 4-bit)
    scale_int4: torch.Tensor      # (d_out, d_in // groupsize), dtype=float16
    zero_int4: torch.Tensor       # (d_out, d_in // groupsize), dtype=float16
    
    # --- Sparse INT8 correction ---
    correction_values: torch.Tensor   # (N_high, brow, bcol), dtype=int8
    correction_scales: torch.Tensor   # (N_high,) or (N_high, brow), dtype=float16
    correction_indices: torch.Tensor  # (N_high, 2), dtype=int32 — (block_row, block_col)
    
    # --- Metadata ---
    block_shape: Tuple[int, int]      # (brow, bcol)
    grid_shape: Tuple[int, int]       # (nrow, ncol)
    budget_ratio: float               # 0.05
    n_high_blocks: int                # N_high
    
    # --- Block selection info (for analysis) ---
    sensitivity_scores: torch.Tensor  # (nrow, ncol), dtype=float32
    high_precision_mask: torch.Tensor # (nrow, ncol), dtype=bool
```

### 4.2 Serialization Format

```python
# Save format: single .pt file per layer (or bundled per model)
save_dict = {
    "w_int4_packed": ...,           # packed 4-bit weights
    "scale_int4": ...,
    "zero_int4": ...,
    "correction_values_int8": ...,  # INT8 correction blocks
    "correction_scales": ...,
    "correction_row_indices": ...,  # block row indices
    "correction_col_indices": ...,  # block column indices
    "block_shape": (brow, bcol),
    "budget_ratio": 0.05,
    "sensitivity_scores": ...,      # optional, for analysis
}
```

---

## 5. Implementation Plan — New Files & Modifications

### 5.1 New Files to Create

| File | Purpose |
|------|---------|
| `gptq_submatrix_mixed.py` | Core GPTQ class: `GPTQSubmatrixMixed`, inherits from `GPTQ` |
| `qwen3_gptq_submatrix_mixed.py` | Entry script: model loading, calibration, quantization pipeline |
| `submatrix_utils.py` | Utility functions: block partitioning, sensitivity scoring, mask generation |

### 5.2 Core Class: GPTQSubmatrixMixed

```python
class GPTQSubmatrixMixed(GPTQ):
    """
    GPTQ variant with submatrix-level mixed precision.
    
    Key differences from standard GPTQ:
    1. Before fasterquant: compute block sensitivity scores and select INT8 blocks
    2. During fasterquant: use block-aware bit-width for each (row_segment, col) pair
    3. After fasterquant: decompose into W_int4_full + W_correction_sparse
    """
    
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        # --- New parameters ---
        block_shape=(128, 128),    # (brow, bcol) submatrix dimensions
        budget_ratio=0.05,         # fraction of blocks to use INT8
        sensitivity_metric="quant_error",  # "weight_norm", "quant_error", "hessian_weighted"
    ):
        ...
```

### 5.3 CLI Parameters (New)

```
--enable-submatrix-mixed     Enable submatrix-level mixed precision
--block-rows INT             Block row dimension (default: 128)
--block-cols INT             Block column dimension (default: 128)
--budget-ratio FLOAT         Fraction of blocks for INT8 (default: 0.05)
--sensitivity-metric STR     Block selection metric: weight_norm|quant_error|hessian_weighted
```

### 5.4 Key Implementation Details

#### 5.4.1 Block Sensitivity Scoring (Phase 1)

```python
def compute_block_sensitivity(
    W: torch.Tensor,           # (d_out, d_in) float weight
    block_shape: Tuple[int, int],
    metric: str,
    quantizer: Quantizer = None,
    H_diag: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute sensitivity score for each block.
    
    Returns:
        scores: (nrow, ncol) tensor of sensitivity scores
    """
    brow, bcol = block_shape
    d_out, d_in = W.shape
    nrow = math.ceil(d_out / brow)
    ncol = math.ceil(d_in / bcol)
    scores = torch.zeros(nrow, ncol, device=W.device)
    
    for i in range(nrow):
        for j in range(ncol):
            r0, r1 = i * brow, min((i + 1) * brow, d_out)
            c0, c1 = j * bcol, min((j + 1) * bcol, d_in)
            block = W[r0:r1, c0:c1]
            
            if metric == "weight_norm":
                scores[i, j] = torch.norm(block, p='fro')
            elif metric == "quant_error":
                block_q = fake_quantize_int4_block(block, quantizer)
                scores[i, j] = torch.norm(block - block_q, p='fro')
            elif metric == "hessian_weighted":
                block_q = fake_quantize_int4_block(block, quantizer)
                err = block - block_q
                h_diag_slice = H_diag[c0:c1]
                scores[i, j] = torch.sum(err ** 2 * h_diag_slice.unsqueeze(0))
    
    return scores
```

#### 5.4.2 Modified fasterquant Column Loop

```python
# Inside fasterquant, for each column col_idx:
for i in range(count):
    w = W1[:, i]
    d = Hinv1[i, i]
    col_idx = i1 + i
    block_col = col_idx // bcol
    
    # Determine per-row-segment bit-width
    q = torch.zeros_like(w)
    for block_row in range(nrow):
        r0 = block_row * brow
        r1 = min((block_row + 1) * brow, self.rows)
        
        if high_precision_mask[block_row, block_col]:
            # INT8 for this segment
            q[r0:r1] = int8_fakequant_segment(w[r0:r1])
        else:
            # INT4 for this segment (use current group scale)
            q[r0:r1] = quantize(
                w[r0:r1].unsqueeze(1),
                self.quantizer.scale[r0:r1],
                self.quantizer.zero[r0:r1],
                self.quantizer.maxq,
            ).flatten()
    
    # Standard GPTQ error propagation (unchanged)
    Q1[:, i] = q
    Losses1[:, i] = (w - q) ** 2 / d ** 2
    err1 = (w - q) / d
    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    Err1[:, i] = err1
```

#### 5.4.3 Post-Quantization Decomposition

```python
def decompose_mixed_precision(
    Q: torch.Tensor,              # (d_out, d_in) quantized weight (FakeQuant float)
    high_precision_mask: torch.Tensor,  # (nrow, ncol) bool
    block_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose quantized weight into dense INT4 + sparse INT8 correction.
    
    Returns:
        W_int4_full: (d_out, d_in) — all blocks quantized to INT4
        correction_values: list of (brow, bcol) tensors — INT8 - INT4 differences
        correction_indices: (N_high, 2) — block positions
    """
    ...
```

---

## 6. Hyperparameter Recommendations

### 6.1 Block Shape Selection

| d_out × d_in | Recommended (brow, bcol) | N_total | N_high (5%) |
|---------------|--------------------------|---------|-------------|
| 2560 × 2560 | (128, 128) | 400 | 20 |
| 2560 × 6912 | (128, 128) | 1080 | 54 |
| 6912 × 2560 | (128, 128) | 1080 | 54 |
| 2560 × 2560 | (64, 64) | 1600 | 80 |

**Constraint**: `bcol` should be a divisor of `groupsize` (128) or vice versa, to align with GPTQ group boundaries.

**Recommended default**: `(brow=128, bcol=128)` — aligns with GPTQ groupsize=128.

### 6.2 Budget Ratio

- **5%** (default): ~5% additional compute and memory overhead
- **2%**: Minimal overhead, may miss some important blocks
- **10%**: More coverage, but higher overhead

### 6.3 Sensitivity Metric

- Start with `quant_error` (Metric B)
- If results are promising, try `hessian_weighted` (Metric C) for potential improvement
- `weight_norm` (Metric A) as fast baseline

---

## 7. Experiment Design

### 7.1 Ablation Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| block_shape | (64,64), (128,128), (256,256) | Block granularity |
| budget_ratio | 0.02, 0.05, 0.10 | INT8 budget |
| sensitivity_metric | weight_norm, quant_error, hessian_weighted | Selection strategy |
| act_order | ON, OFF | GPTQ ordering |
| smooth | ON, OFF | SmoothQuant preprocessing |

### 7.2 Evaluation

- WikiText-2 PPL (Anone, A8, A4g128, A4g128+down:int8)
- Compare against V7 best: smooth_ta_r16 (PPL=10.3428)
- Compare against GPTQ 4-bit raw baseline (PPL=10.3845)

---

## 8. Open Questions & Design Decisions

### Q1: Block selection — before or after act-order permutation?

**Option A**: Compute sensitivity on original column order, then apply act-order permutation.
- Pro: Sensitivity reflects true weight structure
- Con: After permutation, block boundaries may not align with outlier clusters

**Option B**: Apply act-order permutation first, then compute sensitivity on permuted matrix.
- Pro: Block selection operates on the same column order as GPTQ processing
- Con: Permutation may scatter outliers across blocks

**Recommendation**: Option B — compute sensitivity after act-order permutation, so that block selection is consistent with GPTQ's processing order.

### Q2: How to handle blocks at matrix boundaries (padding)?

If d_out or d_in is not divisible by brow/bcol, the last row/column of blocks will be smaller.

**Recommendation**: Allow variable-size boundary blocks. Sensitivity scoring and quantization should handle arbitrary block sizes.

### Q3: Should block selection be per-layer or global?

**Per-layer** (recommended for V9): Each layer independently selects its top-5% blocks.
**Global**: Pool all blocks across all layers, select global top-5%. Some layers may get more INT8 blocks than others.

### Q4: actorder permutation interaction with block structure

When `actorder=True`, columns are permuted by Hessian diagonal importance. This changes which columns fall into which block. The block sensitivity scoring should be done **after** permutation.

After GPTQ completes, use `invperm` to restore original column order. The high_precision_mask must also be permuted back.

---

## 9. Compatibility Notes

### 9.1 With Existing Codebase

- New files only; no modification to existing `qwen3_gptq.py`, `gptq_tail_absorb.py`, etc.
- Reuses `qwen3_gptq.py` utility functions: `get_qwen3()`, `get_wikitext2_or_fallback_loader()`, etc.
- Reuses `gptq/gptq.py` base class and `gptq/quant.py` quantizer

### 9.2 With Benchmark Pipeline

- Output format compatible with `benchmark/eval_ppl.py`
- FakeQuant mode: quantized weights stored as float16 (same as V7/V8)
- Metadata JSON includes block selection statistics

### 9.3 With SmoothQuant

- Supports `--init-state-dict` for loading SmoothQuant preprocessed weights
- Pipeline: Smooth → Block Sensitivity → GPTQ Mixed Precision
