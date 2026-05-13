# Fused Kernel Dataflow 图的正确性审查

> 目的：按论文审稿视角，把用户提供的 fused-kernel 数据流图（左：latency 柱状分解；右：两个 kernel 的 dataflow 方框图）与真实的 CUDA 源码做对比，给出一份**事实层面**的修正清单。
>
> 参考源码：
> - [`csrc/activation_quant/activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu)
> - [`csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)
> - [`csrc_naive/*_naive.cu`](../cuda_kernel/csrc_naive/)

---

## 一句话结论 (TL;DR)

整体拓扑（两个 fused kernel、`Quantize → Dense/Sparse → Reduce`）**方向是对的**，但图里有 **4 处与代码事实不符**的地方，审稿人很容易抓；另外有 **2 处加分项**可以让 fusion / pipeline 的故事讲得更清楚。

---

## ❌ 问题 1 — "Layout Pack" 不是一个独立 stage

**图里画的是什么。** 在 `Fused Kernel 1` 里，`Quantize` 和 `Layout Pack` 被画成两个并列的子方框。

**代码事实是什么。**
[`activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu) 的头部 contract 里写得非常清楚：

> "Pass 1 (max-abs) and Pass 2 (**quant+pack+sum**) share one CTA-local reduction shim; we do *not* split kernels because the work per token is small enough that a second kernel launch dominates."

也就是说**量化、SINT4 打包、per-group sum** 是**同一个 kernel、同一个 pass**里完成的，`pack` 不是独立 stage。仓库全局搜索也确认：**不存在任何运行时的 `layout_pack` / `pack_kernel` / `repack` device function**；整份代码里出现的 `LayoutPack` / `layout` 字样全部来自 CUTLASS 的 host 侧类型别名（`cutlass::layout::RowMajor` 等）。

**审稿风险。** 把 `Pack` 单独画出来恰恰**削弱了**你想表达的 "kernel 1 is fused" 的主张。审稿人会问：*"既然 Pack 是独立 box，为什么外框要叫 Fused？"*

**修法（二选一）：**
- **(A) 推荐。** 把两个子方框合并成一个：`Quantize (fused pack + per-group Σ)`。最贴近代码，叙事也最干净。
- **(B) 如果坚持要画两个子步骤。** 把 `Layout Pack` 改名为 `Pack + Σ`，并用**虚线**画子方框，表示它们是**同一个 kernel 内的两个 pass**，不是两个独立 kernel。

---

## ❌ 问题 2 — 箭头标签 `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ` 的方向/语义都不对

**图里画的是什么。** 一条箭头从 `Dense Compute` **出来**指向 `Reduce Add`，标签写的是 `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ`。Sparse 分支上对称地写着 `Q_X̂ · Q_R̂, s_X̂, s_R̂`。

**代码事实是什么。**
[`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)：

- `Dense Compute` 的**输入**才是 `Q_X̂`（SINT4 激活 tile）、`Q_Ŵ`（UINT4 dense 权重 tile），再加上 dense 分支的反量化系数 `(s_Ŵ, z_Ŵ, sum_X)`。
- `Dense Compute` 的**输出**是一块**已经反量化到 fp32 的 partial tile**（对应那行 `y_fp[im][in_sub][r] += corrected * s;`），**不是**两个仍然量化的张量的乘积。kernel 里**根本不存在**一个叫 `Q_X̂ · Q_Ŵ` 的命名张量：MMA 算完之后，`z · sum_X` 减法和 `· s` 乘法都在寄存器里就地完成，所以离开 `Dense Compute` 的其实已经是 `Y_dense (fp32)`。
- Sparse 分支**没有 zero-point**（`W_high` 是 sign-extended s4，直接喂给 `mma.s4.s4`，参考 `sparse_gemm_naive.cu` 的注释块），所以 sparse 箭头标签**绝对不能**带 `z_R̂`。

**修法（二选一，但全图要风格一致）：**
- **(A) 把标签放在*输入*边上。** `Quantize` → `Dense Compute` 的箭头上写 `Q_X̂, s_X̂`；另一组箭头从 HBM 进入 `Dense Compute`，承载 `Q_Ŵ, s_Ŵ, z_Ŵ`。`Dense Compute` 的输出箭头就写成 `Y_dense (fp32 partial)`。
- **(B) 把标签放在*输出*边上。** 把 `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ` 改成 `Y_dense (fp32)`，也就是用反量化后的语义命名。Sparse 分支对应改为 `Y_sparse = 16 · acc · s_R̂`（这个 `16×` 来自 bit 级的 high-nibble split，见 naive sparse kernel 里的 `// Per-block fold` 注释）。

---

## ❌ 问题 3 — `Reduce Add` 画成与 Dense/Sparse 同级的方框，与左图柱状图互相矛盾

**图里画的是什么。** 在 `Fused Kernel 2` 里，`Reduce Add` 被画成一个完整大小的橙色方框，和 `Dense Compute`、`Sparse Compute` **并列**。

**代码事实是什么。** `Y_low` 和 `Y_high` **从不实体化**——它们在 kernel 里根本不是被命名的张量。Dense 和 Sparse 两条分支都把结果累加进**同一组 fp32 寄存器**：

```cpp
// dense 分支
y_fp[im][in_sub][r] += corrected * s;                        // R27
// sparse 分支
y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;
// 收尾
y_fp[im][in_sub][r] *= sxn_cache[in_sub][r & 1];
// 然后一次 HBM 写出 Y_total
```

所以 "Reduce Add" 是一条**作用于单个输出元素的 `+=` 寄存器指令**，它**不占单独的 cycle**，也不是一个 stage。

**为什么这是最该修的一处。** 左边的柱状图其实已经讲对了故事：Naive 柱子里有一段橙色的 `Reduce Add` (8.2 μs)，而 Ours 柱子里**没有**——橙色段**消失了**，被绿色的 `Fused` 段替代。这个"消失"恰恰是你最强的 fusion claim。**右图**如果把 `Reduce Add` 重新画成一个平级方框，**就会和左图自相矛盾**，审稿人一对照就会觉得两张子图不一致。

**修法：**
- 把 `Reduce Add` 缩小成 `Fused Kernel 2` 右下角的一个**虚线小标注**，标签写 `Reduce Add (in-register +=)` 或者 `Σ (registers)`。
- 或者干脆不画 box，直接把 `+=` 写在 `Dense Compute` 和 `Sparse Compute` 汇流到 `Y_total` 的那条箭头上。

无论哪种画法，目标都是让审稿人一眼能把**左图里消失的橙色段**与**右图里缩小/折叠的小方框**对应上。

---

## ❌ 问题 4 — 两条独立的 `Q_X̂, s_X̂` 箭头会让人误以为激活被读了两次 HBM

**图里画的是什么。** Fused Kernel 1 输出的 `Q_X̂, s_X̂` 分成两条独立箭头，分别进入 `Dense Compute` 和 `Sparse Compute`。

**代码事实是什么。** Fused Kernel 2 里，`X_s4` 通过 `cp.async` **一次性**从 HBM 读进 shared-memory 环形 buffer `sX[kStages][kBn][bytes_per_group]`，dense 和 sparse 两个 MMA pass **共享**这同一块 SMEM tile。`scale_x` 同样只进一份 `__shared__ __half s_scale_x[kBn]`，每个 CTA 只 load 一次。

**审稿风险。** 照现在这样画，读者会默认激活被从 HBM 读了两次——这会**直接抹掉** fusion 在 HBM traffic 上的那部分贡献。

**修法。** 只画**一条**箭头进入 `Fused Kernel 2` 外框左边缘（标 `cp.async → SMEM`），**框内**再分叉到 `Dense Compute` 和 `Sparse Compute`。这样才符合 SMEM 复用的事实。

---

## 💡 加分项 1 — 把 `cp.async` 双缓冲放到图上

`Fused Kernel 2` 这块才是这个 kernel 真正跟 Atom / QServe 等 W4A4 baseline 拉开差距的地方（那些 baseline 大多没有 `cp.async` overlap）。建议在外框的左边缘加一条**虚线反向小箭头**，标签 `cp.async (kStages = 2)`，再配一句 caption（"the HBM→SMEM transfer of tile g+1 overlaps with the Tensor-Core MMA of tile g"），一张图就能承担不少任务——把 2.09× 的加速**诚实地**拆分成 "fusion 贡献 + pipeline 贡献" 两部分，而不是全甩给 fusion。

## 💡 加分项 2 — 左右两图颜色对齐

- **已经做对的部分。** 左图柱状图和右图方框图在 `Quantize` / `Dense` / `Sparse` / `Reduce Add` 上色调已经一致。
- **唯一的小缺口。** 左图里新增的粉色 `Fused Compute` 段（21.7 μs）在右图里没有同名 box——它真正的视觉对应物是**`Fused Kernel 2` 的粉色外框本身**，不是内部某个 box。Caption 里补一句即可：
  > "The pink `Fused Kernel 2` box in the right panel corresponds to the pink `Fused Compute` segment in the left panel (21.73 μs)."

---

## 📝 优先级清单

| # | 问题 | 当前图怎么画的 | 建议改法 | 优先级 |
|---|---|---|---|---|
| 1 | `Layout Pack` 不是独立 stage | `Quantize → Layout Pack` 两个子框 | 合并成一个 `Quantize + Pack + Σ` box | ⭐⭐⭐ 必改 |
| 2 | `Q_X̂·Q_Ŵ, s_X̂, s_Ŵ` 箭头方向/语义不对 | 画在 `Dense Compute` 的**出**边 | 要么挪到**入**边，要么把出边标签改成 `Y_dense (fp32)` | ⭐⭐⭐ 必改 |
| 3 | `Reduce Add` 平级方框与左图柱状图矛盾 | 与 Dense/Sparse 并列的橙色大方框 | 缩成虚线小标注 `Reduce Add (in-register +=)` | ⭐⭐⭐ 必改 |
| 4 | 两条独立的 `Q_X̂, s_X̂` 箭头暗示两次 HBM 读 | 在进入 Fused Kernel 2 之前就分叉 | 一条箭头进外框，框内再分叉 | ⭐⭐ 建议改 |
| 5 | 没有体现 `cp.async` 双缓冲 | 无 | FK2 左边缘加虚线 `cp.async (kStages=2)` 标注 | ⭐ 可选加分 |
| 6 | Sparse 分支不能带 zero-point | 需检查 | Sparse 标签只能有 `Q_R̂, s_R̂`，不能有 `z_R̂` | ⭐ 校对项 |

---

## 📚 附录 — 代码证据（逐条对应）

- **Fused Kernel 1 内 pack 和 Σ 都被融合进量化 kernel** —
  [`activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu)
  头部 contract 的原话："Pass 1 (max-abs) and Pass 2 (quant+pack+sum) share one CTA-local reduction shim; we do *not* split kernels."
- **Dense 分支的 dequant 在寄存器内完成（没有 `Q_X̂·Q_Ŵ` 这个命名张量）** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)，
  `y_fp[im][in_sub][r] += corrected * s;`（R27 注释那一行）。
- **Sparse 分支只有 scale，没有 zero-point** —
  [`sparse_gemm_naive.cu`](../cuda_kernel/csrc_naive/sparse_gemm_naive.cu)
  里的注释块："W_high has no zero-point (sign-extended s4), which is exactly what mma.s4.s4 reads natively."
- **Reduce Add 是寄存器级 `+=`，不是 stage** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)
  里 dense 那行（R27）和 sparse 那行
  `y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;`
  累加的**都是同一组** `y_fp[...]` 寄存器；收尾的 `y_fp[...] *= sxn_cache[...]` 产生唯一一份 `Y_total` 张量。
- **`X_s4` 在每个 CTA 只被 `cp.async` 读一次，Dense / Sparse 两个 MMA pass 共享 SMEM** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)，
  shared 声明
  `__shared__ alignas(16) uint8_t sX[kStages][kBn][bytes_per_group];`
  加上 prologue 里的 `s_scale_x[kBn]` 一次性 load。

---

## 下一步选项

告诉我要走哪一条，我直接出一版改好的图：

- **(a) matplotlib 重绘** — 沿用 [`plot_latency_breakdown.py`](./plot_latency_breakdown.py) 的色板，让柱状图和 dataflow 图看起来像一张图的两部分。
- **(b) TikZ / LaTeX 重绘** — 对齐 [`pipeline_spacetime.tex`](../cuda_kernel/docs/figures/pipeline_spacetime/pipeline_spacetime.tex) 的风格，可以直接 `\input` 进论文，和现有的 space-time 图放在一起。

另外需要你告诉我一件事：**当前这张图是用什么工具画的**（PPT / draw.io / matplotlib / TikZ）？这决定了我是在原图上打 patch，还是从头重绘。
