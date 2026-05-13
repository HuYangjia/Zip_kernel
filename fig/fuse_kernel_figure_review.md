# Fuse Kernel Dataflow Diagram — Correctness Review

> Purpose: a paper-grade review of the user's fused-kernel dataflow figure
> (left: latency breakdown bar chart; right: two-kernel dataflow diagram)
> against the actual CUDA source, with a prioritized fix list.
>
> Sources consulted:
> - [`csrc/activation_quant/activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu)
> - [`csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)
> - [`csrc_naive/*_naive.cu`](../cuda_kernel/csrc_naive/)

---

## TL;DR

The overall topology (two fused kernels, `Quantize → Dense/Sparse → Reduce`)
is **directionally correct**, but the diagram has **4 fact-level mismatches**
with the CUDA source that a careful reviewer will flag, plus **2
nice-to-have improvements** that would make the fusion/pipeline story
land harder.

---

## ❌ Issue 1 — "Layout Pack" is *not* a separate stage

**What the figure shows.** Inside `Fused Kernel 1`, `Quantize` and
`Layout Pack` are drawn as two sibling boxes.

**What the code actually does.**
[`activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu)
states explicitly in its header contract:

> "Pass 1 (max-abs) and Pass 2 (**quant+pack+sum**) share one CTA-local
> reduction shim; we do *not* split kernels because the work per token
> is small enough that a second kernel launch dominates."

So *quantization*, *SINT4 byte packing*, and *per-group sum* are **one
pass of one kernel**, not separate stages. A repo-wide search confirms
there is **no runtime `layout_pack` / `pack_kernel` / `repack` device
function**; the only `LayoutPack`/`layout` tokens in the tree come from
CUTLASS host-side type aliases (`cutlass::layout::RowMajor`, etc.).

**Reviewer risk.** Splitting `Pack` off visually weakens the very claim
the figure is trying to make ("kernel 1 is fused"). A reviewer will ask:
*"If pack is its own box, why is the outer box labelled Fused?"*

**Suggested fix (pick one):**
- **(A) Preferred.** Collapse the two inner boxes into one:
  `Quantize (fused pack + per-group Σ)`. Closest to the code, cleanest
  story.
- **(B) If you insist on showing two sub-steps.** Rename `Layout Pack`
  to `Pack + Σ`, and draw a **dashed** sub-box to signal these are
  **two passes inside one kernel**, not two kernels.

---

## ❌ Issue 2 — Arrow label `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ` has wrong direction / wrong semantics

**What the figure shows.** An arrow exits `Dense Compute` toward
`Reduce Add` carrying the label `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ`. A symmetric
label `Q_X̂ · Q_R̂, s_X̂, s_R̂` appears on the sparse branch.

**What the code actually does.**
[`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu):

- **Inputs** to `Dense Compute` are `Q_X̂` (SINT4 activation tile),
  `Q_Ŵ` (UINT4 dense weight tile), plus the *dense* dequant coefs
  `(s_Ŵ, z_Ŵ, sum_X)`.
- **Output** of `Dense Compute` is an **already-dequantized fp32 partial
  tile** (the line `y_fp[im][in_sub][r] += corrected * s;`), *not* the
  product of two still-quantized tensors. There is no named tensor
  `Q_X̂ · Q_Ŵ` anywhere in the kernel; once the MMA finishes, the
  `z · sum_X` subtract and `· s` multiply happen in-register, so what
  leaves `Dense Compute` is already `Y_dense (fp32)`.
- The sparse branch has **no zero-point** (`W_high` is sign-extended
  s4 fed directly to `mma.s4.s4`, see `sparse_gemm_naive.cu` comment
  block), so any sparse arrow label that implies a `z_R̂` is doubly
  wrong.

**Suggested fix (pick one consistent policy):**
- **(A) Put the label on the *input* edge.** Arrow from `Quantize`
  into `Dense Compute` carrying `Q_X̂, s_X̂`; a second set of arrows
  from HBM into `Dense Compute` for `Q_Ŵ, s_Ŵ, z_Ŵ`. The output arrow
  of `Dense Compute` then carries simply `Y_dense (fp32 partial)`.
- **(B) Keep the label on the *output* edge.** Change
  `Q_X̂ · Q_Ŵ, s_X̂, s_Ŵ` → `Y_dense (fp32)`, i.e. use the post-dequant
  semantic name. For the sparse branch correspondingly:
  `Y_sparse = 16 · acc · s_R̂` (the `16×` comes from the bit-level
  high-nibble split; see the `// Per-block fold` comment in the naive
  sparse kernel).

---

## ❌ Issue 3 — `Reduce Add` drawn as a peer-level box contradicts the left bar chart

**What the figure shows.** In `Fused Kernel 2`, `Reduce Add` is a
full-sized orange box sitting as a **peer** of `Dense Compute` and
`Sparse Compute`.

**What the code actually does.** `Y_low` and `Y_high` **never
materialize** — they are not even named tensors inside the kernel.
Both branches accumulate into the **same** fp32 register tile:

```cpp
// dense branch
y_fp[im][in_sub][r] += corrected * s;                        // R27
// sparse branch
y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;
// final
y_fp[im][in_sub][r] *= sxn_cache[in_sub][r & 1];
// then a single HBM write into Y_total
```

So "Reduce Add" is a **per-output-element `+=` register instruction**,
not a stage with its own cycles.

**Why this is the most important one to fix.** The **left** panel
(the bar chart) already tells the correct story: Naive has an orange
`Reduce Add` segment (8.2 μs), and Ours does **not** — the orange
segment is *gone*, replaced by the green `Fused` segment. That
disappearance is your strongest fusion claim. The **right** panel
reintroducing `Reduce Add` as a peer box directly contradicts the
left panel, so the two figures will read as inconsistent.

**Suggested fix:**
- Shrink `Reduce Add` to a small **dashed** annotation tucked into the
  bottom-right of `Fused Kernel 2`, labelled
  `Reduce Add  (in-register +=)` or `Σ (registers)`.
- Alternatively, draw no box at all and write the `+=` directly on the
  arrow where `Dense Compute` and `Sparse Compute` merge into
  `Y_total`.

Either way, a reviewer should instantly match "the missing orange
segment on the left" with "the collapsed tiny box on the right".

---

## ❌ Issue 4 — Two independent arrows from `Q_X̂, s_X̂` imply two HBM reads of the activation

**What the figure shows.** From the `Q_X̂, s_X̂` output of Fused Kernel 1,
two independent arrows fan out into `Dense Compute` and `Sparse Compute`.

**What the code actually does.** Fused Kernel 2 reads `X_s4` from HBM
**once** via `cp.async` into the shared-memory ring buffer
`sX[kStages][kBn][bytes_per_group]`, and the dense and sparse MMA
passes **share that same SMEM tile**. `scale_x` likewise lives in one
`__shared__ __half s_scale_x[kBn]` loaded once per CTA.

**Reviewer risk.** As drawn, a reader will assume the activation is
loaded from HBM twice — which would undercut the HBM-traffic part of
the fusion story.

**Suggested fix.** Route **one** arrow into the left edge of the
`Fused Kernel 2` outer box (labelled `cp.async → SMEM`), and only
**inside** the outer box fan it out to `Dense Compute` and
`Sparse Compute`. That visually matches the SMEM-reuse fact.

---

## 💡 Nice-to-have 1 — surface `cp.async` double-buffering on the figure

The `Fused Kernel 2` box is where this kernel actually differs from
Atom / QServe / other W4A4 baselines (most of them have no
`cp.async` overlap). A small **dashed** back-arrow on the left edge of
the outer box, labelled `cp.async (kStages = 2)`, plus a single caption
line ("HBM→SMEM transfer of tile g+1 overlaps with Tensor-Core MMA of
tile g") does a lot of work for this figure — it lets you **honestly
decompose** the 2.09× into "fusion savings + pipeline savings" rather
than attributing everything to fusion.

## 💡 Nice-to-have 2 — colour alignment between the two panels

- **Good.** Left bar chart and right dataflow boxes already share the
  same hues for `Quantize` / `Dense` / `Sparse` / `Reduce Add`.
- **Small gap.** The pink `Fused Compute` segment on the left (21.7 μs)
  has no named peer on the right — its true visual peer is the **pink
  outer border of `Fused Kernel 2` itself**, not any inner box.
  One caption line fixes this:
  > "The pink `Fused Kernel 2` box in the right panel corresponds to
  > the pink `Fused Compute` segment in the left panel (21.73 μs)."

---

## 📝 Prioritized change list

| # | Issue | Current figure | Suggested fix | Priority |
|---|---|---|---|---|
| 1 | `Layout Pack` is not a separate stage | `Quantize → Layout Pack` two boxes | Merge into one `Quantize + Pack + Σ` box | ⭐⭐⭐ must-fix |
| 2 | Arrow label `Q_X̂·Q_Ŵ, s_X̂, s_Ŵ` — wrong direction / semantics | Output edge of `Dense Compute` | Either move to **input** edge, or change output label to `Y_dense (fp32)` | ⭐⭐⭐ must-fix |
| 3 | `Reduce Add` peer-level box contradicts left bar chart | Full orange sibling box | Shrink to dashed inline note `Reduce Add (in-register +=)` | ⭐⭐⭐ must-fix |
| 4 | Two independent `Q_X̂, s_X̂` arrows imply two HBM reads | Fan-out before entering Fused Kernel 2 | One arrow into the outer box, fan-out inside | ⭐⭐ should-fix |
| 5 | `cp.async` double-buffering not visible | No indication | Add dashed `cp.async (kStages=2)` marker on FK2's left edge | ⭐ nice-to-have |
| 6 | Sparse branch must not carry a zero-point symbol | Check label | Sparse label: `Q_R̂, s_R̂` only (no `z_R̂`) | ⭐ sanity-check |

---

## 📚 Appendix — ground-truth evidence (verbatim code pointers)

- **Fused pack + Σ inside the quant kernel** —
  [`activation_quant.cu`](../cuda_kernel/csrc/activation_quant/activation_quant.cu),
  header contract lines "Pass 1 (max-abs) and Pass 2 (quant+pack+sum)
  share one CTA-local reduction shim; we do *not* split kernels".
- **Dense branch in-register fold (no `Q_X̂·Q_Ŵ` tensor ever written out)** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu),
  line containing `y_fp[im][in_sub][r] += corrected * s;` (R27 comment).
- **Sparse branch has only a scale, no zero-point** —
  [`sparse_gemm_naive.cu`](../cuda_kernel/csrc_naive/sparse_gemm_naive.cu),
  block comment "W_high has no zero-point (sign-extended s4), which is
  exactly what mma.s4.s4 reads natively."
- **Reduce Add is in-register `+=`, not a stage** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu),
  the dense line (R27) and the sparse line
  `y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;`
  both target the *same* `y_fp[...]` register tile; the final writeback
  `y_fp[...] *= sxn_cache[...]` produces a single `Y_total` tensor.
- **`X_s4` loaded once per CTA via `cp.async` and shared between dense
  and sparse MMA passes** —
  [`fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu),
  shared declaration
  `__shared__ alignas(16) uint8_t sX[kStages][kBn][bytes_per_group];`
  plus `s_scale_x[kBn]` loaded once in the prologue.

---

## Next-step options

Tell me which route you prefer and I will produce a corrected version:

- **(a) matplotlib redraw** — reuse the palette from
  [`plot_latency_breakdown.py`](./plot_latency_breakdown.py) so the
  bar chart and the dataflow diagram look like one figure.
- **(b) TikZ/LaTeX redraw** — match the style of
  [`pipeline_spacetime.tex`](../cuda_kernel/docs/figures/pipeline_spacetime/pipeline_spacetime.tex)
  so the dataflow figure can be `\input`'d directly into the paper
  alongside the existing space-time diagram.

Also useful to know: what tool produced the current figure
(PowerPoint / draw.io / matplotlib / TikZ)?  That determines whether
I patch it in place or re-render from scratch.
