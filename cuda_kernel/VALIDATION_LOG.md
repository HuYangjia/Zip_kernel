## `cuda_kernel` Validation Log

RTX 4090 / SM89 / torch 2.8.0+cu126 / triton 3.4.0.  One-host
experiment journal: every run, decision, and delta lives here so we
can resume after a context switch without re-deriving history.

All timestamps in UTC+8 (server `autodl`).

---

## Run 2026-04-24 11:52: first JIT + first parity attempt

### Build

- Installed `ninja==1.13.0` in conda env `zip` (was missing).
- `HKUST_V9_CUDA_VERBOSE=1 python -c "from kernel.cuda_kernel import ops"`
  compiled all 5 TU (bindings + 4 kernels) in **118.8 s** against
  `arch=compute_89,code=sm_89`.  All 4 Python wrappers report `OK`.
- Register / shared-memory footprint per kernel (ptxas summary):

  | Kernel              | widest kBn/kBt | regs | smem (B) | spill |
  |---------------------|---------------:|-----:|---------:|------:|
  | activation_quant    | kBt=32         |  36  |   1184   | 0     |
  | dense_gemm          | kBn=64         | 128  |   4480   | 0     |
  | sparse_gemm         | kBn=64         | 128  |   4224   | 0     |
  | fused_dense_sparse  | kBn=64         | 238  |   4480   | 0     |

  No spills anywhere.  `fused` at 238 regs caps occupancy at ~1 block
  per SM; flagged as future optimisation target but not blocking.

### First parity run

Command:

```bash
python -m pytest kernel/cuda_kernel/tests/test_parity.py -x -rA --tb=short
```

Result:

```
test_activation_quant_parity[identity-1-4096]       PASSED
test_activation_quant_parity[identity-16-4096]      FAILED
  CUDA error: invalid configuration argument
```

### Root cause

`activation_quant.cu::launch` dispatched `kBt=16` for `T in (4, 16]`
and `kBt=32` for larger T.  Block dimension is
`(kLanesPerGroup=128, kBt, 1)` → `128 * 16 = 2048` threads, which
exceeds the **1024 threads/block hardware limit on SM89**.  The
kernel compiled (ptxas only validates register/smem budgets) but the
launch was rejected at runtime.

### Fix

`activation_quant.cu`: cap `kBt` at 8 (= 1024 threads/block max).
Prefill path has no user for large-T anyway -- `policy._auto_policy`
already routes `T >= 256` to Triton.

```diff
-    else if (T <= 16)  dispatch(std::integral_constant<int, 16>{});
-    else if (T <= 64)  dispatch(std::integral_constant<int, 32>{});
-    else               dispatch(std::integral_constant<int, 32>{});
+    else               dispatch(std::integral_constant<int, 8>{});
```

### Cross-check: other kernels

`dense_gemm`, `sparse_gemm`, `fused_dense_sparse` all use
`dim3 block(128, 1, 1)` -- 128 threads/block, well under the limit.
No change needed.

### Deliverables

- [x] code patch committed
- [ ] rerun parity (pending)
- [ ] record full parity pass / fail matrix
- [ ] benchmark vs Triton

---
