# Phase 1 — Launch-Tax via CUDA Graph replay

_Generated 2026-04-28T09:11:57.892801Z; min-of-means (warmup=50, 3×100 windows)._

Columns:  
* `t_plain` — eager path wall-clock per forward.  
* `t_graph` — CUDA Graph replay wall-clock per forward (kernel-launch API amortised).  
* `launch_tax = t_plain - t_graph` — aggregate kernel-launch overhead per forward.  
* `t_body` — GPU kernel accumulated time per forward from nsys sqlite; **gold-standard kernel-body time**.  
* `body/plain` — how much of wall-clock is real kernel work vs launch/gap.

| tag | T | plain (us) | graph (us) | launch_tax (us) | tax % of plain | t_body (us) | body/plain |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_q_2048_2048 | 1 | 46.766 | 13.169 | 33.597 | 71.84% | 15.283 | 0.327 |
| worst_T8_kv_1024_2048 | 8 | 55.009 | 16.947 | 38.062 | 69.19% | 18.561 | 0.337 |
| mid_T128_kv_2560_2048 | 128 | 80.558 | 64.152 | 16.406 | 20.37% | 66.993 | 0.832 |
| large_T1024_gu_4096_24576 | 1024 | 1127.066 | 1124.136 | 2.93 | 0.26% | 41274.024 | 36.621 |

