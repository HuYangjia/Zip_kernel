"""C.11 Probe-F: smem-layout bisection (scale/zero packed into __half2).

Compile-flag probe to measure whether the dense-mainloop smem fetch of
scale/zero is on the loser-cluster critical path.

Strategy:
  Two subprocess children, each does its own JIT compile.
    child A: HKUST_PROBE_F=0  -> baseline (s_scale_u4 + s_zero_u4 as
                                 two __half[kBm][G+1] arrays; two
                                 ld.shared.b16 per fold).
    child B: HKUST_PROBE_F=1  -> packed s_sz_u4[kBm][G+1] as __half2;
                                 one ld.shared.b32 per fold (half the
                                 smem-load instructions, half the IMAD
                                 address arithmetic).
  Both children are functionally correct (packing is pure layout
  change).  Same shape set as Probe-B/D/E for direct comparability.

Interpretation:
  delta = (baseline_us - probeF_us) / baseline_us
    delta >= 5%    => B2 confirmed.  Promote to production as C.12
                      smem-pack (needs parity tests + full-shape scan +
                      C.8 dispatcher re-tune).
    delta 2-5%     => B2 contributes but is not the dominant cost.
                      Worth productising anyway (low risk, tidy layout),
                      but also launch Probe-G (MMA-only) to find the
                      remainder.
    delta <= 2%    => B2 rejected too.  The loser kernel is dominated
                      by the mma.m16n8k64 issue pipe itself (sm_89 TC
                      compute ceiling).  Transition to Phase 3 Step 2
                      CUTLASS 3.x rewrite.

Correctness: Probe-F is a layout transform; results should match
baseline within fp32 noise.  We skip the correctness check here (same
policy as Probe-B/D/E) since the three previous probes already validated
the harness; timing is the sole signal.

Usage (from repo root, on remote):
    python kernel/cuda_kernel/tests/c11_probe_f_smem_pack.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]  # .../HKUST

SHAPES = [
    # label,            T,     d_in,  d_out,   cluster
    ("32B gu T=2048",   2048, 5120,  55296, "loser"),
    ("70B gu T=2048",   2048, 8192,  57344, "loser"),
    ("8B  q  T=512",     512, 4096,  4096,  "winner"),
    ("14B q  T=512",     512, 5120,  5120,  "winner"),
]


def _run_child():
    import json
    import torch

    import kernel.cuda_kernel.ops as ops
    from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

    WARMUP, OUTER, INNER = 500, 10, 200

    def bench(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(OUTER):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(INNER):
                fn()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) * 1000.0 / INNER)
        return best

    probe_flag = int(os.environ.get("HKUST_PROBE_F", "0"))
    print(f"# child HKUST_PROBE_F={probe_flag}", flush=True)

    for label, T, d_in, d_out, cluster in SHAPES:
        b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                        seed=T + d_in + d_out)
        def _call():
            return ops.fused_dense_sparse_cuda_int4(
                b["W_low_packed"], b["W_high_packed"],
                b["hp_row_offsets"], b["hp_col_indices"],
                b["X_s4"], b["scale_u4"], b["zero_u4"],
                b["sum_X"], b["scale_x"], d_out, d_in,
            )
        us = bench(_call)
        row = {"label": label, "T": T, "d_in": d_in, "d_out": d_out,
               "cluster": cluster, "probe_f": probe_flag, "us": us}
        print("RESULT " + json.dumps(row), flush=True)


def _run_parent():
    import json

    env_common = os.environ.copy()
    base_build = env_common.get("HKUST_V9_CUDA_BUILD_DIR",
                                str(Path.home() / ".cache" / "hkust_v9_cuda"))

    results = {0: [], 1: []}
    for flag in (0, 1):
        env = env_common.copy()
        env["HKUST_PROBE_F"] = str(flag)
        env["HKUST_PROBE_F_CHILD"] = "1"
        env["HKUST_V9_CUDA_BUILD_DIR"] = base_build + f"_probeF{flag}"
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO) + (os.pathsep + pp if pp else "")
        print(f"\n=== spawning child HKUST_PROBE_F={flag} "
              f"build_dir={env['HKUST_V9_CUDA_BUILD_DIR']} ===", flush=True)

        proc = subprocess.Popen(
            [sys.executable, str(HERE)],
            env=env, cwd=str(REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("RESULT "):
                row = json.loads(line[len("RESULT "):])
                results[flag].append(row)
        proc.wait()
        if proc.returncode != 0:
            print(f"!! child {flag} exited {proc.returncode}", flush=True)

    base = {r["label"]: r for r in results[0]}
    probe = {r["label"]: r for r in results[1]}

    print("\n" + "=" * 96, flush=True)
    print("PROBE-F SUMMARY  (delta = (baseline - probeF_packed) / baseline)",
          flush=True)
    print("=" * 96, flush=True)
    print(f"{'label':<18} {'cluster':<8} {'baseline_us':>12} "
          f"{'probeF_us':>12} {'delta':>8}  hypothesis", flush=True)

    for label in [s[0] for s in SHAPES]:
        if label not in base or label not in probe:
            print(f"  {label:<18} (missing)", flush=True)
            continue
        t_a = base[label]["us"]
        t_b = probe[label]["us"]
        delta = (t_a - t_b) / t_a * 100.0
        cluster = base[label]["cluster"]
        if delta >= 5:
            hyp = "B2 CONFIRMED -> promote to C.12 smem-pack"
        elif delta >= 2:
            hyp = "B2 partial -> productise + Probe-G"
        else:
            hyp = "B2 negligible -> TC-compute ceiling; go CUTLASS 3.x"
        print(f"  {label:<18} {cluster:<8} {t_a:>11.2f}  "
              f"{t_b:>11.2f}  {delta:>+7.2f}%  {hyp}", flush=True)


if __name__ == "__main__":
    if os.environ.get("HKUST_PROBE_F_CHILD") == "1":
        _run_child()
    else:
        _run_parent()
