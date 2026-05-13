"""C.11 Probe-D: cp.async wait-group bisection.

Compile-flag probe to measure how much of the W4A4 mainloop kernel time
is spent waiting on HBM->smem (cp.async.wait_group).

Strategy:
  Two subprocess children, each does its own JIT compile.
    child A: HKUST_PROBE_D=0  -> baseline (correct kernel,
                                 wait_group<N> as usual).
    child B: HKUST_PROBE_D=1  -> wait_group is replaced with
                                 `cp.async.wait_group 99` which
                                 effectively never blocks.  Kernel
                                 output is INCORRECT but wall time
                                 gives an upper bound.
  We bench the same set of shapes in each child and compare.

Interpretation:
  delta = (t_A - t_B) / t_A
    delta ~ 0%      => wait_group costs nothing -> kernel was never
                       pipeline-stalled; bottleneck is elsewhere
                       (compute or smem-layout).  Shelve C.11-A-revival
                       and C.11-D warp-specialisation; go to Probe-F.
    delta 5-15%    => mixed: pipeline contributes but is not the only
                       story.  Revisit C.11-A with dynamic smem +
                       register-budget tuning.
    delta >= 15%   => pipeline-bound; 3-stage / warp-specialisation has
                       real headroom; pursue C.11-D.

Correctness of Probe-D results is NOT checked -- the wait-off kernel
produces races on purpose.  We only care about wall time.

Usage (from repo root, on remote):
    python kernel/cuda_kernel/tests/c11_probe_d_cpasync_waitoff.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]  # .../HKUST

# Two loser-cluster shapes + two winner/reference shapes (same as Probe-B
# so the two bisections can be read side-by-side).
SHAPES = [
    # label,       T,     d_in,  d_out,   cluster
    ("32B gu T=2048",  2048, 5120,  55296, "loser"),
    ("70B gu T=2048",  2048, 8192,  57344, "loser"),
    ("8B  q  T=512",    512, 4096,  4096,  "winner"),   # healthy reference
    ("14B q  T=512",    512, 5120,  5120,  "winner"),   # healthy reference
]


def _run_child():
    import json
    import torch

    import kernel.cuda_kernel.ops as ops
    from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

    # Methodology [[memory:bmmiahpl]]: warmup=500, outer=10, inner=200.
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

    probe_flag = int(os.environ.get("HKUST_PROBE_D", "0"))
    print(f"# child HKUST_PROBE_D={probe_flag}", flush=True)

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
               "cluster": cluster, "probe_d": probe_flag, "us": us}
        print("RESULT " + json.dumps(row), flush=True)


def _run_parent():
    import json

    env_common = os.environ.copy()
    base_build = env_common.get("HKUST_V9_CUDA_BUILD_DIR",
                                str(Path.home() / ".cache" / "hkust_v9_cuda"))

    results = {0: [], 1: []}
    for flag in (0, 1):
        env = env_common.copy()
        env["HKUST_PROBE_D"] = str(flag)
        env["HKUST_PROBE_D_CHILD"] = "1"
        env["HKUST_V9_CUDA_BUILD_DIR"] = base_build + f"_probeD{flag}"
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO) + (os.pathsep + pp if pp else "")
        print(f"\n=== spawning child HKUST_PROBE_D={flag} "
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
    print("PROBE-D SUMMARY  (delta = (baseline - probeD_waitoff) / baseline)",
          flush=True)
    print("=" * 96, flush=True)
    print(f"{'label':<18} {'cluster':<8} {'baseline_us':>12} "
          f"{'probeD_us':>12} {'delta':>8}  hypothesis", flush=True)

    for label in [s[0] for s in SHAPES]:
        if label not in base or label not in probe:
            print(f"  {label:<18} (missing)", flush=True)
            continue
        t_a = base[label]["us"]
        t_b = probe[label]["us"]
        delta = (t_a - t_b) / t_a * 100.0
        cluster = base[label]["cluster"]
        if delta >= 15:
            hyp = "pipeline-bound (pursue C.11-D warp-spec)"
        elif delta >= 5:
            hyp = "pipeline-mixed (revisit C.11-A + dyn smem)"
        else:
            hyp = "compute/smem-bound (shelve pipeline; go to Probe-F)"
        print(f"  {label:<18} {cluster:<8} {t_a:>11.2f}  "
              f"{t_b:>11.2f}  {delta:>+7.2f}%  {hyp}", flush=True)


if __name__ == "__main__":
    if os.environ.get("HKUST_PROBE_D_CHILD") == "1":
        _run_child()
    else:
        _run_parent()
