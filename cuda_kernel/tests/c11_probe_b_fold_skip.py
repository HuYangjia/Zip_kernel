"""C.11 Probe-B: fold-skip bisection.

Compile-flag probe to measure how much of the W4A4 mainloop kernel time
is spent in the per-output scalar dequant ALU (fold_dense) vs the MMA
pipeline + cp.async itself.

Strategy:
  Two subprocess children, each does its own JIT compile.
    child A: HKUST_PROBE_B=0  -> baseline (correct kernel).
    child B: HKUST_PROBE_B=1  -> fold_dense skips z/s/sumxn math,
                                 just accumulates raw int32->fp.
  We bench the same set of shapes in each child and compare.

Interpretation:
  delta = (t_A - t_B) / t_A
  - delta ~ 0%      => fold is not the bottleneck (B1 is cheap
                        relative to the MMA+cp.async backbone).
                        This is the loser-cluster hypothesis and
                        means C.11-B/C (LOP3 / int8 scale) have
                        a small budget ceiling.
  - delta >= 15%    => fold has headroom; C.11-B/C are worth
                        the engineering cost.
  - 5% <= delta <15% => marginal; depends on shape class.

Correctness of Probe-B results is NOT checked -- the fold-skip kernel
produces wrong outputs on purpose.  We only care about wall time.

Usage (from repo root, on remote):
    python kernel/cuda_kernel/tests/c11_probe_b_fold_skip.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]  # .../HKUST

# Two loser-cluster shapes + two winner/reference shapes.
# Keep the list small -- we only need 4 points to disambiguate
# "fold-bound" vs "MMA-bound" hypothesis.
SHAPES = [
    # label,       T,     d_in,  d_out,   cluster
    ("32B gu T=2048",  2048, 5120,  55296, "loser"),
    ("70B gu T=2048",  2048, 8192,  57344, "loser"),
    ("8B  q  T=512",    512, 4096,  4096,  "winner"),   # healthy reference
    ("14B q  T=512",    512, 5120,  5120,  "winner"),   # healthy reference
]


# ---------------------------------------------------------------------------
# child mode: JIT the ext (with whatever HKUST_PROBE_B=... is in env),
# bench the shapes, print JSON lines.
# ---------------------------------------------------------------------------
def _run_child():
    import json
    import torch

    import kernel.cuda_kernel.ops as ops
    from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

    # Methodology [[memory:bmmiahpl]]: warmup=500, outer=10, inner=200.
    # Single trial per (shape, child) is OK because we print raw times
    # and the parent will do the A/B comparison across children; any
    # absolute outlier shows up as an obviously-off number.
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

    probe_flag = int(os.environ.get("HKUST_PROBE_B", "0"))
    print(f"# child HKUST_PROBE_B={probe_flag}", flush=True)

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
               "cluster": cluster, "probe_b": probe_flag, "us": us}
        print("RESULT " + json.dumps(row), flush=True)


# ---------------------------------------------------------------------------
# parent mode: spawn two children, parse RESULT lines, print summary.
# ---------------------------------------------------------------------------
def _run_parent():
    import json

    env_common = os.environ.copy()
    # Force a fresh build dir per probe flag so torch JIT doesn't
    # cache the wrong binary across the A/B.
    base_build = env_common.get("HKUST_V9_CUDA_BUILD_DIR",
                                str(Path.home() / ".cache" / "hkust_v9_cuda"))

    results = {0: [], 1: []}
    for flag in (0, 1):
        env = env_common.copy()
        env["HKUST_PROBE_B"] = str(flag)
        env["HKUST_PROBE_B_CHILD"] = "1"
        env["HKUST_V9_CUDA_BUILD_DIR"] = base_build + f"_probeB{flag}"
        # Make sure the child can `import kernel.cuda_kernel.ops` when cwd
        # is set to REPO (autodl layout).
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO) + (os.pathsep + pp if pp else "")
        print(f"\n=== spawning child HKUST_PROBE_B={flag} "
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

    # pair by label.
    base = {r["label"]: r for r in results[0]}
    probe = {r["label"]: r for r in results[1]}

    print("\n" + "=" * 96, flush=True)
    print("PROBE-B SUMMARY  (delta = (baseline - probeB) / baseline)",
          flush=True)
    print("=" * 96, flush=True)
    print(f"{'label':<18} {'cluster':<8} {'baseline_us':>12} "
          f"{'probeB_us':>12} {'delta':>8}  hypothesis", flush=True)

    for label in [s[0] for s in SHAPES]:
        if label not in base or label not in probe:
            print(f"  {label:<16} (missing)", flush=True)
            continue
        t_a = base[label]["us"]
        t_b = probe[label]["us"]
        delta = (t_a - t_b) / t_a * 100.0
        cluster = base[label]["cluster"]
        if delta >= 15:
            hyp = "fold-bound (pursue C.11-B/C)"
        elif delta >= 5:
            hyp = "fold-mixed (marginal)"
        else:
            hyp = "MMA/pipe-bound (fold not the bottleneck)"
        print(f"  {label:<16} {cluster:<8} {t_a:>11.2f}  "
              f"{t_b:>11.2f}  {delta:>+7.2f}%  {hyp}", flush=True)


if __name__ == "__main__":
    if os.environ.get("HKUST_PROBE_B_CHILD") == "1":
        _run_child()
    else:
        _run_parent()
