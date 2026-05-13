"""r77 Probe-G: bar.sync overhead calibration for warp-spec go/no-go gate.

Compile-flag probe to measure UPPER BOUND on per-g-iter bar.sync overhead,
to decide whether the r77 warp-specialisation (1P+3C) spike is worth
implementing.  See PHASE3_STEP2_WARPSPEC_DESIGN.md for context.

Strategy:
  Four subprocess children, each does its own JIT compile with a different
  HKUST_PROBE_G level (N=0/2/4/8).  Each level inserts N extra full-CTA
  `bar.sync id, 128` barriers per dense g-iter immediately before the
  existing __syncthreads().  N=0 is byte-identical to r72 baseline.

  Full-CTA bar.sync (participant=128) is a strict UPPER BOUND on the real
  warp-spec handshake cost — a true 1P+3C implementation uses partial
  barriers (participant=96 or pair-wise 32+96) which are cheaper.

  Same shape set as Probe-B/D/E/F for direct comparability.

Gate (r77 Day-1 spike decision):
    delta(N=2) <  3%  => warp-spec sync cheap enough; proceed with full spike.
    delta(N=2) 3-10%  => sync cost material; retry with smem-flag spin mechanism.
    delta(N=2) > 10%  => sync dominates; abort warp-spec, switch to Option 2
                         CUTLASS 3.x stream-K back-port.

Interpreting the slope (delta(N)/N) is the real signal; we print it as
"per-bar-sync us" which is the marginal cost of ONE extra full-CTA barrier.

Correctness: Probe-G barriers are synchronisation-only with no data-path
effect, so kernel output is mathematically identical across N.  We skip
numeric parity checks (same policy as Probe-B/D/E/F) since timing is the
sole signal.

Usage (from repo root, on remote):
    python kernel/cuda_kernel/tests/c11_probe_g_barsync.py
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

# r77: 4 sync-density levels.  N=2 is the realistic warp-spec sync density
# (one "data-ready" + one "buffer-free" handshake per g-iter).  N=4 and N=8
# are stress levels to extract the per-bar-sync slope.
PROBE_LEVELS = [0, 2, 4, 8]


def _run_child():
    import json
    import torch

    import kernel.cuda_kernel.ops as ops
    from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

    # Sensitive A/B: per memory:bmmiahpl use (warmup=500, outer=20, inner=200)
    # for <3% detection.  Keep OUTER=10 because we stack 4 levels × 4 shapes;
    # each level is a separate child with its own JIT, so total wall is still
    # manageable.  Median-of-trials is done at the parent level only if we see
    # flaky numbers (first pass is single trial per level).
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

    probe_flag = int(os.environ.get("HKUST_PROBE_G", "0"))
    print(f"# child HKUST_PROBE_G={probe_flag}", flush=True)

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
               "cluster": cluster, "probe_g": probe_flag, "us": us}
        print("RESULT " + json.dumps(row), flush=True)


def _run_parent():
    import json

    env_common = os.environ.copy()
    base_build = env_common.get(
        "HKUST_V9_CUDA_BUILD_DIR",
        str(Path.home() / ".cache" / "hkust_v9_cuda"),
    )

    results: dict[int, list] = {n: [] for n in PROBE_LEVELS}
    for flag in PROBE_LEVELS:
        env = env_common.copy()
        env["HKUST_PROBE_G"] = str(flag)
        env["HKUST_PROBE_G_CHILD"] = "1"
        env["HKUST_V9_CUDA_BUILD_DIR"] = base_build + f"_probeG{flag}"
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO) + (os.pathsep + pp if pp else "")
        print(f"\n=== spawning child HKUST_PROBE_G={flag} "
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

    # Reorganise: by_label[label][N] = us
    by_label: dict[str, dict[int, float]] = {}
    for flag in PROBE_LEVELS:
        for r in results[flag]:
            by_label.setdefault(r["label"], {})[flag] = r["us"]

    # --- Per-shape table -----------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("PROBE-G SUMMARY  (delta_N = (us_N - us_0) / us_0 * 100%)", flush=True)
    print("full-CTA bar.sync overhead is UPPER BOUND on real warp-spec "
          "handshake cost", flush=True)
    print("=" * 100, flush=True)

    header = (
        f"{'label':<18} {'cluster':<7}"
        + "".join(f" {'us_N='+str(n):>10}" for n in PROBE_LEVELS)
        + "".join(f" {'d_N='+str(n):>8}" for n in PROBE_LEVELS if n != 0)
        + f"  {'us/bar':>8}"
    )
    print(header, flush=True)

    # Aggregate for decision
    deltas_at_N2_loser = []
    per_bar_us_list = []

    for shape in SHAPES:
        label, _, _, _, cluster = shape
        row_data = by_label.get(label, {})
        if 0 not in row_data:
            print(f"  {label:<18} (missing baseline)", flush=True)
            continue
        us0 = row_data[0]
        cells = [f"{us0:>10.2f}"]
        deltas: dict[int, float] = {}
        for n in PROBE_LEVELS[1:]:
            if n in row_data:
                d = (row_data[n] - us0) / us0 * 100.0
                deltas[n] = d
                cells.append(f"{row_data[n]:>10.2f}")
            else:
                cells.append(f"{'--':>10}")
        delta_cells = []
        for n in PROBE_LEVELS[1:]:
            if n in deltas:
                delta_cells.append(f"{deltas[n]:>+7.2f}%")
            else:
                delta_cells.append(f"{'--':>8}")

        # Per-bar us (slope): fit using largest N as most stable
        if 8 in row_data:
            n_ref = 8
        elif 4 in row_data:
            n_ref = 4
        elif 2 in row_data:
            n_ref = 2
        else:
            n_ref = None
        if n_ref is not None:
            per_bar_us = (row_data[n_ref] - us0) / n_ref
        else:
            per_bar_us = float("nan")

        print(f"  {label:<18} {cluster:<7} "
              + " ".join(cells)
              + " " + " ".join(delta_cells)
              + f"  {per_bar_us:>7.3f}us", flush=True)

        if cluster == "loser" and 2 in deltas:
            deltas_at_N2_loser.append(deltas[2])
        per_bar_us_list.append(per_bar_us)

    # --- Decision gate -------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("r77 WARP-SPEC GO/NO-GO GATE  (worst-case loser delta at N=2)",
          flush=True)
    print("=" * 100, flush=True)
    if not deltas_at_N2_loser:
        print("  INSUFFICIENT DATA — cannot decide.  Re-run probe.",
              flush=True)
        return

    worst_loser_d2 = max(deltas_at_N2_loser)
    print(f"  worst loser delta @ N=2 (2 full-CTA bar.sync/iter): "
          f"{worst_loser_d2:+.2f}%", flush=True)

    import math
    finite_slopes = [x for x in per_bar_us_list if math.isfinite(x)]
    if finite_slopes:
        median_slope = sorted(finite_slopes)[len(finite_slopes) // 2]
        print(f"  median marginal cost per full-CTA bar.sync: "
              f"{median_slope:.3f}us", flush=True)

    print(flush=True)
    if worst_loser_d2 < 3.0:
        verdict = (
            "GREEN — sync overhead cheap.  PROCEED with r77 warp-spec spike\n"
            "          (full 1P+3C structural rewrite, 32B gu T=2048 dense-only\n"
            "           kStages=2; Day-1 end-of-day gate ≥5% loser speed-up)."
        )
    elif worst_loser_d2 < 10.0:
        verdict = (
            "YELLOW — sync overhead material.  DO NOT go full bar.sync warp-spec.\n"
            "          Retry Probe-G with smem-flag spin mechanism; if smem-flag\n"
            "          cuts overhead to <3%, proceed with spike using smem-flag.\n"
            "          Otherwise abort warp-spec and go to Option 2 (CUTLASS 3.x)."
        )
    else:
        verdict = (
            "RED — sync dominates.  ABORT r77 warp-spec.  The producer/consumer\n"
            "          handshake will eat any pipeline gain.  Switch to\n"
            "          Option 2: CUTLASS 3.x stream-K sm_89 back-port (2–3 weeks).\n"
            "          Update PROJECT_HANDOFF.md to shelve warp-spec line item."
        )
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    if os.environ.get("HKUST_PROBE_G_CHILD") == "1":
        _run_child()
    else:
        _run_parent()
