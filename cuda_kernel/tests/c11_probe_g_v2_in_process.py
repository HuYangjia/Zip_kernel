"""r77 Probe-G v2: IN-PROCESS bar.sync overhead calibration.

v1 rationale & failure mode
---------------------------
v1 (c11_probe_g_barsync.py) used a parent/child subprocess-per-level
scheme.  On autodl RTX 4090 this failed: GPU SM clocks fall to 210 MHz
between children, and the 500-iter warmup is insufficient to ramp small
kernels (T=512) back to boost before measurement.  Result: same shape
timed 2x different across children, corrupting the N-delta signal.

v2 design
---------
Load FOUR copies of the hkust_v9_cuda extension in the SAME Python
process, each with a different -DHKUST_PROBE_G=N compile flag and a
distinct build_directory / module name.  This lets us:

  * keep GPU context alive across levels  (no per-level re-init)
  * interleave calls (ABCDABCDABCD) to neutralise any clock drift
  * do 5 independent trials and take per-(level,shape) medians
    (per memory:bmmiahpl sensitive-AB regime: warmup=500, outer=20,
     inner=200, >=5 trial median)

Shapes: same 4 as Probe-B/D/E/F, loser-first for compute-bound signal.

Gate (r77 Day-1 spike decision):
    delta(N=2) <  3%  => GREEN,  proceed with warp-spec spike.
    delta(N=2) 3-10%  => YELLOW, retry with smem-flag mechanism.
    delta(N=2) > 10%  => RED,    abort warp-spec, go Option 2.

Correctness: Probe-G barriers are synchronisation-only (no data path),
so kernel output is mathematically identical across N.  We still run a
once-per-level parity check against N=0 baseline as a safety net
(max_abs_err vs N=0 must be 0.0).

Usage (from repo root, on remote):
    python kernel/cuda_kernel/tests/c11_probe_g_v2_in_process.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs  # noqa: E402

# ---------------------------------------------------------------------------
# Build-config (mirror ops.py exactly, minus the PROBE_G flag which we
# override per-instance).
# ---------------------------------------------------------------------------
_CK_DIR = REPO / "kernel" / "cuda_kernel"
_CSRC   = _CK_DIR / "csrc"

_SOURCES = [
    str(_CSRC / "bindings.cc"),
    str(_CSRC / "activation_quant" / "activation_quant.cu"),
    str(_CSRC / "dense_gemm"         / "dense_gemm_mma_int4.cu"),
    str(_CSRC / "dense_gemm"         / "dense_gemv_decode.cu"),
    str(_CSRC / "sparse_gemm"        / "sparse_gemm_mma_int4.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_dense_sparse_mma_int4.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_dense_sparse_mma_int4_cutlass.cu"),
    str(_CSRC / "fused_dense_sparse" / "cutlass_dequant.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_gemv_decode.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_gemv_smallT.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_quant_gemv.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_quant_dense_sparse_mma_int4.cu"),
]

_NVCC_FLAGS_BASE = [
    "-O3", "-std=c++17",
    "-gencode=arch=compute_89,code=sm_89",
    "--fmad=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Wno-deprecated-declarations",
]

_CXX_FLAGS = ["-O3", "-std=c++17", "-fvisibility=hidden"]

_INCLUDE_DIRS = [str(_CSRC)]
_cutlass_root = _CK_DIR / "extern" / "cutlass"
if (_cutlass_root / "include" / "cutlass" / "cutlass.h").is_file():
    _INCLUDE_DIRS.append(str(_cutlass_root / "include"))
    _INCLUDE_DIRS.append(str(_cutlass_root / "tools" / "util" / "include"))


def build_ext_for_probe_g(n: int):
    """JIT-build hkust_v9_cuda with -DHKUST_PROBE_G=n.  Returns the loaded
    Python module.  Uses a distinct build_directory per n so binaries
    coexist on disk.
    """
    from torch.utils.cpp_extension import load

    build_dir = Path.home() / ".cache" / f"hkust_v9_cuda_probeG{n}"
    build_dir.mkdir(parents=True, exist_ok=True)

    flags = list(_NVCC_FLAGS_BASE)
    if n > 0:
        flags.append(f"-DHKUST_PROBE_G={n}")

    name = f"hkust_v9_cuda_probeG{n}"
    print(f"[build] loading {name} (HKUST_PROBE_G={n}, "
          f"build_dir={build_dir})", flush=True)
    mod = load(
        name=name,
        sources=_SOURCES,
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=flags,
        extra_include_paths=_INCLUDE_DIRS,
        build_directory=str(build_dir),
        verbose=False,
    )
    return mod


# ---------------------------------------------------------------------------
# Kernel invocation helper (mirrors ops.fused_dense_sparse_cuda_int4 arg
# prep; we inline it to avoid any late binding against the globally-
# loaded ops._ext).
# ---------------------------------------------------------------------------
from kernel.triton_kernel.pack_utils import BCOL  # noqa: E402


def _prepare_fused_args(inputs, d_out, d_in):
    W_low_packed = inputs["W_low_packed"].contiguous()
    X_s4         = inputs["X_s4"].contiguous()
    scale_u4     = inputs["scale_u4"].contiguous().to(torch.float16)
    zero_u4      = inputs["zero_u4"].contiguous().to(torch.float16)
    sum_X        = inputs["sum_X"].contiguous().to(torch.int32)
    scale_x      = inputs["scale_x"].contiguous().to(torch.float16)

    W_high = inputs.get("W_high_packed", None)
    if W_high is None or W_high.numel() == 0:
        W_high = torch.zeros((0, 128, BCOL // 2), dtype=torch.int8,
                             device=W_low_packed.device)
    W_high = W_high.contiguous()
    hp_row_offsets = inputs["hp_row_offsets"].contiguous().to(torch.int32)
    hp_col_indices = inputs["hp_col_indices"].contiguous().to(torch.int32)

    T = X_s4.shape[0]
    Y_total = torch.empty((d_out, T), dtype=torch.float16,
                          device=W_low_packed.device)

    return (W_low_packed, W_high,
            hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, zero_u4, sum_X, scale_x,
            Y_total, int(d_out), int(d_in)), Y_total


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
SHAPES = [
    # label,             T,     d_in,  d_out,   cluster
    ("32B gu T=2048",    2048,  5120,  55296,  "loser"),
    ("70B gu T=2048",    2048,  8192,  57344,  "loser"),
    ("8B  q  T=512",      512,  4096,   4096,  "winner"),
    ("14B q  T=512",      512,  5120,   5120,  "winner"),
]

PROBE_LEVELS = [0, 2, 4, 8]

# Sensitive A/B (memory:bmmiahpl), reduced from (500, 20, 200, 5) so that
# a full 4-level x 4-shape x TRIALS sweep fits in ~15 min wall.  Justified:
#  * 32B gu T=2048 @ 10ms/iter: WARMUP=500 => 5s clock-warm (enough for
#    boost), OUTER*INNER=500 => 5s measurement, noise floor << 1%.
#  * 70B gu T=2048 @ 17ms/iter: same reasoning, measurement ~8.5s.
#  * winner shapes T=512 @ ~80us: measurement is 80us*500=40ms, plenty.
# TRIALS=3 still satisfies "median of >=3 independent trials".
WARMUP = 500
OUTER  = 5
INNER  = 100
TRIALS = 3


def time_forward_us(fn) -> float:
    """Single-trial per-iter us: min-over-outer of mean-over-inner."""
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


def make_call_closures(modules, inputs_per_shape):
    """Return a dict of { (shape_label, level_n): callable_with_no_args }.

    One Y_total output buffer per (shape, level) to avoid tensor-alloc
    noise in the hot path.
    """
    closures = {}
    for (label, T, d_in, d_out, _cluster) in SHAPES:
        inp = inputs_per_shape[label]
        for n in PROBE_LEVELS:
            mod = modules[n]
            args, _Y = _prepare_fused_args(inp, d_out, d_in)
            launch = mod.fused_dense_sparse_mma_int4_launch

            def _call(_launch=launch, _args=args):
                _launch(*_args)

            closures[(label, n)] = _call
    return closures


def parity_check(modules, inputs_per_shape) -> dict:
    """Run each level once per shape, compare vs N=0.  All deltas must be 0.0
    (barriers are synchronisation-only).  Returns {(label, n): max_abs_err}.
    """
    errs = {}
    base_mod = modules[0]
    for (label, T, d_in, d_out, _c) in SHAPES:
        inp = inputs_per_shape[label]
        args0, Y0 = _prepare_fused_args(inp, d_out, d_in)
        base_mod.fused_dense_sparse_mma_int4_launch(*args0)
        torch.cuda.synchronize()
        Y0c = Y0.clone()
        for n in PROBE_LEVELS:
            if n == 0:
                errs[(label, n)] = 0.0
                continue
            mod = modules[n]
            args, Y = _prepare_fused_args(inp, d_out, d_in)
            mod.fused_dense_sparse_mma_int4_launch(*args)
            torch.cuda.synchronize()
            e = (Y.float() - Y0c.float()).abs().max().item()
            errs[(label, n)] = e
    return errs


def main():
    torch.manual_seed(0)
    device = "cuda"

    print("=" * 100)
    print("r77 Probe-G v2 (in-process, interleaved trials)")
    print("warmup=%d outer=%d inner=%d trials=%d"
          % (WARMUP, OUTER, INNER, TRIALS))
    print("=" * 100)

    # --- build all extensions up front ---
    modules = {n: build_ext_for_probe_g(n) for n in PROBE_LEVELS}

    # --- prepare inputs once per shape ---
    inputs_per_shape = {}
    for (label, T, d_in, d_out, _c) in SHAPES:
        inputs_per_shape[label] = make_inputs(
            T, d_out, d_in, hp_ratio=0.05, device=device,
            seed=T + d_in + d_out,
        )

    # --- parity gate ---
    print("\n[parity] checking Probe-G barriers do not alter output ...",
          flush=True)
    errs = parity_check(modules, inputs_per_shape)
    bad = {k: v for k, v in errs.items() if v > 0.0}
    if bad:
        print("!! PARITY FAILURE — barriers changed kernel output:")
        for k, v in bad.items():
            print(f"   {k}: max_abs_err={v}")
        print("   aborting; fix probe_g_sync_overhead() first")
        return 1
    print("   OK: all levels bit-identical to N=0 baseline.", flush=True)

    # --- build closures (same output buffers re-used across trials) ---
    closures = make_call_closures(modules, inputs_per_shape)

    # --- interleaved trials ---
    # storage: trials_us[(label, n)] = [t1, t2, ..., tTRIALS]
    trials_us: dict = {k: [] for k in closures.keys()}

    for trial in range(TRIALS):
        # Each trial runs (shape, level) in an order that interleaves
        # levels so no level's measurement catches a consistent clock
        # transient.  Order: per shape, cycle through levels; next
        # trial, offset the level sequence so level 0 is not always
        # "first to see this shape's cache state".
        offset = trial % len(PROBE_LEVELS)
        rotated_levels = PROBE_LEVELS[offset:] + PROBE_LEVELS[:offset]
        print(f"\n[trial {trial+1}/{TRIALS}] level order: {rotated_levels}",
              flush=True)
        for (label, T, d_in, d_out, _c) in SHAPES:
            for n in rotated_levels:
                us = time_forward_us(closures[(label, n)])
                trials_us[(label, n)].append(us)
                print(f"   {label:<16} N={n}  us={us:.2f}", flush=True)

    # --- median + summary ---
    print("\n" + "=" * 100)
    print("PER-SHAPE MEDIAN TABLE")
    print("=" * 100)
    header = (f"{'label':<18} {'cluster':<7}"
              + "".join(f" {'us_N='+str(n):>10}" for n in PROBE_LEVELS)
              + "".join(f" {'d_N='+str(n):>8}" for n in PROBE_LEVELS[1:])
              + f"  {'us/bar':>8}")
    print(header)

    deltas_at_N2_loser = []
    per_bar_us_list = []

    for (label, T, d_in, d_out, cluster) in SHAPES:
        meds = {n: statistics.median(trials_us[(label, n)])
                for n in PROBE_LEVELS}
        us0 = meds[0]
        cells = [f"{us0:>10.2f}"]
        for n in PROBE_LEVELS[1:]:
            cells.append(f"{meds[n]:>10.2f}")
        deltas = {n: (meds[n] - us0) / us0 * 100.0 for n in PROBE_LEVELS[1:]}
        delta_cells = [f"{deltas[n]:>+7.2f}%" for n in PROBE_LEVELS[1:]]

        if 8 in meds:
            per_bar_us = (meds[8] - us0) / 8.0
        elif 4 in meds:
            per_bar_us = (meds[4] - us0) / 4.0
        else:
            per_bar_us = float("nan")

        print(f"  {label:<18} {cluster:<7} "
              + " ".join(cells)
              + " " + " ".join(delta_cells)
              + f"  {per_bar_us:>7.3f}us", flush=True)

        if cluster == "loser":
            deltas_at_N2_loser.append(deltas[2])
        per_bar_us_list.append(per_bar_us)

    # --- dump raw trials for later analysis ---
    out_path = _CK_DIR / "logs" / "probe_g_v2_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {f"{k[0]}|N={k[1]}": v for k, v in trials_us.items()}
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n[raw] {out_path}", flush=True)

    # --- decision gate ---
    print("\n" + "=" * 100)
    print("r77 WARP-SPEC GO/NO-GO GATE  (worst-case loser delta at N=2)")
    print("=" * 100)
    if not deltas_at_N2_loser:
        print("  INSUFFICIENT DATA")
        return 1
    worst_loser_d2 = max(deltas_at_N2_loser)
    print(f"  worst loser delta @ N=2: {worst_loser_d2:+.2f}%")

    import math
    finite_slopes = [x for x in per_bar_us_list if math.isfinite(x)]
    if finite_slopes:
        median_slope = sorted(finite_slopes)[len(finite_slopes) // 2]
        print(f"  median per-bar-sync cost: {median_slope:.3f}us")

    if worst_loser_d2 < 3.0:
        print("\n  VERDICT: GREEN — PROCEED with r77 warp-spec spike")
        print("  Rationale: full-CTA bar.sync cost is cheap enough that the")
        print("  r77 1P+3C handshake will not eat pipeline gains.  Day-1 gate")
        print("  remains >=5% loser speed-up on 32B gu T=2048.")
    elif worst_loser_d2 < 10.0:
        print("\n  VERDICT: YELLOW — sync material, retry with smem-flag")
        print("  Rationale: full-CTA bar.sync is upper bound.  Implement a")
        print("  128-thread partial bar.sync or smem-flag spin and re-measure")
        print("  before committing to the spike.")
    else:
        print("\n  VERDICT: RED — ABORT warp-spec, switch to Option 2")
        print("  Rationale: sync dominates; producer/consumer handshakes will")
        print("  eat pipeline gains.  Shelve r77, start CUTLASS 3.x back-port.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
