"""CUDA-Event timing helpers for single-kernel / sub-layer microbench.

Contract (matches kernel/tools/profile/_phase1_shapes.py::time_forward_us and
the RTX 4090 protocol [[memory:bmmiahpl]]):

    time_us = min_over_outer( mean_over_inner( per_iter_us ) )

and on top of that we do >=5 independent trials and take the **median**.
Any single timing number is NOT trustworthy — transient outliers up to +49%
have been observed even with (warmup=500, outer=20, inner=200).

Usage
-----
    fn = lambda: torch.nn.functional.linear(x, W)
    stats = measure(fn, warmup=500, outer=20, inner=200, trials=5)
    print(stats.median_us, stats.p25_us, stats.p75_us)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable

import torch


# -----------------------------------------------------------------------------
# Default timing presets
# -----------------------------------------------------------------------------
# "light"            — rough exploration (cheap, may show +/-10% noise)
# "strict"           — publication-grade for small (<50us) kernels; required
#                      for any decode-side number that enters the final report
#                      per [[memory:bmmiahpl]].
# "adaptive_prefill" — tailored for ms-level prefill GEMMs. Each fn() is already
#                      hundreds-of-us to several-ms, so amortising over
#                      inner=200 is wasteful. Empirically spread is <1% on
#                      RTX 4090 with these knobs for any op >=100us. This keeps
#                      STRICT-grade stability while cutting per-point wall time
#                      by ~10x compared to STRICT.
LIGHT = dict(warmup=200, outer=10, inner=100, trials=3)
STRICT = dict(warmup=500, outer=20, inner=200, trials=5)
ADAPTIVE_PREFILL = dict(warmup=50, outer=10, inner=20, trials=5)


@dataclass
class TimingStats:
    median_us: float
    min_us: float
    max_us: float
    trial_us: list[float] = field(default_factory=list)
    warmup: int = 0
    outer: int = 0
    inner: int = 0
    trials: int = 0

    @property
    def spread_pct(self) -> float:
        """(max - min) / median — a quick noise sanity indicator."""
        if self.median_us <= 0:
            return float("nan")
        return (self.max_us - self.min_us) / self.median_us * 100.0

    def as_dict(self) -> dict:
        return {
            "median_us": self.median_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "spread_pct": self.spread_pct,
            "trial_us": list(self.trial_us),
            "warmup": self.warmup,
            "outer": self.outer,
            "inner": self.inner,
            "trials": self.trials,
        }


def _time_one_trial(
    fn: Callable[[], None],
    *,
    warmup: int,
    outer: int,
    inner: int,
    device: torch.device,
) -> float:
    """min-of-outer of (mean-of-inner) micro-seconds for a single trial."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    means_us: list[float] = []
    for _ in range(outer):
        start_ev.record()
        for _ in range(inner):
            fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        # elapsed_time -> milliseconds; /inner -> ms per iter; *1000 -> us
        means_us.append(start_ev.elapsed_time(end_ev) * 1000.0 / inner)
    return min(means_us)


def measure(
    fn: Callable[[], None],
    *,
    warmup: int = 500,
    outer: int = 20,
    inner: int = 200,
    trials: int = 5,
    device: str | torch.device = "cuda",
) -> TimingStats:
    """Run ``fn`` ``trials`` times (each trial = min-of-outer) and return
    median/min/max in microseconds.

    ``fn`` must be a zero-arg callable that launches the kernel(s) under test.
    It is the caller's responsibility to make sure ``fn`` is reentrant (no
    in-place accumulation on inputs that would drift over iterations).
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    dev = torch.device(device)

    trial_us: list[float] = []
    for _ in range(trials):
        t = _time_one_trial(
            fn, warmup=warmup, outer=outer, inner=inner, device=dev
        )
        trial_us.append(t)

    return TimingStats(
        median_us=float(median(trial_us)),
        min_us=float(min(trial_us)),
        max_us=float(max(trial_us)),
        trial_us=trial_us,
        warmup=warmup,
        outer=outer,
        inner=inner,
        trials=trials,
    )


__all__ = ["TimingStats", "LIGHT", "STRICT", "ADAPTIVE_PREFILL", "measure"]
