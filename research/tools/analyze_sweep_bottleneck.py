"""Analyse v9 sweep results to identify next optimisation direction.

Reads a sweep_v9 CSV (produced by kernel/triton_kernel/benchmarks/sweep_v9.py),
then prints:

1. Per-bucket stage share (quant / dense / sparse / combine as % of v9_total)
2. Amdahl upper bound: "if stage X went to 0, what speedup would we hit?"
3. The top shapes where V9 is just below cuBLAS FP16 (speedup between 0.5x
   and 1.0x). These are the low-hanging-fruit candidates: one kernel tweak
   may flip them over.
4. dense_ms / fp16_ms ratio, i.e. how close our 4-bit-weight GEMM is to
   the dense FP16 theoretical floor.
5. Sparse cost when active.
6. Dense bandwidth utilisation (GB/s vs 4090 HBM peak).

Usage:
    python analyze_sweep_bottleneck.py <path-to-sweep.csv>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("d_out", "d_in", "bs"):
            r[k] = int(r[k])
        for k in ("hp_ratio", "v9_total_ms", "fp16_ms", "stage1_quant_ms",
                  "stage2_dense_ms", "stage3_sparse_ms", "stage4_combine_ms",
                  "speedup_vs_fp16"):
            r[k] = float(r[k])
    return rows


def bs_tier(bs):
    if bs <= 16:
        return "decode(1-16)"
    if bs <= 64:
        return "small(32-64)"
    if bs <= 512:
        return "mid(128-512)"
    return "prefill(>=2K)"


def hp_tier(hp):
    return "hp=0" if hp == 0.0 else "hp>0"


def main(csv_path: Path):
    rows = load(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path.name}\n")

    # ----- Part 1: stage share by bucket -----
    print("=" * 100)
    print("STAGE TIME SHARE (% of v9_total) by bucket  [avg over cases]")
    print("=" * 100)
    print("{:<16} {:<6} {:>4} | {:>7} {:>7} {:>7} {:>7} | {:>8} {:>8} | {:>7}".format(
        "bs_tier", "hp", "N", "quant", "dense", "sparse", "comb",
        "v9(ms)", "fp16(ms)", "speed"))

    buckets = defaultdict(list)
    for r in rows:
        buckets[(bs_tier(r["bs"]), hp_tier(r["hp_ratio"]))].append(r)

    for k in sorted(buckets.keys()):
        grp = buckets[k]
        n = len(grp)
        v9 = sum(r["v9_total_ms"] for r in grp) / n
        fp = sum(r["fp16_ms"] for r in grp) / n
        q = sum(r["stage1_quant_ms"] / r["v9_total_ms"] for r in grp) / n
        d = sum(r["stage2_dense_ms"] / r["v9_total_ms"] for r in grp) / n
        s = sum(r["stage3_sparse_ms"] / r["v9_total_ms"] for r in grp) / n
        c = sum(r["stage4_combine_ms"] / r["v9_total_ms"] for r in grp) / n
        sp = sum(r["speedup_vs_fp16"] for r in grp) / n
        print("{:<16} {:<6} {:>4} | {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% | {:>8.3f} {:>8.3f} | {:>6.2f}x".format(
            k[0], k[1], n, 100*q, 100*d, 100*s, 100*c, v9, fp, sp))

    # ----- Part 2: Amdahl upper bound -----
    print()
    print("=" * 100)
    print("UPPER-BOUND: avg speedup if we COMPLETELY eliminate each stage (Amdahl limit)")
    print("=" * 100)
    print("{:<16} {:<6} | {:>9} {:>9} {:>9} {:>9} | {:>9}".format(
        "bs_tier", "hp", "elim_q", "elim_d", "elim_s", "elim_c", "current"))

    for k in sorted(buckets.keys()):
        grp = buckets[k]
        n = len(grp)

        def gain(stage_key, g=grp):
            return sum(r["fp16_ms"] / max(r["v9_total_ms"] - r[stage_key], 1e-6)
                       for r in g) / len(g)

        cur = sum(r["speedup_vs_fp16"] for r in grp) / n
        print("{:<16} {:<6} | {:>8.2f}x {:>8.2f}x {:>8.2f}x {:>8.2f}x | {:>8.2f}x".format(
            k[0], k[1],
            gain("stage1_quant_ms"), gain("stage2_dense_ms"),
            gain("stage3_sparse_ms"), gain("stage4_combine_ms"),
            cur))

    # ----- Part 3: below-1.0x cases -----
    print()
    print("=" * 100)
    print("TOP 15 cases where 0.5x < speedup < 1.0x (closest to beating FP16)")
    print("=" * 100)
    print("{:>6} {:>6} {:>5} {:>5} | {:>8} {:>8} {:>6} | {:>5} {:>5} {:>5} {:>5} | {:>7}".format(
        "d_out", "d_in", "bs", "hp", "v9", "fp16", "spd",
        "q%", "d%", "s%", "c%", "gap_ms"))
    below1 = [r for r in rows if 0.5 < r["speedup_vs_fp16"] < 1.0]
    below1.sort(key=lambda r: -r["speedup_vs_fp16"])
    for r in below1[:15]:
        tot = r["v9_total_ms"]
        print("{:>6} {:>6} {:>5} {:>5.2f} | {:>8.3f} {:>8.3f} {:>5.2f}x | {:>4.1f}% {:>4.1f}% {:>4.1f}% {:>4.1f}% | {:>+7.3f}".format(
            r["d_out"], r["d_in"], r["bs"], r["hp_ratio"],
            r["v9_total_ms"], r["fp16_ms"], r["speedup_vs_fp16"],
            100*r["stage1_quant_ms"]/tot, 100*r["stage2_dense_ms"]/tot,
            100*r["stage3_sparse_ms"]/tot, 100*r["stage4_combine_ms"]/tot,
            r["v9_total_ms"] - r["fp16_ms"]))

    # ----- Part 4: dense / fp16 ratio -----
    print()
    print("=" * 100)
    print("DENSE_ms / FP16_ms ratio (hp=0, pure GEMM compute comparison)")
    print("=" * 100)
    dense_ratios = defaultdict(list)
    for r in rows:
        if r["hp_ratio"] != 0.0:
            continue
        dense_ratios[bs_tier(r["bs"])].append(r["stage2_dense_ms"] / r["fp16_ms"])
    for bt in ["decode(1-16)", "small(32-64)", "mid(128-512)", "prefill(>=2K)"]:
        if bt not in dense_ratios:
            continue
        vals = sorted(dense_ratios[bt])
        print("  {:<16} n={:<3} median={:.2f}x  best={:.2f}x  worst={:.2f}x".format(
            bt, len(vals), vals[len(vals) // 2], vals[0], vals[-1]))

    # ----- Part 5: sparse cost -----
    print()
    print("=" * 100)
    print("SPARSE stage cost (ms) and its share when hp>0")
    print("=" * 100)
    print("{:<16} | {:>5} {:>12} {:>10}".format("bs_tier", "N", "avg_ms", "avg_share"))
    for bt in ["decode(1-16)", "small(32-64)", "mid(128-512)", "prefill(>=2K)"]:
        grp = [r for r in rows if bs_tier(r["bs"]) == bt and r["hp_ratio"] > 0]
        if not grp:
            continue
        n = len(grp)
        avg_ms = sum(r["stage3_sparse_ms"] for r in grp) / n
        avg_frac = sum(r["stage3_sparse_ms"] / r["v9_total_ms"] for r in grp) / n
        print("  {:<16} | {:>5} {:>12.4f} {:>9.1f}%".format(bt, n, avg_ms, 100 * avg_frac))

    # ----- Part 6: dense bandwidth utilisation -----
    print()
    print("=" * 100)
    print("DENSE bandwidth utilisation  (4090 HBM peak ~1008 GB/s)")
    print("bytes/iter = d_out*d_in*0.5 (4-bit W) + bs*d_in (int8 act)")
    print("=" * 100)
    print("{:>6} {:>6} {:>5} | {:>10} {:>10} {:>8} {:>10}".format(
        "d_out", "d_in", "bs", "dense_ms", "GB/s", "vs_peak", "vs_fp16"))
    targets = [(4096, 4096), (11008, 4096), (28672, 4096)]
    for r in rows:
        if r["hp_ratio"] != 0.0:
            continue
        if r["bs"] not in (1, 512, 2048):
            continue
        if (r["d_out"], r["d_in"]) not in targets:
            continue
        bytes_w = r["d_out"] * r["d_in"] * 0.5
        bytes_a = r["bs"] * r["d_in"] * 1
        bytes_total = bytes_w + bytes_a
        gbs = bytes_total / (r["stage2_dense_ms"] / 1000) / 1e9
        pct = 100 * gbs / 1008
        ratio = r["stage2_dense_ms"] / r["fp16_ms"]
        print("{:>6} {:>6} {:>5} | {:>9.3f}ms {:>9.1f} {:>7.1f}% {:>7.2f}x".format(
            r["d_out"], r["d_in"], r["bs"], r["stage2_dense_ms"], gbs, pct, ratio))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
