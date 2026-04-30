"""C.2 — kBn sweep (subprocess-isolated, CLEAN timing).

The previous in-process version suffered from GPU clock / L2 cache
transients between bench runs, making 'auto' mode measurements
inconsistent by up to 40%.  This version spawns a fresh Python
process for each (shape, mode) pair so each timing is measured
against a cold GPU state (relative to previous modes).

Cost: ~15-25 s JIT-import per process × (34 shapes × 5 modes) =
about 40 min.  Acceptable for a final-verification sweep.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sweep_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKER = HERE / "_c2_kbn_worker.py"
WORKER.write_text('''"""Inner worker for c2 sweep — one (shape, kbn_mode) run."""
import os, sys, json

import torch
import kernel.cuda_kernel.ops as ops

args = json.loads(sys.argv[1])
T, d_in, d_out = args["T"], args["d_in"], args["d_out"]
dev = torch.device("cuda:0")

torch.manual_seed(0)
X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
perm = torch.randperm(d_in, device=dev).to(torch.int32)
W_low = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
n_g = d_in // 128
scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()
empty_hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
hp_ro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
hp_ci = torch.zeros(0, dtype=torch.int32, device=dev)
X_s4, scale_x, sum_X = ops.activation_quant_cuda(X, perm)

def run():
    ops.fused_dense_sparse_cuda_int4(
        W_low, empty_hpb, hp_ro, hp_ci,
        X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
    )

# bmmiahpl timing: warmup=500, outer=20, inner=200, min-over-outer.
for _ in range(500):
    run()
torch.cuda.synchronize()
best = float("inf")
for _ in range(20):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(200):
        run()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000.0 / 200
    best = min(best, us)
print(json.dumps({"us": best}))
''')


SHAPES = []
def add(model, proj, T, d_in, d_out):
    SHAPES.append(dict(model=model, proj=proj, T=T, d_in=d_in, d_out=d_out))

for T in (32, 128):
    add("Qwen3-0.6B", "q",  T, 1024, 2048)
    add("Qwen3-0.6B", "o",  T, 2048, 1024)
    add("Qwen3-0.6B", "gu", T, 1024, 6144)
    add("Qwen3-0.6B", "dn", T, 3072, 1024)
    add("Qwen3-1.7B", "q",  T, 2048, 2048)
    add("Qwen3-1.7B", "gu", T, 2048, 12288)
    add("Qwen3-1.7B", "dn", T, 6144, 2048)
    add("Qwen3-4B",   "q",  T, 2560, 4096)
    add("Qwen3-4B",   "gu", T, 2560, 18432)
    add("Qwen3-4B",   "dn", T, 9216, 2560)
    add("Qwen3-8B",   "q",  T, 4096, 4096)
    add("Qwen3-8B",   "kv", T, 4096, 2048)
    add("Qwen3-8B",   "gu", T, 4096, 24576)
    add("Qwen3-8B",   "dn", T, 14336, 4096)
    add("Qwen3-14B",  "q",  T, 5120, 5120)
    add("Qwen3-14B",  "gu", T, 5120, 34816)
    add("Qwen3-14B",  "dn", T, 17408, 5120)

MODES = [("auto", None), ("k8", "8"), ("k16", "16"), ("k32", "32"), ("k64", "64")]


def run_one(sh, kbn_env):
    env = os.environ.copy()
    if kbn_env is None:
        env.pop("HKUST_V9_FUSED_FORCE_KBN", None)
    else:
        env["HKUST_V9_FUSED_FORCE_KBN"] = kbn_env
    proc = subprocess.run(
        [sys.executable, str(WORKER), json.dumps(sh)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    if proc.returncode != 0:
        return float("nan"), proc.stderr[-300:]
    for line in reversed(proc.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)["us"], None
            except Exception:
                pass
    return float("nan"), proc.stdout[-300:]


def main():
    rows = []
    header = (f"{'model':<10} {'proj':<4} {'T':>4} {'d_in':>5} {'d_out':>6} " +
              " ".join(f"{lbl:>7}" for lbl, _ in MODES) +
              f"  {'best':>6} {'gain%':>6}")
    print(header)
    print("-" * len(header))
    for i, sh in enumerate(SHAPES):
        us_by_mode = {}
        for lbl, env in MODES:
            us, err = run_one(sh, env)
            us_by_mode[lbl] = us
            if err:
                print(f"[{i+1}] {sh} {lbl}: FAILED {err[:80]}", flush=True)
        auto_us = us_by_mode["auto"]
        forced = {k: v for k, v in us_by_mode.items() if k != "auto"}
        best_mode = min(forced, key=lambda k: forced[k] if forced[k] == forced[k] else 1e9)
        gain = (auto_us - forced[best_mode]) / auto_us * 100 if auto_us == auto_us else 0
        row = {**sh, "us_by_mode": us_by_mode, "best_forced": best_mode,
               "gain_pct_vs_auto": gain}
        rows.append(row)
        print(f"{sh['model']:<10} {sh['proj']:<4} {sh['T']:>4} {sh['d_in']:>5} {sh['d_out']:>6} " +
              " ".join(f"{us_by_mode[lbl]:>7.2f}" for lbl, _ in MODES) +
              f"  {best_mode:>6} {gain:>+5.1f}%", flush=True)

    out_path = OUT_DIR / "c2_kbn_sweep_subprocess.json"
    out_path.write_text(json.dumps({"rows": rows, "n": len(rows)}, indent=2))
    print(f"\nWrote {out_path}")

    # Wins
    k16_wins = [r for r in rows if r["best_forced"] == "k16" and r["gain_pct_vs_auto"] > 3.0]
    print(f"\nShapes where kBn=16 beats auto by >3%: {len(k16_wins)}")
    for r in sorted(k16_wins, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']:<10} {r['proj']:<4} T={r['T']:>3} d=({r['d_in']},{r['d_out']})  "
              f"gain {r['gain_pct_vs_auto']:+.1f}%")

    other_wins = [r for r in rows if r["best_forced"] != "k16" and r["gain_pct_vs_auto"] > 3.0]
    print(f"\nDispatcher oversights (non-k16 mode beats auto >3%): {len(other_wins)}")
    for r in sorted(other_wins, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']:<10} {r['proj']:<4} T={r['T']:>3} d=({r['d_in']},{r['d_out']}) "
              f"best={r['best_forced']} gain {r['gain_pct_vs_auto']:+.1f}%")


if __name__ == "__main__":
    main()
