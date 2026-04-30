"""分析 dispatch_sweep 结果：预估每个 shape 在新 heuristic 下会跑哪个 config。"""
import json
from pathlib import Path

data = json.loads(Path('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r62_f2/dispatch_sweep.json').read_text())


def ceil_div(a, b):
    return (a + b - 1) // b


def simulate_new_dispatcher(d_out, d_in, T):
    """新 heuristic 的模拟：只改 split-K, 其他不动。"""
    n_groups = d_in // 128
    n_cta_m_at_128 = ceil_div(d_out, 128)

    # kBm pick (unchanged R44/R52 gate)
    r44_ok = (
        (T <= 8 and d_out <= 4096)
        or (T <= 16 and d_out <= 3072)
        or (16 < T <= 32 and d_out <= 4096 and d_in <= 4096)
        or (48 <= T <= 64 and d_out <= 4096)
        or (T == 96 and d_out <= 2048)
        or (T == 128 and 512 <= d_out <= 2048 and d_in >= 2048)
    )
    kbm = 64 if (r44_ok and n_cta_m_at_128 * ceil_div(T, 32) <= 64) else 128
    n_cta_m = ceil_div(d_out, kbm)

    # kBn pick (unchanged)
    def waves_at(kBn_c):
        return n_cta_m * ceil_div(T, kBn_c)

    if T <= 8:
        kbn = 8
    elif n_groups >= 64 and waves_at(64) >= 32:
        kbn = 64
    elif waves_at(64) >= 128:
        kbn = 64
    elif waves_at(32) >= 64:
        kbn = 32
    else:
        kbn = 8
    # R44 demote
    if kbm == 64 and 32 <= T <= 96 and kbn >= 32 and d_out <= 2048:
        kbn = 8

    # New split-K heuristic (F2)
    hp = 0
    if hp == 0 and n_groups >= 16 and T >= 32:
        grid_mn_at_kbn64 = n_cta_m_at_128 * ceil_div(T, 64)
        want_sk = ceil_div(128, max(grid_mn_at_kbn64, 1))
        if want_sk >= 4: want_sk = 4
        elif want_sk >= 2: want_sk = 2
        else: want_sk = 1
        if want_sk == 4 and n_groups % 4 == 0:
            sk = 4
        elif want_sk >= 2 and n_groups % 2 == 0:
            sk = 2
        else:
            sk = 1
    else:
        sk = 1
    return kbm, kbn, sk


print(f"{'proj':<13} {'T':<4} {'new_cfg':<13} {'pred_us':>8} {'old_auto':>9} {'best':>7} {'Δ vs auto':>10}")
total_saved = 0.0
total_auto = 0.0
missed_opportunity = 0.0
for r in data:
    kbm, kbn, sk = simulate_new_dispatcher(r['d_out'], r['d_in'], r['T'])
    label = f"{kbm}/{kbn}/{sk}"
    # Find in results
    hit = next((x for x in r['results'] if x.get('label') == label), None)
    pred_us = hit['us'] if hit and hit.get('us') is not None else None
    auto_us = r['auto_us']
    best_us = r['best_us']
    delta = (auto_us - pred_us) if pred_us is not None else None
    total_auto += auto_us
    if pred_us is not None:
        total_saved += (auto_us - pred_us)
        missed_opportunity += (pred_us - best_us)
    delta_str = f"{delta:+.2f}us" if delta is not None else "N/A"
    print(f"{r['proj']:<13} {r['T']:<4} {label:<13} "
          f"{(f'{pred_us:.2f}' if pred_us else 'N/A'):>8} "
          f"{auto_us:>8.2f} {best_us:>7.2f} {delta_str:>10}")

print()
print(f"Total auto_us: {total_auto:.1f}")
print(f"Total predicted saved: {total_saved:.1f}us ({total_saved/total_auto*100:.1f}%)")
print(f"Still missed vs best: {missed_opportunity:.1f}us")
