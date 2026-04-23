"""Compare two sweep_v9 CSVs and print bucketed deltas."""
import csv
import sys


def load(p):
    return {
        (int(r['d_out']), int(r['d_in']), int(r['bs']), float(r['hp_ratio'])): r
        for r in csv.DictReader(open(p))
    }


def bucket(bs):
    if bs <= 16:
        return 'decode(<=16)'
    if bs <= 64:
        return 'small(32..64)'
    if bs <= 512:
        return 'mid(128..512)'
    return 'prefill(>=1024)'


def main(base_path, new_path):
    b = load(base_path)
    n = load(new_path)
    common = [k for k in b if k in n]
    print(f"base_rows={len(b)}, new_rows={len(n)}, common={len(common)}\n")

    agg = {}
    for k in common:
        tier = bucket(k[2])
        hpt = 'hp=0' if k[3] == 0.0 else 'hp>0'
        key = (tier, hpt)
        if key not in agg:
            agg[key] = {'v9_b': 0, 'v9_n': 0, 'q_b': 0, 'q_n': 0, 'n': 0, 'fp': 0}
        agg[key]['v9_b'] += float(b[k]['v9_total_ms'])
        agg[key]['v9_n'] += float(n[k]['v9_total_ms'])
        agg[key]['q_b'] += float(b[k]['stage1_quant_ms'])
        agg[key]['q_n'] += float(n[k]['stage1_quant_ms'])
        agg[key]['fp'] += float(n[k]['fp16_ms'])
        agg[key]['n'] += 1

    hdr = f"{'bucket':<18}{'hp':>5} {'N':>4}{'q_base':>10}{'q_new':>10}{'dq%':>8}{'v9_base':>10}{'v9_new':>10}{'dv%':>8}{'spdup':>9}"
    print(hdr)
    print('-' * len(hdr))
    for (t, h) in sorted(agg):
        r = agg[(t, h)]
        dq = 100 * (r['q_n'] - r['q_b']) / r['q_b'] if r['q_b'] > 0 else 0
        dv = 100 * (r['v9_n'] - r['v9_b']) / r['v9_b'] if r['v9_b'] > 0 else 0
        sp = r['fp'] / r['v9_n'] if r['v9_n'] > 0 else 0
        print(f"{t:<18}{h:>5} {r['n']:>4}{r['q_b']:>10.3f}{r['q_n']:>10.3f}{dq:>+7.1f}%{r['v9_b']:>10.3f}{r['v9_n']:>10.3f}{dv:>+7.1f}%{sp:>8.3f}x")

    # Top individual quant reductions (abs us)
    def q_abs_us(k, src):
        return float(src[k]['stage1_quant_ms']) * 1000

    print('\n--- TOP 10 per-case quant absolute reductions ---')
    best = sorted(common, key=lambda k: (q_abs_us(k, n) - q_abs_us(k, b)))
    for k in best[:10]:
        qb = q_abs_us(k, b); qn = q_abs_us(k, n)
        vb = float(b[k]['v9_total_ms']); vn = float(n[k]['v9_total_ms'])
        dq = 100 * (qn - qb) / qb if qb > 0 else 0
        dv = 100 * (vn - vb) / vb if vb > 0 else 0
        print(f"  d_out={k[0]:>5} d_in={k[1]:>5} bs={k[2]:>5} hp={k[3]:.2f}:  q {qb:>7.1f}us -> {qn:>7.1f}us ({dq:+6.1f}%)  v9 {vb:>7.3f}ms -> {vn:>7.3f}ms ({dv:+6.1f}%)")

    print('\n--- TOP 5 per-case v9 regressions (if any) ---')
    worst = sorted(common, key=lambda k: -(float(n[k]['v9_total_ms']) - float(b[k]['v9_total_ms'])) / float(b[k]['v9_total_ms']) if float(b[k]['v9_total_ms']) > 0 else 0)
    for k in worst[:5]:
        qb = q_abs_us(k, b); qn = q_abs_us(k, n)
        vb = float(b[k]['v9_total_ms']); vn = float(n[k]['v9_total_ms'])
        dv = 100 * (vn - vb) / vb if vb > 0 else 0
        print(f"  d_out={k[0]:>5} d_in={k[1]:>5} bs={k[2]:>5} hp={k[3]:.2f}:  q {qb:>6.1f}us -> {qn:>6.1f}us  v9 {vb:>7.3f} -> {vn:>7.3f} ({dv:+6.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: compare_sweeps.py <base.csv> <new.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
