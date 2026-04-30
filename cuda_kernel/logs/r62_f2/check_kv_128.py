"""检查 kv_proj T=128 / q_proj T=128 等 shape 的所有 config 时间，理解 heuristic 边界"""
import json
from pathlib import Path

data = json.loads(Path(__file__).parent.joinpath('dispatch_sweep.json').read_text())

targets = [
    ('kv_proj', 128),
    ('kv_proj', 32),
    ('kv_proj', 512),
    ('q_proj', 128),
    ('o_proj', 128),
]
for r in data:
    if (r['proj'], r['T']) not in targets:
        continue
    print(f"\n=== {r['proj']} T={r['T']} (d_out={r['d_out']}, d_in={r['d_in']}, ng={r['n_groups']}) ===")
    print(f"    auto: {r['auto_us']:.2f}us")
    configs = sorted(r['results'], key=lambda c: c.get('us') or float('inf'))
    for c in configs:
        us = c.get('us')
        if us is None:
            continue
        print(f"    {c['label']:<12} {us:6.2f}us")
