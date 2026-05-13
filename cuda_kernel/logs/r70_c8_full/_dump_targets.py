"""Dump r70 C.8 speedups for the 5 C.8 TARGET shapes + quick_verify's 5 losers.

Purpose: resolve the conflict between c8_quick_verify.py and bench_qwen3_shapes
on the same (T=1024 or T=2048) loser shapes.
"""
import json
from pathlib import Path

R70 = "/root/Zip_kernel/kernel/cuda_kernel/logs/r70_c8_full/qwen3_20260502_141424/bench.json"
R69 = "/root/Zip_kernel/kernel/cuda_kernel/logs/r69_c7_prefill/bench.json"

d70 = json.load(open(R70))
d69 = json.load(open(R69))

def find(data, m, p, T, din, dout):
    for r in data["records"]:
        if (r.get("kernel") == "end_to_end" and r["model"] == m
                and r["proj"] == p and r["T"] == T
                and r["d_in"] == din and r["d_out"] == dout):
            return r
    return None

# C.8 TARGET shapes — inspect at ALL available T (r70 has 1..1024; r69 has 1024..8192)
# The 5 quick_verify losers: they all probed T=2048 (32B gu, 70B gu) or T=1024.
TARGETS = [
    ("Qwen2.5-32B", "gate_up_proj", 5120, 55296),
    ("LLaMA3-70B",  "gate_up_proj", 8192, 57344),
    ("LLaMA3-70B",  "kv_proj",      8192,  2048),
    ("Qwen3-1.7B",  "down_proj",    6144,  2048),
    ("Qwen3-4B",    "down_proj",    9728,  2560),
]

print("{:<14} {:<14} {:>5} {:>5} {:>6} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7}".format(
    "model", "proj", "T", "d_in", "d_out",
    "fp16_r69", "cuda_r69", "sp_r69",
    "fp16_r70", "cuda_r70", "sp_r70"))
print("-" * 110)
for m, p, din, dout in TARGETS:
    for T in [1, 8, 128, 512, 1024, 2048, 4096, 8192]:
        r69 = find(d69, m, p, T, din, dout)
        r70 = find(d70, m, p, T, din, dout)
        if r69 is None and r70 is None:
            continue
        f69 = f"{r69['fp16_us']:.1f}" if r69 else "-"
        c69 = f"{r69['cuda_us']:.1f}" if r69 else "-"
        s69 = f"{r69['cuda_speedup_vs_fp16']:.3f}x" if r69 else "-"
        f70 = f"{r70['fp16_us']:.1f}" if r70 else "-"
        c70 = f"{r70['cuda_us']:.1f}" if r70 else "-"
        s70 = f"{r70['cuda_speedup_vs_fp16']:.3f}x" if r70 else "-"
        print("{:<14} {:<14} {:>5} {:>5} {:>6} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7}".format(
            m, p, T, din, dout, f69, c69, s69, f70, c70, s70))
    print()
