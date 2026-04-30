"""Standalone VRAM/shape preview — no kernel imports."""

MODELS = [
    ("Qwen3-14B",  5120, 17408, 40, 8),
    ("Qwen2.5-32B", 5120, 27648, 40, 8),
    ("LLaMA3-70B",  8192, 28672, 64, 8),
]
HEAD_DIM = 128

def enumerate_projs(hidden, interm, n_q, n_kv):
    q_out = n_q * HEAD_DIM
    kv_out = n_kv * HEAD_DIM
    return [
        ("q_proj",       hidden,  q_out),
        ("kv_proj",      hidden,  kv_out * 2),
        ("o_proj",       q_out,   hidden),
        ("gate_up_proj", hidden,  interm * 2),
        ("down_proj",    interm,  hidden),
    ]

print(f"{'model':<14} {'proj':<13} {'d_in':>6} {'d_out':>6} {'W_int4_MB':>10}")
total_by_model = {}
for name, h, i, nq, nkv in MODELS:
    tot = 0.0
    for proj, di, do in enumerate_projs(h, i, nq, nkv):
        w_mb = di * do * 0.5 / (1 << 20)
        tot += w_mb
        print(f"{name:<14} {proj:<13} {di:>6} {do:>6} {w_mb:>9.1f}")
    total_by_model[name] = tot
    print(f"{'':14} {'TOTAL per layer':<13} {'':>6} {'':>6} {tot:>9.1f}")
    print()

# Worst-case peak: LLaMA3-70B gate_up T=512
d_in_max = 8192
d_out_max = 28672 * 2
T = 512
print("# Peak transient buffers for LLaMA3-70B T=512 gate_up:")
print(f"  W int4:      {d_in_max * d_out_max * 0.5 / (1<<20):>7.1f} MB")
print(f"  X fp16:      {T * d_in_max * 2 / (1<<20):>7.1f} MB")
print(f"  X_s4 int4:   {T * d_in_max * 0.5 / (1<<20):>7.1f} MB")
print(f"  Y fp16:      {T * d_out_max * 2 / (1<<20):>7.1f} MB")
print(f"  scale_u4 :   {d_out_max * (d_in_max // 128) * 2 / (1<<20):>7.1f} MB")
print(f"  zero_u4 :    {d_out_max * (d_in_max // 128) * 2 / (1<<20):>7.1f} MB")
# Bench setup doubles weights (BF16 reference + INT4): BF16 weight also:
print(f"  BF16 W ref:  {d_in_max * d_out_max * 2 / (1<<20):>7.1f} MB")
print(f"  total (Y_partial+misc ~ +100 MB):  ~{d_in_max * d_out_max * (0.5 + 2) / (1<<20) + 200:.0f} MB")
print()
print("All fits comfortably in RTX 4090 24 GB.  (One shape at a time — bench re-allocs per shape.)")
