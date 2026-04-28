"""One-off sanity check: compare manual roofline math against CSV output."""
import csv
from pathlib import Path

HBM = 1008e9 * 0.85
FP16 = 165.2e12 * 0.85
INT4 = 660.6e12 * 0.85


def fp16_roof(T, di, do):
    flops = 2 * T * di * do
    byts = (di * do + T * di + T * do) * 2
    return max(flops / FP16, byts / HBM) * 1e6


def cuda_T1(di, do):
    ng = di // 128
    byts = di * 2 + di * do * 0.5 + do * ng * 2 * 2 + do * 2
    flops = 2 * di * do
    return max(flops / INT4, byts / HBM) * 1e6


def cuda_multi(T, di, do):
    ng = di // 128
    bq = T * di * 2 + T * di / 2 + T * 2 + T * ng * 4
    tq = bq / HBM * 1e6
    bg = di * do * 0.5 + T * di * 0.5 + do * ng * 2 * 2 + T * do * 2
    fg = 2 * T * di * do
    tg = max(fg / INT4, bg / HBM) * 1e6
    return tq, tg, tq + tg


print("A manual fp16:", round(fp16_roof(1, 1024, 2048), 3))
print("A manual cuda:", round(cuda_T1(1024, 2048), 3))
q, g, s = cuda_multi(128, 2048, 2048)
print("B manual fp16:", round(fp16_roof(128, 2048, 2048), 3),
      "cuda q/g/sum:", round(q, 3), round(g, 3), round(s, 3))
q, g, s = cuda_multi(1024, 4096, 4096)
print("C manual fp16:", round(fp16_roof(1024, 4096, 4096), 3),
      "cuda q/g/sum:", round(q, 3), round(g, 3), round(s, 3))

CSV_PATH = (Path(__file__).resolve().parent.parent
            / "logs" / "qwen3_20260428_111515" / "roofline_compare.csv")
with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

checks = [
    ("Qwen3-0.6B", "q_proj", "1", 1024, 2048),
    ("Qwen3-1.7B", "o_proj", "128", 2048, 2048),
    ("Qwen3-8B", "q_proj", "1024", 4096, 4096),
]
for (m, p, T, di, do) in checks:
    hits = [x for x in rows
            if x["model"] == m and x["proj"] == p and x["T"] == T
            and int(x["d_in"]) == di and int(x["d_out"]) == do]
    if hits:
        h = hits[0]
        print(f"CSV  {m} {p} T={T}: fp16_roof={h['fp16_roof_us']} "
              f"quant={h['cuda_quant_roof_us']} gemm={h['cuda_gemm_roof_us']} "
              f"cuda_roof={h['cuda_roof_us']}")
    else:
        print("NOT FOUND", m, p, T, di, do)
