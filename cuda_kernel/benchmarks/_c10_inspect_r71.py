import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/Zip_kernel/kernel/cuda_kernel/logs/r71_c83_revert/qwen3_20260502_143226/bench.json"
d = json.load(open(path))
recs = d["records"]

for r in recs:
    m = r.get("model", "")
    p = r.get("proj", "")
    k = r.get("kernel", "")
    if (("32B" in m) or ("70B" in m)) and p == "gate_up_proj" and k == "dense_gemm":
        print(f"{m:<14} {p:<8} T={r['T']:>5} fp16={r['fp16_us']:>9.2f}us cuda={r['cuda_us']:>9.2f}us sp={r['cuda_speedup_vs_fp16']:.3f}x")
