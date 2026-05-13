"""Probe: 14B prefill bs=16 gate_up_fused single launch with CUDA_LAUNCH_BLOCKING.

Goal: identify whether the CUDA illegal-memory-access at T=32768, d_out=34816
is (a) a kernel-internal shape assumption, (b) an HBM allocation failure,
or (c) something else. We launch the SAME path the bench uses (legacy_mma:
activation_quant_cuda + fused_dense_sparse_cuda_int4) in isolation, in a
fresh CUDA context (launch-blocking), with meminfo before/after each step.
"""

import os, sys, traceback, gc
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.insert(0, "/root")

from kernel.bench.configs.qwen3_shapes import QWEN3_BY_NAME
from kernel.bench.layer.qwen3_w4a4_ops import build_four_op_callables

def mb(x): return x / (1024**2)

def meminfo(tag):
    free, tot = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"[mem @ {tag:<25}] free={mb(free):8.1f} MiB  alloc={mb(alloc):8.1f} MiB  reserved={mb(reserved):8.1f} MiB  total={mb(tot):8.1f} MiB", flush=True)

device = torch.device("cuda:0")
print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
meminfo("startup")

# 14B prefill seqlen=2048, bs=16 → T=32768
cfg = QWEN3_BY_NAME["Qwen3-14B"]
print(f"cfg: {cfg.name}  hidden={cfg.hidden}  q_out={cfg.q_out}  kv_out={cfg.kv_out}  intermediate={cfg.intermediate}", flush=True)

try:
    print("\n=== building four-op bundles (bs=16, seqlen=2048) ===", flush=True)
    four = build_four_op_callables(cfg, batch=16, seqlen=2048, device=device)
    meminfo("after bundle build")

    for op_name, bundle, fn in four.as_list():
        print(f"\n--- op={op_name}  T={bundle.T}  d_in={bundle.d_in}  d_out={bundle.d_out}  path={bundle.path} ---", flush=True)
        meminfo(f"before {op_name} launch")
        try:
            y = fn()
            torch.cuda.synchronize(device)
            print(f"  launch OK, output shape = {tuple(y.shape)} dtype={y.dtype}", flush=True)
            meminfo(f"after {op_name} launch")
            del y
        except Exception as e:
            print(f"  LAUNCH FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            meminfo(f"after {op_name} FAILURE")
            break
except Exception as e:
    print(f"BUNDLE BUILD FAILED: {e}", flush=True)
    traceback.print_exc()

meminfo("end")
