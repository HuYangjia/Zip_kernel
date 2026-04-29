"""Micro-probe: measure pure CUDA launch overhead on the autodl GPU.

Launches a trivial no-op kernel N times, to establish the floor of what
activation_quant can ever hope to achieve at T=1 D=4096.
"""
import statistics
import torch


@torch.no_grad()
def bench(fn, warmup=500, outer=15, inner=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(outer):
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e1.record()
        for _ in range(inner):
            fn()
        e2.record()
        torch.cuda.synchronize()
        s.append(e1.elapsed_time(e2) / inner * 1000.0)
    return statistics.median(s)


def main():
    x = torch.empty(1, device="cuda")

    # Kernel-free torch op (in-place no-op).
    t1 = bench(lambda: x.zero_())
    # Two-launch pattern.
    t2 = bench(lambda: (x.zero_(), x.zero_()))
    # Single empty-tensor alloc.
    t3 = bench(lambda: torch.empty(16, device="cuda"))
    # Launch overhead pure probe.
    import torch.nn.functional as F
    y = torch.empty(16, device="cuda")
    t4 = bench(lambda: torch.add(x, x, out=x))

    print(f"x.zero_():         {t1:.2f} us")
    print(f"2x x.zero_():      {t2:.2f} us")
    print(f"torch.empty(16):   {t3:.2f} us")
    print(f"torch.add:         {t4:.2f} us")


if __name__ == "__main__":
    main()
