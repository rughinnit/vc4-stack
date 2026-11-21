#!/usr/bin/env python3
"""Timing for the vc4 backend: wall time, engine-reported GPU time, GFLOP/s.

Run:  cd /home/raspi/work/torch-vc4 && .venv/bin/python tests/bench.py [reps]
"""
import sys
import time

import torch
import torch_vc4


def bench(m, n, k, reps=5):
    a = torch.randn(m, k, device=torch_vc4.device())
    b = torch.randn(k, n, device=torch_vc4.device())
    (a @ b)  # warm: kernel generation + upload
    best = 1e9
    gpu = None
    for _ in range(reps):
        torch_vc4.reset_stats()
        t0 = time.perf_counter()
        c = a @ b
        dt = time.perf_counter() - t0
        s = torch_vc4.stats()
        g = s["kernel_ms"] + s["pack_ms"]
        best = min(best, dt)
        gpu = g if gpu is None else min(gpu, g)
    flops = 2.0 * m * n * k
    ref = a.cpu().double() @ b.cpu().double()
    err = float((c.cpu().double() - ref).abs().max())
    print(f"{m}x{n}x{k}: wall {1e3*best:8.3f} ms ({flops/best/1e9:5.2f} GFLOP/s)"
          f"   engine gpu {gpu:8.3f} ms ({flops/(gpu*1e-3)/1e9:5.2f} GFLOP/s)"
          f"   max|err| {err:.2e}")
    del a, b, c


def bench_ew(n, reps=5):
    a = torch.randn(n, device=torch_vc4.device())
    b = torch.randn(n, device=torch_vc4.device())
    for name, fn in (("relu", lambda: torch.relu(a)), ("add", lambda: a + b),
                     ("mul", lambda: a * b)):
        fn()
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        print(f"elementwise {name} n={n}: {1e3*best:.4f} ms "
              f"({n/best/1e9:.3f} G elem/s)")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(torch_vc4.arena())
    for shape in ((64, 64, 64), (128, 128, 128), (256, 256, 256)):
        bench(*shape, reps=reps)
    bench_ew(256 * 256, reps=reps)
