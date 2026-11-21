"""Per-op cost: vc4 wall time vs the CPU, small and medium tensors.

Run on the Pi:  .venv/bin/python tests/bench_ops.py

Every vc4 launch is a synchronous LAUNCH_WAIT round trip to vc4d (~0.2-0.5 ms
per launch on this board), so the interesting number for small tensors is
"how many launches does this op cost", and the honest headline is that almost
everything small is launch-bound and slower than the CPU.  The script prints
both the wall times and the launches per call (from vc4_get_stats).
"""

import sys
import time

import torch
import torch.nn.functional as F

import torch_vc4 as tv

D = tv.device()
REPS = 5


def timeit(fn, reps=REPS):
    fn()                                     # warm (JIT, cache lines)
    torch_vc4_sync()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3                        # ms


def torch_vc4_sync():
    tv.synchronize()                          # no-op, every launch is sync


def launches_per_call(fn, n=3):
    tv.reset_stats()
    for _ in range(n):
        fn()
    return tv.stats()["launches"] / n


def U(t):
    return t.to(D)


def bench(name, dev_fn, cpu_fn):
    vc = timeit(dev_fn)
    cp = timeit(cpu_fn)
    ln = launches_per_call(dev_fn)
    print("%-30s vc4 %8.2f ms   cpu %8.3f ms   %6.1fx   %4.1f launches"
          % (name, vc, cp, vc / cp, ln))
    return vc, cp, ln


def main():
    torch.manual_seed(0)
    print("per-op wall time, best of %d, stock clocks; sync every launch\n"
          % REPS)
    print("%-30s %-12s %-12s %-8s %s" % ("op", "vc4", "cpu", "ratio",
                                         "launches"))

    # ---- matmul / linear (256^3 is where the QPU overtakes this ARM core)
    for n in (32, 128, 256):
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        av, bv = U(a), U(b)
        bench("mm %dx%dx%d" % (n, n, n), lambda: torch.mm(av, bv),
              lambda: torch.mm(a, b))
    x = torch.randn(1, 1024)
    w = torch.randn(4, 1024)
    xv, wv = U(x), U(w)
    bench("linear 1x1024x4", lambda: F.linear(xv, wv),
          lambda: F.linear(x, w))

    # ---- conv2d
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(16, 3, 3, 3)
    xv, wv = U(x), U(w)
    bench("conv 3->16 k3 s1 p1 8x8", lambda: F.conv2d(xv, wv, None, padding=1),
          lambda: F.conv2d(x, w, None, padding=1))
    x = torch.randn(1, 16, 32, 32)
    w = torch.randn(16, 16, 3, 3)
    xv, wv = U(x), U(w)
    bench("conv 16->16 k3 s1 p1 32x32",
          lambda: F.conv2d(xv, wv, None, padding=1),
          lambda: F.conv2d(x, w, None, padding=1))
    w1 = torch.randn(16, 16, 1, 1)
    w1v = U(w1)
    bench("conv 16->16 k1 s1 32x32",
          lambda: F.conv2d(xv, w1v, None),
          lambda: F.conv2d(x, w1, None))
    wd = torch.randn(16, 1, 3, 3)
    wdv = U(wd)
    bench("conv depthwise 16 k3 32x32",
          lambda: F.conv2d(xv, wdv, None, padding=1, groups=16),
          lambda: F.conv2d(x, wd, None, padding=1, groups=16))
    bench("conv 16->16 k3 s2 p1 32x32",
          lambda: F.conv2d(xv, wv, None, stride=2, padding=1),
          lambda: F.conv2d(x, w, None, stride=2, padding=1))

    # ---- pooling
    x = torch.randn(1, 16, 16, 16)
    xv = U(x)
    bench("max_pool2d 2x2 16x16", lambda: F.max_pool2d(xv, 2),
          lambda: F.max_pool2d(x, 2))
    bench("max_pool2d 3x3 s2 p1", lambda: F.max_pool2d(xv, 3, 2, 1),
          lambda: F.max_pool2d(x, 3, 2, 1))
    bench("adaptive_avg_pool2d 1x1", lambda: F.adaptive_avg_pool2d(xv, 1),
          lambda: F.adaptive_avg_pool2d(x, 1))

    # ---- elementwise / softmax / resize
    for n in (1024, 65536):
        x = torch.randn(n)
        xv = U(x)
        bench("relu %d" % n, lambda: torch.relu(xv), lambda: torch.relu(x))
        bench("sigmoid %d" % n, lambda: torch.sigmoid(xv),
              lambda: torch.sigmoid(x))
    x = torch.randn(64, 10)
    xv = U(x)
    bench("softmax 64x10", lambda: F.softmax(xv, -1), lambda: F.softmax(x, -1))
    x = torch.randn(1, 16, 8, 8)
    xv = U(x)
    bench("upsample 2x 16x8x8",
          lambda: F.interpolate(xv, scale_factor=2, mode="nearest"),
          lambda: F.interpolate(x, scale_factor=2, mode="nearest"))
    a = torch.randn(1, 16, 8, 8)
    b = torch.randn(1, 16, 8, 8)
    av, bv = U(a), U(b)
    bench("cat 16+16 x 8x8", lambda: torch.cat([av, bv], 1),
          lambda: torch.cat([a, b], 1))
    x = torch.randn(1, 16, 16, 16)
    xv = U(x)
    rm, rv = torch.zeros(16), torch.ones(16)
    g, be = torch.ones(16), torch.zeros(16)
    bench("batch_norm eval 16x16x16",
          lambda: F.batch_norm(xv, U(rm), U(rv), U(g), U(be), False, .1, 1e-5),
          lambda: F.batch_norm(x, rm, rv, g, be, False, .1, 1e-5))
    x = torch.randn(1024)
    xv = U(x)
    bench("sum 1024", lambda: xv.sum(), lambda: x.sum())
    x = torch.randn(4096)
    xv = U(x)
    bench("sum 4096 (4 GEMM chunks)", lambda: xv.sum(), lambda: x.sum())

    # ---- the floor: one dispatch + one launch
    x = torch.randn(64)
    xv = U(x)
    print()
    t = timeit(lambda: torch.zeros(4, device=D), 20)
    print("torch.zeros(4) on vc4 (empty + host fill): %.3f ms" % t)
    t = timeit(lambda: xv.view(64), 200)
    print("view (metadata only, no launch):            %.4f ms" % t)
    t = timeit(lambda: torch.relu(xv), 20)
    print("relu 64 floats (one launch):                %.3f ms" % t)


if __name__ == "__main__":
    sys.exit(main())
