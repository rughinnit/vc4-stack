"""Measure the COMPLETENESS.md op table: accuracy vs the CPU and wall time at
three sizes (vc4 through vc4d vs the same PyTorch op on the CPU).

Run on the Pi:  .venv/bin/python tests/completeness.py > completeness.table
"""

import time

import torch
import torch.nn.functional as F

import torch_vc4 as tv

torch.set_num_threads(4)
D = tv.device()
REPS = 5
ROWS = []


def U(t):
    return t.to(D)


def timeit(fn, reps=REPS):
    """Median of `reps` after two warm-up calls (protocol: threads=4)."""
    fn()
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def err(got, ref):
    g, r = got.cpu().double(), ref.double()
    e = (g - r).abs()
    return (e.max().item() if e.numel() else 0.0,
            (e / r.abs().clamp_min(1.0)).max().item() if e.numel() else 0.0)


def row(name, backend, sizes, build):
    vs, cs = [], []
    ma = mr = 0.0
    for sh in sizes:
        df, cf, got, ref = build(sh)
        if got is not None:
            a, b = err(got, ref)
            ma, mr = max(ma, a), max(mr, b)
        vs.append(timeit(df))
        cs.append(timeit(cf))
    ROWS.append((name, backend, ma, mr, vs, cs))
    print("%-38s %-9s %.2e/%.2e  vc4 %s  cpu %s"
          % (name, backend, ma, mr,
             "/".join("%.3f" % v for v in vs),
             "/".join("%.4f" % v for v in cs)), flush=True)


def main():
    torch.manual_seed(3)
    print("# raw completeness measurements (generated)\n")
    print("# protocol: Pi 3B stock clocks, torch.set_num_threads(4),")
    print("#           vc4d active, no other GPU client, median of %d" % REPS)
    print("#           after 2 warm-ups; vc4 wall includes the vc4d round trip.\n")

    # gemm
    def b_mm(sh):
        n, k, m = sh
        a, b = torch.randn(n, k), torch.randn(k, m)
        av, bv = U(a), U(b)
        return (lambda: torch.mm(av, bv), lambda: torch.mm(a, b),
                torch.mm(av, bv), torch.mm(a, b))
    row("aten::mm", "device", [(32, 32, 32), (128, 128, 128),
                               (256, 256, 256)], b_mm)
    n = 256
    a, b = torch.randn(n, n), torch.randn(n, n)
    av, bv = U(a), U(b)
    torch.mm(av, bv)
    tv.reset_stats()
    for _ in range(5):
        torch.mm(av, bv)
    st = tv.stats()
    print("# mm 256^3: engine GPU %.3f ms/call (kernel+pack), %d launches/5\n"
          % (st["kernel_ms"] / 5.0, int(st["launches"])))

    def b_linear(sh):
        n, k, m = sh
        x, w, b = torch.randn(n, k), torch.randn(m, k), torch.randn(m)
        xv, wv, bv = U(x), U(w), U(b)
        return (lambda: F.linear(xv, wv, bv), lambda: F.linear(x, w, b),
                F.linear(xv, wv, bv), F.linear(x, w, b))
    row("aten::linear", "device",
        [(1, 256, 64), (32, 256, 64), (128, 512, 128)], b_linear)

    # conv
    def b_conv(sh):
        n, c, h, w, co, k, s, p = sh
        x, wt = torch.randn(n, c, h, w), torch.randn(co, c, k, k)
        xv, wv = U(x), U(wt)
        return (lambda: F.conv2d(xv, wv, padding=p, stride=s),
                lambda: F.conv2d(x, wt, padding=p, stride=s),
                F.conv2d(xv, wv, padding=p, stride=s),
                F.conv2d(x, wt, padding=p, stride=s))
    row("conv2d dense s1", "device",
        [(1, 3, 8, 8, 8, 3, 1, 1), (2, 16, 32, 32, 16, 3, 1, 1),
         (1, 32, 32, 32, 32, 3, 1, 1)], b_conv)
    row("conv2d dense s2", "device (phase decomposition)",
        [(1, 3, 16, 16, 8, 3, 2, 1), (2, 16, 32, 32, 16, 3, 2, 1),
         (1, 32, 32, 32, 32, 3, 2, 1)], b_conv)

    def b_dw(sh):
        n, c, h, w = sh
        x, wt = torch.randn(n, c, h, w), torch.randn(c, 1, 3, 3)
        xv, wv = U(x), U(wt)
        return (lambda: F.conv2d(xv, wv, padding=1, groups=c),
                lambda: F.conv2d(x, wt, padding=1, groups=c),
                F.conv2d(xv, wv, padding=1, groups=c),
                F.conv2d(x, wt, padding=1, groups=c))
    row("conv2d depthwise 3x3", "device",
        [(1, 6, 16, 16), (2, 16, 16, 16), (1, 32, 24, 24)], b_dw)

    # pools / resize
    def b_pool(sh):
        n, c, h, w = sh
        x = torch.randn(n, c, h, w)
        xv = U(x)
        return (lambda: F.max_pool2d(xv, 2), lambda: F.max_pool2d(x, 2),
                F.max_pool2d(xv, 2), F.max_pool2d(x, 2))
    row("max_pool2d 2x2", "device",
        [(1, 4, 16, 16), (2, 16, 32, 32), (1, 32, 64, 32)], b_pool)

    def b_apool(sh):
        n, c, h, w = sh
        x = torch.randn(n, c, h, w)
        xv = U(x)
        return (lambda: F.adaptive_avg_pool2d(xv, 1),
                lambda: F.adaptive_avg_pool2d(x, 1),
                F.adaptive_avg_pool2d(xv, 1), F.adaptive_avg_pool2d(x, 1))
    row("adaptive_avg_pool2d 1x1", "composite",
        [(1, 4, 16, 16), (2, 16, 32, 32), (1, 32, 64, 32)], b_apool)

    def b_up(sh):
        n, c, h, w = sh
        x = torch.randn(n, c, h, w)
        xv = U(x)
        return (lambda: F.interpolate(xv, scale_factor=2, mode="nearest"),
                lambda: F.interpolate(x, scale_factor=2, mode="nearest"),
                F.interpolate(xv, scale_factor=2, mode="nearest"),
                F.interpolate(x, scale_factor=2, mode="nearest"))
    row("upsample_nearest2d 2x", "composite",
        [(1, 4, 8, 8), (2, 16, 16, 16), (1, 32, 32, 32)], b_up)

    def b_cat(sh):
        n, c, h, w = sh
        a, b = torch.randn(n, c, h, w), torch.randn(n, c, h, w)
        av, bv = U(a), U(b)
        return (lambda: torch.cat([av, bv], 1), lambda: torch.cat([a, b], 1),
                torch.cat([av, bv], 1), torch.cat([a, b], 1))
    row("aten::cat (channel)", "composite (host byte copy)",
        [(1, 8, 16, 16), (2, 16, 32, 32), (1, 32, 64, 64)], b_cat)

    # softmax
    def b_sm(sh):
        rows, cols = sh
        x = torch.randn(rows, cols)
        xv = U(x)
        return (lambda: F.softmax(xv, -1), lambda: F.softmax(x, -1),
                F.softmax(xv, -1), F.softmax(x, -1))
    row("softmax <=16 classes", "device",
        [(8, 10), (64, 10), (512, 16)], b_sm)
    row("softmax >16 classes", "composite (sigmoid/exp + max/sum)",
        [(8, 17), (64, 100), (512, 256)], b_sm)

    # elementwise
    def b_relu(sh):
        n = sh
        x = torch.randn(n)
        xv = U(x)
        return (lambda: torch.relu(xv), lambda: torch.relu(x),
                torch.relu(xv), torch.relu(x))
    row("aten::relu", "device", [64, 4096, 65536], b_relu)

    def b_add(sh):
        n = sh
        a, b = torch.randn(n), torch.randn(n)
        av, bv = U(a), U(b)
        return (lambda: av + bv, lambda: a + b, av + bv, a + b)
    row("aten::add.Tensor (same shape)", "device",
        [64, 4096, 65536], b_add)

    def b_badd(sh):
        n = sh
        a, b = torch.randn(n, n), torch.randn(n, 1)
        av, bv = U(a), U(b)
        return (lambda: av + bv, lambda: a + b, av + bv, a + b)
    row("aten::add.Tensor (broadcast)", "composite (expand copy + device)",
        [16, 64, 256], b_badd)

    def b_abs(sh):
        n = sh
        x = torch.randn(n)
        xv = U(x)
        return (lambda: torch.abs(xv), lambda: torch.abs(x),
                torch.abs(xv), torch.abs(x))
    row("aten::abs", "device", [64, 4096, 65536], b_abs)

    def b_floor(sh):
        n = sh
        x = torch.randn(n) * 3e7
        xv = U(x)
        return (lambda: torch.floor(xv), lambda: torch.floor(x),
                torch.floor(xv), torch.floor(x))
    row("aten::floor", "device", [64, 4096, 65536], b_floor)

    def b_ceil(sh):
        n = sh
        x = torch.randn(n) * 3e7
        xv = U(x)
        return (lambda: torch.ceil(xv), lambda: torch.ceil(x),
                torch.ceil(xv), torch.ceil(x))
    row("aten::ceil", "device", [64, 4096, 65536], b_ceil)

    def b_round(sh):
        n = sh
        x = torch.randn(n) * 1e7
        xv = U(x)
        return (lambda: torch.round(xv), lambda: torch.round(x),
                torch.round(xv), torch.round(x))
    row("aten::round", "device", [64, 4096, 65536], b_round)

    def b_exp(sh):
        n = sh
        x = torch.randn(n) * 4.0
        xv = U(x)
        return (lambda: torch.exp(xv), lambda: torch.exp(x),
                torch.exp(xv), torch.exp(x))
    row("aten::exp", "device", [64, 4096, 65536], b_exp)

    def b_log(sh):
        n = sh
        x = torch.rand(n) * 1e4 + 1e-6
        xv = U(x)
        return (lambda: torch.log(xv), lambda: torch.log(x),
                torch.log(xv), torch.log(x))
    row("aten::log", "device", [64, 4096, 65536], b_log)

    def b_erf(sh):
        n = sh
        x = torch.randn(n) * 2.0
        xv = U(x)
        return (lambda: torch.erf(xv), lambda: torch.erf(x),
                torch.erf(xv), torch.erf(x))
    row("aten::erf", "device", [64, 4096, 65536], b_erf)

    def b_gelu0(sh):
        n = sh
        x = torch.randn(n) * 2.0
        xv = U(x)
        return (lambda: F.gelu(xv, approximate="none"),
                lambda: F.gelu(x, approximate="none"),
                F.gelu(xv, approximate="none"), F.gelu(x, approximate="none"))
    row("gelu(approximate='none')", "composite (erf)",
        [64, 4096, 65536], b_gelu0)

    def b_tanh(sh):
        n = sh
        x = torch.randn(n)
        xv = U(x)
        return (lambda: torch.tanh(xv), lambda: torch.tanh(x),
                torch.tanh(xv), torch.tanh(x))
    row("aten::tanh", "composite (2*sigmoid(2x)-1)",
        [64, 4096, 65536], b_tanh)

    def b_sqrt(sh):
        n = sh
        x = torch.rand(n) + 0.5
        xv = U(x)
        return (lambda: torch.sqrt(xv), lambda: torch.sqrt(x),
                torch.sqrt(xv), torch.sqrt(x))
    row("aten::sqrt", "composite (x*rsqrt(x))",
        [64, 4096, 65536], b_sqrt)

    # reductions
    def b_sum(sh):
        a, b = sh
        x = torch.randn(a, b)
        xv = U(x)
        return (lambda: torch.sum(xv, dim=1), lambda: torch.sum(x, dim=1),
                torch.sum(xv, dim=1), torch.sum(x, dim=1))
    row("aten::sum(dim)", "composite (GEMM)",
        [(4, 16), (64, 64), (256, 256)], b_sum)

    def b_amax(sh):
        a, b = sh
        x = torch.randn(a, b)
        xv = U(x)
        return (lambda: torch.amax(xv, dim=1), lambda: torch.amax(x, dim=1),
                torch.amax(xv, dim=1), torch.amax(x, dim=1))
    row("aten::amax(dim)", "composite (relu max tree)",
        [(4, 16), (32, 64), (64, 128)], b_amax)

    def b_argmax(sh):
        a, b = sh
        x = torch.randn(a, b)
        xv = U(x)
        return (lambda: torch.argmax(xv, dim=1), lambda: torch.argmax(x, dim=1),
                torch.argmax(xv, dim=1), torch.argmax(x, dim=1))
    row("aten::argmax(dim)", "composite (host index scan)",
        [(4, 16), (32, 64), (64, 128)], b_argmax)

    # indexing
    def b_isel(sh):
        a, b = sh
        x = torch.randn(a, b)
        idx = torch.randperm(a)[: max(1, a // 2)]
        xv, iv = U(x), U(idx)
        return (lambda: torch.index_select(xv, 0, iv),
                lambda: torch.index_select(x, 0, idx),
                torch.index_select(xv, 0, iv), torch.index_select(x, 0, idx))
    row("aten::index_select", "composite (host word gather)",
        [(64, 64), (512, 128), (2048, 256)], b_isel)

    # batch norm
    rows = []
    def b_bn(sh):
        n, c, h, w = sh
        x = torch.randn(n, c, h, w)
        rm, rv = torch.zeros(c), torch.ones(c)
        g, be = torch.ones(c), torch.zeros(c)
        xv, rmv, rvv = U(x), U(rm), U(rv)
        gv, bev = U(g), U(be)
        return (lambda: F.batch_norm(xv, rmv, rvv, gv, bev, False, 0.1, 1e-5),
                lambda: F.batch_norm(x, rm, rv, g, be, False, 0.1, 1e-5),
                F.batch_norm(xv, rmv, rvv, gv, bev, False, 0.1, 1e-5),
                F.batch_norm(x, rm, rv, g, be, False, 0.1, 1e-5))
    row("batch_norm (eval)", "composite (device elementwise)",
        [(1, 4, 16, 16), (2, 16, 32, 32), (1, 32, 64, 32)], b_bn)

    def b_bnt(sh):
        n, c, h, w = sh
        x = torch.randn(n, c, h, w)
        rm, rv = torch.zeros(c), torch.ones(c)
        g, be = torch.ones(c), torch.zeros(c)
        xv, rmv, rvv = U(x), U(rm), U(rv)
        gv, bev = U(g), U(be)
        return (lambda: F.batch_norm(xv, rmv, rvv, gv, bev, True, 0.1, 1e-5),
                lambda: F.batch_norm(x, rm.clone(), rv.clone(), g, be, True,
                                     0.1, 1e-5),
                F.batch_norm(xv, rmv, rvv, gv, bev, True, 0.1, 1e-5),
                F.batch_norm(x, rm, rv, g, be, True, 0.1, 1e-5))
    row("batch_norm (train)", "composite (device reductions)",
        [(1, 4, 16, 16), (2, 16, 32, 32), (1, 32, 64, 32)], b_bnt)

    print("\n## markdown\n")
    for name, backend, ma, mr, vs, cs in ROWS:
        print("| `%s` | %s | %.2e / %.2e | %s | %s |"
              % (name, backend, ma, mr,
                 " / ".join("%.3f" % v for v in vs),
                 " / ".join("%.4f" % v for v in cs)))
    return 0


if __name__ == "__main__":
    sys_exit = main()
