"""Forward parity: every vc4 op vs the CPU, plus the strictness checks.

Run on the Pi:  .venv/bin/python tests/test_ops.py
Exit code 0 = everything matched and every unsupported case raised.

The CPU reference is fp32 (same maths as the engine, different summation
order), so tolerances are reported per op: max abs, and max rel with the
engine's ``|ref| >= 1`` denominator.  Bit-exact ops (pooling, cat, resize,
views) are checked with equality.
"""

import sys

import torch
import torch.nn.functional as F

import torch_vc4 as tv

D = tv.device()
FAIL = []


def _dev(t):
    return t.device.type == "vc4"


def check(name, got, ref, rtol=1e-4, atol=1e-5, exact=False):
    if _dev(got):
        got = got.cpu()
    g = got.double()
    r = ref.double()
    if g.shape != r.shape:
        print("FAIL  %-34s shape %s != %s" % (name, tuple(g.shape),
                                             tuple(r.shape)))
        FAIL.append(name)
        return
    err = (g - r).abs()
    maxabs = err.max().item() if err.numel() else 0.0
    maxrel = (err / r.abs().clamp_min(1.0)).max().item() if err.numel() else 0.0
    if exact:
        ok = torch.equal(got, ref)
        verdict = "exact" if ok else "MISMATCH"
    else:
        ok = bool(torch.allclose(g.float(), r.float(), rtol=rtol, atol=atol))
        verdict = "ok" if ok else "FAIL"
    print("%-5s %-34s shape %-18s maxabs %.3e  maxrel %.3e"
          % (verdict, name, str(tuple(got.shape)), maxabs, maxrel))
    if not ok:
        FAIL.append(name)


def raises(name, fn, needle=None):
    try:
        fn()
    except Exception as e:
        msg = str(e).split("\n")[0]
        ok = needle is None or needle in msg
        print("%-5s %-34s %s" % ("ok" if ok else "FAIL", name, msg[:96]))
        if not ok:
            FAIL.append(name)
        return
    print("FAIL  %-34s did not raise" % name)
    FAIL.append(name)


def U(x):
    return x.to(D)


def main():
    torch.manual_seed(1234)

    # ---------------------------------------------------------- memory/views
    x = torch.randn(1, 4, 8, 8)
    y = U(x)
    check("view [1,4,8,8]->[4,64]", y.view(4, 64), x.view(4, 64), exact=True)
    check("view -1", U(x).view(-1, 8), x.view(-1, 8), exact=True)
    check("flatten(1)", U(x).flatten(1), x.flatten(1), exact=True)
    check("reshape", U(x).reshape(4, 64), x.reshape(4, 64), exact=True)
    check("fill_ (torch.full)", torch.full((3, 4), 2.5, device=D),
          torch.full((3, 4), 2.5), exact=True)
    check("zero_ (torch.zeros)", torch.zeros((5, 6), device=D),
          torch.zeros((5, 6)), exact=True)

    # ------------------------------------------------------------------ conv
    print("\n--- conv2d (NCHW, batch 1 and batched) ---")
    conv_cases = [
        # Cin, Cout, H, W, k, stride, pad, groups
        (3, 5, 8, 8, 3, 1, 1, 1),
        (4, 4, 9, 9, 3, 1, 0, 1),
        (16, 16, 12, 12, 3, 1, 1, 1),
        (5, 7, 9, 9, 3, 2, 1, 1),
        (4, 6, 10, 10, 3, 2, 0, 1),
        (3, 8, 8, 8, 1, 1, 0, 1),
        (3, 8, 8, 8, 1, 2, 0, 1),
        (8, 8, 10, 10, 1, 2, 0, 1),
        (6, 6, 8, 8, 3, 1, 1, 6),      # depthwise
        (6, 6, 9, 9, 3, 2, 1, 6),      # depthwise stride 2
        (6, 6, 8, 8, 1, 1, 0, 6),      # depthwise 1x1
    ]
    for (C, Co, H, W, k, s, p, g) in conv_cases:
        xi = torch.randn(1, C, H, W)
        wi = torch.randn(Co, C // g if g == 1 else 1, k, k)
        bi = torch.randn(Co)
        ref = F.conv2d(xi, wi, bi, stride=s, padding=p, groups=g)
        got = F.conv2d(U(xi), U(wi), U(bi), stride=s, padding=p, groups=g)
        check("conv %d->%d k%d s%d p%d g%d" % (C, Co, k, s, p, g), got, ref,
              atol=5e-4)
    xi = torch.randn(1, 2, 7, 7)
    wi = torch.randn(3, 2, 3, 3)
    check("conv no bias", F.conv2d(U(xi), U(wi), None, padding=1),
          F.conv2d(xi, wi, None, padding=1), atol=5e-4)

    # --------------------------------------------------------------- pooling
    print("\n--- pooling / resize ---")
    x = torch.randn(1, 4, 16, 16)
    check("max_pool2d 2x2/2", F.max_pool2d(U(x), 2), F.max_pool2d(x, 2),
          exact=True)
    check("max_pool2d 3x3/2 p1", F.max_pool2d(U(x), 3, 2, 1),
          F.max_pool2d(x, 3, 2, 1), exact=True)
    x2 = torch.randn(1, 4, 9, 11)
    check("max_pool2d 2x2 (9x11)", F.max_pool2d(U(x2), 2),
          F.max_pool2d(x2, 2), exact=True)
    check("max_pool2d 5x5/1 p2", F.max_pool2d(U(x2), 5, 1, 2),
          F.max_pool2d(x2, 5, 1, 2), exact=True)
    x = torch.randn(1, 5, 16, 16)
    check("adaptive_avg_pool2d 1x1", F.adaptive_avg_pool2d(U(x), 1),
          F.adaptive_avg_pool2d(x, 1), atol=2e-6)
    check("adaptive_avg_pool2d (3,5)", F.adaptive_avg_pool2d(U(x), (3, 5)),
          F.adaptive_avg_pool2d(x, (3, 5)), atol=2e-6)
    check("adaptive_avg_pool2d (7,7)",
          F.adaptive_avg_pool2d(U(x2), (7, 7)),
          F.adaptive_avg_pool2d(x2, (7, 7)), atol=2e-6)
    x3 = torch.randn(1, 3, 4, 4)
    check("upsample nearest 2x",
          F.interpolate(U(x3), scale_factor=2, mode="nearest"),
          F.interpolate(x3, scale_factor=2, mode="nearest"), exact=True)
    check("upsample nearest 3x",
          F.interpolate(U(x3), scale_factor=3, mode="nearest"),
          F.interpolate(x3, scale_factor=3, mode="nearest"), exact=True)
    check("upsample nearest size (7,5)",
          F.interpolate(U(x3), size=(7, 5), mode="nearest"),
          F.interpolate(x3, size=(7, 5), mode="nearest"), exact=True)

    # --------------------------------------------------------------- softmax
    print("\n--- softmax (last dim, <= 16) ---")
    for cols in (4, 10, 16):
        x = torch.randn(5, cols)
        check("softmax %d classes" % cols, F.softmax(U(x), -1),
              F.softmax(x, -1), atol=2e-6)
    x = torch.randn(8, 3, 16)
    check("softmax 3-D", F.softmax(U(x), -1), F.softmax(x, -1), atol=2e-6)

    # ----------------------------------------------------------- elementwise
    print("\n--- elementwise / reductions ---")
    x = torch.randn(6, 5)
    check("sigmoid", torch.sigmoid(U(x)), torch.sigmoid(x), atol=2e-6)
    check("silu", F.silu(U(x)), F.silu(x), atol=2e-6)
    check("relu", torch.relu(U(x)), torch.relu(x), exact=True)
    check("add tensor", U(x) + U(x), x + x, exact=True)
    check("mul tensor", U(x) * U(x), x * x, exact=True)
    check("sub tensor", U(x) - 0.5 * U(x), x - 0.5 * x, atol=1e-6)
    check("mul scalar", U(x) * 2.5, x * 2.5, atol=1e-6)
    check("div scalar", U(x) / 4.0, x / 4.0, atol=1e-7)
    check("add scalar", U(x) + 1.25, x + 1.25, atol=1e-6)
    check("neg", -U(x), -x, exact=True)
    check("pow 2", U(x) ** 2, x ** 2, atol=1e-6)
    check("sum (full)", U(x).sum(), x.sum(), atol=2e-5)
    check("mean (full)", U(x).mean(), x.mean(), atol=2e-6)

    # ------------------------------------------------------------- cat/linear
    print("\n--- cat / linear ---")
    a = torch.randn(1, 3, 5, 5)
    b = torch.randn(1, 4, 5, 5)
    check("cat channel", torch.cat([U(a), U(b)], 1), torch.cat([a, b], 1),
          exact=True)
    x = torch.randn(1, 9)
    w = torch.randn(4, 9)
    bb = torch.randn(4)
    check("linear 2-D", F.linear(U(x), U(w), U(bb)), F.linear(x, w, bb),
          atol=2e-5)
    x4 = torch.randn(2, 3, 5)
    w4 = torch.randn(7, 5)
    check("linear 3-D", F.linear(U(x4), U(w4)), F.linear(x4, w4), atol=2e-5)

    # ---------------------------------------------------------- batch norm
    print("\n--- batch_norm (inference) ---")
    x = torch.randn(1, 5, 6, 6)
    rm = torch.randn(5) * 0.1
    rv = torch.rand(5) + 0.5
    g, be = torch.randn(5), torch.randn(5)
    check("batch_norm eval",
          F.batch_norm(U(x), U(rm), U(rv), U(g), U(be), False, 0.1, 1e-5),
          F.batch_norm(x, rm, rv, g, be, False, 0.1, 1e-5), atol=2e-6)
    w = torch.randn(3, 4, 3, 3)
    bias = torch.randn(3)
    w2, b2 = tv.fold_bn_into_conv(w, bias, rm[:3], rv[:3], g[:3], be[:3])
    ref = F.batch_norm(F.conv2d(x[:, :4], w, bias, padding=1), rm[:3], rv[:3],
                       g[:3], be[:3], False, 0.1, 1e-5)
    got = F.conv2d(x[:, :4], w2, b2, padding=1)
    check("fold_bn_into_conv", got, ref, atol=2e-6)

    # ------------------------------------------------------- strictness
    print("\n--- strict rejection (no CPU fallback) ---")
    x = torch.randn(1, 4, 8, 8)
    vx = U(x)
    raises("torch.sin(vc4)", lambda: torch.sin(vx), "not implemented")
    raises("torch.sort(vc4)", lambda: torch.sort(vx))
    check("torch.max(vc4) reduction", torch.max(vx[0]), torch.max(x[0]),
          atol=2e-5)
    raises("conv2d dilation 2",
           lambda: F.conv2d(vx, U(torch.randn(4, 4, 3, 3)), None,
                            padding=2, dilation=2), "not implemented")
    xb = torch.randn(2, 4, 8, 8)
    wb = torch.randn(4, 4, 3, 3)
    check("conv2d batch 2",
          F.conv2d(U(xb), U(wb), None, padding=1),
          F.conv2d(xb, wb, None, padding=1), atol=5e-4)
    raises("conv2d 5x5",
           lambda: F.conv2d(vx, U(torch.randn(4, 4, 5, 5)), None, padding=2),
           "1x1 and 3x3")
    xs = torch.randn(4, 32)
    check("softmax 32 classes (two-pass)",
          F.softmax(U(xs), -1), F.softmax(xs, -1), atol=5e-6)
    raises("max_pool2d ceil_mode",
           lambda: F.max_pool2d(vx, 3, 2, 1, ceil_mode=True), "ceil_mode")
    xb = torch.randn(2, 4, 8, 8)
    yb = F.batch_norm(U(xb), U(torch.zeros(4)), U(torch.ones(4)),
                      U(torch.ones(4)), U(torch.zeros(4)), True, 0.1, 1e-5)
    check("batch_norm training mode",
          yb, F.batch_norm(xb, torch.zeros(4), torch.ones(4),
                           torch.ones(4), torch.zeros(4), True, 0.1, 1e-5),
          atol=5e-5)
    raises("float64 on vc4",
           lambda: torch.zeros(4, dtype=torch.float64, device=D), "float32")
    raises("mm cpu x vc4", lambda: torch.mm(torch.randn(4, 4), U(
        torch.randn(4, 4))), "expected a vc4 tensor")
    # a metadata-only (non-contiguous) view must not be computed on
    raises("compute on transposed view",
           lambda: torch.add(vx.transpose(2, 3), vx.transpose(2, 3)),
           "contiguous")

    # ------------------------------------------------- allocator must free
    before = torch.ops.torch_vc4.alloc_info()[1]
    for _ in range(3000):
        torch.zeros(4096, device=D)          # 48 MB of churn in a 24 MiB arena
    after = torch.ops.torch_vc4.alloc_info()[1]
    check("allocator reclaims (48 MB churn)", torch.tensor(after),
          torch.tensor(before), exact=True)

    st = tv.stats()
    print("\nlaunches %d, kernel %.1f ms, %.2f GFLOP/s"
          % (st["launches"], st["kernel_ms"], st["gflops_kernel"]))
    if FAIL:
        print("\nFAILED %d check(s): %s" % (len(FAIL), FAIL))
        return 1
    print("\nALL FORWARD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
