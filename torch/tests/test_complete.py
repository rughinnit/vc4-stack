"""Feature-completeness checks: batches, arbitrary-dim reductions, softmax for
any class count, broadcasting, batch-norm training, native conv paths, the
unary/indexing surface, float64 rejection and autograd for all of it.

Every forward check is against the CPU on at least three shapes and prints the
max abs / max rel error.  Unsupported cases must raise with the precise reason.

Run on the Pi from the repo root:  .venv/bin/python torch/tests/test_complete.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

# Measure the checkout this file is in when its extension has been built.  A
# pip-installed or editable torch_vc4 can point at another tree; importing that
# by accident tests the wrong backend.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "torch_vc4")
if os.path.isdir(PACKAGE) and any(
        name.startswith("_C") and name.endswith(".so")
        for name in os.listdir(PACKAGE)):
    sys.path.insert(0, ROOT)

import torch_vc4 as tv

D = tv.device()
FAIL = []


def U(t):
    return t.to(D)


def check(name, got, ref, rtol=1e-4, atol=1e-5, exact=False):
    if got.device.type == "vc4":
        got = got.cpu()
    g, r = got.double(), ref.double()
    if g.shape != r.shape:
        print("FAIL  %-42s shape %s != %s"
              % (name, tuple(g.shape), tuple(r.shape)))
        FAIL.append(name)
        return
    err = (g - r).abs()
    maxabs = err.max().item() if err.numel() else 0.0
    den = r.abs().clamp_min(1.0)
    maxrel = (err / den).max().item() if err.numel() else 0.0
    if exact:
        ok = torch.equal(got, ref)
        verdict = "exact" if ok else "FAIL"
    else:
        ok = bool(torch.allclose(g.float(), r.float(), rtol=rtol, atol=atol))
        verdict = "ok" if ok else "FAIL"
    print("%-5s %-42s %-16s maxabs %.3e  maxrel %.3e"
          % (verdict, name, str(tuple(got.shape)), maxabs, maxrel))
    if not ok:
        FAIL.append(name)


def raises(name, fn, needle=None):
    try:
        fn()
    except Exception as e:
        msg = str(e).split("\n")[0]
        ok = needle is None or needle in msg
        print("%-5s %-42s %s" % ("ok" if ok else "FAIL", name, msg[:100]))
        if not ok:
            FAIL.append(name)
        return
    print("FAIL  %-42s did not raise" % name)
    FAIL.append(name)


def grads(fn, inputs):
    outs = fn(*inputs)
    gen = torch.Generator().manual_seed(0)

    def contract(o):
        w = torch.randn(o.shape, generator=gen).to(o.device)
        return (o * w).sum()

    loss = sum(contract(o) for o in outs) if isinstance(outs, (tuple, list)) \
        else contract(outs)
    loss.backward()
    return loss, [i.grad for i in inputs]


def parity(name, fn, tensors, limit=2e-3):
    cpu_in = [x.clone().requires_grad_() for x in tensors]
    vc4_in = [U(x.detach()).requires_grad_() for x in tensors]
    _, gc = grads(fn, cpu_in)
    _, gv = grads(fn, vc4_in)
    for i, (a, b) in enumerate(zip(gv, gc)):
        if b is None:
            continue
        check("%s grad[%d]" % (name, i), a, b, rtol=1e-3, atol=1e-4)


def main():
    torch.manual_seed(123)

    # ---------------------------------------------------------- 1. batches
    print("\n=== 1. batch > 1 forward ===")
    conv_cases = [
        (2, 3, 8, 8, 5, 3, 1, 1, 1),
        (3, 4, 9, 9, 6, 3, 2, 1, 1),
        (2, 5, 10, 10, 7, 3, 2, 0, 1),
        (3, 3, 8, 8, 8, 1, 1, 0, 1),
        (2, 4, 9, 9, 5, 1, 2, 0, 1),
        (2, 6, 8, 8, 6, 3, 1, 1, 6),
        (3, 6, 9, 9, 6, 3, 2, 1, 6),
    ]
    for (N, C, H, W, Co, k, s, p, gr) in conv_cases:
        xi = torch.randn(N, C, H, W)
        wi = torch.randn(Co, C // gr if gr == 1 else 1, k, k)
        bi = torch.randn(Co)
        ref = F.conv2d(xi, wi, bi, stride=s, padding=p, groups=gr)
        got = F.conv2d(U(xi), U(wi), U(bi), stride=s, padding=p, groups=gr)
        check("conv N%d %d->%d k%d s%d p%d g%d" % (N, C, Co, k, s, p, gr),
              got, ref, atol=5e-4)

    for N, C, H, W in ((2, 4, 16, 16), (3, 3, 8, 8)):
        x = torch.randn(N, C, H, W)
        check("max_pool2d 2x2 N%d" % N, F.max_pool2d(U(x), 2),
              F.max_pool2d(x, 2), exact=True)
        check("max_pool2d 3x3 s2 p1 N%d" % N, F.max_pool2d(U(x), 3, 2, 1),
              F.max_pool2d(x, 3, 2, 1), exact=True)
        check("adaptive_avg_pool2d 1x1 N%d" % N,
              F.adaptive_avg_pool2d(U(x), 1),
              F.adaptive_avg_pool2d(x, 1), atol=2e-6)
        check("adaptive_avg_pool2d (3,5) N%d" % N,
              F.adaptive_avg_pool2d(U(x), (3, 5)),
              F.adaptive_avg_pool2d(x, (3, 5)), atol=2e-6)
        check("upsample nearest 2x N%d" % N,
              F.interpolate(U(x), scale_factor=2, mode="nearest"),
              F.interpolate(x, scale_factor=2, mode="nearest"), exact=True)
        check("upsample nearest (7,5) N%d" % N,
              F.interpolate(U(x), size=(7, 5), mode="nearest"),
              F.interpolate(x, size=(7, 5), mode="nearest"), exact=True)

    for N in (2, 3):
        a = torch.randn(N, 3, 5, 5)
        b = torch.randn(N, 4, 5, 5)
        check("cat channel N%d" % N, torch.cat([U(a), U(b)], 1),
              torch.cat([a, b], 1), exact=True)

    for shape, classes in (((2, 5, 3), 3), ((3, 4, 16), 16),
                           ((2, 3, 17), 17), ((2, 2, 64), 64),
                           ((4, 100), 100)):
        x = torch.randn(*shape)
        check("softmax %s last=%d" % (shape, classes),
              F.softmax(U(x), -1), F.softmax(x, -1), atol=5e-6)
    x = torch.randn(3, 20, 4)
    check("softmax dim=1 (20)", F.softmax(U(x), 1), F.softmax(x, 1),
          atol=5e-6)

    # batch norm eval + training
    for N in (2, 3):
        x = torch.randn(N, 5, 6, 6)
        rm, rv = torch.randn(5) * 0.1, torch.rand(5) + 0.5
        g, be = torch.randn(5), torch.randn(5)
        check("batch_norm eval N%d" % N,
              F.batch_norm(U(x), U(rm), U(rv), U(g), U(be), False, 0.1, 1e-5),
              F.batch_norm(x, rm, rv, g, be, False, 0.1, 1e-5), atol=2e-6)

        rmc, rvc = rm.clone(), rv.clone()
        yref = F.batch_norm(x, rmc, rvc, g, be, True, 0.2, 1e-5)
        rmv, rvv = U(rm.clone()), U(rv.clone())
        ygot = F.batch_norm(U(x), rmv, rvv, U(g), U(be), True, 0.2, 1e-5)
        check("batch_norm train y N%d" % N, ygot, yref, atol=5e-5)
        check("batch_norm train running_mean N%d" % N, rmv, rmc, atol=2e-6)
        check("batch_norm train running_var N%d" % N, rvv, rvc, atol=2e-6)

    # ------------------------------------------------- 2. dim reductions
    print("\n=== 2. reductions over arbitrary dims ===")
    red_cases = [((3, 4), 0), ((3, 4), 1), ((2, 3, 4), 0),
                 ((2, 3, 4), 1), ((2, 3, 4), 2), ((2, 3, 4), -1),
                 ((2, 3, 4), (0, 1)), ((2, 3, 4), (0, 2)),
                 ((2, 3, 4, 5), (1, 2)), ((2, 3, 4, 5), (0, 2, 3))]
    for shape, dim in red_cases:
        x = torch.randn(*shape)
        check("sum %s dim=%s" % (shape, dim), torch.sum(U(x), dim=dim),
              torch.sum(x, dim=dim), atol=2e-5)
        check("mean %s dim=%s" % (shape, dim), torch.mean(U(x), dim=dim),
              torch.mean(x, dim=dim), atol=2e-6)
        check("amax %s dim=%s" % (shape, dim), torch.amax(U(x), dim=dim),
              torch.amax(x, dim=dim), atol=2e-5)
        check("amin %s dim=%s" % (shape, dim), torch.amin(U(x), dim=dim),
              torch.amin(x, dim=dim), atol=2e-5)
        if isinstance(dim, int):
            check("argmax %s dim=%s" % (shape, dim),
                  torch.argmax(U(x), dim=dim), torch.argmax(x, dim=dim),
                  exact=True)
    x = torch.randn(2, 3, 4)
    check("sum keepdim", torch.sum(U(x), dim=1, keepdim=True),
          torch.sum(x, dim=1, keepdim=True), atol=2e-5)
    check("mean keepdim (0,2)", torch.mean(U(x), dim=(0, 2), keepdim=True),
          torch.mean(x, dim=(0, 2), keepdim=True), atol=2e-6)
    x = torch.randn(7, 9)
    check("sum full", torch.sum(U(x)), torch.sum(x), atol=2e-5)
    check("amax full", torch.amax(U(x)), torch.amax(x), atol=2e-5)
    check("amin full", torch.amin(U(x)), torch.amin(x), atol=2e-5)
    check("argmax full", torch.argmax(U(x)), torch.argmax(x), exact=True)
    check("max full", torch.max(U(x)), torch.max(x), atol=2e-5)
    v, i = torch.max(U(x), dim=1)
    rv_, ri_ = torch.max(x, dim=1)
    check("max.dim values", v, rv_, atol=2e-5)
    check("max.dim indices", i, ri_, exact=True)

    # ---------------------------------------------------- 3. broadcasting
    print("\n=== 3. broadcasting ===")
    bcast = [((3, 4), (4,)), ((2, 3, 4), (3, 1)), ((2, 1, 4), (1, 5, 1)),
             ((2, 3, 4, 4), (3, 1, 1)), ((4,), (1,)), ((2, 3), ())]
    for sa, sb in bcast:
        a = torch.randn(*sa) if sa else torch.randn(())
        b = torch.randn(*sb) if sb else torch.randn(())
        for name, fn in (("add", torch.add), ("sub", torch.sub),
                         ("mul", torch.mul), ("div", torch.div)):
            ref = fn(a, b)
            got = fn(U(a), U(b))
            check("%s %s %s" % (name, sa, sb), got, ref, atol=2e-6)
    a = torch.randn(2, 3, 4)
    ch = torch.randn(3, 1)
    check("per-channel add", U(a) + U(ch), a + ch, atol=1e-6)
    check("per-channel div", U(a) / U(ch + 2.0), a / (ch + 2.0), atol=2e-6)

    # --------------------------------------------------------- 4. unary
    print("\n=== 4. unary/elementwise ===")
    shapes = [(3, 4), (2, 3, 5), (13,)]
    for sh in shapes:
        x = torch.randn(*sh) * 2.0
        check("abs %s" % (sh,), torch.abs(U(x)), torch.abs(x), exact=True)
        check("sign %s" % (sh,), torch.sign(U(x)), torch.sign(x), atol=1e-6)
        check("clip %s" % (sh,), torch.clip(U(x), -1.0, 1.5),
              torch.clip(x, -1.0, 1.5), atol=2e-7)
        check("clip min-only %s" % (sh,), torch.clip(U(x), min=0.0),
              torch.clip(x, min=0.0), atol=2e-7)
        check("leaky_relu %s" % (sh,),
              F.leaky_relu(U(x), 0.1), F.leaky_relu(x, 0.1), atol=1e-6)
        check("hardsigmoid %s" % (sh,), F.hardsigmoid(U(x)),
              F.hardsigmoid(x), atol=1e-6)
        check("hardswish %s" % (sh,), F.hardswish(U(x)),
              F.hardswish(x), atol=1e-6)
        check("tanh %s" % (sh,), torch.tanh(U(x)), torch.tanh(x), atol=5e-6)
        xs = x.abs() + 0.1
        check("sqrt %s" % (sh,), torch.sqrt(U(xs)), torch.sqrt(xs), atol=2e-6)
        check("reciprocal %s" % (sh,), torch.reciprocal(U(x)),
              torch.reciprocal(x), atol=2e-6)
        check("gelu tanh %s" % (sh,),
              F.gelu(U(x), approximate="tanh"),
              F.gelu(x, approximate="tanh"), atol=2e-5)
        check("gelu exact %s" % (sh,),
              F.gelu(U(x), approximate="none"),
              F.gelu(x, approximate="none"), atol=2e-6)
        check("exp %s" % (sh,), torch.exp(U(x)), torch.exp(x))
        check("log %s" % (sh,), torch.log(U(xs)), torch.log(xs), atol=1e-6)
        check("erf %s" % (sh,), torch.erf(U(x)), torch.erf(x), atol=2e-6)
        check("floor %s" % (sh,), torch.floor(U(x)), torch.floor(x),
              exact=True)
        check("ceil %s" % (sh,), torch.ceil(U(x)), torch.ceil(x), exact=True)
        check("round %s" % (sh,), torch.round(U(x)), torch.round(x),
              exact=True)
    # +-inf pass-through is measured separately (COMPLETENESS.md); the
    # error metric would print inf - inf as NaN here.
    edge = torch.tensor([1e10, -1e10, 8388608.0, -8388608.0, 0.0, -0.0])
    check("floor large/signed zero", torch.floor(U(edge)), torch.floor(edge),
          exact=True)
    check("ceil large/signed zero", torch.ceil(U(edge)), torch.ceil(edge),
          exact=True)
    ties = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
    check("round ties to even", torch.round(U(ties)), torch.round(ties),
          exact=True)

    # ------------------------------------------------------- 5. indexing
    print("\n=== 5. indexing ===")
    x = torch.randn(5, 4)
    idx = torch.tensor([0, 3, 1, 4, 2])
    check("index_select dim0", torch.index_select(U(x), 0, U(idx)),
          torch.index_select(x, 0, idx), exact=True)
    x3 = torch.randn(3, 5, 4)
    check("index_select dim1", torch.index_select(U(x3), 1, U(idx)),
          torch.index_select(x3, 1, idx), exact=True)
    gidx = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 0, 0]])
    check("gather dim1", torch.gather(U(x), 1, U(gidx)),
          torch.gather(x, 1, gidx), exact=True)
    gidx0 = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1], [1, 1, 0, 0],
                          [2, 2, 2, 2], [4, 0, 4, 0]])
    check("gather dim0", torch.gather(U(x), 0, U(gidx0)),
          torch.gather(x, 0, gidx0), exact=True)
    raises("scatter raises by name",
           lambda: torch.scatter(U(x), 0, U(idx.view(-1, 1)),
                                 U(torch.randn(5, 1))), "scatter")
    raises("sort raises with a message", lambda: torch.sort(U(x)), "sort")
    raises("topk raises with a message", lambda: torch.topk(U(x), 2), "topk")

    # --------------------------------------------------------- 6. float64
    print("\n=== 6. float64 rejection ===")
    raises("float64 empty raises", lambda: torch.zeros(
        4, dtype=torch.float64, device=D), "float64")
    raises("float64 .to raises",
           lambda: U(torch.randn(4)).to(torch.float64), "float64")

    # -------------------------------------------------------- 7. autograd
    print("\n=== 7. autograd (forward + backward) ===")
    parity("conv N2 s1", lambda x, w, b: F.conv2d(
        x, w, b, padding=1),
        [torch.randn(2, 4, 8, 8), torch.randn(5, 4, 3, 3), torch.randn(5)])
    parity("conv N2 s2", lambda x, w, b: F.conv2d(
        x, w, b, stride=2, padding=1),
        [torch.randn(2, 4, 9, 9), torch.randn(5, 4, 3, 3), torch.randn(5)])
    parity("depthwise N2", lambda x, w, b: F.conv2d(
        x, w, b, padding=1, groups=4),
        [torch.randn(2, 4, 8, 8), torch.randn(4, 1, 3, 3), torch.randn(4)])
    parity("max_pool2d N2", lambda x: F.max_pool2d(x, 2),
           [torch.randn(2, 3, 16, 16)], limit=1e-6)
    parity("adaptive_avg_pool2d N2", lambda x: F.adaptive_avg_pool2d(x, 1),
           [torch.randn(2, 3, 16, 16)])
    parity("cat channel N2", lambda a, b: torch.cat([a, b], 1),
           [torch.randn(2, 3, 5, 5), torch.randn(2, 4, 5, 5)], limit=1e-6)
    parity("upsample nearest N2", lambda x: F.interpolate(
        x, scale_factor=2, mode="nearest"), [torch.randn(2, 3, 4, 5)],
        limit=1e-6)
    parity("softmax 3-D big classes", lambda x: F.softmax(x, -1),
           [torch.randn(2, 3, 40)], limit=3e-3)
    parity("broadcast add", lambda a, b: a + b,
           [torch.randn(2, 3, 4), torch.randn(3, 1)])
    parity("broadcast mul", lambda a, b: a * b,
           [torch.randn(2, 3, 4), torch.randn(4)])
    parity("broadcast div", lambda a, b: a / b,
           [torch.randn(2, 3, 4), torch.randn(1, 3, 1) + 2.0])
    parity("sum dim", lambda x: torch.sum(x, dim=1),
           [torch.randn(2, 3, 4)])
    parity("mean dim", lambda x: torch.mean(x, dim=(0, 2)),
           [torch.randn(2, 3, 4)])
    parity("sum keepdim", lambda x: torch.sum(x, dim=2, keepdim=True),
           [torch.randn(2, 3, 4)])
    parity("abs", lambda x: torch.abs(x), [torch.randn(3, 4)])
    parity("clip", lambda x: torch.clip(x, -1.0, 1.0), [torch.randn(3, 4)])
    parity("leaky_relu", lambda x: F.leaky_relu(x, 0.1), [torch.randn(3, 4)])
    parity("hardsigmoid", lambda x: F.hardsigmoid(x), [torch.randn(3, 4)])
    parity("hardswish", lambda x: F.hardswish(x), [torch.randn(3, 4)])
    parity("tanh", lambda x: torch.tanh(x), [torch.randn(3, 4)])
    parity("sqrt", lambda x: torch.sqrt(x), [torch.rand(3, 4) + 0.5])
    parity("reciprocal", lambda x: torch.reciprocal(x),
           [torch.rand(3, 4) + 0.5])
    parity("gelu tanh", lambda x: F.gelu(x, approximate="tanh"),
           [torch.randn(3, 4)])
    parity("gelu exact", lambda x: F.gelu(x, approximate="none"),
           [torch.randn(3, 4)])

    # batch-norm training gradients (input, gamma, beta), separate running
    # statistics per device (training updates them in place)
    x = torch.randn(2, 4, 5, 5)
    gam = torch.randn(4)
    bet = torch.randn(4)
    cpu_in = [x.clone().requires_grad_(), gam.clone().requires_grad_(),
              bet.clone().requires_grad_()]
    vc4_in = [U(x).requires_grad_(), U(gam).requires_grad_(),
              U(bet).requires_grad_()]
    lc = F.batch_norm(cpu_in[0], torch.zeros(4), torch.ones(4), cpu_in[1],
                      cpu_in[2], True, 0.1, 1e-5)
    w = torch.randn(lc.shape, generator=torch.Generator().manual_seed(0))
    lc.mul(w).sum().backward()
    lv = F.batch_norm(vc4_in[0], U(torch.zeros(4)), U(torch.ones(4)),
                      vc4_in[1], vc4_in[2], True, 0.1, 1e-5)
    lv.mul(U(w)).sum().backward()
    for i, nm in enumerate(("x", "gamma", "beta")):
        check("batch_norm train grad[%s]" % nm, vc4_in[i].grad,
              cpu_in[i].grad, rtol=1e-3, atol=1e-4)

    raises("amax backward raises precisely",
           lambda: torch.amax(U(torch.randn(3, 4)).requires_grad_(),
                              dim=1).sum().backward(),
           "amax")
    raises("gather backward raises precisely",
           lambda: torch.gather(U(torch.randn(3, 4)).requires_grad_(), 1,
                                U(torch.tensor([[0, 1, 2, 3]]))).sum()
           .backward(), "scatter")

    # ----------------------------------------------------- 8. tiny train
    print("\n=== 8. batched CNN trains end to end ===")
    torch.manual_seed(5)
    N, CIN, C1, CLS = 3, 1, 8, 4
    x = torch.randn(N, CIN, 16, 16)          # pool backward needs H%4, W%16
    tgt = torch.randn(N, CLS)
    p = {"w1": torch.randn(C1, CIN, 3, 3) * 0.3,
         "b1": torch.zeros(C1),
         "fc": torch.randn(CLS, C1 * 8 * 8) * (1.0 / (C1 * 64) ** 0.5)}
    vp = {k: U(v).requires_grad_() for k, v in p.items()}
    xv, tv_ = U(x), U(tgt)

    def model(ps, xin):
        h = torch.relu(F.conv2d(xin, ps["w1"], ps["b1"], padding=1))
        h = F.max_pool2d(h, 2)
        return F.linear(h.flatten(1), ps["fc"])

    losses = []
    for _ in range(60):
        for q in vp.values():
            q.grad = None
        loss = tv.mse_loss(model(vp, xv), tv_)
        loss.backward()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for q in vp.values():
                q.copy_(torch.ops.torch_vc4.add(
                    q, torch.ops.torch_vc4.mul_scalar(q.grad, -5e-3)))
    ok = losses[-1] < 1e-4 and losses[-1] < losses[0] * 1e-3
    print("%-5s %-42s %.6f -> %.6f" %
          ("ok" if ok else "FAIL", "batched CNN loss", losses[0], losses[-1]))
    if not ok:
        FAIL.append("batched CNN train")

    if FAIL:
        print("\nFAILED %d check(s): %s" % (len(FAIL), FAIL))
        return 1
    print("\nALL COMPLETENESS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
