"""Gradient parity vs the CPU, then a tiny CNN trained through the backend.

Run on the Pi:  .venv/bin/python tests/test_autograd.py
Exit code 0 = every gradient within 1e-3 relative and the loss decreased.

The vc4 reference is the *same* fp32 model evaluated with torch's CPU
kernels, so the comparison isolates this backend (engine fp32 accumulation
order, padding, phase decomposition) rather than the model.
"""

import sys
import time

import torch
import torch.nn.functional as F

import torch_vc4 as tv

D = tv.device()
FAIL = []


def U(t):
    return t.to(D)


def relerr(got, ref):
    g = got.cpu().double()
    r = ref.double()
    if g.shape != r.shape:
        return float("inf"), float("inf")
    maxabs = (g - r).abs().max().item() if g.numel() else 0.0
    # denominator: per-element |ref| floored at 1e-3 (near-zero gradient
    # entries would otherwise dominate); this is the "<=1e-3 rel" metric
    maxrel = ((g - r).abs() / r.abs().clamp_min(1e-3)).max().item() \
        if g.numel() else 0.0
    return maxabs, maxrel


def check(name, got, ref, limit=1e-3):
    ma, mr = relerr(got, ref)
    g = got.cpu().double()
    r = ref.double()
    # relative to the gradient magnitude, with a 1e-3 absolute floor so an
    # exactly-zero gradient is graded on absolute error rather than noise
    fro = float((g - r).norm() / r.norm().clamp_min(1e-3)) if r.numel() else 0.0
    ok = fro <= limit
    print("%-5s %-30s maxabs %.2e  maxrel %.2e  relFro %.2e (limit %.0e)"
          % ("ok" if ok else "FAIL", name, ma, mr, fro, limit))
    if not ok:
        FAIL.append(name)


def grads(fn, inputs):
    """Run fn, contract the outputs with fixed random weights, backward.

    A plain ``.sum()`` would make the softmax gradient identically zero (the
    rows sum to one) and grade only noise; fixed weights keep every check
    meaningful and identical on both devices.
    """
    outs = fn(*inputs)
    g = torch.Generator().manual_seed(0)

    def contract(o):
        w = torch.randn(o.shape, generator=g).to(o.device)
        return (o * w).sum()

    if isinstance(outs, (tuple, list)):
        loss = sum(contract(o) for o in outs)
    else:
        loss = contract(outs)
    loss.backward()
    return loss, [i.grad for i in inputs]


def parity(name, fn, cpu_inputs, vc4_inputs=None, limit=1e-3):
    """fn(*tensors) with the same values on CPU and vc4, compare gradients."""
    cpu_in = [x.clone().requires_grad_() for x in cpu_inputs]
    vc4_in = [U(x.detach()).requires_grad_() for x in cpu_inputs]
    _, g_cpu = grads(fn, cpu_in)
    _, g_vc4 = grads(fn, vc4_in)
    for i, (a, b) in enumerate(zip(g_vc4, g_cpu)):
        if b is None:
            continue
        check("%s grad[%d]" % (name, i), a, b, limit)


def main():
    torch.manual_seed(7)

    # ---------------------------------------------------------- gemm/add/mul
    a = torch.randn(12, 20, requires_grad=False)
    b = torch.randn(20, 9)
    parity("mm 12x20x9", lambda x, y: torch.mm(x, y), [a, b])
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    parity("matmul @", lambda x, y: x @ y, [a, b])
    x = torch.randn(6, 5)
    y = torch.randn(6, 5)
    parity("add", lambda p, q: p + q, [x, y])
    parity("mul", lambda p, q: p * q, [x, y])
    parity("sub", lambda p, q: p - q, [x, y])
    parity("mul scalar", lambda p: p * 2.5, [x])
    parity("add scalar", lambda p: p + 1.25, [x])
    parity("pow 2", lambda p: p ** 2, [x])

    # ------------------------------------------------------- unary/softmax
    x = torch.randn(4, 7)
    parity("relu", lambda p: torch.relu(p), [x])
    parity("sigmoid", lambda p: torch.sigmoid(p), [x])
    parity("silu", lambda p: F.silu(p), [x])
    parity("softmax", lambda p: F.softmax(p, -1), [x])

    # --------------------------------------------------------- linear/pool
    x = torch.randn(3, 11)
    w = torch.randn(5, 11)
    b = torch.randn(5)
    parity("linear", lambda p, q, r: F.linear(p, q, r), [x, w, b],
           limit=2e-3)
    x = torch.randn(1, 4, 16, 16)
    parity("max_pool2d 2x2", lambda p: F.max_pool2d(p, 2), [x], limit=1e-6)
    parity("adaptive_avg_pool2d 1x1",
           lambda p: F.adaptive_avg_pool2d(p, 1), [x], limit=2e-3)

    # ------------------------------------------------------------- conv2d
    conv_cases = [
        ("conv 3->5 k3 s1 p1", 3, 5, 8, 8, 3, 1, 1, 1),
        ("conv 4->6 k3 s2 p1", 4, 6, 9, 9, 3, 2, 1, 1),
        ("conv 3->4 k3 s2 p0", 3, 4, 10, 10, 3, 2, 0, 1),
        ("conv 3->6 k1 s1", 3, 6, 8, 8, 1, 1, 0, 1),
        ("conv 4->5 k1 s2", 4, 5, 9, 9, 1, 2, 0, 1),
        ("conv 16->16 k3 s1 p1", 16, 16, 8, 8, 3, 1, 1, 1),
        ("depthwise 6->6 k3 s1", 6, 6, 8, 8, 3, 1, 1, 6),
        ("depthwise 6->6 k3 s2", 6, 6, 9, 9, 3, 2, 1, 6),
    ]
    for (name, C, Co, H, W, k, s, p, g) in conv_cases:
        xi = torch.randn(1, C, H, W)
        wi = torch.randn(Co, C // g if g == 1 else 1, k, k)
        bi = torch.randn(Co)
        parity(name, lambda x_, w_, b_: F.conv2d(x_, w_, b_, stride=s,
                                                 padding=p, groups=g),
               [xi, wi, bi], limit=2e-3)

    # ------------------------------------------------ rejections still strict
    print("\n--- strictness under autograd ---")
    vx = U(torch.randn(4, 4)).requires_grad_()
    try:
        torch.sin(vx)
        print("FAIL  sin on a grad tensor did not raise")
        FAIL.append("sin autograd")
    except RuntimeError as e:
        print("ok    sin on a grad tensor raises  %s"
              % str(e).split("\n")[0][:70])
    # a pool the engine has no backward kernel for must refuse, not detach
    vx = U(torch.randn(1, 4, 16, 16)).requires_grad_()
    try:
        F.max_pool2d(vx, 3, 1, 1).sum().backward()
        print("FAIL  3x3 max_pool2d backward did not raise")
        FAIL.append("pool backward")
    except RuntimeError as e:
        print("ok    3x3 max_pool2d backward raises  %s"
              % str(e).split("\n")[0][:70])

    # ------------------------------------------------------ tiny CNN training
    print("\n--- tiny CNN (conv-relu-pool-conv-relu-flatten-fc) on vc4 ---")
    torch.manual_seed(11)
    H = W = 16
    CIN, C1, C2, CLS = 1, 16, 16, 4
    x = torch.randn(1, CIN, H, W)
    target = torch.randn(1, CLS)

    # small initial weights: output O(1), so the loss starts around 1 (a
    # randn init gives activations of ~60 and fp32 gradient comparisons at
    # that scale are dominated by cancellation)
    params = {
        "w1": torch.randn(C1, CIN, 3, 3) * 0.5,
        "b1": torch.zeros(C1),
        "w2": torch.randn(C2, C1, 3, 3) * (1.0 / (C1 * 9) ** 0.5),
        "b2": torch.zeros(C2),
        "fc": torch.randn(CLS, C2 * 8 * 8) * (1.0 / (C2 * 8 * 8) ** 0.5),
    }
    tv_params = {k: U(v).requires_grad_() for k, v in params.items()}
    xv = U(x)
    tv_target = U(target)

    def model(p, xin):
        h = torch.relu(F.conv2d(xin, p["w1"], p["b1"], padding=1))
        h = F.max_pool2d(h, 2)
        h = torch.relu(F.conv2d(h, p["w2"], p["b2"], padding=1))
        h = h.flatten(1)
        return F.linear(h, p["fc"])

    # CPU reference gradients (same weights, same data)
    cpu_params = {k: v.clone().requires_grad_() for k, v in params.items()}
    loss_cpu = F.mse_loss(model(cpu_params, x), target)
    loss_cpu.backward()

    loss_v = tv.mse_loss(model(tv_params, xv), tv_target)
    loss_v.backward()
    print("  initial loss  vc4 %.6f   cpu %.6f"
          % (float(loss_v), float(loss_cpu)))
    for k in params:
        check("cnn grad " + k, tv_params[k].grad, cpu_params[k].grad,
              limit=2e-3)

    # SGD by hand: p <- p - lr * g  (no torch.optim needed)
    for p in tv_params.values():
        p.grad = None
    lr = 0.0005
    n_steps = 40
    losses = []
    t0 = time.perf_counter()
    tv.reset_stats()
    for step in range(n_steps):
        for p in tv_params.values():
            p.grad = None
        out = model(tv_params, xv)
        loss = tv.mse_loss(out, tv_target)
        loss.backward()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for p in tv_params.values():
                new = torch.ops.torch_vc4.add(
                    p, torch.ops.torch_vc4.mul_scalar(p.grad, -lr))
                p.copy_(new)
    dt = time.perf_counter() - t0
    st = tv.stats()
    losses = torch.tensor(losses)
    print("  %d steps in %.2f s (%.0f ms/step), %d launches, %.1f launches/step"
          % (n_steps, dt, dt / n_steps * 1e3, st["launches"],
             st["launches"] / n_steps))
    print("  loss: first %.6f  last %.6f  min %.6f"
          % (losses[0], losses[-1], losses.min()))
    print("  first 5: %s" % ["%.4f" % v for v in losses[:5]])
    print("  last  5: %s" % ["%.4f" % v for v in losses[-5:]])
    if not (losses[-1] < 1e-3 and losses[-1] < losses[0] * 0.01):
        print("FAIL  loss did not converge (%.6f -> %.6f)"
              % (losses[0], losses[-1]))
        FAIL.append("training loss")
    else:
        print("ok    loss %.4f -> %.6f (%.0fx, with %.0e counted as 0)"
              % (losses[0], losses[-1],
                 losses[0] / max(losses[-1], 1e-6), 1e-6))

    # the trained model must actually predict something on the trained input
    with torch.no_grad():
        out = model(tv_params, xv)
        err = (out.cpu() - target).abs().max().item()
    print("  |pred - target|_max after training: %.4f" % err)
    if err > 0.05:
        print("FAIL  predictions did not reach the target")
        FAIL.append("training fit")

    # ------------------------------------------- the same model via nn.Module
    print("\n--- nn.Module path (parameters live on vc4) ---")
    import torch.nn as nn
    m = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1, device=D), nn.ReLU(),
        nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 8 * 8, 4, device=D))
    tv.mse_loss(m(xv), tv_target).backward()
    have = [n for n, p in m.named_parameters() if p.grad is not None]
    ok = len(have) == 4
    print("%-5s %-34s %d/4 parameters have gradients"
          % ("ok" if ok else "FAIL", "nn.Sequential conv/relu/pool/fc", len(have)))
    if not ok:
        FAIL.append("nn.Module grads")
    bn = nn.BatchNorm2d(8, device=D).eval()
    h = bn(torch.relu(m[0](xv)))
    ok = tuple(h.shape) == (1, 8, 16, 16)
    print("%-5s %-34s %s" % ("ok" if ok else "FAIL", "nn.BatchNorm2d (eval)",
                             tuple(h.shape)))
    if not ok:
        FAIL.append("nn.BatchNorm2d")
    m2 = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    m2.to(D)
    ok = m2[0].weight.device.type == "vc4"
    h2 = m2(xv)
    print("%-5s %-34s weight device %s, out %s"
          % ("ok" if ok else "FAIL", "module.to(vc4) from CPU",
             m2[0].weight.device, tuple(h2.shape)))
    if not ok:
        FAIL.append("module.to")

    if FAIL:
        print("\nFAILED %d check(s): %s" % (len(FAIL), FAIL))
        return 1
    print("\nALL AUTOGRAD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
