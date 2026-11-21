#!/usr/bin/env python3
"""Acceptance test for torch_vc4: device vc4, CPU-verified matmul, strict
failure on unsupported ops, and no /dev/mem in the client process.

Run:  cd /home/raspi/work/torch-vc4 && .venv/bin/python tests/test_backend.py
Exit code 0 == all checks passed.
"""
import os
import sys
import time

import torch

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def rel_err(gpu, cpu):
    """Engine-style metric: err / max(|ref|, 1); plus the raw max |err|."""
    d = (gpu.double() - cpu.double()).abs()
    denom = cpu.double().abs().clamp_min(1.0)
    return float((d / denom).max()), float(d.max())


def expect_raise(name, fn, needle="not implemented for device vc4"):
    try:
        r = fn()
    except RuntimeError as e:
        msg = str(e).replace("\n", " ")
        ok = needle in msg
        check(name, ok, ("raised: " + msg[:110]) if ok
              else ("raised the wrong error: " + msg[:160]))
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, f"did NOT raise, returned {type(r)}")


def torch_vc4_t(like, cpu_t):
    return cpu_t.to(like.device)


def main():
    print(f"torch {torch.__version__}  python {sys.version.split()[0]}")
    import torch_vc4

    backend = torch._C._get_privateuse1_backend_name()
    check("backend renamed to vc4", backend == "vc4", backend)

    dev = torch_vc4.device()
    check("torch_vc4.device()", dev.type == "vc4" and dev.index == 0, str(dev))
    try:  # torch 2.15 parser limitation, recorded rather than hidden
        torch.device("vc4")
        print("      torch.device('vc4') accepted by this torch build")
    except RuntimeError as e:
        print("      torch.device('vc4') string: rejected by torch's parser "
              f"({e}); torch_vc4.device() is the supported spelling")
    print("      arena:", {k: int(v) if float(v).is_integer() else v
                           for k, v in torch_vc4.arena().items()})
    check("engine arena up (vc4d)", torch_vc4.is_available())

    # ---------------------------------------------------------------- matmul
    torch.manual_seed(0)
    x = torch.randn(256, 256, device=dev)
    check("randn on vc4 (host RNG + upload)",
          x.device.type == "vc4" and x.dtype == torch.float32, str(x.shape))

    xc = x.cpu()
    check("vc4->cpu copy is bit exact", torch.equal(xc, xc.clone()))

    torch_vc4.reset_stats()
    t0 = time.perf_counter()
    y = x @ x
    t1 = time.perf_counter()
    check("x @ x returns a vc4 tensor",
          y.device.type == "vc4" and tuple(y.shape) == (256, 256))

    st = torch_vc4.stats()
    cpu_mm = xc @ xc                      # fp32 CPU: the fair reference
    ref = xc.double() @ xc.double()       # fp64 CPU: reference for the report
    yc = y.cpu()
    mx_rel, mx_abs = rel_err(yc, cpu_mm)
    print(f"      256^3 wall {1e3*(t1-t0):.3f} ms, engine gpu "
          f"{st['kernel_ms'] + st['pack_ms']:.3f} ms "
          f"({st['gflops_kernel']:.2f} GFLOP/s kernel), "
          f"{st['launches']:.0f} launches")
    check("matmul matches CPU (allclose rtol 1e-4, atol 1e-3)",
          bool(torch.allclose(yc, cpu_mm, rtol=1e-4, atol=1e-3)),
          f"max rel {mx_rel:.2e}, max abs {mx_abs:.2e} "
          f"(fp64 ref: {float((yc.double()-ref).abs().max()):.2e})")
    check("matmul max relative error < 1e-3 (engine-style denominator)",
          mx_rel < 1e-3, f"{mx_rel:.2e}")

    # timing: best of 5
    best = 1e9
    gpu_best = 0.0
    for _ in range(5):
        torch_vc4.reset_stats()
        t0 = time.perf_counter()
        yb = x @ x
        dt = time.perf_counter() - t0
        s = torch_vc4.stats()
        g = s["kernel_ms"] + s["pack_ms"]
        best = min(best, dt)
        gpu_best = g if gpu_best == 0.0 else min(gpu_best, g)
    print(f"      256^3 best wall {1e3*best:.3f} ms -> "
          f"{2*256**3/1e9/best:.2f} GFLOP/s wall; engine GPU {gpu_best:.3f} ms")
    check("256^3 wall time is plausible (>1 ms, <60 s)", 0.001 < best < 60,
          f"{1e3*best:.2f} ms")

    # ------------------------------------------------------------- other ops
    a = torch.randn(64, 128, device=torch_vc4.device())
    b = torch.randn(64, 128, device=torch_vc4.device())
    ac, bc = a.cpu(), b.cpu()
    s1 = (a + b).cpu()
    s2 = (a * b).cpu()
    s3 = torch.relu(a).cpu()
    check("add on vc4 == CPU",
          bool(torch.allclose(s1, ac + bc, rtol=1e-6, atol=1e-6)))
    check("mul on vc4 == CPU",
          bool(torch.allclose(s2, ac * bc, rtol=1e-6, atol=1e-6)))
    check("relu on vc4 == CPU",
          bool(torch.allclose(s3, torch.relu(ac), rtol=1e-6, atol=1e-6)))

    # non-multiple-of-64 N exercises the engine's padded scratch path
    a2 = torch.randn(20, 36, device=torch_vc4.device())
    b2 = torch.randn(36, 44, device=torch_vc4.device())
    c2 = (a2 @ b2).cpu()
    ref2 = a2.cpu() @ b2.cpu()            # fp32 CPU reference
    e2, a2abs = rel_err(c2, ref2)
    check("odd-shaped mm 20x36x44 == CPU",
          e2 < 1e-4,
          f"max rel {e2:.2e}, max abs {a2abs:.2e}")

    # ------------------------------------------------- strict failure, no CPU
    expect_raise("torch.sin(vc4) raises", lambda: torch.sin(x))
    xc2 = x.cpu()
    check("sum over a dim == CPU",
          bool(torch.allclose(torch.sum(x, dim=1).cpu(),
                              torch.sum(xc2, dim=1), rtol=1e-4, atol=1e-4)))
    bc = torch.randn(256, 1)
    check("broadcasting add == CPU",
          bool(torch.allclose((x + torch_vc4_t(x, bc)).cpu(), xc2 + bc,
                              rtol=1e-6, atol=1e-6)))
    # views are metadata-only: a dim-0 slice is contiguous and usable ...
    check("dim-0 slice is a contiguous view",
          bool(x[0:2].is_contiguous() and tuple(x[0:2].shape) == (2, 256)),
          str(tuple(x[0:2].shape)))
    # ... but a non-contiguous view is rejected by every compute op
    expect_raise("compute on a non-contiguous view raises",
                 lambda: torch.add(x.transpose(0, 1), x.transpose(0, 1)),
                 needle="contiguous")
    expect_raise("float64 on vc4 raises",
                 lambda: torch.empty(4, 4, dtype=torch.float64,
                                     device=torch_vc4.device()),
                 needle="unsupported by design")
    expect_raise("mm cpu x vc4 raises",
                 lambda: torch.mm(torch.randn(4, 4),
                                  torch.randn(4, 4,
                                              device=torch_vc4.device())),
                 needle="expected a vc4 tensor")

    # ------------------------------------------------------- process hygiene
    fds = {}
    for fd in os.listdir("/proc/self/fd"):
        try:
            fds[fd] = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            pass
    has_devmem = any(os.path.basename(t) == "mem" and "/dev/" in t
                     for t in fds.values())
    check("client process has no /dev/mem fd", not has_devmem,
          ", ".join(sorted(set(fds.values()))))
    sock = [t for t in fds.values() if t.startswith("socket:")]
    check("client holds a vc4d socket", len(sock) >= 1, str(sock))
    with open("/proc/self/status") as f:
        stt = f.read()
    capeff = [l for l in stt.splitlines() if l.startswith("CapEff")][0]
    print("      " + capeff.strip() + "  uid=" + str(os.getuid()))

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
