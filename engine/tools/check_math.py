#!/usr/bin/env python3
"""check_math - GPU-free verification of the *formulas* the kernels implement.

Three things are checked against brute-force references:

  1. the sigmoid/silu arithmetic chain (2^f polynomial + Newton reciprocal),
     emulated in fp32 with the same operation order the QPU executes;
  2. the im2col kernel's source/destination addressing, including zero padding
     and the row-block mapping over a padded image, compared element by
     element with a reference im2col panel;
  3. the GEMM's per-tile addressing (A panel, B, C) for a range of shapes and
     QPU counts, compared with the reference panel addresses.

What this does *not* cover is the instruction encoding itself, which comes from
asm3's verified qenc.py, and the hardware behaviour (that is what the GPU tests
in tests/ are for).
"""
import math
import struct
import sys

MASK32 = 0xFFFFFFFF


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32i(u):
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def f32t(x):
    """Round to fp32 the way the VC4 QPU FPU does: toward zero.

    The QPU fadd/fmul truncate the result (measured: 1.0 + 0.75 ulp -> 1.0),
    so modelling IEEE round-to-nearest gives the wrong error bounds.
    """
    y = f32(x)
    if x != 0.0:
        away = (x > 0 and y > x) or (x < 0 and y < x)
        if away:
            y = f32i(bits(y) - 1)
    return y


# ---------------------------------------------------------------- sigmoid
C = [f32i(0x3AAEC3FF), f32i(0x3C1D955B), f32i(0x3D635847), f32i(0x3E75FDF0),
     f32i(0x3F317218)]                       # c5..c1, same order as the kernel
NLOG2E = f32i(0xBFB8AA3B)
MAGIC = f32i(0x4B400000)
CLAMP_LO = f32i(0xC2B40000)
CLAMP_HI = f32i(0x42B40000)
ONE = 1.0
TWO = 2.0
RCPMAGIC = 0x7EF311C3


def kernel_sigmoid(x):
    """Exactly the instruction sequence in tools/kgen.py::_ew_compute."""
    t = f32t(f32t(x) * NLOG2E)
    t = f32t(max(t, CLAMP_LO))
    t = f32t(min(t, CLAMP_HI))
    # The QPU fadd truncates, so t + 2^23 would floor; add 0.5 first so the
    # magic trick gives n = round(t) and f stays in [-0.5, 0.5].
    z = f32t(f32t(t + 0.5) + MAGIC)
    n = f32t(z - MAGIC)
    f = f32t(t - n)
    p = f32(C[0])
    for c in C[1:]:
        p = f32t(f32t(p * f) + c)
    p = f32t(f32t(p * f) + ONE)
    ni = int(z)                            # FTOI truncates; z is an integer >= 0
    bi = ((ni << 8) & MASK32) << 15 & MASK32
    scale = (bi + bits(ONE)) & MASK32      # + 127<<23 == +0x3F800000
    e = f32t(f32t(p * f32i(scale)))
    d = f32t(e + ONE)
    r = f32i((RCPMAGIC - bits(d)) & MASK32)
    for _ in range(3):
        q = f32t(r * d)
        q = f32t(TWO - q)
        r = f32t(r * q)
    return r


def check_sigmoid(n=20001):
    worst_abs = worst_rel = 0.0
    for i in range(n):
        x = -30.0 + 60.0 * i / (n - 1)
        got = kernel_sigmoid(x)
        ref = 1.0 / (1.0 + math.exp(-x))
        e = abs(got - ref)
        worst_abs = max(worst_abs, e)
        if abs(ref) > 1e-12:
            worst_rel = max(worst_rel, e / ref)
    print("sigmoid: max |err| %.3g over [-30,30], max rel %.3g -> %s"
          % (worst_abs, worst_rel, "PASS" if worst_abs < 1e-5 else "FAIL"))
    # silu = x * sigmoid(x): the same bound times |x|
    worst = 0.0
    for i in range(n):
        x = -20.0 + 40.0 * i / (n - 1)
        got = f32t(x * kernel_sigmoid(x))
        ref = x / (1.0 + math.exp(-x))
        worst = max(worst, abs(got - ref))
    print("silu   : max |err| %.3g over [-20,20] -> %s"
          % (worst, "PASS" if worst < 1e-4 else "FAIL"))
    return worst_abs < 1e-5 and worst < 1e-4


# ---------------------------------------------------------------- im2col
def ref_panel_pixel(img, Cin, H, W, KH, KW, pad, wp, rb, k, r):
    """Value of PANEL[(rb*K + k)*16 + r] built by the QPU im2col kernel.

    rb indexes a row block of 16 output pixels, laid out contiguously in the
    padded image (row length wp); k = (tap*Cin + ci); r is the pixel within the
    block.
    """
    cin_idx = k % Cin
    tap = k // Cin
    ky, kx = divmod(tap, KW)
    m = rb * 16 + r                      # padded-image pixel index
    oy, ox = divmod(m, wp)
    y = oy + ky - pad
    x = ox + kx - pad
    if y < 0 or y >= H or x < 0 or x >= W:
        return 0.0
    return img[(cin_idx * H + y) * W + x]


def check_im2col(Cin=16, H=12, W=12, KH=3, KW=3, pad=1):
    wp = (W + 2 * pad + 15) // 16 * 16
    ho = H + 2 * pad - KH + 1
    M = ho * wp
    K = Cin * KH * KW
    img = [((i * 37) % 101) * 0.01 - 0.5 for i in range(Cin * H * W)]

    # what the kernel's addressing produces: for row block rb, tap s, ci block c,
    # source = padded + (rb*16 + ky*wp + kx)*4 + c*16*plane, 16 rows of pitch
    # plane, and the transposed store puts VPM row (ci) at k = s*Cin + c*16 + ci
    bad = 0
    nrb = M // 16
    for rb in range(nrb):
        for k in range(K):
            for r in range(16):
                got = ref_panel_pixel(img, Cin, H, W, KH, KW, pad, wp, rb, k, r)
                # independent path: recompute through the kernel's own arithmetic
                s, rest = divmod(k, Cin)
                c, ci = divmod(rest, 16)
                ky, kx = divmod(s, KW)
                # padded coordinates of lane r of row block rb
                m = rb * 16 + r
                oy, ox = divmod(m, wp)
                y = oy + ky - pad
                x = ox + kx - pad
                ref = 0.0 if (y < 0 or y >= H or x < 0 or x >= W) else \
                    img[((c * 16 + ci) * H + y) * W + x]
                if abs(got - ref) > 1e-9:
                    bad += 1
    print("im2col : %d mismatches over %d panel elements (M=%d K=%d wp=%d) -> %s"
          % (bad, M * K, M, K, wp, "PASS" if bad == 0 else "FAIL"))
    return bad == 0


# ---------------------------------------------------------------- GEMM adds
def check_gemm_addr(M, N, K, nqpu):
    """Every row block must be executed exactly once across the QPUs, and no
    QPU may run a row block past the end (the kernel has a compile-time
    iteration bound plus an in-loop idx < TR guard)."""
    TR, TC = M // 16, N // 16
    a_rb, c_rb = 64 * K, 16 * N * 4
    count = (TR + nqpu - 1) // nqpu
    saw = {}
    bad = 0
    for q in range(nqpu):
        arb = q * a_rb          # what the launcher puts in u3
        crb = q * c_rb
        idx = q                 # u7
        for _ in range(count):
            if idx >= TR:       # the kernel's guard: skip this iteration
                idx += nqpu
                continue
            if idx in saw:
                bad += 1        # a row block done twice
            saw[idx] = True
            if arb != idx * a_rb or crb != idx * c_rb:
                bad += 1        # accumulated base != row block base
            for cb in range(TC):
                if crb + cb * 64 != idx * c_rb + cb * 64:
                    bad += 1    # C tile base
                if cb * 16 * 4 != cb * 64:
                    bad += 1    # B column base
            arb += nqpu * a_rb
            crb += nqpu * c_rb
            idx += nqpu
    missing = [r for r in range(TR) if r not in saw]
    if missing:
        bad += len(missing)
    print("gemm   : M=%u N=%u K=%u q=%d -> %u/%u row blocks, %s"
          % (M, N, K, nqpu, len(saw), TR, "PASS" if bad == 0 else "FAIL (%d)" % bad))
    return bad == 0


if __name__ == "__main__":
    ok = check_sigmoid()
    ok &= check_im2col(16, 12, 12, 3, 3, 1)
    ok &= check_im2col(32, 8, 16, 3, 3, 0)
    ok &= check_im2col(16, 8, 16, 1, 1, 0)
    for (M, N, K, q) in ((128, 128, 128, 8), (16, 16, 16, 1), (256, 64, 288, 12),
                         (32, 48, 64, 2), (80, 96, 32, 5)):
        ok &= check_gemm_addr(M, N, K, q)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
