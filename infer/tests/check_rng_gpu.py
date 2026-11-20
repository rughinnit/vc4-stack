#!/usr/bin/env python3
"""check_rng_gpu.py -- the counter-based QPU RNG, on the board.

Runs the ``VC4_UOP_RNG`` kernel (``kops.b_rng``) over N counter elements and
checks that the 10^6 draws behave like U[0,1): range, mean, variance, a
32-bin chi-square, a Kolmogorov-Smirnov distance, serial correlation at lags
1..8, a 2-D lattice test on consecutive pairs, bit balance, reproducibility
for one (seed, call) and de-correlation between calls.

Then the same for the two Box-Muller normals the RandomNormal handler builds
from it (mean, variance, skew, kurtosis, KS against the normal CDF).

    TMPDIR=$HOME/scratch/tmp python3 tests/check_rng_gpu.py [--n 1000000]

Exit status 0 only if every check passes.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding as vc4infer                  # noqa: E402

UOP_RNG = getattr(vc4infer, "UOP_RNG", None)


# The arena of a shared daemon can be small (another client holds part of
# it), and the engine has no free(): draw in blocks so the peak footprint is
# two chunk-sized buffers rather than two n-sized ones.  Each block uses its
# own launch key, so the blocks are independent draws of the same stream --
# the statistics are over the pooled 10^6 values, not over one launch.
CHUNK = int(os.environ.get("VC4_RNG_CHUNK", "250000"))


def counters(eng, n):
    c = np.arange(n, dtype=np.uint32).view(np.float32)
    return eng.upload(c)


def draw_blocks(eng, n, seed, base_call):
    """n uniforms as ceil(n / CHUNK) independent blocks.

    The counter and output buffers are allocated once and reused: the engine
    has no free(), so a per-block allocation would accumulate 2*CHUNK floats
    per block and fail on a small shared arena.
    """
    m = min(CHUNK, n)
    cb = counters(eng, m)
    ob = eng.alloc(m * 4)
    parts = []
    got, k = 0, 0
    while got < n:
        m = min(CHUNK, n - got)
        key = rng_key(seed, base_call + k)
        eng.unary(UOP_RNG, cb, ob, m, float(key & 0xFFFF),
                  float(key >> 16), 0.0)
        parts.append(eng.download(ob, m * 4).copy())
        got += m
        k += 1
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def rng_key(seed, call):
    """The stream key the RandomUniform handler derives (its rng_key)."""
    return (int(seed) * 0x9E3779B1 + int(call) * 0x85EBCA77) & 0xFFFFFFFF


def draw(eng, n, seed, call, cb=None):
    if cb is None:
        cb = counters(eng, n)
    out = eng.alloc(n * 4)
    k = rng_key(seed, call)
    eng.unary(UOP_RNG, cb, out, n, float(k & 0xFFFF), float(k >> 16), 0.0)
    return eng.download(out, n * 4).copy()


def chi2_p(stat, df):
    """Wilson-Hilferty tail for a chi-square (adequate for df > 10)."""
    if df <= 0:
        return 1.0
    z = ((stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9 * df))) / \
        math.sqrt(2.0 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ks_u01(u):
    n = u.size
    s = np.sort(u)
    f = np.arange(1, n + 1) / n
    return float(max(np.max(np.abs(f - s)), np.max(np.abs(s - (f - 1.0 / n)))))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--seed", type=float, default=12345.0)
    args = ap.parse_args(argv)
    if UOP_RNG is None:
        print("vc4infer.UOP_RNG is missing: rebuild the library")
        return 2
    n = args.n
    cache = os.environ.get("VC4INFER_CACHE") or None
    eng = vc4infer.Engine(cache_dir=cache)
    fails = []
    try:
        print("arena            %.1f MiB (blocks of %d)"
              % (eng.arena_bytes / 2 ** 20, CHUNK))
        u = draw_blocks(eng, n, args.seed, 0)
        ok = True
        lo, hi = float(u.min()), float(u.max())
        print("draws            %d" % n)
        print("range            [%.9f, %.9f)" % (lo, hi))
        if not (0.0 <= lo and hi < 1.0):
            fails.append("range")
        mean = float(u.mean())
        var = float(u.var())
        print("mean             %.6f   (0.5 +- %.6f at 3 sigma)"
              % (mean, 3 * (1 / 12.0) ** 0.5 / n ** 0.5))
        if abs(mean - 0.5) > 3 * (1 / 12.0) ** 0.5 / n ** 0.5:
            fails.append("mean")
        print("variance         %.6f   (1/12 = 0.083333 +- 0.00086)"
              % var)
        if abs(var - 1 / 12.0) > 0.00086:
            fails.append("variance")

        for bins in (16, 256):
            h = np.histogram(u, bins=bins, range=(0.0, 1.0))[0]
            stat = float(((h - n / bins) ** 2 / (n / bins)).sum())
            p = chi2_p(stat, bins - 1)
            print("chi2 %-4d bins    stat %.2f  df %d  p %.3f"
                  % (bins, stat, bins - 1, p))
            if p < 1e-4:
                fails.append("chi2-%d" % bins)

        d = ks_u01(u)
        lim = 1.95 / math.sqrt(n)
        print("KS distance      %.6f   (limit %.6f at p=1e-3)" % (d, lim))
        if d > lim:
            fails.append("ks")

        for lag in (1, 2, 3, 4, 8):
            c = float(np.corrcoef(u[:-lag], u[lag:])[0, 1])
            print("serial corr lag%d %.6f   (limit +-%.6f)"
                  % (lag, c, 4.0 / math.sqrt(n)))
            if abs(c) > 4.0 / math.sqrt(n) + 1e-6:
                fails.append("corr%d" % lag)

        # 2-D lattice: a plain Weyl/LCG stream lies on lines and shows it here
        h2 = np.histogram2d(u[:-1], u[1:], bins=64, range=[[0, 1], [0, 1]])[0]
        e2 = (n - 1) / 4096.0
        stat = float(((h2 - e2) ** 2 / e2).sum())
        p = chi2_p(stat, 4095)
        print("2-D chi2 64x64   stat %.1f  df 4095  p %.3f" % (stat, p))
        if p < 1e-4:
            fails.append("2d")

        # bit balance: every one of the 24 output bits should be a fair coin
        bits = (u * float(1 << 24)).astype(np.uint32)
        worst = 0.0
        for b in range(24):
            frac = float(((bits >> b) & 1).mean())
            worst = max(worst, abs(frac - 0.5))
        print("bit balance      worst |p-0.5| %.6f  (limit %.6f)"
              % (worst, 3 * 0.5 / math.sqrt(n)))
        if worst > 3 * 0.5 / math.sqrt(n):
            fails.append("bits")

        # the per-call stream: a graph that draws one element per launch walks
        # the *call* axis, so that axis gets its own uniformity test
        m = 4096
        cb1 = counters(eng, 1)
        ob = eng.alloc(64)
        ku = np.empty(m, np.float64)
        for i in range(m):
            k = rng_key(args.seed, i)
            eng.unary(UOP_RNG, cb1, ob, 1, float(k & 0xFFFF),
                      float(k >> 16), 0.0)
            ku[i] = eng.download(ob, 4)[0]
        h = np.histogram(ku, bins=16, range=(0.0, 1.0))[0]
        stat = float(((h - m / 16) ** 2 / (m / 16)).sum())
        p = chi2_p(stat, 15)
        d = ks_u01(ku)
        print("call-stream      chi2(16) %.2f p %.3f   KS %.4f (limit %.4f)"
              % (stat, p, d, 1.95 / math.sqrt(m)))
        if p < 1e-3 or d > 1.95 / math.sqrt(m):
            fails.append("call-stream")

        # reproducibility and de-correlation
        cb = counters(eng, 4096)
        a1 = draw(eng, 4096, args.seed, 7.0, cb)
        a2 = draw(eng, 4096, args.seed, 7.0, cb)
        a3 = draw(eng, 4096, args.seed, 8.0, cb)
        a4 = draw(eng, 4096, args.seed + 1.0, 7.0, cb)
        same = bool(np.array_equal(a1, a2))
        diff_call = int(np.count_nonzero(a1 == a3))
        diff_seed = int(np.count_nonzero(a1 == a4))
        print("reproducible     %s (%d/4096 equal for one (seed,call))"
              % (same, int(np.count_nonzero(a1 == a2))))
        print("decorrelated     call: %d/4096 equal, seed: %d/4096 equal"
              % (diff_call, diff_seed))
        if not same:
            fails.append("reproducible")
        if diff_call > 3 or diff_seed > 3:
            fails.append("decorrelated")

        # Box-Muller normals, as the RandomNormal handler composes them
        n2 = min(n, 1 << 20)
        u1 = draw_blocks(eng, n2, args.seed, 11).astype(np.float64)
        u2 = draw_blocks(eng, n2, args.seed, 211).astype(np.float64)
        u1 = np.maximum(u1, 1.0 / (1 << 24))
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        m, v = float(z.mean()), float(z.var())
        sk = float(((z - m) ** 3).mean() / v ** 1.5)
        ku = float(((z - m) ** 4).mean() / v ** 2)
        print("box-muller       mean %.5f  var %.5f  skew %.5f  kurt %.5f"
              % (m, v, sk, ku))
        if abs(m) > 3 / math.sqrt(n2) or abs(v - 1) > 0.01 or \
                abs(sk) > 0.02 or abs(ku - 3) > 0.05:
            fails.append("boxmuller")
        # a KS against the normal CDF on a sorted sample (n=4096 is enough
        # resolution for a 1e-3 test)
        zs = np.sort(z[:4096])
        f = (np.arange(1, 4097) - 0.5) / 4096.0
        cdf = 0.5 * (1.0 + np.array([math.erf(x / math.sqrt(2.0))
                                     for x in zs]))
        d = float(np.max(np.abs(f - cdf)))
        lim = 1.95 / math.sqrt(4096)
        print("box-muller KS    %.5f  (limit %.5f)" % (d, lim))
        if d > lim:
            fails.append("boxmuller-ks")
    finally:
        eng.close()
    print("")
    if fails:
        print("FAILED: %s" % ", ".join(fails))
        return 1
    print("all RNG checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
