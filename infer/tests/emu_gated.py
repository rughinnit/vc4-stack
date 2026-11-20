#!/usr/bin/env python3
"""emu_gated - run the five formerly gated kops kernels on the ALU/TMU model.

The model executes one group (the branch back is skipped, so the 16 vector
bodies run once) and stops at the first VPM store; A0..A15 (or A0..A3) then
hold exactly what the store writes.  Every TMU read is checked against the
memory the caller would have mapped, so a wild address shows up here instead
of wedging the board.

    python3 emu_gated.py gather|blit|reduce|cumsum|sel3 [op]
"""
import os
import struct
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "tools"))
sys.path.insert(0, _here)
import kgen
import kops
import emu_alu

A = 0xE0000000          # arena bus base (ctx->gpu_bus)
O = 0xE1000000


def bits(v):
    return int(struct.unpack('<I', struct.pack('<f', float(v)))[0])


class Mem(dict):
    misses = 0
    store_bad = 0
    def get(self, k, d=None):
        if k not in self:
            Mem.misses += 1
        return dict.get(self, k, d)


def run(kind, opts, unif, mem, nvec_out=16):
    p = kops.build(kgen, kind, opts)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "emu_gated.bin")
    p.write(path)
    words = emu_alu.load(path)
    m = emu_alu.Machine(words)
    m.mem = Mem(mem)
    m.unif = list(unif) + [0] * (16 - len(unif))
    stop = len(words)
    for i, w in enumerate(words):
        if ((w >> 60) & 0xF) == 1 and ((w >> 38) & 0x3F) == 50 and \
                ((w >> 44) & 1):
            stop = i
            break
    Mem.misses = 0
    m.run(0, stop)
    out = [float(m.getf('A', r)[lane]) for r in range(nvec_out)
           for lane in range(16)]
    if m.store is not None:
        # the DMA store address is a per-lane register: it must point into the
        # caller's output buffer.  Without this check a store that used a
        # body temp as its address reads back plausible registers and still
        # writes the wrong memory (that was gather/sel3/cumsum/reduce: A_CADR
        # was redefined by build_copy2d to A27 = the body's first temp).
        st = m.store[0]
        bad = [hex(int(a)) for a in st if not (O <= int(a) < O + 256 * 4)]
        if bad:
            print("  !! store address outside the output buffer: %s" % bad[:4])
            Mem.store_bad = len(bad)
        else:
            Mem.store_bad = 0
    return out, m, Mem.misses, len(words)


def run_ewb(op, abus, bbus, n, mem):
    """One nvec=4 group of a kops EWB body: 4 vectors x 16 lanes."""
    p = kops.build_ewb(kgen, op)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "emu_ewb.bin")
    p.write(path)
    words = emu_alu.load(path)
    m = emu_alu.Machine(words)
    m.mem = Mem(mem)
    m.unif = [abus, bbus, O, 1, 0, 1] + [0] * 10
    stop = len(words)
    for i, w in enumerate(words):
        if ((w >> 60) & 0xF) == 1 and ((w >> 38) & 0x3F) == 50 and \
                ((w >> 44) & 1):
            stop = i
            break
    Mem.misses = 0
    m.run(0, stop)
    out = [float(m.getf('A', r)[lane]) for r in range(4) for lane in range(16)]
    return out, m, Mem.misses, len(words)


def report(name, got, want, tol=0.0):
    bad = 0
    for g, w in zip(got, want):
        if tol:
            if abs(g - w) > tol + tol * abs(w):
                bad += 1
        elif bits(g) != bits(w):
            bad += 1
    print("%s %-22s %4d/%d bad" % ("PASS" if not bad else "FAIL", name, bad,
                                   len(want)))
    if bad:
        for i, (g, w) in enumerate(zip(got, want)):
            if (tol and abs(g - w) > tol + tol * abs(w)) or \
                    (not tol and bits(g) != bits(w)):
                if i < 500:
                    print("   [%d] got %.9g want %.9g" % (i, g, w))
    return bad


def main():
    kind = sys.argv[1]
    op = sys.argv[2] if len(sys.argv) > 2 else None
    opts = {}
    total_bad = 0

    if kind == 'gather':
        n = 256
        src = A + 0x1000
        tab = A + 0x2000
        mem = {}
        want = []
        for i in range(n):
            v = float(i) * 0.5 - 40.0
            mem[src + 4 * i] = bits(v)
            if i % 11 == 5:
                mem[tab + 4 * i] = 0xFFFFFFFF       # -1 -> fill
                want.append(-7.5)
            else:
                k = (n - 1 - i)
                mem[tab + 4 * i] = (src + 4 * k) - A
                want.append(float(k) * 0.5 - 40.0)
        got, m, miss, nw = run(kind, opts, [tab, O, 1, 0, 1, A, bits(-7.5)],
                               mem)
        total_bad += report("gather n=%d" % n, got, want)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    elif kind == 'blit':
        mem = {}
        want = []
        for i in range(4):
            v = float(i) + 0.25
            mem[A + 4 * i] = bits(v)
            want.append(v)
        got, m, miss, nw = run(kind, opts, [A, O, 1, 0, 1], mem, nvec_out=1)
        total_bad += report("blit group0", got, want)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    elif kind == 'cumsum':
        cols = 256
        mem = {}
        a = np.array([((i * 37) % 19) * 0.25 - 2.0 for i in range(cols)],
                     np.float64)
        for i in range(cols):
            mem[A + 4 * i] = bits(a[i])
        # u6 is the carry-seed address; the first chunk reads a zero word.
        mem[A + 0x3000] = bits(0.0)
        got, m, miss, nw = run(kind, opts, [A, O, 1, 0, 1, cols, A + 0x3000],
                               mem)
        want = list(np.cumsum(a))
        total_bad += report("cumsum cols=%d" % cols, got, want, tol=2e-6)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    elif kind == 'sel3':
        op = op or 'where'
        n = 256
        mem = {}
        cond, bv, cv, want = [], [], [], []
        for i in range(n):
            c = [0.0, -0.0, 1.0, -1.0, 2.5, -2.5][i % 6]
            cond.append(c)
            bv.append(float(i) + 0.5)
            cv.append(-float(i) - 0.25)
            if op == 'where':
                want.append(bv[-1] if c != 0.0 else cv[-1])
            elif op == 'clip3':
                want.append(min(max(bv[-1], c), cv[-1]))
            else:
                want.append(c if c > 0 else c * bv[-1])
        for i in range(n):
            mem[A + 4 * i] = bits(cond[i])
            mem[A + 0x1000 + 4 * i] = bits(bv[i])
            mem[A + 0x2000 + 4 * i] = bits(cv[i])
        got, m, miss, nw = run(kind, {'op': op},
                               [A, O, 1, 0, 1, A + 0x1000, A + 0x2000, 0], mem)
        total_bad += report("sel3 %s n=%d" % (op, n), got, want)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    elif kind == 'reduce':
        op = op or 'sum'
        outer, red = 256, 5
        mem = {}
        data = np.array([((i * 37) % 1013 - 500) * 0.125 for i in
                         range(outer * red)], np.float64)
        for i, v in enumerate(data):
            mem[A + 4 * i] = bits(v)
        vstr = red * 64
        u = [A, O, 1, 0, 1, red, bits(1.0 / red), vstr, 0, 0,
             (1 - 1) * vstr * 16, (1 - 1) * 1024, (1 - 1) * 1024, red * 4]
        got, m, miss, nw = run(kind, {'op': op}, u, mem)
        want = []
        for o in range(outer):
            r = data[o * red:(o + 1) * red]
            if op == 'sum':
                want.append(float(r.sum()))
            elif op == 'mean':
                want.append(float(r.sum() / red))
            elif op == 'max':
                want.append(float(r.max()))
            elif op == 'min':
                want.append(float(r.min()))
            elif op == 'argmax':
                want.append(float(np.argmax(r)))
            else:
                want.append(float(np.argmin(r)))
        if op in ('argmax', 'argmin'):
            # the kernel writes raw int32 indices; the float view is a denormal
            got = [float(bits(g) & 0xFFFFFFFF) for g in got]
        total_bad += report("reduce %s %dx%d" % (op, outer, red), got, want,
                            tol=0.0 if op in ('argmax', 'argmin', 'max', 'min')
                            else 3e-5)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    elif kind == 'gelu_erf_bwd':
        import math
        n = 64                                   # 4 vectors per group
        xs = [((i * 37) % 211 - 105) * 0.19 for i in range(n)]
        gs = [0.25 + (i % 7) * 0.5 for i in range(n)]
        bbus = A + 0x1000
        mem = {}
        for i in range(n):
            mem[A + 4 * i] = bits(xs[i])
            mem[bbus + 4 * i] = bits(gs[i])
        got, m, miss, nw = run_ewb('gelu_erf_bwd', A, bbus, n, mem)
        want = []
        for i in range(n):
            x = xs[i]
            d = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + \
                x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
            want.append(gs[i] * d)
        total_bad += report("gelu_erf_bwd n=%d" % n, got, want, tol=2e-5)
        print("   TMU outside the mapped arena: %d, words %d" % (miss, nw))

    if Mem.store_bad:
        total_bad += 1
    print("RESULT", "FAIL" if total_bad else "PASS")
    return 1 if total_bad else 0


sys.exit(main())
