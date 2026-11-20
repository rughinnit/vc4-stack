#!/usr/bin/env python3
"""kops - generic QPU kernels for the numeric op long tail of the ONNX runner.

`build(kgen_module, kind, opts)` is called from kgen.main().  Every kernel
reuses the verified building blocks in kgen.py: the QPU encoder (qenc.py),
the verified VPM->DMA stores (`emit_store16` / `_store4`), the TMU
issue/pop rule and the Builder's no-forwarding checker.

Kinds
    uop     --op NAME      out = f(a)               unary math (UOPS)
    sel3    --op NAME      out = f(a,b,c)           where / clip3 / prelu
    gather                 out[o] = arena[tab[o]]   int32 byte-offset table
    reduce  --op NAME      out[o] = reduce_r a[o*red + r]
    cumsum                 out[r][c] = sum_{c'<=c} a[r][c']  (padded cols)

Kernel ABI (one 64 B uniform block per QPU; all offsets are arena bytes)
    uop     u0 a bus, u1 out bus, u2 groups, u3 qid, u4 nqpu,
            u5,u6,u7 float parameters (op attributes)
    sel3    u0 a, u1 b, u2 c, u3 out, u4 groups, u5 qid, u6 nqpu, u7 op
    gather  u0 table bus, u1 out bus, u2 groups, u3 qid, u4 nqpu,
            u5 arena bus base, u6 fill (float bits)
    reduce  u0 a bus, u1 out bus, u2 groups for this QPU, u3 qid, u4 nqpu,
            u5 red, u6 op, u7 in offset, u8 in gap, u9 in step,
            u10 out offset, u11 out step, u12 out gap, u13 1/red,
            u14 lanestride (red*4)
    cumsum  u0 a bus, u1 out bus, u2 groups, u3 qid, u4 nqpu, u5 cols

A "group" is NVEC vectors of 16 lanes; QPU q runs groups q, q+nqpu, ...
Each vector produces 16 consecutive output floats.  The QPU FPU truncates
(round toward zero) on both fadd and fmul, so every transcendental here is
range reduction + minimax polynomial + exact integer scaling.
"""
import struct

# ---------------------------------------------------------------- registers
A_VPMQ, A_CADR = 25, 28                     # reserved: the store sequences
B_STEP, B_CSTEP, B_VPMQ7 = 22, 24, 23
A_AB, A_OB, A_CNT, A_QID, A_NQ = 16, 17, 18, 19, 20
A_IB, A_OB2, A_P0, A_P1, A_P2 = 21, 22, 23, 24, 26
A_T0, A_T1, A_T2, A_T3 = 27, 29, 30, 31
# reduce kernel loop state (distinct from the ctx A temps, which reduce does
# not use)
A_RA, A_RIDX, A_NR, A_BESTI = 23, 24, 31, 26
A_INBASE, A_OUTBASE = 30, 29
B_C64, B_GAP, B_NEG1, B_V4, B_V9 = 21, 17, 20, 19, 18

M32 = 0xFFFFFFFF
ACC0, ACC1, ACC2, ACC3 = 100, 101, 102, 103


def _fb(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def _d(reg):
    if reg >= 200:
        return reg - 200, 1
    if reg >= 100:
        return 32 + reg - 100, 0
    return reg, 0


def _s(k, reg):
    if reg >= 200:
        return k.breg(reg - 200)
    if reg >= 100:
        return k.acc(reg - 100)
    return k.areg(reg)


def _bank(reg):
    if reg >= 200:
        return 'B'
    if reg >= 100:
        return 'C'
    return 'A'


class E:
    """ALU emitter with automatic bank-conflict staging.

    The VC4 ALU can read at most one A-file and one B-file register per
    instruction (qenc raises otherwise).  Constants live in B and temps in A,
    so `B op B` and `A op A` words are common; the emitter stages the second
    operand through acc2, which no body or core uses for data.
    """

    def __init__(self, k, b, ctx):
        self.k, self.b, self.ctx = k, b, ctx

    def _fit(self, a, bb):
        """Make the operands addressable by one ALU word.

        The VC4 word has one A and one B read slot, so two operands from the
        same file are staged through ACC2.  `self.staged` records that the
        move happened: the *multiply* ALU cannot read an add-ALU accumulator
        result in the next word (measured; NUMERICS.md), so mul() pads two
        words before consuming it.  The add ALU may read it immediately.
        """
        self.staged = False
        if bb is None or a == bb:
            return a, bb
        ba, bbb = _bank(a), _bank(bb)
        if ba == bbb and ba in ('A', 'B'):
            self.mov(ACC2, bb)
            self.staged = True
            return a, ACC2
        return a, bb

    def op(self, op, out, a, bb=None, sf=0):
        a, bb = self._fit(a, bb)
        o, ws = _d(out)
        d = dict(add_op=op, add_out=o, ws=ws, add_a=_s(self.k, a), sf=sf)
        if bb is not None:
            d['add_b'] = _s(self.k, bb)
        self.b.e(d)

    def opi(self, op, out, a, imm):
        if _bank(a) == 'B':
            # a B register and a small immediate cannot share one word
            self.mov(ACC2, a)
            a = ACC2
        o, ws = _d(out)
        self.b.e(dict(add_op=op, add_out=o, ws=ws, add_a=_s(self.k, a),
                      add_b=self.k.imm(imm)))

    def mul(self, out, a, bb):
        a, bb = self._fit(a, bb)
        if getattr(self, 'staged', False):
            self.b.e(self.k.nop(), self.k.nop())    # add-ALU ACC -> mul ALU
        if 100 <= out < 200:
            o = 32 + out - 100
        else:
            o = 32 + 3
        self.b.e(dict(mul_op=self.k.MUL_FMUL, mul_out=o, ws=0,
                      mul_a=_s(self.k, a), mul_b=_s(self.k, bb)))
        self.b.e(self.k.nop(), self.k.nop())
        if not (100 <= out < 200):
            self.mov(out, ACC3)

    def mul24(self, out, a, bb):
        a, bb = self._fit(a, bb)
        if getattr(self, 'staged', False):
            self.b.e(self.k.nop(), self.k.nop())
        assert 100 <= out < 200, "MUL24 result must be an accumulator"
        self.b.e(dict(mul_op=self.k.MUL_MUL24, mul_out=32 + out - 100, ws=0,
                      mul_a=_s(self.k, a), mul_b=_s(self.k, bb)))

    def mov(self, out, a):
        o, ws = _d(out)
        self.b.e(dict(add_op=self.k.ADD_OR, add_out=o, ws=ws,
                      add_a=_s(self.k, a), add_b=_s(self.k, a)))


class Ctx:
    """Constant lookup + scratch registers for one vector body."""

    def __init__(self, consts):
        self.c = consts
        # A_T0..3, then the prologue-only A registers (bus bases, qid, nqpu):
        # all dead once the group loop starts.
        self.a_free = [A_T0, A_T1, A_T2, A_T3, A_AB, A_OB, A_QID, A_NQ]
        used = set(v[1] for v in consts.values() if v[0] == 'B')
        self.b_free = [n for n in range(0, 16) if n not in used]
        self.b_free += [n for n in range(25, 32) if n not in used]
        self.a0 = list(self.a_free)
        self.b0 = list(self.b_free)

    def reset(self):
        """Every vector body starts with the whole scratch pool again."""
        self.a_free = list(self.a0)
        self.b_free = list(self.b0)

    def no_a(self):
        self.a_free = []
        self.a0 = []

    def k(self, name):
        bank, reg = self.c[name]
        return 200 + reg if bank == 'B' else reg

    def ta(self):
        if self.a_free:
            return self.a_free.pop(0)
        return 200 + self.b_free.pop(0)

    def tb(self):
        if self.b_free:
            return 200 + self.b_free.pop(0)
        return self.a_free.pop(0)

    def rel(self, *rs):
        for r in rs:
            if r >= 200:
                self.b_free.insert(0, r - 200)
            else:
                self.a_free.insert(0, r)


# ---------------------------------------------------------------- helpers
def gt_mask(e, out, d):
    k = e.k
    e.opi(k.ADD_SUB, out, d, 1)
    e.opi(k.ADD_ASR, out, out, 31)
    e.op(k.ADD_NOT, out, out, out)


def eq0_mask(e, out, d, t):
    k = e.k
    e.op(k.ADD_OR, t, d, d)
    e.opi(k.ADD_SUB, t, t, 1)
    e.opi(k.ADD_ASR, t, t, 31)
    e.opi(k.ADD_ASR, out, d, 31)
    e.op(k.ADD_NOT, out, out, out)
    e.op(k.ADD_AND, out, out, t)


def sel(e, out, mask, a_true, b_false, t):
    k = e.k
    e.op(k.ADD_AND, t, mask, a_true)
    e.op(k.ADD_NOT, out, mask, mask)
    e.op(k.ADD_AND, out, out, b_false)
    e.op(k.ADD_OR, out, out, t)


def keep_sel(e, out, mask, a_true):
    """out = mask ? a_true : out, safe when the false operand is `out`.

    `sel` writes `out` before reading the false operand, so passing `out` as
    b_false makes the padded read see the new ~mask instead of the old value.
    Here the AND with the old value comes first: a same-word read returns the
    pre-write register (the one forwarding rule the QPU does have).  Only one
    scratch register is needed, which matters because the long bodies are at
    the B-register budget already.
    """
    k = e.k
    t = e.ctx.tb()
    e.op(k.ADD_NOT, t, mask, mask)
    e.op(k.ADD_AND, out, out, t)        # old out & ~mask (same-word read)
    e.op(k.ADD_AND, t, mask, a_true)
    e.op(k.ADD_OR, out, out, t)
    e.ctx.rel(t)


def core_exp2(e, t, out, z, sc):
    """out = 2^t for t in [-126,128]; t,out,z,sc distinct registers.

    If a caller aliases t and out (the natural "exp in place" call) the
    fraction is copied first: the seed `out = E5` below would otherwise
    destroy the argument before the Horner loop reads it.
    """
    k = e.k
    copied = False
    if t == out:
        t = e.ctx.ta() if _bank(out) == 'A' else e.ctx.tb()
        e.mov(t, out)
        copied = True
    e.op(k.ADD_FMAX, t, t, e.ctx.k('EXP_LO'))
    e.op(k.ADD_FMIN, t, t, e.ctx.k('EXP_HI'))
    e.op(k.ADD_FADD, z, t, e.ctx.k('HALF'))
    e.op(k.ADD_FADD, z, z, e.ctx.k('MAGIC'))
    e.op(k.ADD_FTOI, sc, z, z)
    e.opi(k.ADD_SHL, sc, sc, 8)
    e.opi(k.ADD_SHL, sc, sc, 15)
    e.op(k.ADD_ADD, sc, sc, e.ctx.k('ONE'))
    e.op(k.ADD_FSUB, z, z, e.ctx.k('MAGIC'))
    e.op(k.ADD_FSUB, t, t, z)
    e.op(k.ADD_OR, out, e.ctx.k('E5'), e.ctx.k('E5'))
    for c in ('E4', 'E3', 'E2', 'E1'):
        e.mul(out, out, t)
        e.op(k.ADD_FADD, out, out, e.ctx.k(c))
    e.mul(out, out, t)
    e.op(k.ADD_FADD, out, out, e.ctx.k('ONE'))
    e.mul(out, out, sc)
    if copied:
        e.ctx.rel(t)


def core_rcp(e, d, out, q):
    k = e.k
    copied = False
    if d == out:
        # the seed below overwrites , so an in-place call must keep the
        # denominator somewhere else first
        d = e.ctx.ta() if _bank(out) == 'A' else e.ctx.tb()
        e.mov(d, out)
        copied = True
    e.op(k.ADD_SUB, out, e.ctx.k('RCPM'), d)
    for _ in range(4):
        e.mul(q, out, d)
        e.op(k.ADD_FSUB, q, e.ctx.k('TWO'), q)
        e.mul(out, out, q)
    if copied:
        e.ctx.rel(d)


def core_rsqrt(e, a, out, q):
    k = e.k
    e.opi(k.ADD_SHR, out, a, 1)
    e.op(k.ADD_SUB, out, e.ctx.k('RSQRTC'), out)
    for _ in range(4):
        e.mul(q, a, out)
        e.mul(q, q, out)
        e.mul(q, q, e.ctx.k('HALF'))
        e.op(k.ADD_FSUB, q, e.ctx.k('THREEH'), q)
        e.mul(out, out, q)


def core_tanh(e, x, out, t1, t2, t3):
    k = e.k
    ax = e.ctx.ta()
    e.op(k.ADD_AND, ax, x, e.ctx.k('ABSM'))
    e.op(k.ADD_FMIN, ax, ax, e.ctx.k('T44'))
    e.mul(t1, ax, e.ctx.k('NLOG2E2'))
    core_exp2(e, t1, out, t2, t3)
    e.op(k.ADD_FADD, t2, out, e.ctx.k('ONE'))
    core_rcp(e, t2, t1, t3)
    e.op(k.ADD_FSUB, t3, e.ctx.k('ONE'), out)
    e.mul(t3, t3, t1)
    e.op(k.ADD_AND, t1, x, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, out, t3, t1)
    e.ctx.rel(ax)


def core_erf(e, x, out, t1, t2, t3):
    k = e.k
    a = e.ctx.ta()
    e.op(k.ADD_AND, a, x, e.ctx.k('ABSM'))
    e.op(k.ADD_FMIN, a, a, e.ctx.k('SIX'))
    e.mul(t1, a, e.ctx.k('ERFC1'))
    e.op(k.ADD_FADD, t1, t1, e.ctx.k('ONE'))
    core_rcp(e, t1, t2, t3)
    p = e.ctx.ta()
    e.op(k.ADD_OR, p, e.ctx.k('ERF5'), e.ctx.k('ERF5'))
    for c in ('ERF4', 'ERF3', 'ERF2', 'ERF1'):
        e.mul(p, p, t2)
        e.op(k.ADD_FADD, p, p, e.ctx.k(c))
    e.mul(p, p, t2)
    e.mul(t1, a, a)
    e.mul(t1, t1, e.ctx.k('NLOG2E'))
    core_exp2(e, t1, t1, t3, t2)
    e.mul(p, p, t1)
    e.op(k.ADD_FSUB, p, e.ctx.k('ONE'), p)
    e.op(k.ADD_AND, t1, x, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, out, p, t1)
    e.ctx.rel(a, p)


# ---------------------------------------------------------------- bodies
def b_abs(e, IN, OUT):
    e.op(e.k.ADD_AND, OUT, IN, e.ctx.k('ABSM'))


def b_neg(e, IN, OUT):
    e.op(e.k.ADD_XOR, OUT, IN, e.ctx.k('SIGNB'))


def b_identity(e, IN, OUT):
    e.mov(OUT, IN)


def b_clip(e, IN, OUT):
    k = e.k
    e.op(k.ADD_FMAX, IN, IN, e.ctx.k('P_LO'))
    e.op(k.ADD_FMIN, OUT, IN, e.ctx.k('P_HI'))


def b_leakyrelu(e, IN, OUT):
    k = e.k
    t = e.ctx.tb()
    e.op(k.ADD_FMAX, OUT, IN, e.ctx.k('ZERO'))
    e.op(k.ADD_FMIN, t, IN, e.ctx.k('ZERO'))
    e.mul(t, t, e.ctx.k('ALPHA'))
    e.op(k.ADD_FADD, OUT, OUT, t)
    e.ctx.rel(t)


def _trunc(e, IN):
    """n = ceil(|x|) as int, g = -1 iff |x| has a fraction, sg = sign mask."""
    k = e.k
    ax = e.ctx.tb()
    n = e.ctx.tb()
    fr = e.ctx.tb()
    g = e.ctx.tb()
    sg = e.ctx.tb()
    e.op(k.ADD_AND, ax, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FTOI, n, ax, ax)
    e.op(k.ADD_ITOF, fr, n, n)
    e.op(k.ADD_FSUB, fr, ax, fr)
    gt_mask(e, g, fr)
    e.op(k.ADD_SUB, n, n, g)
    e.opi(k.ADD_ASR, sg, IN, 31)
    e.ctx.rel(ax, fr)
    return n, g, sg


def b_floor(e, IN, OUT):
    k = e.k
    n, g, sg = _trunc(e, IN)
    fl = e.ctx.tb()
    e.op(k.ADD_ADD, fl, n, g)
    t = e.ctx.tb()
    e.op(k.ADD_SUB, n, e.ctx.k('ZERO'), n)
    sel(e, OUT, sg, n, fl, t)
    e.op(k.ADD_ITOF, OUT, OUT, OUT)
    # |x| >= 2^23 is already an integer: return x.  This also passes inf
    # through instead of overflowing the int32 FTOI (floor(-1e10) was wrong).
    e.op(k.ADD_AND, n, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, n, n, e.ctx.k('T23'))
    gt_mask(e, sg, n)
    keep_sel(e, OUT, sg, IN)
    e.ctx.rel(n, g, sg, fl, t)


def b_ceil(e, IN, OUT):
    k = e.k
    n, g, sg = _trunc(e, IN)
    fl = e.ctx.tb()
    e.op(k.ADD_ADD, fl, n, g)
    t = e.ctx.tb()
    e.op(k.ADD_SUB, fl, e.ctx.k('ZERO'), fl)
    sel(e, OUT, sg, fl, n, t)
    e.op(k.ADD_ITOF, OUT, OUT, OUT)
    e.op(k.ADD_AND, n, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, n, n, e.ctx.k('T23'))
    gt_mask(e, sg, n)
    keep_sel(e, OUT, sg, IN)
    e.ctx.rel(n, g, sg, fl, t)


def b_round(e, IN, OUT):
    """ONNX Round: nearest, ties to even, all integer work."""
    k = e.k
    n = e.ctx.tb()
    nf = e.ctx.tb()
    fr = e.ctx.tb()
    d = e.ctx.tb()
    mk = e.ctx.tb()
    lsb = e.ctx.tb()
    e.op(k.ADD_FTOI, n, IN, IN)
    e.op(k.ADD_ITOF, nf, n, n)
    e.op(k.ADD_FSUB, fr, IN, nf)
    e.op(k.ADD_AND, lsb, n, e.ctx.k('ONE_I'))
    e.op(k.ADD_FSUB, d, fr, e.ctx.k('HALF'))
    gt_mask(e, mk, d)
    e.op(k.ADD_SUB, n, n, mk)
    eq0_mask(e, mk, d, e.ctx.tb())
    e.op(k.ADD_AND, mk, mk, lsb)
    e.op(k.ADD_ADD, n, n, mk)
    e.op(k.ADD_FADD, d, fr, e.ctx.k('HALF'))
    e.opi(k.ADD_ASR, mk, d, 31)
    e.op(k.ADD_ADD, n, n, mk)
    eq0_mask(e, mk, d, e.ctx.tb())
    e.op(k.ADD_AND, mk, mk, lsb)
    e.op(k.ADD_SUB, n, n, mk)
    e.op(k.ADD_ITOF, OUT, n, n)
    m = e.ctx.tb()
    e.op(k.ADD_AND, m, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, m, m, e.ctx.k('T23'))
    gt_mask(e, mk, m)
    keep_sel(e, OUT, mk, IN)
    e.ctx.rel(n, nf, fr, d, mk, lsb, m)


def b_sign(e, IN, OUT):
    """ONNX Sign: -1, 0, +1; both zeros map to +0.

    The integer "float bits > 0" test is used on |x| (so the sign bit cannot
    turn -0.0 into a wrapped positive) and then split by the sign bit.
    """
    k = e.k
    a = e.ctx.tb()
    m = e.ctx.tb()
    sg = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    gt_mask(e, m, a)                       # all-ones iff |x| != 0
    e.opi(k.ADD_ASR, sg, IN, 31)           # all-ones iff sign bit set
    e.op(k.ADD_AND, a, m, e.ctx.k('ONE'))  # +1 (or 0)
    e.op(k.ADD_AND, sg, sg, m)
    e.op(k.ADD_AND, sg, sg, e.ctx.k('NEGONE'))
    e.op(k.ADD_OR, OUT, a, sg)
    e.ctx.rel(a, m, sg)


def b_exp(e, IN, OUT):
    k = e.k
    t = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('LOG2E'))
    core_exp2(e, t, OUT, e.ctx.tb(), e.ctx.tb())
    e.ctx.rel(t)


def core_ln(e, IN, OUT, terms=5):
    """ln(IN) for a positive normal: the verified `Log` body, factored out.

    `b_log` is the wrapper with `terms=5`, so the emitted words are unchanged
    (the same sequence that was element-verified on 2026-11-20).  The
    asinh/acosh/atanh bodies reuse it with `terms=4`: their argument is always
    near or above 1, and dropping the u^9 term costs 1.5e-9 relative, but
    saves the B-register slot the five-term form would need.

    Note: this *overwrites* IN with max(IN, 2^-126), exactly like the body it
    came from, so callers passing an argument they still need must copy it.
    """
    k = e.k
    ee = e.ctx.tb()
    m = e.ctx.tb()
    d = e.ctx.tb()
    mk = e.ctx.tb()
    e.op(k.ADD_FMAX, IN, IN, e.ctx.k('TINY'))
    e.opi(k.ADD_SHR, ee, IN, 23)
    e.op(k.ADD_SUB, ee, ee, e.ctx.k('C127'))
    e.op(k.ADD_AND, m, IN, e.ctx.k('MANT'))
    e.op(k.ADD_OR, m, m, e.ctx.k('ONE'))
    e.op(k.ADD_FSUB, d, m, e.ctx.k('SQRT2'))
    gt_mask(e, mk, d)
    e.mul(d, m, e.ctx.k('HALF'))
    # m = mk ? m/2 : m.  The old code wrote ~mk into m and then AND-ed m with
    # itself, losing the mantissa (NaN for every power of two).
    keep_sel(e, m, mk, d)
    e.op(k.ADD_AND, mk, mk, e.ctx.k('ONE_I'))
    e.op(k.ADD_ADD, ee, ee, mk)
    e.ctx.rel(d, mk)
    num = e.ctx.tb()
    den = e.ctx.tb()
    e.op(k.ADD_FSUB, num, m, e.ctx.k('ONE'))
    e.op(k.ADD_FADD, den, m, e.ctx.k('ONE'))
    core_rcp(e, den, OUT, e.ctx.tb())
    e.mul(num, num, OUT)
    e.ctx.rel(den)
    u2 = e.ctx.tb()
    p = e.ctx.tb()
    e.mul(u2, num, num)
    top = 5 if terms >= 5 else 4
    e.op(k.ADD_OR, p, e.ctx.k('L%d' % (top - 1)),
         e.ctx.k('L%d' % (top - 1)))
    for c in ('L3', 'L2', 'L1', 'L0') if top == 5 else ('L2', 'L1', 'L0'):
        e.mul(p, p, u2)
        e.op(k.ADD_FADD, p, p, e.ctx.k(c))
    e.mul(p, p, num)
    e.op(k.ADD_FADD, p, p, p)
    e.op(k.ADD_ITOF, ee, ee, ee)
    e.mul(ee, ee, e.ctx.k('LN2'))
    e.op(k.ADD_FADD, OUT, ee, p)
    e.ctx.rel(num, u2, p, m, ee)


def b_log(e, IN, OUT):
    core_ln(e, IN, OUT)


def b_sqrt(e, IN, OUT):
    """sqrt(x) = x * (1/sqrt(x)); the executor restricts x to finite, >= 0."""
    k = e.k
    a = e.ctx.tb()
    y = e.ctx.tb()
    e.op(k.ADD_FMAX, a, IN, e.ctx.k('TINY'))
    core_rsqrt(e, a, y, e.ctx.tb())
    e.mul(OUT, IN, y)
    e.ctx.rel(a, y)


def b_reciprocal(e, IN, OUT):
    """1/x for every fp32 x the datapath can represent.

    The Newton core works on a positive normal magnitude.  |x| < 2^-126
    (zero and denormals, which the FPU flushes) maps to +-inf, which is also
    what a float32 reference produces: 1/1.4e-45 overflows.  The sign bit is
    restored at the end, so negative inputs (a case the old max(x,TINY) clamp
    destroyed) are correct.  The result is a denormal - and therefore 0 - for
    |x| > 2^126; that is the one documented range limit.
    """
    k = e.k
    a = e.ctx.tb()
    r = e.ctx.tb()
    sg = e.ctx.tb()
    d = e.ctx.tb()
    m = e.ctx.tb()
    t = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_AND, sg, IN, e.ctx.k('SIGNB'))
    # m = all-ones iff |x| >= 2^-126 (the smallest normal); both zeros and
    # denormals take the inf path
    e.op(k.ADD_SUB, d, a, e.ctx.k('TINYI'))
    e.op(k.ADD_ADD, d, d, e.ctx.k('ONE_I'))
    gt_mask(e, m, d)
    e.op(k.ADD_FMAX, a, a, e.ctx.k('TINY'))    # no-op for normals
    e.op(k.ADD_FMIN, a, a, e.ctx.k('BIG'))
    core_rcp(e, a, r, e.ctx.tb())
    e.op(k.ADD_OR, r, r, sg)
    e.op(k.ADD_OR, t, sg, e.ctx.k('POSINF'))
    sel(e, OUT, m, r, t, d)
    e.ctx.rel(a, r, sg, d, m, t)


def b_tanh(e, IN, OUT):
    core_tanh(e, IN, OUT, e.ctx.tb(), e.ctx.tb(), e.ctx.tb())


def b_erf(e, IN, OUT):
    """A&S 7.1.26, then the exact +-1 saturation for |x| >= 4.

    The truncating FPU rounds 1-eps down to 1-2^-24, so the polynomial alone
    makes the composed Gelu-erf return x*(2^-24)/2 (-298 at x=-1e10).  The
    true fp32 erf is exactly +-1 from about |x| = 4.
    """
    k = e.k
    core_erf(e, IN, OUT, e.ctx.tb(), e.ctx.tb(), e.ctx.tb())
    a = e.ctx.tb()
    m = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, a, a, e.ctx.k('SIX'))
    gt_mask(e, m, a)
    e.op(k.ADD_AND, a, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, a, a, e.ctx.k('ONE'))
    keep_sel(e, OUT, m, a)
    e.ctx.rel(a, m)


def b_elu(e, IN, OUT):
    k = e.k
    m = e.ctx.tb()
    t = e.ctx.tb()
    u = e.ctx.tb()
    gt_mask(e, m, IN)
    e.op(k.ADD_FMIN, t, IN, e.ctx.k('ZERO'))
    e.mul(u, t, e.ctx.k('LOG2E'))
    core_exp2(e, u, u, e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FSUB, u, u, e.ctx.k('ONE'))
    e.mul(u, u, e.ctx.k('ALPHA'))
    sel(e, OUT, m, IN, u, t)
    e.ctx.rel(m, t, u)


def b_selu(e, IN, OUT):
    k = e.k
    m = e.ctx.tb()
    t = e.ctx.tb()
    u = e.ctx.tb()
    gt_mask(e, m, IN)
    e.op(k.ADD_FMIN, t, IN, e.ctx.k('ZERO'))
    e.mul(u, t, e.ctx.k('LOG2E'))
    core_exp2(e, u, u, e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FSUB, u, u, e.ctx.k('ONE'))
    e.mul(u, u, e.ctx.k('ALPHA'))
    sel(e, OUT, m, IN, u, t)
    e.mul(OUT, OUT, e.ctx.k('GAMMA'))
    e.ctx.rel(m, t, u)


def core_softplus(e, IN, out):
    """softplus(x) = max(x,0) + log1p(exp(-|x|)).

    t = exp(-|x|); u = t/(2+t) = tanh(log(1+t)/2); log1p(t) is the atanh
    series 2*(u + u^3/3 + u^5/5 + ...).  The |x| form never exponentiates a
    positive argument, so it cannot overflow.
    """
    k = e.k
    ax = e.ctx.tb()
    a = e.ctx.tb()
    e.op(k.ADD_AND, ax, IN, e.ctx.k('ABSM'))
    e.mul(a, ax, e.ctx.k('NLOG2E'))
    z = e.ctx.tb()
    sc = e.ctx.tb()
    core_exp2(e, a, a, z, sc)
    e.ctx.rel(z, sc)
    e.op(k.ADD_FADD, a, a, e.ctx.k('ONE'))
    u = e.ctx.tb()
    den = e.ctx.tb()
    q = e.ctx.tb()
    e.op(k.ADD_FSUB, u, a, e.ctx.k('ONE'))
    e.op(k.ADD_FADD, den, a, e.ctx.k('ONE'))
    core_rcp(e, den, den, q)
    e.ctx.rel(q)
    e.mul(u, u, den)
    u2 = e.ctx.tb()
    p = e.ctx.tb()
    e.mul(u2, u, u)
    e.op(k.ADD_OR, p, e.ctx.k('L4'), e.ctx.k('L4'))
    for c in ('L3', 'L2', 'L1', 'L0'):
        e.mul(p, p, u2)
        e.op(k.ADD_FADD, p, p, e.ctx.k(c))
    e.mul(p, p, u)
    e.op(k.ADD_FADD, p, p, p)
    e.op(k.ADD_FMAX, ax, IN, e.ctx.k('ZERO'))
    e.op(k.ADD_FADD, out, ax, p)
    e.ctx.rel(ax, a, u, den, u2, p)


def b_softplus(e, IN, OUT):
    core_softplus(e, IN, OUT)


def b_mish(e, IN, OUT):
    """mish(x) = x * tanh(softplus(x))."""
    k = e.k
    sp = e.ctx.tb()
    core_softplus(e, IN, sp)
    core_tanh(e, sp, OUT, e.ctx.tb(), e.ctx.tb(), e.ctx.tb())
    e.mul(OUT, OUT, IN)
    e.ctx.rel(sp)


def b_softsign(e, IN, OUT):
    k = e.k
    a = e.ctx.tb()
    d = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FADD, d, a, e.ctx.k('ONE'))
    core_rcp(e, d, a, e.ctx.tb())
    e.mul(OUT, IN, a)
    e.ctx.rel(a, d)


def b_gelu(e, IN, OUT):
    """ONNX Gelu, tanh approximation: 0.5*x*(1+tanh(0.79788456*(x+0.044715x^3))).

    The tanh core is a truncating 1 - 2t/(1+t) form, so it saturates at
    1-2^-24 rather than 1.  At |u| > 44 the true tanh is exactly +-1 in fp32,
    and multiplying that 2^-24 residue by a large x would be a real error
    (x=-1e10 produced -298), so the saturated half is selected explicitly.
    """
    k = e.k
    x3 = e.ctx.tb()
    u = e.ctx.tb()
    m = e.ctx.tb()
    t = e.ctx.tb()
    e.mul(x3, IN, IN)
    e.mul(x3, x3, IN)
    e.mul(x3, x3, e.ctx.k('GELU_C'))
    e.op(k.ADD_FADD, x3, x3, IN)
    e.mul(u, x3, e.ctx.k('GELU_S'))
    core_tanh(e, u, OUT, e.ctx.tb(), e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FADD, OUT, OUT, e.ctx.k('ONE'))
    e.mul(OUT, OUT, IN)
    e.mul(OUT, OUT, e.ctx.k('HALF'))
    # u > 44 -> x ; u < -44 -> 0
    e.op(k.ADD_FADD, t, u, e.ctx.k('T44'))
    e.op(k.ADD_XOR, t, t, e.ctx.k('SIGNB'))
    gt_mask(e, m, t)
    keep_sel(e, OUT, m, e.ctx.k('ZERO'))
    e.op(k.ADD_FSUB, t, u, e.ctx.k('T44'))
    gt_mask(e, m, t)
    keep_sel(e, OUT, m, IN)
    e.ctx.rel(x3, u, m, t)


def b_gelu_erf(e, IN, OUT):
    k = e.k
    u = e.ctx.tb()
    e.mul(u, IN, e.ctx.k('INV_SQRT2'))
    core_erf(e, u, OUT, e.ctx.tb(), e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FADD, OUT, OUT, e.ctx.k('ONE'))
    e.mul(OUT, OUT, IN)
    e.mul(OUT, OUT, e.ctx.k('HALF'))
    e.ctx.rel(u)


def b_hardsigmoid(e, IN, OUT):
    k = e.k
    e.mul(OUT, IN, e.ctx.k('ALPHA'))
    e.op(k.ADD_FADD, OUT, OUT, e.ctx.k('BETA'))
    e.op(k.ADD_FMAX, OUT, OUT, e.ctx.k('ZERO'))
    e.op(k.ADD_FMIN, OUT, OUT, e.ctx.k('ONE'))


def b_affine(e, IN, OUT):
    """out = a * p0 + p1 (two scalar uniforms; the gradient-broadcast scale)."""
    k = e.k
    e.mul(OUT, IN, e.ctx.k('P0'))
    e.op(k.ADD_FADD, OUT, OUT, e.ctx.k('P1'))


def b_hardswish(e, IN, OUT):
    k = e.k
    t = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('ALPHA'))
    e.op(k.ADD_FADD, t, t, e.ctx.k('BETA'))
    e.op(k.ADD_FMAX, t, t, e.ctx.k('ZERO'))
    e.op(k.ADD_FMIN, t, t, e.ctx.k('ONE'))
    e.mul(OUT, IN, t)
    e.ctx.rel(t)


def _sincos(e, IN):
    """Range reduction and both polynomials: (sin r, cos r, q&1, q>>1)."""
    k = e.k
    t = e.ctx.tb()
    q = e.ctx.tb()
    r = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('TWO_OVER_PI'))
    e.op(k.ADD_FADD, q, t, e.ctx.k('HALF'))
    e.op(k.ADD_FADD, q, q, e.ctx.k('MAGIC'))
    e.op(k.ADD_FSUB, t, q, e.ctx.k('MAGIC'))
    e.op(k.ADD_FTOI, q, q, q)
    q2 = e.ctx.tb()
    e.op(k.ADD_AND, q2, q, e.ctx.k('THREE'))
    e.mul(r, t, e.ctx.k('PIO2_HI'))
    e.op(k.ADD_FSUB, r, IN, r)
    h = e.ctx.tb()
    e.mul(h, t, e.ctx.k('PIO2_MID'))
    e.op(k.ADD_FSUB, r, r, h)
    e.mul(h, t, e.ctx.k('PIO2_LO'))
    e.op(k.ADD_FSUB, r, r, h)
    r2 = e.ctx.tb()
    e.mul(r2, r, r)
    sp = e.ctx.tb()
    cp = e.ctx.tb()
    e.op(k.ADD_OR, sp, e.ctx.k('S4'), e.ctx.k('S4'))
    for c in ('S3', 'S2', 'S1'):
        e.mul(sp, sp, r2)
        e.op(k.ADD_FADD, sp, sp, e.ctx.k(c))
    e.mul(sp, sp, r2)
    e.op(k.ADD_FADD, sp, sp, e.ctx.k('ONE'))
    e.mul(sp, sp, r)
    e.op(k.ADD_OR, cp, e.ctx.k('C4P'), e.ctx.k('C4P'))
    for c in ('C3P', 'C2P', 'C1P'):
        e.mul(cp, cp, r2)
        e.op(k.ADD_FADD, cp, cp, e.ctx.k(c))
    e.mul(cp, cp, r2)
    e.op(k.ADD_FADD, cp, cp, e.ctx.k('ONE'))
    b0 = e.ctx.tb()
    b1 = e.ctx.tb()
    e.opi(k.ADD_AND, b0, q2, 1)
    e.opi(k.ADD_SHR, b1, q2, 1)
    e.ctx.rel(t, q, r, h, r2)
    return sp, cp, b0, b1


def b_sin(e, IN, OUT):
    """sin with quadrant q = round(x*2/pi): q&1 picks cos, q>>1 the sign."""
    k = e.k
    sp, cp, b0, b1 = _sincos(e, IN)
    m = e.ctx.tb()
    t = e.ctx.tb()
    e.op(k.ADD_SUB, m, e.ctx.k('ZERO'), b0)   # -1 iff b0 (q odd)
    sel(e, OUT, m, cp, sp, t)           # odd -> cos(r), even -> sin(r)
    e.opi(k.ADD_SHL, b1, b1, 31)
    e.op(k.ADD_XOR, OUT, OUT, b1)
    e.ctx.rel(sp, cp, b0, b1, m, t)


def b_cos(e, IN, OUT):
    """cos with quadrant q = round(x*2/pi): q&1 picks -sin, q>>1 the sign."""
    k = e.k
    sp, cp, b0, b1 = _sincos(e, IN)
    m = e.ctx.tb()
    t = e.ctx.tb()
    e.op(k.ADD_XOR, sp, sp, e.ctx.k('SIGNB'))
    e.op(k.ADD_SUB, m, e.ctx.k('ZERO'), b0)   # -1 iff b0 (q odd)
    sel(e, OUT, m, sp, cp, t)           # odd -> -sin(r), even -> cos(r)
    e.opi(k.ADD_SHL, b1, b1, 31)
    e.op(k.ADD_XOR, OUT, OUT, b1)
    e.ctx.rel(sp, cp, b0, b1, m, t)


# ------------------------------------------- inverse / hyperbolic (2026-11-20)
def core_asin(e, IN, OUT):
    """asin(IN) for |IN| <= 1; the executor refuses anything else.

    Two branches, one polynomial.  With a = |x|,
        a <= 0.5:  asin(a) = a + a^3 Q(a^2)                (Q on [0, 0.25])
        a >  0.5:  s = sqrt((1-a)/2)  (in [0, 0.5]; 1-a is exact there)
                   asin(a) = pi/2 - 2*(s + s^3 Q(s^2))
    The second branch is fdlibm's: it never subtracts two numbers near 1, so
    asin(1-2^-24) keeps its absolute accuracy (the sqrt carries the 2^-12
    structure of the answer).  The -2p is formed as 2p and one fsub.
    """
    k = e.k
    a = e.ctx.tb()
    m = e.ctx.tb()
    u = e.ctx.tb()
    s = e.ctx.tb()
    q = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, m, a, e.ctx.k('HALF'))
    gt_mask(e, m, m)
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), a)     # exact for a in [0.5, 1]
    e.mul(u, u, e.ctx.k('HALF'))
    e.op(k.ADD_FMAX, u, u, e.ctx.k('TINY'))    # keeps 1/sqrt(0) out of the body
    core_rsqrt(e, u, s, q)
    e.mul(s, s, u)                             # s = sqrt((1-a)/2)
    sel(e, u, m, s, a, q)                      # z = m ? s : a, both in [0, .5]
    e.mul(q, u, u)                             # t = z^2
    e.op(k.ADD_OR, s, e.ctx.k('AS4'), e.ctx.k('AS4'))
    for c in ('AS3', 'AS2', 'AS1', 'AS0'):
        e.mul(s, s, q)
        e.op(k.ADD_FADD, s, s, e.ctx.k(c))
    e.mul(s, s, q)
    e.mul(s, s, u)
    e.op(k.ADD_FADD, s, s, u)                  # asin(z)
    e.op(k.ADD_FADD, q, s, s)                  # 2 asin(z)
    e.op(k.ADD_FSUB, q, e.ctx.k('PI2'), q)     # pi/2 - 2 asin(z)
    keep_sel(e, s, m, q)
    e.op(k.ADD_AND, q, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, OUT, s, q)


def b_asin(e, IN, OUT):
    core_asin(e, IN, OUT)


def b_acos(e, IN, OUT):
    """acos = pi/2 - asin.

    At x -> 1 the subtraction cancels and costs one ulp of pi/2 (1.2e-7
    absolute against a 1e-4 absolute tolerance); measured, not assumed
    (docs/OPS-UNARY.md).
    """
    core_asin(e, IN, OUT)
    e.op(e.k.ADD_FSUB, OUT, e.ctx.k('PI2'), OUT)


def b_atan(e, IN, OUT):
    """atan: r = min(|x|, 1/|x|) in [0,1], then pi/2 - atan(r) for |x| > 1.

    The reciprocal is the verified Newton core (the same one log/softsign
    use); |x| is clamped to BIG first so a non-finite input cannot make it
    NaN.
    """
    k = e.k
    a = e.ctx.tb()
    m = e.ctx.tb()
    r = e.ctx.tb()
    p = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FSUB, m, a, e.ctx.k('ONE'))
    gt_mask(e, m, m)
    e.op(k.ADD_FMIN, p, a, e.ctx.k('BIG'))
    core_rcp(e, p, r, e.ctx.tb())              # r = 1/|x| (finite)
    sel(e, p, m, r, a, e.ctx.tb())             # arg = m ? 1/a : a
    e.mul(r, p, p)                             # t = arg^2
    e.op(k.ADD_OR, OUT, e.ctx.k('AT6'), e.ctx.k('AT6'))
    for c in ('AT5', 'AT4', 'AT3', 'AT2', 'AT1', 'AT0'):
        e.mul(OUT, OUT, r)
        e.op(k.ADD_FADD, OUT, OUT, e.ctx.k(c))
    e.mul(OUT, OUT, r)
    e.mul(OUT, OUT, p)
    e.op(k.ADD_FADD, OUT, OUT, p)              # atan(arg)
    e.op(k.ADD_FSUB, r, e.ctx.k('PI2'), OUT)   # pi/2 - atan(1/a)
    keep_sel(e, OUT, m, r)
    e.op(k.ADD_AND, r, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, OUT, OUT, r)


def b_sinh(e, IN, OUT):
    """sinh = sign(x)*(t - 1/t)/2 with t = e^|x|.

    |x| <= 88.6: above that e^|x| overflows and 1/t flushes, so the answer
    would be a finite lie instead of +-inf - the executor refuses it.
    """
    k = e.k
    a = e.ctx.tb()
    t = e.ctx.tb()
    q = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.mul(t, a, e.ctx.k('LOG2E'))
    core_exp2(e, t, t, e.ctx.tb(), e.ctx.tb())
    # the reciprocal's bit-trick seed is only valid below 2^126; e^|x| reaches
    # 3e38 at the top of the domain, where 1/t is 1e-38 and its exact value is
    # irrelevant to the sum, so clamp the input instead of the algorithm.
    e.op(k.ADD_FMIN, q, t, e.ctx.k('R125'))
    core_rcp(e, q, q, e.ctx.tb())
    e.op(k.ADD_FSUB, t, t, q)
    e.mul(t, t, e.ctx.k('HALF'))
    # x = +-0: (t - 1/t)/2 leaves the reciprocal core's 3e-8 residue, so force
    # the exact signed zero the FPU defines
    gt_mask(e, q, a)
    e.op(k.ADD_AND, t, t, q)
    e.op(k.ADD_AND, q, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, OUT, t, q)


def b_cosh(e, IN, OUT):
    """cosh = (t + 1/t)/2 with t = e^|x|; same |x| <= 88.6 limit."""
    k = e.k
    a = e.ctx.tb()
    t = e.ctx.tb()
    q = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.mul(t, a, e.ctx.k('LOG2E'))
    core_exp2(e, t, t, e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FMIN, q, t, e.ctx.k('R125'))    # see b_sinh
    core_rcp(e, q, q, e.ctx.tb())
    e.op(k.ADD_FADD, t, t, q)
    e.mul(OUT, t, e.ctx.k('HALF'))
    # cosh(0) is exactly 1; the reciprocal core's residue would make it
    # 0.99999994
    eq0_mask(e, q, a, e.ctx.tb())
    keep_sel(e, OUT, q, e.ctx.k('ONE'))


def b_tan(e, IN, OUT):
    """tan from the same Cody-Waite reduction as sin/cos.

        q = round(x*2/pi), r = x - q*pi/2   (|r| <= pi/4)
        tan(x) = tan(r)       q even
        tan(x) = -1/tan(r)    q odd

    tan(r) is one degree-11 odd polynomial, so the pole accuracy is the
    *relative* accuracy of the reduced r, not the difference of two O(1)
    values: near a pole r is small and tan(r) = r*(1+...) keeps its relative
    error.  What the kernel cannot fix is the reduction's absolute error in r
    (~|x|*2^-24 of pi/2), which becomes a relative error when |r| is small;
    the executor therefore refuses inputs with |cos(x)| below the measured
    threshold in docs/OPS-UNARY.md.
    """
    k = e.k
    t = e.ctx.tb()
    q = e.ctx.tb()
    r = e.ctx.tb()
    h = e.ctx.tb()
    r2 = e.ctx.tb()
    p = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('TWO_OVER_PI'))
    e.op(k.ADD_FADD, q, t, e.ctx.k('HALF'))
    e.op(k.ADD_FADD, q, q, e.ctx.k('MAGIC'))
    e.op(k.ADD_FSUB, t, q, e.ctx.k('MAGIC'))
    e.op(k.ADD_FTOI, q, q, q)
    e.mul(r, t, e.ctx.k('PIO2_HI'))
    e.op(k.ADD_FSUB, r, IN, r)
    e.mul(h, t, e.ctx.k('PIO2_MID'))
    e.op(k.ADD_FSUB, r, r, h)
    e.mul(h, t, e.ctx.k('PIO2_LO'))
    e.op(k.ADD_FSUB, r, r, h)
    e.mul(r2, r, r)
    e.op(k.ADD_OR, p, e.ctx.k('TN5'), e.ctx.k('TN5'))
    for c in ('TN4', 'TN3', 'TN2', 'TN1', 'TN0'):
        e.mul(p, p, r2)
        e.op(k.ADD_FADD, p, p, e.ctx.k(c))
    e.mul(p, p, r2)
    e.mul(p, p, r)
    e.op(k.ADD_FADD, p, p, r)                  # tan(r)
    e.op(k.ADD_AND, h, p, e.ctx.k('ABSM'))
    e.op(k.ADD_FMAX, h, h, e.ctx.k('TINY'))    # 0 -> 1/2^-126, never inf
    core_rcp(e, h, h, r2)
    e.op(k.ADD_AND, r2, p, e.ctx.k('SIGNB'))
    e.op(k.ADD_XOR, r2, r2, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, h, h, r2)                   # -1/tan(r)
    e.opi(k.ADD_SHL, r2, q, 31)
    e.opi(k.ADD_ASR, r2, r2, 31)               # all-ones iff the quadrant is odd
    sel(e, OUT, r2, h, p, t)


def core_asinh_acosh(e, IN, OUT, acosh):
    """log(a + sqrt(a^2 +- 1)) through the verified log core.

    The forms that keep the small-argument accuracy: 1-a is exact, x^2-1 is
    factored as (x-1)(x+1) (so nothing cancels), and |a| >= 2^63 switches to
    log|a| + ln2 because a^2 would overflow.  The log core's (m-1)/(m+1)
    series then has an absolute error of about one ulp of 1, which is what
    both functions need near their branch point.
    """
    k = e.k
    a = e.ctx.tb()
    big = e.ctx.tb()
    t = e.ctx.tb()
    s = e.ctx.tb()
    u = e.ctx.tb()
    if acosh:
        e.op(k.ADD_OR, a, IN, IN)
    else:
        e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_SUB, big, a, e.ctx.k('T63'))
    gt_mask(e, big, big)
    if acosh:
        e.op(k.ADD_FSUB, t, a, e.ctx.k('ONE'))     # exact in [1,2]
        e.op(k.ADD_FADD, u, a, e.ctx.k('ONE'))
        e.mul(t, t, u)                             # x^2-1, factored
    else:
        e.mul(t, a, a)
        e.op(k.ADD_FADD, t, t, e.ctx.k('ONE'))
    e.op(k.ADD_FMAX, t, t, e.ctx.k('TINY'))
    core_rsqrt(e, t, s, u)
    e.mul(s, s, t)                                 # sqrt(a^2 +- 1)
    e.op(k.ADD_FADD, u, a, s)
    keep_sel(e, u, big, a)                         # the log argument
    # The log core is the widest point of the body (six live temps, and this
    # body's constant list leaves only eight): free everything else and
    # rebuild |x| from the input afterwards.
    e.ctx.rel(t, s, a)
    core_ln(e, u, OUT, 4)
    e.ctx.rel(u)
    a = e.ctx.tb()
    t = e.ctx.tb()
    s = e.ctx.tb()
    if not acosh:
        e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    else:
        e.op(k.ADD_OR, a, IN, IN)
    e.op(k.ADD_AND, t, big, e.ctx.k('LN2'))
    e.op(k.ADD_FADD, OUT, OUT, t)                  # + ln2 on the big half
    # non-finite: |a| >= inf -> +-inf, a nan stays nan.  keep_sel reads its
    # a_true, so the constant registers are used directly: cloning one into a
    # temp would need a register the pool no longer has (and `u` is dead).
    e.op(k.ADD_SUB, t, a, e.ctx.k('POSINF'))
    e.op(k.ADD_ADD, t, t, e.ctx.k('ONE_I'))
    gt_mask(e, s, t)
    keep_sel(e, OUT, s, e.ctx.k('POSINF'))
    e.op(k.ADD_SUB, t, a, e.ctx.k('POSINF'))
    gt_mask(e, s, t)
    keep_sel(e, OUT, s, e.ctx.k('NAN'))
    if not acosh:
        # x = +-0: log(0+sqrt(1)) is 0 only to the reciprocal core's residue,
        # so force the exact signed zero
        gt_mask(e, s, a)
        e.op(k.ADD_AND, OUT, OUT, s)
        e.op(k.ADD_AND, t, IN, e.ctx.k('SIGNB'))
        e.op(k.ADD_OR, OUT, OUT, t)


def b_asinh(e, IN, OUT):
    core_asinh_acosh(e, IN, OUT, False)


def b_acosh(e, IN, OUT):
    core_asinh_acosh(e, IN, OUT, True)


def b_atanh(e, IN, OUT):
    """atanh = ln((1+a)/(1-a))/2, sign restored; |x| < 1 (refused outside).

    1-a is exact for a in [0.5,1], and the log core's (m-1) form keeps the
    error near a = 0 at the ulp of 1 (6e-8), well inside the 1e-4 absolute
    tolerance.
    """
    k = e.k
    a = e.ctx.tb()
    t = e.ctx.tb()
    e.op(k.ADD_AND, a, IN, e.ctx.k('ABSM'))
    e.op(k.ADD_FADD, t, a, e.ctx.k('ONE'))
    e.op(k.ADD_FSUB, a, e.ctx.k('ONE'), a)      # 1-|x| (exact in [0.5,1])
    e.op(k.ADD_FMAX, a, a, e.ctx.k('TINY'))
    core_rcp(e, a, a, e.ctx.tb())
    e.mul(t, t, a)                              # (1+a)/(1-a)
    core_ln(e, t, OUT, 4)
    e.mul(OUT, OUT, e.ctx.k('HALF'))
    # x = +-0: atanh(0) is 0, not the reciprocal core's residue (`a` has been
    # reused for 1/(1-|x|) by now, so test |x| from the input)
    e.op(k.ADD_AND, t, IN, e.ctx.k('ABSM'))
    gt_mask(e, t, t)
    e.op(k.ADD_AND, OUT, OUT, t)
    e.op(k.ADD_AND, t, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_OR, OUT, OUT, t)


# ---------------------------------------------------------------- constants
STD = {
    'ABSM': (0x7FFFFFFF, 0), 'SIGNB': (0x80000000, 0),
    'ONE': (0x3F800000, 0), 'TWO': (0x40000000, 0),
    'HALF': (0x3F000000, 0), 'MAGIC': (0x4B400000, 0),
    'RCPM': (0x7EF311C3, 0), 'ONE_I': (1, 1), 'NEG1': (M32, 0),
    'T23': (_fb(8388608.0), 0),   # every float this large is integral
    'TINY': (0x00800000, 0), 'TINYI': (0x00800000, 1),
    'C127': (127, 1), 'MANT': (0x007FFFFF, 0),
    'ZERO': (0, 0), 'THREE': (3, 1),
    'EXP_LO': (_fb(-126.0), 0), 'EXP_HI': (_fb(128.0), 0),
    'E5': (0x3AAEC3FF, 0), 'E4': (0x3C1D955B, 0),
    'E3': (0x3D635847, 0), 'E2': (0x3E75FDF0, 0), 'E1': (0x3F317218, 0),
    'NEGONE': (0xBF800000, 0),
    'LOG2E': (_fb(1.4426950408889634), 0),
    'NLOG2E': (_fb(-1.4426950408889634), 0),
    'NLOG2E2': (_fb(-2.8853900817779268), 0),
    'T44': (_fb(44.0), 0), 'THREEH': (_fb(1.5), 0),
    'RSQRTC': (0x5F3759DF, 0), 'BIG': (_fb(3.0e38), 0),
    'SQRT2': (_fb(1.4142135623730951), 0),
    'LN2': (_fb(0.6931471805599453), 0),
    'L4': (_fb(1.0 / 9), 0), 'L3': (_fb(1.0 / 7), 0),
    'L2': (_fb(1.0 / 5), 0), 'L1': (_fb(1.0 / 3), 0),
    'L0': (_fb(1.0), 0),
    'ERFC1': (_fb(0.3275911), 0), 'ERF1': (_fb(0.254829592), 0),
    'ERF2': (_fb(-0.284496736), 0), 'ERF3': (_fb(1.421413741), 0),
    'ERF4': (_fb(-1.453152027), 0), 'ERF5': (_fb(1.061405429), 0),
    'SIX': (_fb(6.0), 0), 'T4': (_fb(4.0), 0),
    'TWO_OVER_PI': (_fb(0.6366197723675814), 0),
    'PIO2_HI': (0x3FC90000, 0),      # pi/2 split: 1.5703125
    'PIO2_MID': (0x39FDAA22, 0),     # 4.8382679e-4
    'PIO2_LO': (0x2C346000, 0),      # 2.5632829e-12
    'S1': (_fb(-1.0 / 6), 0), 'S2': (_fb(1.0 / 120), 0),
    'S3': (_fb(-1.0 / 5040), 0), 'S4': (_fb(1.0 / 362880), 0),
    'C1P': (_fb(-0.5), 0), 'C2P': (_fb(1.0 / 24), 0),
    'C3P': (_fb(-1.0 / 720), 0), 'C4P': (_fb(1.0 / 40320), 0),
    'GELU_C': (_fb(0.044715), 0), 'GELU_S': (_fb(0.7978845608028654), 0),
    'INV_SQRT2': (_fb(0.7071067811865476), 0),
    'NONE': (_fb(-1.0), 0), 'NHALF': (_fb(-0.5), 0),
    'TWO_OVER_SQRTPI': (_fb(2.0 / 1.77245385090551602729), 0),
    'INV_SQRT2PI': (_fb(0.39894228040143267794), 0),
    'NEGINF': (0xFF800000, 0), 'POSINF': (0x7F800000, 0),
    # the counter-based RNG body: 16-bit halves of the two "lowbias32"
    # multipliers and of the two seed-mix multipliers, the low-half mask, and
    # the 2^-24 that turns a 24-bit integer into [0,1)
    'M16': (0x0000FFFF, 1),
    'K1L': (0x352D, 1), 'K1H': (0x7FEB, 1),
    'K2L': (0xA68B, 1), 'K2H': (0x846C, 1),
    'G1L': (0x79B1, 1), 'G1H': (0x9E37, 1),
    'G2L': (0xCA77, 1), 'G2H': (0x85EB, 1),
    'RSCALE': (0x33800000, 0),
    'NAN': (0x7FC00000, 0), 'INFI': (0x7F800000, 1),
    # inverse / hyperbolic (2026-11-20): minimax coefficients, fitted in
    # float64 and rounded to fp32 (docs/OPS-UNARY.md has the fit errors)
    'PI2': (0x3FC90FDB, 0),                       # pi/2
    'R125': (0x7E800000, 0),      # 2^125: the Newton reciprocal's seed range
    'T63': (0x5F000000, 0),                       # 2^63, an integer compare
    'AT0': (0xBEAAAA98, 0), 'AT1': (0x3E4CBDC1, 0), 'AT2': (0xBE114220, 0),
    'AT3': (0x3DD5A3A5, 0), 'AT4': (0xBD8A713C, 0), 'AT5': (0x3CF9A2FB, 0),
    'AT6': (0xBBD696D4, 0),                       # atan, deg 13 odd
    'AS0': (0x3E2AAAAF, 0), 'AS1': (0x3D999346, 0), 'AS2': (0x3D386070, 0),
    'AS3': (0x3CD8E023, 0), 'AS4': (0x3D1CA8E0, 0),   # asin, deg 9 odd
    'TN0': (0x3EAAAAA7, 0), 'TN1': (0x3E088B7B, 0), 'TN2': (0x3D5C373A, 0),
    'TN3': (0x3CBE2686, 0), 'TN4': (0x3BA3FC1D, 0), 'TN5': (0x3C082E24, 0),
}                                                 # tan, deg 11 odd

E_CORE = ('LOG2E', 'E5', 'E4', 'E3', 'E2', 'E1', 'MAGIC', 'HALF', 'ONE',
          'EXP_LO', 'EXP_HI', 'ONE_I')
RCP_C = ('RCPM', 'TWO')
LOG_C = ('SQRT2', 'MANT', 'C127', 'ONE', 'LN2', 'TINY')
LOGP_C = ('L4', 'L3', 'L2', 'L1', 'L0')
SIN_C = ('TWO_OVER_PI', 'MAGIC', 'HALF', 'PIO2_HI', 'PIO2_MID', 'PIO2_LO', 'THREE',
         'S1', 'S2', 'S3', 'S4', 'C1P', 'C2P', 'C3P', 'C4P', 'ONE', 'SIGNB',
         'ONE_I', 'ZERO')
# the log core without the u^9 term: enough for fp32 (the dropped term is
# 1.5e-9 at the largest series argument).  asinh/acosh/atanh use this form
# because their constant list (21) plus a 5-term log does not fit the pool.
LOG_C4 = ('SQRT2', 'MANT', 'C127', 'ONE', 'LN2', 'TINY')
LOGP_C4 = ('L3', 'L2', 'L1', 'L0')
LOG4_C = LOG_C4 + LOGP_C4 + RCP_C
ASIN_C = ('HALF', 'ONE', 'TINY', 'ABSM', 'SIGNB', 'PI2',
          'RSQRTC', 'THREEH', 'AS0', 'AS1', 'AS2', 'AS3', 'AS4')
ATAN_C = ('BIG', 'ONE', 'ABSM', 'SIGNB', 'PI2') + RCP_C + \
    ('AT0', 'AT1', 'AT2', 'AT3', 'AT4', 'AT5', 'AT6')
SINH_C = ('LOG2E',) + E_CORE[1:-1] + RCP_C + ('ABSM', 'SIGNB', 'R125')
COSH_C = ('LOG2E',) + E_CORE[1:-1] + RCP_C + ('ABSM', 'R125')
TAN_C = ('TWO_OVER_PI', 'MAGIC', 'HALF', 'PIO2_HI', 'PIO2_MID', 'PIO2_LO',
         'ABSM', 'SIGNB', 'TINY') + RCP_C + \
    ('TN0', 'TN1', 'TN2', 'TN3', 'TN4', 'TN5')
ASINH_C = LOG4_C + ('RSQRTC', 'THREEH', 'HALF', 'ABSM', 'SIGNB', 'T63',
                    'POSINF', 'NAN', 'ONE_I')
ACOSH_C = LOG4_C + ('RSQRTC', 'THREEH', 'HALF', 'T63', 'POSINF', 'NAN',
                    'ONE_I', 'SIGNB')
ATANH_C = LOG4_C + ('HALF', 'ABSM', 'SIGNB', 'ONE_I')
# erf needs every E_CORE-2^f constant except ONE_I (integer one, used by
# log/round only): the 23-slot B budget after excluding B8 is tight.
E_CORE_ERF = tuple(c for c in E_CORE[1:] if c != 'ONE_I')
ERF_C = ('ABSM', 'SIX', 'ERFC1', 'ERF1', 'ERF2', 'ERF3', 'ERF4',
         'ERF5', 'NLOG2E', 'ONE', 'SIGNB') + E_CORE_ERF + RCP_C


# ================================================================ rng (uop)
# A counter-based device RNG.  ``docs/OPS-RNN.md`` sec. 5 has the record; the
# algorithm is documented here because this body *is* the definition.
#
#   s = mul32(FTOI(seed), 0x9E3779B1) + mul32(FTOI(call), 0x85EBCA77)
#   z = counter_bits ^ s                       (counter = the element index)
#   z ^= z >> 16;  z = mul32(z, 0x7FEB352D)
#   z ^= z >> 15;  z = mul32(z, 0x846CA68B)
#   z ^= z >> 16
#   u = ITOF(z >> 8) * 2^-24                   in [0, 1)
#
# The two multiplies are Pelle Evensen's "lowbias32" finaliser, which is a
# bijection on 32 bits with measured avalanche ~16 bits/bit; xoring the seed
# mixture into the counter first makes the value a function of (seed, call,
# element index) only -- there is no state to advance, so the same three
# numbers always give the same draw and a second launch of the same node is
# de-correlated by the call counter.  Not cryptographic: an observer who knows
# the seed can reproduce the stream, and the 3-round finaliser is not a
# cipher.  Statistical evidence (1e6 draws) is in docs/OPS-RNN.md sec. 5.
def _mul32(e, out, x, klo, khi):
    """out = x * (khi << 16 | klo) mod 2^32, x a 32-bit word.

    The QPU has only mul24 (24x24 -> low 32 bits), so a 32-bit multiply by a
    constant is three 16x16 products: the two low-half terms add in the low
    half, the cross terms shift into the high half (anything above 2^32
    vanishes), which is exact.
    """
    k = e.k
    C = e.ctx
    al, ah = C.ta(), C.ta()
    p0, p1 = C.ta(), C.ta()
    e.op(k.ADD_AND, al, x, C.k('M16'))
    e.opi(k.ADD_SHR, ah, x, 16)
    e.mul24(ACC0, al, C.k(klo))
    e.op(k.ADD_OR, p0, ACC0, ACC0)
    e.mul24(ACC1, ah, C.k(klo))
    e.op(k.ADD_OR, p1, ACC1, ACC1)
    e.mul24(ACC2, al, C.k(khi))
    e.op(k.ADD_ADD, p1, p1, ACC2)          # mul -> add is exact at 0 nops
    e.opi(k.ADD_SHL, p1, p1, 16)
    e.op(k.ADD_ADD, out, p0, p1)
    C.rel(al, ah, p0, p1)


def b_rng(e, IN, OUT):
    """out = uniform in [0,1) from the counter word IN.

    The two parameters are the 16-bit halves of a 32-bit stream key that the
    ARM derives once per launch from the operator's seed and its launch
    counter.  That is a *key schedule*, not a draw: it produces one integer
    per launch and no per-element randomness.  Every element value is the
    QPU hash below.
    """
    k = e.k
    C = e.ctx
    lo, hi = C.ta(), C.ta()
    t = C.ta()
    z = C.ta()
    e.op(k.ADD_FTOI, lo, C.k('KEYLO'), C.k('KEYLO'))
    e.op(k.ADD_FTOI, hi, C.k('KEYHI'), C.k('KEYHI'))
    e.opi(k.ADD_SHL, hi, hi, 16)
    e.op(k.ADD_OR, t, lo, hi)
    e.op(k.ADD_XOR, z, IN, t)
    C.rel(lo, hi)
    e.opi(k.ADD_SHR, t, z, 16)
    e.op(k.ADD_XOR, z, z, t)
    _mul32(e, z, z, 'K1L', 'K1H')
    e.opi(k.ADD_SHR, t, z, 15)
    e.op(k.ADD_XOR, z, z, t)
    _mul32(e, z, z, 'K2L', 'K2H')
    e.opi(k.ADD_SHR, t, z, 16)
    e.op(k.ADD_XOR, z, z, t)
    e.opi(k.ADD_SHR, z, z, 8)
    e.op(k.ADD_ITOF, OUT, z, z)
    e.mul(OUT, OUT, C.k('RSCALE'))
    C.rel(t, z)



UOPS = {
    'abs':        (('ABSM',), b_abs, ()),
    'neg':        (('SIGNB',), b_neg, ()),
    'identity':   ((), b_identity, ()),
    'clip':       ((), b_clip, ('P_LO', 'P_HI')),
    'leakyrelu':  (('ZERO',), b_leakyrelu, ('ALPHA',)),
    'floor':      (('ABSM', 'SIGNB', 'ZERO', 'T23'), b_floor, ()),
    'ceil':       (('ABSM', 'SIGNB', 'ZERO', 'T23'), b_ceil, ()),
    'round':      (('ONE_I', 'HALF', 'ABSM', 'T23'), b_round, ()),
    'sign':       (('ONE', 'NEGONE', 'SIGNB', 'ABSM'), b_sign, ()),
    'exp':        (E_CORE, b_exp, ()),
    'log':        (LOG_C + LOGP_C + RCP_C + ('HALF', 'ONE_I', 'ABSM',
                    'SIGNB', 'NEGINF', 'POSINF', 'NAN', 'INFI'), b_log, ()),
    'sqrt':       (('TINY', 'RSQRTC', 'HALF', 'THREEH', 'SIGNB', 'NAN',
                    'POSINF', 'INFI', 'ONE_I'), b_sqrt, ()),
    'reciprocal': (('TINY', 'BIG', 'ABSM', 'SIGNB', 'POSINF',
                    'TINYI', 'ONE_I') + RCP_C, b_reciprocal, ()),
    'tanh':       (('ABSM', 'T44', 'NLOG2E2', 'SIGNB', 'ONE') + E_CORE[1:]
                   + RCP_C, b_tanh, ()),
    'erf':        (ERF_C, b_erf, ()),
    'elu':        (('ZERO', 'ONE') + E_CORE, b_elu, ('ALPHA',)),
    'selu':       (('ZERO', 'ONE') + E_CORE, b_selu, ('ALPHA', 'GAMMA')),
    'softplus':   (('ABSM', 'NLOG2E', 'ONE', 'ZERO') + LOGP_C + RCP_C
                   + E_CORE[1:], b_softplus, ()),
    'softsign':   (('ABSM', 'ONE') + RCP_C, b_softsign, ()),
    'gelu':       (('GELU_C', 'GELU_S', 'ABSM', 'T44', 'NLOG2E2', 'SIGNB',
                    'ONE', 'HALF', 'ZERO') + E_CORE[1:] + RCP_C, b_gelu, ()),
    'gelu_erf':   (('INV_SQRT2', 'HALF', 'ONE') + ERF_C, b_gelu_erf, ()),
    'hardsigmoid': (('ZERO', 'ONE'), b_hardsigmoid, ('ALPHA', 'BETA')),
    'hardswish':  (('ZERO', 'ONE'), b_hardswish, ('ALPHA', 'BETA')),
    'mish':       (('ABSM', 'NLOG2E', 'ONE', 'ZERO') + LOGP_C + RCP_C
                   + ('T44', 'NLOG2E2', 'SIGNB') + E_CORE_ERF, b_mish, ()),
    'affine':     ((), b_affine, ('P0', 'P1')),
    # rng: nine constants plus the two parameters (seed, call counter)
    'rng':        (('M16', 'K1L', 'K1H', 'K2L', 'K2H', 'RSCALE'),
                   b_rng, ('KEYLO', 'KEYHI')),
    'sin':        (SIN_C, b_sin, ()),
    'cos':        (SIN_C, b_cos, ()),
    'acos':       (ASIN_C, b_acos, ()),
    'acosh':      (ACOSH_C, b_acosh, ()),
    'asin':       (ASIN_C, b_asin, ()),
    'asinh':      (ASINH_C, b_asinh, ()),
    'atan':       (ATAN_C, b_atan, ()),
    'atanh':      (ATANH_C, b_atanh, ()),
    'tan':        (TAN_C, b_tan, ()),
    'sinh':       (SINH_C, b_sinh, ()),
    'cosh':       (COSH_C, b_cosh, ()),
}

HEAVY = {'log', 'tanh', 'erf', 'elu', 'selu', 'softplus', 'softsign', 'gelu', 'mish',
         'gelu_erf', 'sin', 'cos', 'round', 'sqrt', 'reciprocal',
         'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'tan', 'sinh',
         'cosh'}


def _consts_for(k, names, extra=()):
    names = list(dict.fromkeys(list(names) + list(extra)))
    free = [n for n in range(0, 16)] + [n for n in range(25, 32)]
    out = {}
    for n in names:
        if n not in STD:
            raise KeyError("kops: no such constant %r" % (n,))
        if n == 'ONE' and 'E1' in names and STD['E1'][0] == 0x3F800000:
            continue
        if not free:
            raise RuntimeError("kops: %d constants do not fit the B file"
                               % len(names))
        out[n] = ('B', free.pop(0))
    if 'ONE' in names and 'ONE' not in out and 'E1' in out:
        out['ONE'] = out['E1']
    return out


# ---------------------------------------------------------------- prologue
def _prologue(k, b, nvec, vpmq16, consts):
    gb = nvec * 64
    shift = {4: 2, 8: 3, 16: 4}[nvec] + 6
    b.e(k.ldi(64, B_C64, ws=1))
    b.e(k.ldi(gb, B_CSTEP, ws=1))
    b.e(k.ldi(M32, B_NEG1, ws=1))
    b.e(k.add_(k.ADD_SHL, B_STEP, k.areg(A_NQ), k.imm(shift), ws=1))
    b.e(k.add_(k.ADD_ADD, A_T2, k.areg(A_NQ), k.breg(B_NEG1)))
    b.e(k.add_(k.ADD_SHL, B_GAP, k.areg(A_T2), k.imm(shift), ws=1))
    if vpmq16:
        b.e(k.add_(k.ADD_SHL, A_VPMQ, k.areg(A_QID), k.imm(2)))
        b.e(k.add_(k.ADD_SHL, B_VPMQ7, k.areg(A_VPMQ), k.imm(7), ws=1))
    else:
        b.e(k.add_(k.ADD_SHL, B_V4, k.areg(A_QID), k.imm(2), ws=1))
        b.e(k.add_(k.ADD_SHL, B_V9, k.areg(A_QID), k.imm(9), ws=1))
    b.e(k.add_(k.ADD_SHL, A_T1, k.areg(A_QID), k.imm(shift)))     # q*gb
    b.e(k.add_(k.ADD_SHL, A_T0, k.areg(k.R_ELEM_NUM), k.imm(2)))  # lane*4
    b.e(k.nop(), k.nop())


def _stream_base(k, b, bus_reg, ptr_reg):
    b.e(k.add_(k.ADD_OR, ACC3, k.areg(bus_reg), k.areg(bus_reg)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, ACC3, k.acc(3), k.areg(A_T1)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, ptr_reg, k.acc(3), k.areg(A_T0)))
    b.e(k.nop(), k.nop())


def _store(k, b, nvec, vpmq16):
    p = b.p
    b.check = False
    if vpmq16:
        k.emit_store16(p, 0)
    else:
        k._store4(b, B_V4, B_V9, A_CADR, 0)
        if nvec == 8:
            for i in range(4):
                b.e(k.add_or(i, k.areg(4 + i), k.areg(4 + i)))
            b.e(k.add_(k.ADD_ADD, A_CADR, k.areg(A_CADR), k.imm(0)))
    b.check = True


def _load(k, b, addr_reg, acc=0, tmu=0):
    w = k.W_TMU0_S if tmu == 0 else k.W_TMU1_S
    sig = k.SIG_LOAD_TMU0 if tmu == 0 else k.SIG_LOAD_TMU1
    b.e(k.add_or(w, k.areg(addr_reg), k.areg(addr_reg), ws=1))
    b.e(k.nop(), k.nop(), k.nop(), k.nop())
    b.e(k.nop(), k.nop(), k.nop(), k.nop())
    b.e(k.alu(sig=sig, cond_add=k.COND_NEVER, cond_mul=k.COND_NEVER))
    b.e(k.add_or(32 + acc, k.acc(4), k.acc(4)))


def _consts_load(k, b, consts):
    for name, (bank, reg) in consts.items():
        if bank != 'B':
            continue
        val = STD[name]
        b.e(k.ldi(val[0] & M32, reg, ws=1), k.nop())


def _head(k, nvec, vpmq16, consts):
    b = k.Builder()
    p = b.p
    b.e(k.add_or(36, k.imm(1), k.imm(1)))
    p._builder = b
    return b, p


def _tail(k, b, extra=0):
    pass


def _group_tail(k, b, nvec, gap_regs=()):
    p = b.p
    b.e(k.add_(k.ADD_ADD, A_OB2, k.areg(A_OB2), k.breg(B_STEP)))
    b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(B_GAP)))
    for r in gap_regs:
        b.e(k.add_(k.ADD_ADD, r, k.areg(r), k.breg(B_GAP)))
    b.e(k.add_(k.ADD_SUB, A_CNT, k.areg(A_CNT), k.imm(1), sf=1))
    p.branch_to('grp', cond=k.BR_ALL_Z_CLEAR)
    b.e(k.nop(), k.nop(), k.nop())
    b.e(k.end_program())
    b.e(k.nop(), k.nop(), k.nop())


# ---------------------------------------------------------------- builders
UOP_ORDER = ['abs', 'neg', 'identity', 'clip', 'leakyrelu', 'floor', 'ceil',
             'round', 'sign', 'exp', 'log', 'sqrt', 'reciprocal', 'tanh',
             'erf', 'elu', 'selu', 'softplus', 'softsign', 'gelu', 'gelu_erf',
             'hardsigmoid', 'hardswish', 'sin', 'cos', 'affine', 'mish',
             'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'tan', 'sinh',
             'cosh', 'rng']

# The unary ops ride on kgen.build_ew's *verified* harness (TMU priming, the
# 16-vector group, the VPM->DMA store, the counter branch).  Only the body is
# new, so a body bug can make a wrong number but cannot change control flow.
# Register budget inside build_ew: B0..B20 and B26 are free for constants
# (B21..B25, B27..B29 are harness state), A16..A18, A20, A24, A27, A30, A31 are
# dead once the group loop starts.
UOP_B = list(range(0, 21)) + [26]
UOP_A = [16, 17, 18, 20, 24, 27, 30, 31]


def _uop_info(code):
    op = UOP_ORDER[code]
    return UOPS[op]


def ew_consts(k, b, code):
    """Load this op's constants (called from build_ew's prologue)."""
    names, body, params = _uop_info(code)
    names = list(dict.fromkeys(names))     # HALF/ONE appear twice in ERF_C
    # B30/B31 carry the first two op parameters; an op with fewer parameters
    # can use the spare ones for constants (gelu_erf needs 23)
    # B8 is the harness's b operand (written with x before every body):
    # never a constant.  B30/B31 are the parameter slots.
    pool = [r for r in UOP_B if r != 8]
    pool += [r for r in (30, 31) if r not in (30, 31)[:len(params)]]
    consts = {}
    for n in names:
        if n == 'ONE' and 'E1' in names and STD['E1'][0] == 0x3F800000:
            continue
        if n not in STD:
            raise KeyError("kops: no such constant %r" % (n,))
        if not pool:
            raise RuntimeError("kops: too many constants for the ew harness")
        consts[n] = ('B', pool.pop(0))
    if 'ONE' in names and 'ONE' not in consts:
        consts['ONE'] = consts['E1']
    for i, name in enumerate(params):
        if i < 2:
            probe = (30 + i, 1)                     # B30 / B31
            consts[name] = ('B', 30 + i)
            b.read_uni(30 + i, ws=1)
        else:
            consts[name] = ('A', 24)
            b.read_uni(24)
    for n, (bank, reg) in consts.items():
        if bank != 'B' or n in params:
            continue
        val = STD[n]
        b.e(k.ldi(val[0] & M32, reg, ws=1), k.nop())
    k._kops_consts[code] = consts


def ew_body(k, b, code, j):
    names, body, params = _uop_info(code)
    consts = k._kops_consts[code]
    used_b = set(v[1] for v in consts.values() if v[0] == 'B')
    used_a = set(v[1] for v in consts.values() if v[0] == 'A')
    ctx = Ctx(consts)
    ctx.a_free = [r for r in UOP_A if r not in used_a]
    ctx.b_free = [r for r in UOP_B if r != 8 and r not in used_b]
    ctx.a0 = list(ctx.a_free)
    ctx.b0 = list(ctx.b_free)
    e = E(k, b, ctx)
    body(e, 26, j)          # input A26 (E_AV), result into A j


def build_uop(k, op, nvec=None):
    if op not in UOP_ORDER:
        raise ValueError("kops: unknown uop %r" % (op,))
    if not hasattr(k, '_kops_consts'):
        k._kops_consts = {}
    # --nvec is authoritative: vc4_gen.c's uop_nvec() and the group size the
    # host pads/launches with must agree with what was generated
    nv = nvec if nvec else (4 if any(op in names for names in [HEAVY]) else 16)
    return k.build_ew(100 + UOP_ORDER.index(op), nvec=nv)


# ------------------------------------------------------- binary / backward ops
# These ride on the same verified build_ew harness.  Operands: A26 = a (the
# forward input, or the left operand), B8 = b (the incoming gradient, or the
# right operand).  B8 is the harness's second TMU stream, so these ops must
# not use it for constants: UOP_B2 is the unary pool minus B8.
#
# Every derivative body is written against the *same* primitives the forward
# op uses (core_exp2 / core_rcp / core_tanh / core_erf), so the analytic
# gradient is the derivative of the kernel that actually ran, not of an ideal
# formula.
UOP_B2 = [r for r in UOP_B if r != 8]

SIG_C = ('NLOG2E',) + E_CORE[1:] + ('RCPM', 'TWO')
RCP_C2 = ('TINY', 'BIG', 'RCPM', 'TWO')
TANH_C2 = UOPS['tanh'][0]
ERF_C2 = UOPS['erf'][0]
EXP_C2 = UOPS['exp'][0]


def k_sigmoid(e, IN, OUT):
    """OUT = 1/(1+2^(-x*log2 e)) with the forward kernels' own primitives."""
    k = e.k
    t = e.ctx.tb()
    d = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('NLOG2E'))
    core_exp2(e, t, d, e.ctx.tb(), e.ctx.tb())
    e.op(k.ADD_FADD, d, d, e.ctx.k('ONE'))
    core_rcp(e, d, OUT, e.ctx.tb())


def k_recip(e, IN, OUT):
    k = e.k
    a = e.ctx.tb()
    e.op(k.ADD_FMAX, a, IN, e.ctx.k('TINY'))
    e.op(k.ADD_FMIN, a, a, e.ctx.k('BIG'))
    core_rcp(e, a, OUT, e.ctx.tb())


def k_gt(e, x):
    """-1 where bits(x) > 0 (i.e. x > +0), else 0.  Integer, exact."""
    m = e.ctx.tb()
    gt_mask(e, m, x)
    return m


def b2_sub(e, IN, G, OUT):
    e.op(e.k.ADD_FSUB, OUT, IN, G)


def _uni(body):
    """Adapt a UOPS body (e, IN, OUT) to the binary ewb signature."""
    def f(e, IN, G, OUT):
        _ = G
        body(e, IN, OUT)
    return f


def b2_max(e, IN, G, OUT):
    e.op(e.k.ADD_FMAX, OUT, IN, G)


def b2_min(e, IN, G, OUT):
    e.op(e.k.ADD_FMIN, OUT, IN, G)


def b2_div(e, IN, G, OUT):
    t = e.ctx.tb()
    k_recip(e, G, t)
    e.mul(OUT, IN, t)


def b2_eqmask(e, IN, G, OUT):
    """OUT = 1.0 iff the *bit patterns* of a and b are equal, else 0.0.

    Used for max-pool argmax routing: the forward max is a bit-selected
    operand, so bit equality is exactly "this element produced the maximum".
    """
    k = e.k
    t = e.ctx.tb()
    u = e.ctx.tb()
    e.op(k.ADD_SUB, t, IN, G)
    eq0_mask(e, u, t, e.ctx.tb())
    e.op(k.ADD_AND, OUT, u, e.ctx.k('ONE'))


def b2_sigmoid_bwd(e, IN, G, OUT):
    k = e.k
    s = e.ctx.tb()
    u = e.ctx.tb()
    k_sigmoid(e, IN, s)
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), s)
    e.mul(s, s, u)
    e.mul(OUT, G, s)


def b2_silu_bwd(e, IN, G, OUT):
    """d/dx [x*s] = s + x*s*(1-s) = s + y*(1-s)."""
    k = e.k
    s = e.ctx.tb()
    y = e.ctx.tb()
    u = e.ctx.tb()
    k_sigmoid(e, IN, s)
    e.mul(y, IN, s)
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), s)
    e.mul(u, u, y)
    e.op(k.ADD_FADD, s, s, u)
    e.mul(OUT, G, s)


def b2_tanh_bwd(e, IN, G, OUT):
    k = e.k
    t = e.ctx.tb()
    u = e.ctx.tb()
    _uni(UOPS['tanh'][1])(e, IN, None, t)
    e.mul(u, t, t)
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), u)
    e.mul(OUT, G, u)


def b2_exp_bwd(e, IN, G, OUT):
    y = e.ctx.tb()
    _uni(UOPS['exp'][1])(e, IN, None, y)
    e.mul(OUT, G, y)


def b2_log_bwd(e, IN, G, OUT):
    t = e.ctx.tb()
    k_recip(e, IN, t)
    e.mul(OUT, G, t)


def b2_sqrt_bwd(e, IN, G, OUT):
    """d/dx sqrt = 1/(2 sqrt x) = 0.5 * rsqrt(x)."""
    k = e.k
    r = e.ctx.tb()
    core_rsqrt(e, IN, r, e.ctx.tb())
    e.mul(r, r, e.ctx.k('HALF'))
    e.mul(OUT, G, r)


def b2_abs_bwd(e, IN, G, OUT):
    """d|x|/dx = sign(x): a bit copy of g with x's sign bit (exact)."""
    k = e.k
    t = e.ctx.tb()
    e.op(k.ADD_AND, t, IN, e.ctx.k('SIGNB'))
    e.op(k.ADD_XOR, OUT, G, t)


def b2_clip_bwd(e, IN, G, OUT):
    """g inside [lo, hi], 0 outside.  The endpoints count as inside (the
    subgradient ONNX's Clip uses)."""
    k = e.k
    t = e.ctx.tb()
    u = e.ctx.tb()
    e.op(k.ADD_FMAX, t, IN, e.ctx.k('P_LO'))
    e.op(k.ADD_FMIN, t, t, e.ctx.k('P_HI'))
    e.op(k.ADD_SUB, u, t, IN)                 # 0 iff the value was clipped
    eq0_mask(e, u, u, e.ctx.tb())
    e.op(k.ADD_AND, u, u, e.ctx.k('ONE'))
    e.mul(OUT, G, u)


def b2_leakyrelu_bwd(e, IN, G, OUT):
    k = e.k
    m = k_gt(e, IN)
    t = e.ctx.tb()
    sel(e, t, m, e.ctx.k('ONE'), e.ctx.k('ALPHA'), e.ctx.tb())
    e.mul(OUT, G, t)


def b2_hardsigmoid_bwd(e, IN, G, OUT):
    """d/dx clip(alpha*x+beta,0,1) = alpha inside, 0 in the saturated tails."""
    k = e.k
    t = e.ctx.tb()
    u = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('ALPHA'))
    e.op(k.ADD_FADD, t, t, e.ctx.k('BETA'))
    e.op(k.ADD_FMAX, t, t, e.ctx.k('ZERO'))
    e.op(k.ADD_FMIN, t, t, e.ctx.k('ONE'))
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), t)
    m = k_gt(e, t)
    m2 = k_gt(e, u)
    e.op(k.ADD_AND, m, m, m2)
    e.op(k.ADD_AND, m, m, e.ctx.k('ALPHA'))
    e.mul(OUT, G, m)


def b2_hardswish_bwd(e, IN, G, OUT):
    """d/dx [x*s] = s + x*s'(x), s = clip(alpha*x+beta,0,1)."""
    k = e.k
    s = e.ctx.tb()
    t = e.ctx.tb()
    u = e.ctx.tb()
    e.mul(t, IN, e.ctx.k('ALPHA'))
    e.op(k.ADD_FADD, t, t, e.ctx.k('BETA'))
    e.op(k.ADD_FMAX, s, t, e.ctx.k('ZERO'))
    e.op(k.ADD_FMIN, s, s, e.ctx.k('ONE'))
    e.op(k.ADD_FSUB, u, e.ctx.k('ONE'), s)
    m = k_gt(e, s)
    m2 = k_gt(e, u)
    e.op(k.ADD_AND, m, m, m2)
    e.op(k.ADD_AND, m, m, e.ctx.k('ALPHA'))
    e.mul(t, IN, m)
    e.op(k.ADD_FADD, s, s, t)
    e.mul(OUT, G, s)


def b2_gelu_bwd(e, IN, G, OUT):
    """tanh approximation: 0.5(1+t) + 0.5*x*(1-t^2)*GELU_S*(1+3*GELU_C*x^2)."""
    k = e.k
    x2 = e.ctx.tb()
    u = e.ctx.tb()
    t = e.ctx.tb()
    d = e.ctx.tb()
    e.mul(x2, IN, IN)
    e.mul(u, x2, e.ctx.k('GELU_C'))
    e.op(k.ADD_FADD, u, u, e.ctx.k('ONE'))
    e.mul(u, u, e.ctx.k('THREE'))             # 1 + 3c x^2
    e.mul(d, x2, IN)
    e.mul(d, d, e.ctx.k('GELU_C'))
    e.op(k.ADD_FADD, d, d, IN)                # x + c x^3
    e.mul(d, d, e.ctx.k('GELU_S'))
    b_tanh(e, d, t)
    e.mul(d, u, e.ctx.k('GELU_S'))            # dt/dx
    e.mul(x2, t, t)
    e.op(k.ADD_FSUB, x2, e.ctx.k('ONE'), x2)  # 1 - t^2
    e.mul(d, d, x2)                           # (1-t^2) * GELU_S*(1+3cx^2)
    e.mul(d, d, IN)
    e.op(k.ADD_FADD, d, d, t)
    e.op(k.ADD_FADD, d, d, e.ctx.k('ONE'))
    e.mul(d, d, e.ctx.k('HALF'))
    e.mul(OUT, G, d)


def b2_gelu_erf_bwd(e, IN, G, OUT):
    """0.5(1+erf(x/sqrt2)) + x*exp(-x^2/2)/sqrt(2 pi)."""
    k = e.k
    u = e.ctx.tb()
    t = e.ctx.tb()
    e.mul(u, IN, e.ctx.k('INV_SQRT2'))
    _uni(UOPS['erf'][1])(e, u, None, t)
    e.op(k.ADD_FADD, t, t, e.ctx.k('ONE'))
    e.mul(t, t, e.ctx.k('HALF'))              # 0.5(1+erf)
    e.mul(u, IN, IN)
    e.mul(u, u, e.ctx.k('NHALF'))             # -x^2/2
    _uni(UOPS['exp'][1])(e, u, None, u)
    e.mul(u, u, IN)
    e.mul(u, u, e.ctx.k('INV_SQRT2PI'))
    e.op(k.ADD_FADD, t, t, u)
    e.mul(OUT, G, t)


def b2_erf_bwd(e, IN, G, OUT):
    k = e.k
    u = e.ctx.tb()
    t = e.ctx.tb()
    e.mul(u, IN, IN)
    e.mul(u, u, e.ctx.k('NONE'))              # -x^2
    _uni(UOPS['exp'][1])(e, u, None, t)
    e.mul(t, t, e.ctx.k('TWO_OVER_SQRTPI'))
    e.mul(OUT, G, t)


def b2_rcp(e, IN, G, OUT):
    """1/x with 3 Newton steps: the 4-step UOPS body is 2146 instructions at
    16 vectors per group and the code slot holds 2048."""
    k = e.k
    a = e.ctx.tb()
    e.op(k.ADD_FMAX, a, IN, e.ctx.k('TINY'))
    e.op(k.ADD_FMIN, a, a, e.ctx.k('BIG'))
    e.op(k.ADD_SUB, OUT, e.ctx.k('RCPM'), a)
    for _ in range(3):
        q = e.ctx.tb()
        e.mul(q, OUT, a)
        e.op(k.ADD_FSUB, q, e.ctx.k('TWO'), q)
        e.mul(OUT, OUT, q)
        e.ctx.rel(q)
    e.ctx.rel(a)


EWB = {
    'sub':            ((), b2_sub, ()),
    'max':            ((), b2_max, ()),
    'min':            ((), b2_min, ()),
    'div':            (RCP_C2, b2_div, ()),
    'eqmask':         (('ONE',), b2_eqmask, ()),
    'sigmoid_bwd':    (SIG_C, b2_sigmoid_bwd, ()),
    'silu_bwd':       (SIG_C, b2_silu_bwd, ()),
    'tanh_bwd':       (TANH_C2, b2_tanh_bwd, ()),
    'exp_bwd':        (EXP_C2, b2_exp_bwd, ()),
    'log_bwd':        (RCP_C2, b2_log_bwd, ()),
    'sqrt_bwd':       (('RSQRTC', 'HALF', 'THREEH'), b2_sqrt_bwd, ()),
    'abs_bwd':        (('SIGNB',), b2_abs_bwd, ()),
    'clip_bwd':       (('ONE',), b2_clip_bwd, ('P_LO', 'P_HI')),
    'leakyrelu_bwd':  (('ONE',), b2_leakyrelu_bwd, ('ALPHA',)),
    'hardsigmoid_bwd': (('ZERO', 'ONE'), b2_hardsigmoid_bwd, ('ALPHA', 'BETA')),
    'hardswish_bwd':  (('ZERO', 'ONE'), b2_hardswish_bwd, ('ALPHA', 'BETA')),
    'erf_bwd':        (EXP_C2 + ('NONE', 'TWO_OVER_SQRTPI'), b2_erf_bwd, ()),
    # unary helpers on the same (verified) 16-vector store path; the body is
    # the one UOPS pinned, together with the constant list it was written for
    'exp':            (UOPS['exp'][0], _uni(UOPS['exp'][1]), ()),
    'log':            (UOPS['log'][0], _uni(UOPS['log'][1]), ()),
    'sqrt':           (UOPS['sqrt'][0], _uni(UOPS['sqrt'][1]), ()),
    'rcp':            (('TINY', 'BIG', 'RCPM', 'TWO'), b2_rcp, ()),
    'tanh':           (UOPS['tanh'][0], _uni(UOPS['tanh'][1]), ()),
}

EWB_ORDER = list(EWB.keys())

# Bodies that do not fit the 16 KiB code slot at 16 vectors per group.
# The 4-vector store (_store4 via build_ew's nvec!=16 arm) does not write its
# result on this board and the blit kernel built on it wedges the QPU, so no
# op here may select it: every body below is generated with 16 vectors per
# group and must fit the 16 KiB code slot.  b_gelu (the tanh-form GELU
# derivative) does not, so that one is composed in C from the ops that do.
EWB_HEAVY = set()


def ewb_consts(k, b, code):
    names, body, params = EWB[EWB_ORDER[code]]
    pool = list(UOP_B2)
    apool = list(UOP_A)
    consts = {}
    for n in dict.fromkeys(names):
        if n not in STD:
            raise KeyError("kops: no such constant %r" % (n,))
        if pool:
            consts[n] = ('B', pool.pop(0))
        elif apool:
            # The B file is exhausted: a constant can live in the A file
            # (ws=0 selects it).  E._fit stages the A+A bank conflict.
            consts[n] = ('A', apool.pop(0))
        else:
            raise RuntimeError("kops: too many constants for the ew harness")
    for i, name in enumerate(params):
        consts[name] = ('B', 30 + i)
        b.read_uni(30 + i, ws=1)
    for n, (bank, reg) in consts.items():
        if n in params:
            continue
        val = STD[n]
        b.e(k.ldi(val[0] & M32, reg, ws=1 if bank == 'B' else 0), k.nop())
    k._kops_consts[code] = consts


def ewb_body(k, b, code, j):
    names, body, params = EWB[EWB_ORDER[code]]
    consts = k._kops_consts[code]
    used_b = set(v[1] for v in consts.values() if v[0] == 'B')
    used_a = set(v[1] for v in consts.values() if v[0] == 'A')
    ctx = Ctx(consts)
    ctx.a_free = [r for r in UOP_A if r not in used_a]
    ctx.b_free = [r for r in UOP_B2 if r not in used_b]
    ctx.a0 = list(ctx.a_free)
    ctx.b0 = list(ctx.b_free)
    e = E(k, b, ctx)
    body(e, 26, 8, j)          # a = A26 (E_AV), b/g = B8 (B_VB)


def build_ewb(k, op):
    if op not in EWB:
        raise ValueError("kops: unknown binary/backward ew op %r" % (op,))
    if not hasattr(k, '_kops_consts'):
        k._kops_consts = {}
    nv = 4 if op in EWB_HEAVY else 16
    return k.build_ew(200 + EWB_ORDER.index(op), nvec=nv)


def build_sel3(k, op):
    if op not in ('where', 'clip3', 'prelu'):
        raise ValueError("kops: unknown sel3 op %r" % (op,))
    consts = _consts_for(k, ('ABSM',))
    ctx = Ctx(consts)
    nvec = 16
    b, p = _head(k, nvec, True, consts)
    b.read_uni(A_AB)          # a
    b.read_uni(A_OB)          # out
    b.read_uni(A_CNT)
    b.read_uni(A_QID)
    b.read_uni(A_NQ)
    b.read_uni(A_P0)          # b
    b.read_uni(A_P1)          # c
    b.read_uni(A_P2)          # op (compile-time; kept for ABI stability)
    _consts_load(k, b, consts)
    _prologue(k, b, nvec, True, consts)
    _stream_base(k, b, A_AB, A_IB)
    _stream_base(k, b, A_OB, A_OB2)
    A_BP, A_CP = A_P0, A_P1
    for reg, bus in ((A_BP, A_BP), (A_CP, A_CP)):
        b.e(k.add_(k.ADD_OR, ACC1, k.areg(bus), k.areg(bus)))
        b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_ADD, ACC1, k.acc(1), k.areg(A_T1)))
        b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_ADD, ACC1, k.acc(1), k.areg(A_T0)))
        b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_OR, reg, k.acc(1), k.acc(1)))
        b.e(k.nop(), k.nop())
    p.label('grp')
    b.e(k.add_or(A_CADR, k.areg(A_OB2), k.areg(A_OB2)))
    for j in range(nvec):
        _load(k, b, A_IB, 0)
        _load(k, b, A_BP, 1, tmu=1)
        _load(k, b, A_CP, 2, tmu=0)
        ctx.reset()
        e = E(k, b, ctx)
        if op == 'where':
            m = ctx.tb()
            e.op(k.ADD_AND, m, ACC0, ctx.k('ABSM'))
            gt_mask(e, m, m)
            t = ctx.tb()
            sel(e, ACC1, m, ACC1, ACC2, t)
            ctx.rel(m, t)
        elif op == 'clip3':
            e.op(k.ADD_FMAX, ACC1, ACC1, ACC0)
            e.op(k.ADD_FMIN, ACC1, ACC1, ACC2)
        else:
            m = ctx.tb()
            t = ctx.tb()
            u = ctx.tb()
            gt_mask(e, m, ACC0)
            e.mul(t, ACC0, ACC1)
            sel(e, ACC1, m, ACC0, t, u)
            ctx.rel(m, t, u)
        b.e(k.add_or(j, k.acc(1), k.acc(1)))
        b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(B_C64)))
        b.e(k.add_(k.ADD_ADD, A_BP, k.areg(A_BP), k.breg(B_C64)))
        b.e(k.add_(k.ADD_ADD, A_CP, k.areg(A_CP), k.breg(B_C64)))
    _store(k, b, nvec, True)
    _group_tail(k, b, nvec, gap_regs=(A_BP, A_CP))
    return p


def build_gather(k):
    consts = _consts_for(k, ('ABSM',))
    ctx = Ctx(consts)
    nvec = 16
    b, p = _head(k, nvec, True, consts)
    b.read_uni(A_AB)          # table
    b.read_uni(A_OB)
    b.read_uni(A_CNT)
    b.read_uni(A_QID)
    b.read_uni(A_NQ)
    b.read_uni(A_P0)          # arena bus base
    b.read_uni(A_P1)          # fill
    b.read_uni(A_P2)          # unused
    _consts_load(k, b, consts)
    _prologue(k, b, nvec, True, consts)
    _stream_base(k, b, A_AB, A_IB)
    _stream_base(k, b, A_OB, A_OB2)
    p.label('grp')
    b.e(k.add_or(A_CADR, k.areg(A_OB2), k.areg(A_OB2)))
    for j in range(nvec):
        _load(k, b, A_IB, 0)                    # acc0 = table vector
        e = E(k, b, ctx)
        m = ctx.ta()
        tg = ctx.ta()
        e.mov(tg, ACC0)
        e.opi(k.ADD_ASR, m, tg, 31)             # -1 iff negative
        e.op(k.ADD_NOT, ACC0, m, m)
        e.op(k.ADD_AND, ACC0, tg, ACC0)         # 0 for negative entries
        e.op(k.ADD_ADD, ACC0, ACC0, A_P0)       # absolute bus address
        # The TMU setup word takes its address from an *A-file* register, not
        # from an accumulator (`areg(100)` encodes a different register, and a
        # TMU read of that value wedges the QPU).  Stage the per-lane address
        # in A30, which is dead once the group loop starts.  The builder's
        # no-forwarding check does not see accumulator reads (the `acc(n)`
        # descriptor carries NOP_ADDR, not the accumulator index), so the gap
        # has to be explicit: two words between the write of r0 and the read.
        b.e(k.nop(), k.nop())
        b.e(k.add_or(30, k.acc(0), k.acc(0)))
        b.e(k.nop(), k.nop())
        _load(k, b, 30, 0, tmu=1)               # per-lane gather
        e.op(k.ADD_AND, m, m, A_P1)             # fill where negative
        e.op(k.ADD_NOT, ACC1, m, m)
        e.op(k.ADD_AND, ACC1, ACC1, ACC0)
        e.op(k.ADD_OR, ACC1, ACC1, m)
        ctx.rel(m, tg)
        b.e(k.add_or(j, k.acc(1), k.acc(1)))
        b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(B_C64)))
    _store(k, b, nvec, True)
    _group_tail(k, b, nvec)
    return p


RED_OPS = ('sum', 'mean', 'max', 'min', 'argmax', 'argmin')




HARNESS_B = {17, 18, 19, 20, 21, 22, 23, 24}


def _free_bs(consts, n):
    used = set(v[1] for v in consts.values() if v[0] == 'B') | HARNESS_B
    out = [r for r in list(range(0, 16)) + list(range(25, 32)) if r not in used]
    if len(out) < n:
        raise RuntimeError("kops: need %d more B registers" % n)
    return out[:n]


HARNESS_B = {17, 18, 19, 20, 21, 22, 23, 24}


def _free_bs(consts, n):
    used = set(v[1] for v in consts.values() if v[0] == 'B') | HARNESS_B
    out = [r for r in list(range(0, 16)) + list(range(25, 32)) if r not in used]
    if len(out) < n:
        raise RuntimeError("kops: need %d more B registers" % n)
    return out[:n]


def build_reduce(k, op):
    """out[o] = reduce_r a[o*red + r]; one vector = 16 consecutive o.

    The host passes every byte stride (the q offset cannot be derived in the
    kernel: q*256*red*4 would overflow 24-bit address arithmetic)."""
    if op not in RED_OPS:
        raise ValueError("kops: unknown reduce op %r" % (op,))
    consts = _consts_for(k, ('NEGINF', 'POSINF', 'ZERO'))
    b_red, b_inv, b_ogap, b_ostep, b_ls, b_ingap = _free_bs(consts, 6)
    b_vstr = _free_bs(consts, 7)[6]
    ctx = Ctx(consts)
    ctx.no_a()
    nvec = 16
    b, p = _head(k, nvec, True, consts)
    b.read_uni(A_AB)
    b.read_uni(A_OB)
    b.read_uni(A_CNT)
    b.read_uni(A_QID)
    b.read_uni(A_NQ)
    b.read_uni(b_red, ws=1)
    b.read_uni(b_inv, ws=1)
    b.read_uni(b_vstr, ws=1)
    b.read_uni(A_INBASE)
    b.read_uni(A_OUTBASE)
    b.read_uni(b_ingap, ws=1)
    b.read_uni(b_ogap, ws=1)
    b.read_uni(b_ostep, ws=1)
    b.read_uni(b_ls, ws=1)
    _consts_load(k, b, consts)
    b.e(k.ldi(256, B_CSTEP, ws=1))
    b.e(k.add_(k.ADD_SHL, A_VPMQ, k.areg(A_QID), k.imm(2)))
    b.e(k.add_(k.ADD_SHL, B_VPMQ7, k.areg(A_QID), k.imm(7), ws=1))
    # lane * lane_stride  and  lane * 4
    b.e(k.add_(k.ADD_OR, ACC0, k.areg(k.R_ELEM_NUM), k.areg(k.R_ELEM_NUM)))
    b.e(k.nop(), k.nop())
    b.e(dict(mul_op=k.MUL_MUL24, mul_out=32 + 1, mul_a=k.acc(0),
             mul_b=k.breg(b_ls), ws=0, sig=k.SIG_NONE))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_OR, ACC3, k.areg(A_AB), k.areg(A_AB)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, ACC3, k.acc(3), k.areg(A_INBASE)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, A_IB, k.acc(3), k.acc(1)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_SHL, ACC2, k.acc(0), k.imm(2)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_OR, ACC3, k.areg(A_OB), k.areg(A_OB)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, ACC3, k.acc(3), k.areg(A_OUTBASE)))
    b.e(k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, A_OB2, k.acc(3), k.acc(2)))
    b.e(k.nop(), k.nop())
    p.label('grp')
    b.e(k.add_or(A_CADR, k.areg(A_OB2), k.areg(A_OB2)))
    for j in range(nvec):
        b.e(k.add_(k.ADD_OR, A_RA, k.areg(A_IB), k.areg(A_IB)))
        b.e(k.add_(k.ADD_OR, A_NR, k.breg(b_red), k.breg(b_red)))
        b.e(k.ldi(0, A_RIDX, ws=0))
        b.e(k.ldi(0, A_BESTI, ws=0))
        if op in ('sum', 'mean'):
            b.e(k.add_or(ACC0, k.breg(consts['ZERO'][1]), k.breg(consts['ZERO'][1])))
        elif op == 'min':
            b.e(k.add_or(ACC0, k.breg(consts['POSINF'][1]), k.breg(consts['POSINF'][1])))
        else:
            b.e(k.add_or(ACC0, k.breg(consts['NEGINF'][1]), k.breg(consts['NEGINF'][1])))
        b.e(k.nop(), k.nop())
        p.label('r%d' % j)
        _load(k, b, A_RA, 1)
        ctx.reset()
        e = E(k, b, ctx)
        if op in ('sum', 'mean'):
            e.op(k.ADD_FADD, ACC0, ACC0, ACC1)
        elif op == 'max':
            e.op(k.ADD_FMAX, ACC0, ACC0, ACC1)
        elif op == 'min':
            e.op(k.ADD_FMIN, ACC0, ACC0, ACC1)
        else:
            t = ctx.tb()
            u = ctx.tb()
            if op == 'argmax':
                e.op(k.ADD_FSUB, t, ACC1, ACC0)
            else:
                e.op(k.ADD_FSUB, t, ACC0, ACC1)
            gt_mask(e, t, t)
            sel(e, ACC0, t, ACC1, ACC0, u)
            sel(e, A_BESTI, t, A_RIDX, A_BESTI, u)
            ctx.rel(t, u)
        b.e(k.add_(k.ADD_ADD, A_RA, k.areg(A_RA), k.imm(4)))
        b.e(k.add_(k.ADD_ADD, A_RIDX, k.areg(A_RIDX), k.imm(1)))
        b.e(k.add_(k.ADD_SUB, A_NR, k.areg(A_NR), k.imm(1), sf=1))
        p.branch_to('r%d' % j, cond=k.BR_ALL_Z_CLEAR)
        b.e(k.nop(), k.nop(), k.nop())
        if op == 'mean':
            e.mul(ACC0, ACC0, 200 + b_inv)
        if op in ('argmax', 'argmin'):
            b.e(k.add_or(j, k.areg(A_BESTI), k.areg(A_BESTI)))
        else:
            b.e(k.add_or(j, k.acc(0), k.acc(0)))
        b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(b_vstr)))
    _store(k, b, nvec, True)
    b.e(k.add_(k.ADD_ADD, A_OB2, k.areg(A_OB2), k.breg(b_ostep)))
    b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(b_ingap)))
    b.e(k.add_(k.ADD_SUB, A_CNT, k.areg(A_CNT), k.imm(1), sf=1))
    p.branch_to('grp', cond=k.BR_ALL_Z_CLEAR)
    b.e(k.nop(), k.nop(), k.nop())
    b.e(k.end_program())
    b.e(k.nop(), k.nop(), k.nop())
    return p


def build_cumsum(k):
    """Inclusive prefix sum over the last axis.  `cols` must be a multiple of
    256 (the C layer zero-pads and slices); rows are independent."""
    consts = _consts_for(k, ('ONE_I', 'ZERO'))
    b_m1, b_m2, b_m4, b_m8, b_t, b_carry = _free_bs(consts, 6)
    ctx = Ctx(consts)
    nvec = 16
    b, p = _head(k, nvec, True, consts)
    b.read_uni(A_AB)
    b.read_uni(A_OB)
    b.read_uni(A_CNT)
    b.read_uni(A_QID)
    b.read_uni(A_NQ)
    b.read_uni(A_P0)
    b.read_uni(A_P1)
    b.read_uni(A_P2)
    _consts_load(k, b, consts)
    _prologue(k, b, nvec, True, consts)
    _stream_base(k, b, A_AB, A_IB)
    _stream_base(k, b, A_OB, A_OB2)
    # mask_s = all ones iff lane >= s
    for s, reg in ((1, b_m1), (2, b_m2), (4, b_m4), (8, b_m8)):
        b.e(k.add_(k.ADD_SUB, ACC0, k.areg(k.R_ELEM_NUM), k.imm(s)))
        b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_ASR, ACC0, k.acc(0), k.imm(31)))
        b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_NOT, 200 + reg, k.acc(0), k.acc(0), ws=1))
        b.e(k.nop(), k.nop())
    p.label('grp')
    b.e(k.add_or(A_CADR, k.areg(A_OB2), k.areg(A_OB2)))
    b.e(k.add_(k.ADD_OR, 200 + b_carry, k.breg(consts['ZERO'][1]),
               k.breg(consts['ZERO'][1]), ws=1))
    b.e(k.nop(), k.nop())
    for j in range(nvec):
        _load(k, b, A_IB, 0)
        b.e(k.add_(k.ADD_OR, ACC1, k.acc(0), k.acc(0)))
        b.e(k.nop(), k.nop())
        for s, reg in ((1, b_m1), (2, b_m2), (4, b_m4), (8, b_m8)):
            b.e(k.mul_rot(2, 1, s))
            b.e(k.nop(), k.nop())
            b.e(k.add_(k.ADD_AND, 200 + b_t, k.acc(2), k.breg(reg), ws=1))
            b.e(k.nop(), k.nop())
            b.e(k.add_(k.ADD_ADD, ACC1, k.acc(1), k.breg(b_t)))
            b.e(k.nop(), k.nop())
        b.e(k.add_(k.ADD_FADD, ACC1, k.acc(1), k.breg(b_carry)))
        b.e(k.nop(), k.nop())
        b.e(k.mul_rot(2, 1, 1))                 # lane0 <- lane15
        b.e(k.nop(), k.nop())
        b.e(k.add_or(37, k.acc(2), k.acc(2)))   # replicate lane 0
        b.e(k.add_(k.ADD_OR, 200 + b_carry, k.acc(5), k.acc(5), ws=1))
        b.e(k.nop(), k.nop())
        b.e(k.add_or(j, k.acc(1), k.acc(1)))
        b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(B_C64)))
    _store(k, b, nvec, True)
    _group_tail(k, b, nvec)
    return p


# ------------------------------------------------------------------ copy2d
# The layout workhorse: a 3-level affine copy.  One vector is 16 *consecutive
# destination words* (the verified VPM->DMA store), and its 16 source words
# are read one per lane from src + lane*slane, so slane=4 is a contiguous
# copy and any other stride is a gather (channel plane <-> pixel-major,
# NHWC <-> NCHW, channel slice/split/concat, transpose).
#
#   row   (vpr vectors, each a 16-word vector, `units` vectors per store)
#     block (nblk blocks, `bstep` bytes apart in both streams)
#
# The block level exists because a channel concat of 32..256 channels is
# 2..16 consecutive 16-channel blocks per pixel: inside a row the block step
# is 64 B, while the step to the next row is a different constant.  QPU q
# takes rows q, q+nqpu, ...  Lanes >= nlane (and every lane when nlane==0)
# store the fill value instead of the source word, which gives the -inf
# MaxPool border, the zero channel padding of the first conv, and the short
# (< 16 words) column runs of a pitched NHWC->NCHW reshape.
#
# ABI (16 words per QPU)
#   u0  src bus (this QPU's first row)     u1  dst bus
#   u2  rows this QPU handles              u3  src row advance (nqpu*srow)
#   u4  dst row advance (nqpu*drow)        u5  vectors per row
#   u6  src vector advance                 u7  dst vector advance
#   u8  src lane advance                   u9  4*qid
#   u10 (4*qid)<<7                         u11 lanes to copy (0..16, 0 = none)
#   u12 blocks per row                     u13 src block advance (bytes)
#   u14 dst block advance (bytes)          u15 DMA row stride (dvec - 64)
# The fill value is a *compile-time* option (--fill <float bits>), not a
# uniform: any float whose bits are >= 0xC0000000 (e.g. -inf, 0xFF800000) is
# refused by vc4d's uniform policy because it looks like a GPU bus address.
A_SRC, A_DST, A_ROWS = 16, 17, 18
A_SB, A_DB, A_S, A_D, A_VCNT, A_BCNT = 19, 20, 21, 22, 23, 24
A_TA, A_CADR = 26, 27
A_MASK, A_NMASK, A_FILLN, A_STRCO = 28, 29, 30, 31
B_SVEC, B_DVEC, B_SLANE, B_SSTEP, B_DSTEP = 0, 1, 2, 3, 4
B_C4D, B_STRIDE, B_VPMQ, B_VPMQ7, B_FILL, B_C64, B_VPR = 5, 6, 7, 8, 9, 10, 11
B_NLANE, B_NBLK, B_SBSTEP, B_DBSTEP, B_LV = 12, 13, 14, 15, 16


def _ushift(units):
    assert units in (1, 2, 4)
    return {1: 0, 2: 1, 4: 2}[units]


def _store_var(b, k, cadr, vpmq, vpmq7, strco, stride, units=4, depth=16):
    """kgen._store4 with the row stride and the chunk step in registers.

    The VDW stride field is "memory step between the rows minus 64", so a row
    stride of dvec bytes needs dvec-64 in a B register; the constant
    0xC0010000 (VDW, block mode) lives in an A register because the ALU can
    read one A and one B file per word.  ``units`` is the number of 16-word
    vectors per store and ``depth`` the number of words per vector, so
    units=1, depth=k stores the k-word tail of a row that is not a multiple
    of 16 floats (the head's 25/100-wide extents).
    """
    assert units in (1, 2, 4) and 1 <= depth <= 16
    b.e(k.ldi(0x1A00, 37, ws=1))                 # ACC5 = vpw_generic(row 0)
    b.e(dict(add_op=k.ADD_ADD, add_out=k.W_VPW_SETUP, ws=1,
             add_a=k.acc(5), add_b=k.breg(vpmq)))
    for r in range(units):
        b.e(k.add_or(k.W_VPM, k.areg(r), k.areg(r)))
    b.e(k.ldi((2 << 30) | (units << 23) | (depth << 16) | (1 << 14), 37,
              ws=1))
    b.e(dict(add_op=k.ADD_ADD, add_out=k.W_VPW_SETUP, ws=1,
             add_a=k.acc(5), add_b=k.breg(vpmq7)))
    b.e(dict(add_op=k.ADD_ADD, add_out=k.W_VPW_SETUP, ws=1,
             add_a=k.areg(strco), add_b=k.breg(stride)))
    b.e(k.add_or(k.W_VPW_ADDR, k.areg(cadr), k.areg(cadr), ws=1))
    b.e(k.add_or(k.W_NOP, k.breg(k.R_VPW_WAIT), k.breg(k.R_VPW_WAIT)))
    b.e(k.nop(), k.nop(), k.nop(), k.nop())


def build_copy2d(k, fill=0, units=4, depth=16):
    b, p = _head(k, 16, True, {})
    b.read_uni(A_SRC)                    # u0
    b.read_uni(A_DST)                    # u1
    b.read_uni(A_ROWS)                   # u2
    b.read_uni(B_SSTEP, ws=1)            # u3
    b.read_uni(B_DSTEP, ws=1)            # u4
    b.read_uni(B_VPR, ws=1)              # u5
    b.read_uni(B_SVEC, ws=1)             # u6
    b.read_uni(B_DVEC, ws=1)             # u7
    b.read_uni(B_SLANE, ws=1)            # u8
    b.read_uni(B_VPMQ, ws=1)             # u9
    b.read_uni(B_VPMQ7, ws=1)            # u10
    b.read_uni(B_NLANE, ws=1)            # u11
    b.read_uni(B_NBLK, ws=1)             # u12
    b.read_uni(B_SBSTEP, ws=1)           # u13
    b.read_uni(B_DBSTEP, ws=1)           # u14
    b.read_uni(B_STRIDE, ws=1)           # u15
    # ACC2 = dvec, reused for the derived words (chunk step, store stride)
    b.e(k.ldi(fill & M32, B_FILL, ws=1))
    b.e(k.ldi(0xC0010000, A_STRCO))
    b.e(k.add_or(34, k.breg(B_DVEC), k.breg(B_DVEC)))
    b.e(k.add_(k.ADD_SHL, B_C4D, k.acc(2), k.imm(_ushift(units)), ws=1))
    # lane mask: mask = -1 for lane < nlane, 0 otherwise
    b.e(k.add_(k.ADD_SUB, A_TA, k.areg(k.R_ELEM_NUM), k.breg(B_NLANE)))
    b.e(k.add_(k.ADD_ASR, A_MASK, k.areg(A_TA), k.imm(31)))
    b.e(k.add_(k.ADD_NOT, A_NMASK, k.areg(A_MASK), k.areg(A_MASK)))
    b.e(k.add_(k.ADD_AND, A_FILLN, k.breg(B_FILL), k.areg(A_NMASK)))
    # lane * slane, in the B file so a vector address is one A + one B read
    b.e(k.add_(k.ADD_OR, 33, k.areg(k.R_ELEM_NUM),
               k.areg(k.R_ELEM_NUM)))                          # ACC1 = lane
    b.e(dict(mul_op=k.MUL_MUL24, mul_out=32 + 2, mul_a=k.acc(1),
             mul_b=k.breg(B_SLANE), ws=0, sig=k.SIG_NONE))
    b.e(k.add_or(B_LV, k.acc(2), k.acc(2), ws=1))
    p.label('row')
    b.e(k.add_or(A_SB, k.areg(A_SRC), k.areg(A_SRC)))
    b.e(k.add_or(A_DB, k.areg(A_DST), k.areg(A_DST)))
    b.e(k.add_or(A_BCNT, k.breg(B_NBLK), k.breg(B_NBLK)))
    p.label('blk')
    b.e(k.add_or(A_S, k.areg(A_SB), k.areg(A_SB)))
    b.e(k.add_or(A_D, k.areg(A_DB), k.areg(A_DB)))
    b.e(k.add_or(A_VCNT, k.breg(B_VPR), k.breg(B_VPR)))
    p.label('chunk')
    b.e(k.add_or(A_CADR, k.areg(A_D), k.areg(A_D)))
    for r in range(units):
        # lanes >= nlane read at the vector base instead of past the source
        b.e(k.add_(k.ADD_AND, 33, k.breg(B_LV), k.areg(A_MASK)))
        b.e(k.add_(k.ADD_ADD, A_TA, k.acc(1), k.areg(A_S)))
        b.e(k.add_or(k.W_TMU0_S, k.areg(A_TA), k.areg(A_TA), ws=1))
        b.e(k.nop(), k.nop(), k.nop(), k.nop())
        b.e(k.nop(), k.nop(), k.nop(), k.nop())
        b.e(k.alu(sig=k.SIG_LOAD_TMU0, cond_add=k.COND_NEVER,
                  cond_mul=k.COND_NEVER))
        b.e(k.add_or(32, k.acc(4), k.acc(4)))          # ACC0 = src word
        b.e(k.add_(k.ADD_AND, 33, k.acc(0), k.areg(A_MASK)))
        b.e(k.add_or(35, k.acc(1), k.areg(A_FILLN)))
        b.e(k.add_or(r, k.acc(3), k.acc(3)))           # store row r
        b.e(k.add_(k.ADD_ADD, A_S, k.areg(A_S), k.breg(B_SVEC)))
    _store_var(b, k, A_CADR, B_VPMQ, B_VPMQ7, A_STRCO, B_STRIDE,
               units, depth)
    b.e(k.add_(k.ADD_ADD, A_D, k.areg(A_D), k.breg(B_C4D)))
    b.e(k.add_(k.ADD_SUB, A_VCNT, k.areg(A_VCNT), k.imm(units), sf=1))
    p.branch_to('chunk', cond=k.BR_ALL_Z_CLEAR)
    b.e(k.nop(), k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, A_SB, k.areg(A_SB), k.breg(B_SBSTEP)))
    b.e(k.add_(k.ADD_ADD, A_DB, k.areg(A_DB), k.breg(B_DBSTEP)))
    b.e(k.add_(k.ADD_SUB, A_BCNT, k.areg(A_BCNT), k.imm(1), sf=1))
    p.branch_to('blk', cond=k.BR_ALL_Z_CLEAR)
    b.e(k.nop(), k.nop(), k.nop())
    b.e(k.add_(k.ADD_ADD, A_SRC, k.areg(A_SRC), k.breg(B_SSTEP)))
    b.e(k.add_(k.ADD_ADD, A_DST, k.areg(A_DST), k.breg(B_DSTEP)))
    b.e(k.add_(k.ADD_SUB, A_ROWS, k.areg(A_ROWS), k.imm(1), sf=1))
    p.branch_to('row', cond=k.BR_ALL_Z_CLEAR)
    b.e(k.nop(), k.nop(), k.nop())
    b.e(k.end_program())
    b.e(k.nop(), k.nop(), k.nop())
    return p


def build(k, kind, opts):
    if kind == 'copy2d':
        return build_copy2d(k, int(opts.get('fill', 0)),
                            int(opts.get('units', 4)),
                            int(opts.get('depth', 16)))
    if kind == 'ewb':
        return build_ewb(k, opts.get('op'))
    if kind == 'uop':
        nv = opts.get('nvec')
        return build_uop(k, opts.get('op'), int(nv) if nv else None)
    if kind == 'sel3':
        return build_sel3(k, opts.get('op'))
    if kind == 'gather':
        return build_gather(k)
    if kind == 'reduce':
        return build_reduce(k, opts.get('op'))
    if kind == 'cumsum':
        return build_cumsum(k)
    if kind == 'blit':
        return build_copy(k)
    raise ValueError("kops: unknown kind %r" % (kind,))


def build_copy(k):  # kind "blit"
    """Contiguous QPU copy: 64 B per group (one 16-float vector).

    The 4-row store writes the same 64 B four times (stride 0), which is
    idempotent, so a single vector per group is enough.  The C layer requires
    a length that is a multiple of 64 B and uses the gather table for
    everything else (strided rows, permutations, broadcasts)."""
    nvec = 4
    consts = _consts_for(k, ())
    b, p = _head(k, nvec, False, consts)
    b.read_uni(A_AB)
    b.read_uni(A_OB)
    b.read_uni(A_CNT)
    b.read_uni(A_QID)
    b.read_uni(A_NQ)
    b.read_uni(A_P0)
    b.read_uni(A_P1)
    b.read_uni(A_P2)
    # A_BSTEP: group stride in bytes (64 unless the caller wants another unit)
    b.e(k.ldi(64, B_C64, ws=1))
    b.e(k.ldi(64, B_CSTEP, ws=1))
    b.e(k.ldi(M32, B_NEG1, ws=1))
    b.e(k.add_(k.ADD_SHL, B_V4, k.areg(A_QID), k.imm(2), ws=1))
    b.e(k.add_(k.ADD_SHL, B_V9, k.areg(A_QID), k.imm(9), ws=1))
    b.e(k.add_(k.ADD_SHL, A_T1, k.areg(A_QID), k.imm(8)))
    b.e(k.add_(k.ADD_SHL, A_T0, k.areg(k.R_ELEM_NUM), k.imm(2)))
    b.e(k.add_(k.ADD_SHL, B_STEP, k.areg(A_NQ), k.imm(8), ws=1))
    b.e(k.add_(k.ADD_ADD, A_T2, k.areg(A_NQ), k.breg(B_NEG1)))
    b.e(k.add_(k.ADD_SHL, B_GAP, k.areg(A_T2), k.imm(8), ws=1))
    _stream_base(k, b, A_AB, A_IB)
    _stream_base(k, b, A_OB, A_OB2)
    p.label('grp')
    b.e(k.add_or(A_CADR, k.areg(A_OB2), k.areg(A_OB2)))
    for j in range(4):
        _load(k, b, A_IB, 0)
        b.e(k.add_or(j, k.acc(0), k.acc(0)))
        b.e(k.add_(k.ADD_ADD, A_IB, k.areg(A_IB), k.breg(B_C64)))
    _store(k, b, 4, False)
    _group_tail(k, b, nvec)
    return p
