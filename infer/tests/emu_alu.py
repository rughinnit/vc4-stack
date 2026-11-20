#!/usr/bin/env python3
"""emu_alu - execute a kgen blob's ALU words on 16 lanes, bit-exactly.

The VC4 FPU rounds toward zero and flushes denormals on input and output;
integer ops wrap at 32 bits.  A product or sum of two fp32 values is exact in
float64, so each op is "compute in float64, truncate to fp32" - the rule the
board was measured to obey (docs/NUMERICS.md).  Registers hold raw uint32 bit
patterns.

    from emu_alu import Machine
    m = Machine(words)
    m.setf('A', 26, x_lanes)
    m.seti('B', 0, 0x7FFFFFFF)
    m.run(pc0, pc1)
    m.getf('A', 0)
"""
import struct

import numpy as np

ADD_NOP, ADD_FADD, ADD_FSUB, ADD_FMIN, ADD_FMAX = 0, 1, 2, 3, 4
ADD_FTOI, ADD_ITOF = 7, 8
ADD_ADD, ADD_SUB, ADD_SHR, ADD_ASR, ADD_ROR, ADD_SHL = 12, 13, 14, 15, 16, 17
ADD_MIN, ADD_MAX, ADD_AND, ADD_OR, ADD_XOR, ADD_NOT = 18, 19, 20, 21, 22, 23
MUL_NOP, MUL_FMUL = 0, 1
MUL_MUL24 = 2
SIG_BRANCH = 15
MUL_V8MIN = 4
SIG_ALU_IMMEDIATE = 13
SIG_LOAD_IMMEDIATE = 14
SIG_END_PROGRAM = 3

F32_MAX = 3.4028234663852886e38


def f32_bits(x):
    """float64 -> fp32 bits, round toward zero, denormals flushed to zero."""
    out = np.zeros(len(x), np.uint32)
    for i, v in enumerate(x):
        if v != v:
            out[i] = 0x7FC00000
            continue
        a = abs(v)
        if a == 0.0:
            out[i] = 0
            continue
        if a < 2.0 ** -126:
            out[i] = 0
            continue
        if a > F32_MAX:
            out[i] = 0x7F800000 if v > 0 else 0xFF800000
            continue
        b = struct.unpack("<I", struct.pack("<f", v))[0]
        g = struct.unpack("<f", struct.pack("<I", b))[0]
        if abs(g) > a:
            b = (b - 1) if v > 0 else (b + 1)
            g = struct.unpack("<f", struct.pack("<I", b))[0]
            if abs(g) > a:
                b = (b - 1) if v > 0 else (b + 1)
        out[i] = b
    return out


def to_f64(bits):
    """uint32 bits -> float64, flushing denormal inputs to zero."""
    f = bits.view(np.float32).astype(np.float64)
    f[np.abs(f) < 2.0 ** -126] = 0.0
    return f


def s32(u):
    u = int(u) & 0xFFFFFFFF
    return u - 0x100000000 if u & 0x80000000 else u


def sext6(v):
    v &= 0x3F
    return v - 0x40 if v & 0x20 else v


class Machine:
    def __init__(self, words):
        self.w = list(words)
        self.A = [np.zeros(16, np.uint32) for _ in range(64)]
        self.B = [np.zeros(16, np.uint32) for _ in range(64)]
        # special registers used by the harness prologue
        self.A[38] = np.arange(16, dtype=np.uint32)   # elem_num (A file)
        self.B[38] = np.zeros(16, np.uint32)          # qpu_num  (B file)
        self.acc = [np.zeros(16, np.uint32) for _ in range(6)]
        self.mem = {}                  # bus address -> uint32
        self.unif = [0] * 16           # uniform port words
        self.upos = 0
        self.tmu_q = [[], []]          # addresses latched at issue (TMU0/1)
        self.store = None              # (addr, rows) captured at VPW_ADDR
        self.z = np.zeros(16, bool)    # per-lane zero flag (sf=1)
        self.n = np.zeros(16, bool)    # per-lane sign flag (sf=1)
        self.writes = []               # (special, value) for debugging

    def setf(self, bank, reg, vals):
        a = np.asarray(vals, np.float64)
        if a.size == 1:
            a = np.full(16, a.item())
        assert a.size == 16
        getattr(self, bank)[reg] = f32_bits(a)

    def seti(self, bank, reg, bits):
        getattr(self, bank)[reg] = np.full(16, bits & 0xFFFFFFFF, np.uint32)

    def getf(self, bank, reg):
        return getattr(self, bank)[reg].view(np.float32).astype(np.float64)

    def geti(self, bank, reg):
        return getattr(self, bank)[reg]

    def _branch_taken(self, cond):
        """The QPU branch conditions, from the per-lane flags."""
        z = self.z
        if cond == 0:
            return bool(z.all())
        if cond == 1:
            return not bool(z.any())
        if cond == 2:
            return bool(z.any())
        if cond == 3:
            return not bool(z.all())
        if cond == 4:
            return bool(self.n.all())
        if cond == 5:
            return not bool(self.n.any())
        if cond == 6:
            return bool(self.n.any())
        if cond == 7:
            return not bool(self.n.all())
        return True                     # 15 = always; carry conds unused here

    def run(self, pc, stop, trace=None, max_steps=200000):
        pc = int(pc)
        steps = 0
        while pc < stop:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("emu: branch loop did not terminate")
            i = pc
            self.pc = i
            pc += 1
            w = self.w[i]
            sig = (w >> 60) & 0xF
            if sig == SIG_END_PROGRAM:
                break
            if sig == SIG_BRANCH:
                cond = (w >> 52) & 0xF
                if self._branch_taken(cond):
                    # byte offset from the word *after the 3 delay slots*
                    # (measured against Prog's label table).  The delay slots
                    # every generator here emits are nops, so they are skipped.
                    off = int(np.uint32(w & 0xFFFFFFFF).view(np.int32))
                    rel = (w >> 51) & 1
                    pc = (i + 4 + off // 8) if rel else (off // 8)
                continue
            if sig in (4, 5, 6, 12):
                continue
            if sig in (10, 11):
                # TMU pop: the address was latched when the issue word wrote
                # special register 48; 16 lanes read 4 B each from it
                q = self.tmu_q[0 if sig == 10 else 1]
                if q:
                    addr = q.pop(0)
                    for lane in range(16):
                        a = int(addr[lane])
                        self.acc[4][lane] = self.mem.get(a, self.acc[4][lane])
                continue
            if sig in (0, 2, 7, 8, 9, 15):
                continue
            add_op = (w >> 24) & 0x1F
            mul_op = (w >> 29) & 7
            ws = (w >> 44) & 1
            add_out = (w >> 38) & 0x3F
            mul_out = (w >> 32) & 0x3F
            ra = (w >> 18) & 0x3F
            rb = (w >> 12) & 0x3F
            am, bm = (w >> 9) & 7, (w >> 6) & 7
            ma, mb = (w >> 3) & 7, w & 7
            cond_add, cond_mul = (w >> 49) & 7, (w >> 46) & 7
            if sig == SIG_LOAD_IMMEDIATE:
                value = np.full(16, w & 0xFFFFFFFF, np.uint32)
                if add_out < 32:
                    (self.A if not ws else self.B)[add_out] = value
                elif add_out < 38:
                    self.acc[add_out - 32] = value
                continue
            if sig == SIG_ALU_IMMEDIATE:
                pass
            operands = []
            iv = np.full(16, rb & 0x3F, np.uint32)   # small immediate is unsigned 0..63 here
            for mux in (am, bm, ma, mb):
                if mux == 8:
                    operands.append(iv)
                elif mux < 6:
                    operands.append(self.acc[mux].copy())
                elif mux == 6:
                    if ra == 32:
                        operands.append(np.full(16, self.unif[self.upos] if self.upos < len(self.unif) else 0, np.uint32))
                    else:
                        operands.append(self.A[ra].copy())
                else:
                    # in an ALU-immediate word raddr_b holds the small
                    # immediate, not a B-file address (qenc rewrites the mux)
                    operands.append(iv if sig == SIG_ALU_IMMEDIATE
                                    else self.B[rb].copy())
            add_ok = cond_add in (1,)
            mul_ok = cond_mul in (1,)
            if add_op != ADD_NOP and add_ok:
                r = self._add(add_op, operands[0], operands[1])
                if (w >> 45) & 1:                    # sf: set the flags
                    self.z = r == 0
                    self.n = (r & 0x80000000) != 0
                lda = self.w[i] >> 38 & 0x3F
                if ws and lda == 56:
                    self.tmu_q[0].append(operands[0].copy())
                if ws and lda == 60:
                    self.tmu_q[1].append(operands[0].copy())
                if ws and lda == 50:
                    self.store = (operands[0].copy(), None)
                if add_out < 32:
                    (self.A if not ws else self.B)[add_out] = r
                elif ws and add_out == 37:
                    # the rep_all write port: r5 gets lane 0 in every lane
                    self.acc[5] = np.full(16, r[0], np.uint32)
                elif add_out < 38:
                    self.acc[add_out - 32] = r
            if mul_op != MUL_NOP and mul_ok:
                if mul_op == MUL_V8MIN and sig == SIG_ALU_IMMEDIATE and \
                        49 <= (rb & 0x3F) <= 63:
                    # the 16-lane rotate: out[L] = in[(L - n) & 15]
                    n = (rb & 0x3F) - 48
                    r = np.roll(operands[2], n)
                    self._last_mul = r.view(np.float32).astype(np.float64)
                elif mul_op == MUL_MUL24:
                    a24 = operands[2] & 0xFFFFFF
                    b24 = operands[3] & 0xFFFFFF
                    r = ((a24.astype(np.uint64) * b24.astype(np.uint64))
                         & 0xFFFFFFFF).astype(np.uint32)
                    self._last_mul = r.view(np.float32).astype(np.float64)
                else:
                    r = self._mul(operands[2], operands[3])
                if mul_out < 32:
                    (self.A if ws else self.B)[mul_out] = r
                elif mul_out < 38:
                    self.acc[mul_out - 32] = r
            if 6 in (am, bm, ma, mb) and ra == 32:
                self.upos += 1
            if trace is not None:
                trace.append((i, add_op, mul_op, add_out, mul_out,
                              [None if o is None else o.copy() for o in operands],
                              None if not add_ok or add_op == ADD_NOP else self._last_add,
                              None if not mul_ok or mul_op == MUL_NOP else self._last_mul))

    def _add(self, op, a, b):
        if op == ADD_FADD:
            r = f32_bits(to_f64(a) + to_f64(b))
            self._last_add = r.view(np.float32).astype(np.float64)
            return r
        if op == ADD_FSUB:
            return f32_bits(to_f64(a) - to_f64(b))
        if op == ADD_FMIN:
            return f32_bits(np.minimum(to_f64(a), to_f64(b)))
        if op == ADD_FMAX:
            return f32_bits(np.maximum(to_f64(a), to_f64(b)))
        if op == ADD_FTOI:
            return np.array([np.uint32(s32(int(np.trunc(v))) & 0xFFFFFFFF)
                             for v in to_f64(a)], np.uint32)
        if op == ADD_ITOF:
            return f32_bits(np.array([float(s32(v)) for v in a]))
        if op == ADD_ADD:
            return np.array([np.uint32((int(x) + int(y)) & 0xFFFFFFFF)
                             for x, y in zip(a, b)], np.uint32)
        if op == ADD_SUB:
            return np.array([np.uint32((int(x) - int(y)) & 0xFFFFFFFF)
                             for x, y in zip(a, b)], np.uint32)
        if op == ADD_SHR:
            return np.array([np.uint32((int(x) & 0xFFFFFFFF) >> (int(y) & 31))
                             for x, y in zip(a, b)], np.uint32)
        if op == ADD_ASR:
            return np.array([np.uint32((s32(x) >> (int(y) & 31)) & 0xFFFFFFFF)
                             for x, y in zip(a, b)], np.uint32)
        if op == ADD_SHL:
            return np.array([np.uint32((int(x) << (int(y) & 31)) & 0xFFFFFFFF)
                             for x, y in zip(a, b)], np.uint32)
        if op == ADD_MIN:
            return np.array([np.uint32(min(s32(x), s32(y))) for x, y in zip(a, b)],
                            np.uint32)
        if op == ADD_MAX:
            return np.array([np.uint32(max(s32(x), s32(y))) for x, y in zip(a, b)],
                            np.uint32)
        if op == ADD_AND:
            return a & b
        if op == ADD_OR:
            return a | b
        if op == ADD_XOR:
            return a ^ b
        if op == ADD_NOT:
            return ~a
        raise ValueError("emu: unhandled add op %d" % op)

    def _mul(self, a, b):
        r = f32_bits(to_f64(a) * to_f64(b))
        self._last_mul = r.view(np.float32).astype(np.float64)
        return r


def load(path):
    data = open(path, "rb").read()
    return [struct.unpack("<Q", data[i:i + 8])[0] for i in range(0, len(data), 8)]
