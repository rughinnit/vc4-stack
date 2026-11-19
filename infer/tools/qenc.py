#!/usr/bin/env python3
"""Encoder for the VideoCore IV QPU instruction set.

Field layout taken from VC4C src/asm/*.h, validated against `vc4c --hex` output.

Output register encoding:
  - general purpose registers: 0..31, the file is selected by ws (add_out) and
    by the inverted ws (mul_out)
  - accumulators r0..r5: 32..37
  - special (vpm etc.): 48..63
Unused raddr slots are encoded as 39 ("-").
"""

ADD_NOP = 0
ADD_FADD = 1
ADD_FSUB = 2
ADD_FMIN = 3
ADD_FMAX = 4
ADD_FMINABS = 5
ADD_FMAXABS = 6
ADD_FTOI = 7
ADD_ITOF = 8
ADD_ADD = 12
ADD_SUB = 13
ADD_SHR = 14
ADD_ASR = 15
ADD_ROR = 16
ADD_SHL = 17
ADD_MIN = 18
ADD_MAX = 19
ADD_AND = 20
ADD_OR = 21
ADD_XOR = 22
ADD_NOT = 23
ADD_CLZ = 24

MUL_NOP = 0
MUL_FMUL = 1
MUL_MUL24 = 2
MUL_V8MIN = 4          # also the opcode used with a rotator small immediate

ACC0, ACC1, ACC2, ACC3, ACC4, ACC5, REGA, REGB = range(8)
IMM = 8  # pseudo-mux: small immediate encoded in raddr_b with SIG_ALU_IMMEDIATE

COND_NEVER = 0
COND_ALWAYS = 1
COND_ZERO_SET = 2
COND_ZERO_CLEAR = 3
COND_NEG_SET = 4
COND_NEG_CLEAR = 5
COND_CARRY_SET = 6
COND_CARRY_CLEAR = 7

SIG_BREAK = 0
SIG_NONE = 1
SIG_SWITCH_THREAD = 2
SIG_END_PROGRAM = 3
SIG_WAIT_SCORE = 4
SIG_UNLOCK_SCORE = 5
SIG_THREAD_SWITCH_LAST = 6
SIG_LOAD_COVERAGE = 7
SIG_LOAD_COLOR = 8
SIG_LOAD_COLOR_END = 9
SIG_LOAD_TMU0 = 10
SIG_LOAD_TMU1 = 11
SIG_LOAD_ALPHA = 12
SIG_ALU_IMMEDIATE = 13
SIG_LOAD_IMMEDIATE = 14
SIG_BRANCH = 15

BR_ALL_Z_SET = 0
BR_ALL_Z_CLEAR = 1
BR_ANY_Z_SET = 2
BR_ANY_Z_CLEAR = 3
BR_ALL_N_SET = 4
BR_ALL_N_CLEAR = 5
BR_ANY_N_SET = 6
BR_ANY_N_CLEAR = 7
BR_ALL_C_SET = 8
BR_ALL_C_CLEAR = 9
BR_ANY_C_SET = 10
BR_ANY_C_CLEAR = 11
BR_ALWAYS = 15

# out registers
R0, R1, R2, R3, R4, R5 = 32, 33, 34, 35, 36, 37
W_NOP = 39
W_IRQ = 38
W_UNIF_ADDR = 40
W_VPM = 48
W_VPR_SETUP = 49
W_VPW_SETUP = 49
W_VPR_ADDR = 50
W_VPW_ADDR = 50
W_MUTEX_REL = 51
W_TMU0_S = 56
W_TMU0_T = 57
W_TMU0_R = 58
W_TMU0_B = 59
W_TMU1_S = 60

R_UNIF = 32
R_ELEM_NUM = 38       # B file
R_QPU_NUM = 38        # B file
R_VPW_WAIT = 50       # B file
R_VPR_WAIT = 50       # A file

NOP_ADDR = 39


# pack/unpack/PM field geometry, from
# vc4-gpu-tools/include/vc4_qpu_defines.h (QPU_UNPACK_*/QPU_PM_*/QPU_PACK_*):
# unpack is 3 bits at 57..59, PM (MUL pack mode) is bit 56, pack is 4 bits at
# 52..55.  The original encoder used 4 bits at 56 for unpack (which overlapped
# PM) and 5 bits at 52 for pack (which overlapped PM and unpack), so a non-zero
# pack/unpack would have corrupted PM and both fields.  Every kernel word sets
# them to 0, so no emitted blob was ever affected.
UNPACK_SHIFT, UNPACK_MASK = 57, 0x7
PM_SHIFT, PM_MASK = 56, 0x1
PACK_SHIFT, PACK_MASK = 52, 0xF


def alu(add_op=ADD_NOP, mul_op=MUL_NOP, add_out=W_NOP, mul_out=W_NOP, ws=0,
        raddr_a=NOP_ADDR, raddr_b=NOP_ADDR,
        add_a=ACC0, add_b=ACC0, mul_a=ACC0, mul_b=ACC0,
        sig=SIG_NONE, cond_add=COND_ALWAYS, cond_mul=None, sf=0,
        unpack=0, pm=0, pack=0):
    if cond_mul is None:
        cond_mul = COND_ALWAYS if mul_op != MUL_NOP else COND_NEVER
    w = 0
    w |= (sig & 0xF) << 60
    w |= (unpack & UNPACK_MASK) << UNPACK_SHIFT
    w |= (pm & PM_MASK) << PM_SHIFT
    w |= (pack & PACK_MASK) << PACK_SHIFT
    w |= (cond_add & 7) << 49
    w |= (cond_mul & 7) << 46
    w |= (sf & 1) << 45
    w |= (ws & 1) << 44
    w |= (add_out & 0x3F) << 38
    w |= (mul_out & 0x3F) << 32
    w |= (mul_op & 7) << 29
    w |= (add_op & 0x1F) << 24
    w |= (raddr_a & 0x3F) << 18
    w |= (raddr_b & 0x3F) << 12
    w |= (add_a & 7) << 9
    w |= (add_b & 7) << 6
    w |= (mul_a & 7) << 3
    w |= (mul_b & 7) << 0
    return w


def nop(cond_add=COND_NEVER):
    return alu(add_op=ADD_NOP, mul_op=MUL_NOP, cond_add=cond_add, cond_mul=COND_NEVER)


def end_program():
    return alu(sig=SIG_END_PROGRAM, cond_add=COND_NEVER, cond_mul=COND_NEVER)


def ldi(value, out_reg, ws=0):
    w = 0
    w |= (SIG_LOAD_IMMEDIATE & 0xF) << 60
    w |= (0 & 0x7) << 57          # OpLoad::LOAD_IMM_32
    w |= (COND_ALWAYS & 7) << 49
    w |= (ws & 1) << 44
    w |= (out_reg & 0x3F) << 38
    w |= (W_NOP & 0x3F) << 32
    w |= value & 0xFFFFFFFF
    return w


def branch(byte_off_from_next, cond=BR_ALWAYS, relative=True, reg=None):
    w = 0
    w |= (SIG_BRANCH & 0xF) << 60
    w |= (cond & 0xF) << 52
    w |= (1 if relative else 0) << 51
    w |= (1 if reg is not None else 0) << 50
    w |= ((reg or 0) & 0x1F) << 45
    w |= (W_NOP & 0x3F) << 38
    w |= (W_NOP & 0x3F) << 32
    w |= byte_off_from_next & 0xFFFFFFFF
    return w


# operand descriptors: (mux, addr)
def imm(v):
    return (IMM, v & 0x3F)


def acc(n):
    return (n, NOP_ADDR)


def areg(n):
    return (REGA, n)


def breg(n):
    return (REGB, n)


def _ops(add_op=ADD_NOP, mul_op=MUL_NOP, add_out=W_NOP, mul_out=W_NOP, ws=0,
         add_a=None, add_b=None, mul_a=None, mul_b=None,
         sig=SIG_NONE, cond_add=COND_ALWAYS, sf=0, cond_mul=None,
         raddr_b=None):
    """Build an ALU word from operand descriptors.

    raddr_b forces the B address field.  The rotator uses it: the rotate code
    is a B small-immediate address (48..63) while the B data mux is unused.
    """
    a_mux = b_mux = ma_mux = mb_mux = ACC0
    ra_addr = rb_addr = NOP_ADDR
    seen_a = seen_b = None
    for desc, slot in ((add_a, 'a'), (add_b, 'b'), (mul_a, 'ma'), (mul_b, 'mb')):
        if desc is None:
            continue
        mux, addr = desc
        if mux == IMM:
            if seen_b is not None and rb_addr != addr:
                raise ValueError("instruction needs two different B operands")
            seen_b = slot
            mux = REGB
            rb_addr = addr
            sig = SIG_ALU_IMMEDIATE
        elif mux == REGA:
            if seen_a is not None and ra_addr != addr:
                raise ValueError("instruction needs two different A registers")
            seen_a = slot
            ra_addr = addr
        elif mux == REGB:
            if seen_b is not None and rb_addr != addr:
                raise ValueError("instruction needs two different B registers")
            seen_b = slot
            rb_addr = addr
        if slot == 'a':
            a_mux = mux
        elif slot == 'b':
            b_mux = mux
        elif slot == 'ma':
            ma_mux = mux
        else:
            mb_mux = mux
    return alu(add_op=add_op, mul_op=mul_op, add_out=add_out, mul_out=mul_out, ws=ws,
               raddr_a=ra_addr, raddr_b=rb_addr if raddr_b is None else raddr_b,
               add_a=a_mux, add_b=b_mux, mul_a=ma_mux, mul_b=mb_mux,
               sig=sig, cond_add=cond_add, cond_mul=cond_mul, sf=sf)


def fmul_fadd(mul_out, mul_a, mul_b, add_out, add_a, add_b, ws=0, sig=SIG_NONE):
    """One instruction = fmul(mul_out, mul_a, mul_b) and fadd(add_out, add_a, add_b)."""
    return _ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=add_out, mul_out=mul_out, ws=ws,
                add_a=add_a, add_b=add_b, mul_a=mul_a, mul_b=mul_b, sig=sig)


# --- rotator ---------------------------------------------------------------
# The MUL-ALU B small-immediate field doubles as the 16-lane rotator:
#   48      rotate right by r5 (dynamic amount)
#   49..63  rotate right by 1..15 lanes
# Rotating by 0 has no code; it is the plain accumulator move (rot_inst(0)).
# 1.0f is small immediate 32, NOT 48.  The same 48..63 codes alias the
# constants -16..-1 when the field is read as a constant.
# In the rotate form the rotate amount is the MUL B small-immediate address and
# the B data operand is unused.  vc4asm and qpuasm both encode that unused B mux
# as ACC0 and leave the imm bit in the mul_a/src operand, i.e.
#   mul_a=acc(src), mul_b=acc(src), raddr_b=48+n, sig=ALU_IMMEDIATE
# which is what rot_inst() now emits (byte-identical to vc4asm).
# The other form (mul_b=REGB, i.e. the standalone small-immediate mux) is what
# the kernels used before 2026-11-19; the two forms differ in exactly 3 bits
# and both are element-exact on this silicon (A/B under one GPU lock, the same
# gemm6_256 blob with only those 3 bits changed: Cmm=0 for both, 398.5 cyc/k,
# 2.451 ms).  The 2026-11-19 65533-mismatch scare was a separate encoder bug
# (a stray cond_add=NEVER leaked into the word's add slot and disabled the rep);
# it is fixed and guarded by gemm6's own word checks.
ROT_OFFSET = 48


def rot_inst(n, src=ACC0, out=1):
    """Descriptor for a full 16-lane rotate: acc[out] = acc[src] >> n.

    Returns the dict form consumed by _ops().  It deliberately does not set
    ws, so it can be merged into a word whose add slot selects the bank.
    n must be 0..15; 0 is the plain accumulator move (there is no rotate code
    for 0), 1..15 use the B small immediates 49..63.
    """
    if not 0 <= n <= 15:
        raise ValueError("rotate amount must be 0..15")
    # no cond_add here: when merged into a word that also runs an add-slot
    # operation, that operation's condition must survive.  A standalone
    # rotate word gets the _ops default (add NOP, cond ALWAYS), as before.
    d = dict(mul_op=MUL_V8MIN, mul_out=32 + out,
             mul_a=acc(src), mul_b=acc(src))
    if n:
        d.update(raddr_b=ROT_OFFSET + n, sig=SIG_ALU_IMMEDIATE)
    return d


def encode_to_bin(words, path):
    import struct
    with open(path, 'wb') as f:
        for w in words:
            f.write(struct.pack('<Q', w))


def dump(words):
    for i, w in enumerate(words):
        print(f"{i:3d}: 0x{w & 0xFFFFFFFF:08x}, 0x{w >> 32:08x}")


if __name__ == '__main__':
    corpus = [
        ("or ra1, unif, unif", 0x1002006715827d80,
         _ops(add_op=ADD_OR, add_out=1, add_a=areg(R_UNIF), add_b=areg(R_UNIF))),
        ("or ra0, qpu_num, qpu_num", 0x10020027159e6fc0,
         _ops(add_op=ADD_OR, add_out=0, add_a=breg(R_QPU_NUM), add_b=breg(R_QPU_NUM))),
        ("or vpm, rb0, rb0", 0x10020c27159c0fc0,
         _ops(add_op=ADD_OR, add_out=W_VPM, add_a=breg(0), add_b=breg(0))),
        ("add vpw_setup, r5, ra0", 0x10021c670c027b80,
         _ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=areg(0))),
        ("add vpw_addr, ra4, rb1", 0x10021ca70c101dc0,
         _ops(add_op=ADD_ADD, add_out=W_VPW_ADDR, ws=1, add_a=areg(4), add_b=breg(1))),
        ("shl r0, ra0, 2", 0xd002082711002dc0,
         _ops(add_op=ADD_SHL, add_out=R0, add_a=areg(0), add_b=imm(2), sig=SIG_ALU_IMMEDIATE)),
        ("add r0, unif, r0", 0x100208270c827c00,
         _ops(add_op=ADD_ADD, add_out=R0, add_a=areg(R_UNIF), add_b=acc(0))),
        ("add rb0, r4, 1", 0xd00210270c9c19c0,
         _ops(add_op=ADD_ADD, add_out=0, ws=1, add_a=acc(4), add_b=imm(1), sig=SIG_ALU_IMMEDIATE)),
        ("ldi vpw_setup, 0xc0010000", 0xe0021c67c0010000, ldi(0xC0010000, W_VPW_SETUP, ws=1)),
        ("nop", 0x100009e7009e7000, nop()),
        ("nop.load_tmu1", 0xb00009e7009e7000,
         alu(sig=SIG_LOAD_TMU1, cond_add=COND_NEVER, cond_mul=COND_NEVER)),
        ("nop.end", 0x300009e7009e7000, end_program()),
        ("brr.ifallc -41", 0xf08809e7fffffeb8,
         branch(-41 * 8, cond=BR_ALL_C_SET)),
    ]
    ok = True
    for name, want, got in corpus:
        if want != got:
            ok = False
            print(f"BAD {name}: want {want:016x} got {got:016x}")
        else:
            print(f"OK  {name}")
    print("corpus pass" if ok else "corpus FAIL")


# ---------------- disassembler (audit aid) ----------------
_ADD_NAMES = {0: 'nop', 1: 'fadd', 2: 'fsub', 3: 'fmin', 4: 'fmax', 5: 'fminabs', 6: 'fmaxabs',
              7: 'ftoi', 8: 'itof', 12: 'add', 13: 'sub', 14: 'shr', 15: 'asr', 16: 'ror',
              17: 'shl', 18: 'min', 19: 'max', 20: 'and', 21: 'or', 22: 'xor', 23: 'not', 24: 'clz'}
_MUL_NAMES = {0: 'nop', 1: 'fmul', 2: 'mul24'}
_SPECIAL_W = {36: 'tmu_noswap', 37: 'rep_all|r5', 38: 'irq', 40: 'unif_addr', 48: 'vpm',
              49: 'vpw_setup', 50: 'vpw_addr', 51: 'mutex_rel', 56: 'tmu0s', 57: 'tmu0t',
              58: 'tmu0r', 59: 'tmu0b', 60: 'tmu1s', 61: 'tmu1t', 62: 'tmu1r', 63: 'tmu1b'}
_SPECIAL_R = {(0, 32): 'unif', (0, 38): 'elem_num', (1, 38): 'qpu_num', (1, 50): 'vpw_wait',
              (0, 50): 'vpr_wait', (0, 48): 'vpm', (1, 48): 'vpm'}
_SIG_NAMES = {3: 'thrend', 10: 'load_tmu0', 11: 'load_tmu1'}


def _rname(mux, raddr_a, raddr_b, is_imm):
    if mux == REGA:
        return _SPECIAL_R.get((0, raddr_a), f'ra{raddr_a}')
    if mux == REGB:
        if is_imm:
            return str(raddr_b)
        return _SPECIAL_R.get((1, raddr_b), f'rb{raddr_b}')
    if mux == ACC5:
        return 'r5'
    return f'r{mux}'


def _outname(num, ws, add=True):
    if 32 <= num <= 37:
        return f'r{num - 32}'
    if num == 39:
        return '-'
    if num in _SPECIAL_W:
        return _SPECIAL_W[num]
    return (('ra' if ws == 0 else 'rb') + str(num)) if add else (('rb' if ws == 0 else 'ra') + str(num))


def disasm(w):
    sig = (w >> 60) & 0xF
    if sig == SIG_BRANCH:
        cond = (w >> 52) & 0xF
        rel = (w >> 51) & 1
        off = w & 0xFFFFFFFF
        if off >= 0x80000000:
            off -= 0x100000000
        return f'br{"r" if rel else "a"}.{cond} (pc+4){off // 8:+d}'
    if sig == SIG_LOAD_IMMEDIATE:
        ws = (w >> 44) & 1
        out = (w >> 38) & 0x3F
        val = w & 0xFFFFFFFF
        return f'ldi {_outname(out, ws)}, 0x{val:08x}'
    is_imm = (sig == SIG_ALU_IMMEDIATE)
    add_op = (w >> 24) & 0x1F
    mul_op = (w >> 29) & 7
    ws = (w >> 44) & 1
    add_out = (w >> 38) & 0x3F
    mul_out = (w >> 32) & 0x3F
    ra = (w >> 18) & 0x3F
    rb = (w >> 12) & 0x3F
    add_a = (w >> 9) & 7
    add_b = (w >> 6) & 7
    mul_a = (w >> 3) & 7
    mul_b = w & 7
    sf = (w >> 45) & 1
    cond_a = (w >> 49) & 7
    parts = []
    if add_op:
        extra = '' if cond_a == COND_ALWAYS else f'.{cond_a}'
        if sf:
            extra += '.setf'
        parts.append(f'{_ADD_NAMES.get(add_op, add_op)}{extra} {_outname(add_out, ws)}'
                     f', {_rname(add_a, ra, rb, is_imm)}, {_rname(add_b, ra, rb, is_imm)}')
    if mul_op:
        parts.append(f'{_MUL_NAMES.get(mul_op, mul_op)} {_outname(mul_out, ws, add=False)}'
                     f', {_rname(mul_a, ra, rb, is_imm)}, {_rname(mul_b, ra, rb, is_imm)}')
    if not parts:
        s = 'nop'
    else:
        s = '; '.join(parts)
    if sig in _SIG_NAMES:
        s += f'.{_SIG_NAMES[sig]}'
    return s


def listing(words):
    return '\n'.join(f'{i:3d}: 0x{w & 0xFFFFFFFF:08x}, 0x{w >> 32:08x}  // {disasm(w)}'
                     for i, w in enumerate(words))
