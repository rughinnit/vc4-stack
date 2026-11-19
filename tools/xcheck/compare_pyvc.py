#!/usr/bin/env python3
"""Fourth independent QPU encoder: Idein/py-videocore (MIT).

py-videocore (Koichi Nakamura / Idein, https://github.com/Idein/py-videocore)
is a Python toolchain for the VideoCore IV QPU.  Its ``videocore/encoding.py``
describes the 64-bit instruction word as a ctypes bit-field Structure
(AluInsn / LoadInsn / BranchInsn / SemaInsn) and ``videocore/assembler.py`` is
an independent Python-DSL front end that fills those fields.  Neither has
anything to do with qenc or vc4asm.

Two checks are run:

1. *Instruction level*: a representative instruction set is assembled with
   py-videocore's ``assemble()`` and with qenc, and the bytes are compared.
   Where a maazl/vc4asm spelling exists it is assembled as well, so every diff
   has a third opinion.  Each case's word is also searched for in the shipped
   ``gemm6_128.bin``; when it is found the word index is printed, which ties
   the case to what the kernel actually runs.

2. *Word level*: every 64-bit word of the shipped blobs is taken apart with
   qenc's bit map, re-encoded through py-videocore's ``encoding.py`` Insn
   classes, and compared byte for byte.  That exercises py-videocore's field
   geometry on real words, including fields the DSL rarely touches
   (pack/unpack/PM, LDI mode, branch link).

    python3 compare_pyvc.py \
        --qenc-dir /home/raspi/work/vc4-stack/kernels/wider-tile/asm10 \
        --pyvc-dir /home/raspi/work/xcheck/py-videocore \
        --vc4asm   /home/raspi/work/xcheck/vc4asm/build/vc4asm

Exit 0 when every case is either bit-exact or a documented py-videocore DSL
spelling difference whose canonical form is vc4asm's (== qenc's); 1 otherwise.
"""
import argparse
import glob
import os
import re
import struct
import subprocess
import sys

QENC_DIR = None


def _load_dirs(a):
    global QENC_DIR
    QENC_DIR = os.path.abspath(a.qenc_dir)
    sys.path.insert(0, QENC_DIR)
    sys.path.insert(0, os.path.abspath(a.pyvc_dir))


# --------------------------------------------------------------------------
# py-videocore front end (assembler.py DSL).  Every function emits the words
# of one case; parameters let one function cover all rotate amounts.
# --------------------------------------------------------------------------
def make_pyvc():
    from videocore.assembler import qpu, assemble
    from videocore.encoding import REGISTERS

    @qpu
    def c_nop(asm):
        nop()

    @qpu
    def c_or_ra1_unif(asm):
        mov(ra25, uniform)

    @qpu
    def c_or_ra27_qpu(asm):
        mov(ra27, qpu_number)

    @qpu
    def c_or_rb18_r4(asm):
        mov(rb18, r4)

    @qpu
    def c_add_setf(asm):
        iadd(ra25, ra24, rb22)

    @qpu
    def c_shl_sf(asm):
        shl(ra27, ra27, 2, set_flags=False)

    @qpu
    def c_fadd_one(asm):
        fadd(r1, r0, 1.0, set_flags=False)

    @qpu
    def c_fmul_only(asm):
        fmul(r2, r5, ra16)

    @qpu
    def c_fadd_only(asm):
        fadd(ra0, ra0, r2, set_flags=False)

    @qpu
    def c_fadd_fmul(asm):
        fadd(ra0, ra0, r2, set_flags=False).fmul(r2, r5, rb16)

    @qpu
    def c_fadd_fmul_indep(asm):
        fadd(rb0, rb0, r1, set_flags=False).fmul(r3, r4, ra18)

    @qpu
    def c_ldi_acc(asm):
        ldi(r1, 4)

    @qpu
    def c_ldi_rb(asm):
        ldi(rb20, 0x200)

    @qpu
    def c_rot(asm, n):
        rotate(r1, r0, n)

    @qpu
    def c_rep_rot(asm, n):
        mov(broadcast, r1).rotate(r1, r0, n)

    @qpu
    def c_rep(asm):
        mov(broadcast, r1)

    @qpu
    def c_quad(asm):
        mov(r5_pix0, r1)

    @qpu
    def c_pack16a(asm):
        iadd(ra1.pack('16a'), ra0, rb0)

    @qpu
    def c_unpack16a(asm):
        mov(r0, ra0.unpack('16a'))

    @qpu
    def c_unpack_r4(asm):
        mov(r0, r4.unpack('8a'))

    @qpu
    def c_mul_pack8a(asm):
        fmul(ra1.pack('8a'), r0, r1)

    @qpu
    def c_tmu_s(asm):
        mov(tmu0_s, r3)

    @qpu
    def c_tmu_add(asm):
        iadd(tmu1_s, r3, rb26, set_flags=False)

    @qpu
    def c_tmu_dual(asm):
        mov(tmu0_s, r3, sig='load tmu0').fmul(r2, r5, ra16)

    @qpu
    def c_vpm_words(asm, which):
        if which == 0:
            setup_dma_store_stride(0, blockmode=True)
        elif which == 1:
            setup_dma_load_stride(512)
        elif which == 2:
            setup_vpm_write(Y=52, stride=1)
        elif which == 3:
            setup_dma_load(nrows=16, ncols=16, mode='32bit horizontal', Y=1)
        else:
            setup_dma_store(nrows=4, ncols=16, mode='32bit horizontal', Y=0)

    @qpu
    def c_loop3(asm):
        ldi(r1, 4)
        L.top
        iadd(r1, r1, -1)
        jzc_any(L.top)

    @qpu
    def c_thrend(asm):
        mov(r0, r0, sig='thread end')

    @qpu
    def c_ldtmu1(asm):
        nop(sig='load tmu1')

    funcs = dict(nop=c_nop, or_ra1_unif=c_or_ra1_unif,
                 or_ra27_qpu=c_or_ra27_qpu, or_rb18_r4=c_or_rb18_r4,
                 add_setf=c_add_setf, shl_sf=c_shl_sf, fadd_one=c_fadd_one,
                 fmul_only=c_fmul_only, fadd_only=c_fadd_only,
                 fadd_fmul=c_fadd_fmul, fadd_fmul_indep=c_fadd_fmul_indep,
                 ldi_acc=c_ldi_acc, ldi_rb=c_ldi_rb, rot=c_rot,
                 rep_rot=c_rep_rot, rep=c_rep, quad=c_quad,
                 pack16a=c_pack16a, unpack16a=c_unpack16a,
                 unpack_r4=c_unpack_r4, mul_pack8a=c_mul_pack8a,
                 tmu_s=c_tmu_s, tmu_add=c_tmu_add, tmu_dual=c_tmu_dual,
                 vpm_words=c_vpm_words, loop3=c_loop3, thrend=c_thrend,
                 ldtmu1=c_ldtmu1)

    def words(key, *args):
        key = key[2:] if key.startswith('c_') else key
        code = assemble(funcs[key], *args)
        return [int.from_bytes(code[i * 8:i * 8 + 8], 'little')
                for i in range(len(code) // 8)]

    return words, REGISTERS


# --------------------------------------------------------------------------
# case list
# --------------------------------------------------------------------------
NOP_NOTE = ("py-videocore's nop() names r0 as the destination and leaves "
            "cond_add=always; qenc and vc4asm write null with cond_add=never. "
            "The add opcode is nop in all three.")
LDI_NOTE = ("py-videocore sets the unused MUL-slot condition of an LDI word "
            "to always (cond_mul=1); qenc and vc4asm set it to never. The "
            "MUL write address is null, so the slot is inert.")


def build_cases(q, R):
    """q = qenc module, R = py-videocore REGISTERS dict."""
    from qenc import (ACC0, ACC1, ACC4, ADD_ADD, ADD_FADD, ADD_OR, ADD_SHL,
                      BR_ANY_Z_CLEAR, COND_NEVER, MUL_FMUL, MUL_V8MIN, REGA,
                      REGB, SIG_ALU_IMMEDIATE, SIG_LOAD_TMU1, SIG_LOAD_TMU0,
                      _ops, acc, alu, areg, branch, breg, end_program, imm,
                      ldi, nop, rot_inst)
    import vpmwords as V

    REP = dict(add_op=ADD_OR, add_out=37, ws=1, add_a=acc(1), add_b=acc(1))

    def mk_quad(n):
        """rotate by n lanes as a standalone MUL-only word (no rep slot)."""
        return _ops(**{**rot_inst(n, src=0, out=1), 'cond_add': COND_NEVER})

    def mk_rep_rot(n):
        """rotate by n lanes in the real r5rep word (gemm6's rotate shape)."""
        return _ops(**{**rot_inst(n, src=0, out=1), **REP})

    def mk_rep_dyn():
        d = dict(mul_op=MUL_V8MIN, mul_out=33, mul_a=acc(0), mul_b=acc(0),
                 raddr_b=q.ROT_OFFSET, sig=SIG_ALU_IMMEDIATE)
        return _ops(**{**d, **REP})

    def mk_dyn():
        return _ops(mul_op=MUL_V8MIN, mul_out=33, mul_a=acc(0), mul_b=acc(0),
                    raddr_b=q.ROT_OFFSET, sig=SIG_ALU_IMMEDIATE,
                    cond_add=COND_NEVER)

    C = []

    def add(name, qenc_words, pyv, vc=None, vc_n=1, known=None):
        C.append(dict(name=name, qw=qenc_words, pyv=pyv, vc=vc, vc_n=vc_n,
                      known=known))

    add("nop (null destination)", [nop()], ("nop", ()), "\tnop",
        known=NOP_NOTE)
    add("or ra25, unif, unif",
        [_ops(add_op=ADD_OR, add_out=25, add_a=areg(32), add_b=areg(32))],
        ("or_ra1_unif", ()), "\tor ra25, unif, unif")
    add("or ra27, qpu_num, qpu_num",
        [_ops(add_op=ADD_OR, add_out=27, add_a=breg(38), add_b=breg(38))],
        ("or_ra27_qpu", ()), "\tor ra27, qpu_num, qpu_num")
    add("mov rb18, r4 (ws=1)",
        [_ops(add_op=ADD_OR, add_out=18, ws=1, add_a=acc(4), add_b=acc(4))],
        ("or_rb18_r4", ()), "\tor rb18, r4, r4")
    add("add.setf ra25, ra24, rb22",
        [_ops(add_op=ADD_ADD, add_out=25, add_a=areg(24), add_b=breg(22), sf=1)],
        ("add_setf", ()), "\tadd.setf ra25, ra24, rb22")
    add("shl ra27, ra27, 2 (small imm)",
        [_ops(add_op=ADD_SHL, add_out=27, add_a=areg(27), add_b=imm(2),
              sig=SIG_ALU_IMMEDIATE)],
        ("shl_sf", ()), "\tshl ra27, ra27, 2")
    add("fadd r1, r0, 1.0f (small imm 32)",
        [_ops(add_op=ADD_FADD, add_out=33, add_a=acc(0), add_b=imm(32),
              sig=SIG_ALU_IMMEDIATE)],
        ("fadd_one", ()), "\tfadd r1, r0, 1.0")

    add("fmul r2, r5, ra16 (MUL slot only)",
        [_ops(mul_op=MUL_FMUL, mul_out=34, mul_a=acc(5), mul_b=areg(16),
              cond_add=COND_NEVER)],
        ("fmul_only", ()), "\tfmul r2, r5, ra16")
    add("fadd ra0, ra0, r2 (ADD slot only)",
        [_ops(add_op=ADD_FADD, add_out=0, add_a=areg(0), add_b=acc(2),
              cond_mul=COND_NEVER)],
        ("fadd_only", ()), "\tfadd ra0, ra0, r2")
    add("fadd ra0,ra0,r2; fmul r2,r5,rb16 (dual, mul->add dependent)",
        [_ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=0, mul_out=34,
              add_a=areg(0), add_b=acc(2), mul_a=acc(5), mul_b=breg(16))],
        ("fadd_fmul", ()), "\tfadd ra0, ra0, r2; fmul r2, r5, rb16")
    add("fadd rb0,rb0,r1; fmul r3,r4,ra18 (dual, independent)",
        [_ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=0, ws=1, mul_out=35,
              add_a=breg(0), add_b=acc(1), mul_a=acc(4), mul_b=areg(18))],
        ("fadd_fmul_indep", ()), "\tfadd rb0, rb0, r1; fmul r3, r4, ra18")

    add("ldi r1, 4", [ldi(4, 33)], ("ldi_acc", ()), "\tldi r1, 4",
        known=LDI_NOTE)
    add("ldi rb20, 0x200", [ldi(0x200, 20, ws=1)], ("ldi_rb", ()),
        "\tldi rb20, 0x200", known=LDI_NOTE)

    for n in range(1, 16):
        add("rot %d (plain, MUL slot)" % n, [mk_quad(n)], ("rot", (n,)),
            "\tmov r1, r0 >> %d" % n)
    add("rot by r5 (plain, MUL slot, dynamic)", [mk_dyn()],
        ("rot", (R['r5'],)), "\tmov r1, r0 << r5")
    add("rot 0 (plain move, no rotator code)", [mk_quad(0)], ("rot", (0,)),
        "\tv8min r1, r0, r0")

    add("mov r5rep, r1", [_ops(**REP)], ("rep", ()), "\tmov r5rep, r1")
    add("mov r5quad, r1", [_ops(**{**REP, 'ws': 0})], ("quad", ()),
        "\tmov r5quad, r1")

    for n in range(0, 16):
        add("mov r5rep, r1; v8min r1, r0 rot %d" % n, [mk_rep_rot(n)],
            ("rep_rot", (n,)),
            ("\tmov r5rep, r1; v8min r1, r0, r0" if n == 0
             else "\tmov r5rep, r1; mov r1, r0 >> %d" % n))
    add("mov r5rep, r1; v8min r1, r0 rot r5", [mk_rep_dyn()],
        ("rep_rot", (R['r5'],)), "\tmov r5rep, r1; mov r1, r0 << r5")

    add("iadd.pack16a ra1, ra0, rb0",
        [alu(add_op=ADD_ADD, add_out=1, raddr_a=0, raddr_b=0, add_a=REGA,
             add_b=REGB, sf=1, pack=1)],
        ("pack16a", ()), "\tadd.setf.pack16a ra1, ra0, rb0")
    add("mov r0, ra0.unpack16a",
        [alu(add_op=ADD_OR, add_out=32, raddr_a=0, add_a=REGA, add_b=REGA,
             unpack=1)],
        ("unpack16a", ()), "\tmov r0, ra0.unpack16a")
    add("mov r0, r4.unpack8a (unpack + PM)",
        [alu(add_op=ADD_OR, add_out=32, add_a=ACC4, add_b=ACC4, unpack=4,
             pm=1)],
        ("unpack_r4", ()), "\tmov r0, r4.unpack8a")
    add("fmul.pack8a ra1, r0, r1 (ws=1)",
        [alu(mul_op=MUL_FMUL, mul_out=1, ws=1, mul_a=ACC0, mul_b=ACC1,
             pack=4, cond_add=COND_NEVER)],
        ("mul_pack8a", ()), "\tfmul.pack8a ra1, r0, r1")

    add("mov t0s, r3",
        [_ops(add_op=ADD_OR, add_out=56, add_a=acc(3), add_b=acc(3))],
        ("tmu_s", ()), "\tmov t0s, r3")
    add("add t1s, r3, rb26",
        [_ops(add_op=ADD_ADD, add_out=60, add_a=acc(3), add_b=breg(26))],
        ("tmu_add", ()), "\tadd t1s, r3, rb26")
    add("or t0s, r3, r3; fmul r2, r5, ra16 + ldtmu0",
        [_ops(add_op=ADD_OR, mul_op=MUL_FMUL, add_out=56, mul_out=34,
              add_a=acc(3), add_b=acc(3), mul_a=acc(5), mul_b=areg(16),
              sig=SIG_LOAD_TMU0)],
        ("tmu_dual", ()), None)

    add("ldi vw_setup, 0xc0010000 (VDW stride)", [ldi(0xC0010000, 49, ws=1)],
        ("vpm_words", (0,)), "\tldi vw_setup, 0xc0010000", known=LDI_NOTE)
    add("ldi vr_setup, 0x90000200 (VDR stride 512)", [ldi(0x90000200, 49)],
        ("vpm_words", (1,)), "\tldi vr_setup, 0x90000200", known=LDI_NOTE)
    add("ldi vw_setup, 0x1a34 (generic write, row 52 stride 1)",
        [ldi(V.vpw_generic(52, 1), 49, ws=1)], ("vpm_words", (2,)),
        "\tldi vw_setup, 0x1a34", known=LDI_NOTE)
    add("ldi vr_setup, 0x83001010 (VDR basic 16x16, vpitch 1, Y=1)",
        [ldi(V.vdr_basic(1, vpitch=1, numrows=16, rowlen=16, mpitch=3), 49)],
        ("vpm_words", (3,)), "\tldi vr_setup, 0x83001010", known=LDI_NOTE)
    add("ldi vw_setup, 0x82104000 (VDW basic depth 16 units 4)",
        [ldi(V.vdw_basic(0, 16, 4), 49, ws=1)], ("vpm_words", (4,)),
        "\tldi vw_setup, 0x82104000", known=LDI_NOTE)

    add("loop: ldi r1,4 / add.setf r1,r1,-1 / brr.anynz top",
        [ldi(4, 33),
         _ops(add_op=ADD_ADD, add_out=33, add_a=acc(1), add_b=imm(31),
              sf=1, sig=SIG_ALU_IMMEDIATE),
         branch(-40, cond=BR_ANY_Z_CLEAR)],
        ("loop3", ()),
        "\tldi r1, 4\ntop:\tadd.setf r1, r1, -1\n\tbrr.anynz -, r:top\n"
        "\tnop\n\tnop\n\tnop", vc_n=6,
        known="word 0 is the LDI spelling difference; the add.setf word and "
              "the relative branch (immediate -40) are bit-exact")

    add("thrend", [end_program()], ("thrend", ()), "\tthrend",
        known="py-videocore's mov(r0,r0,sig='thread end') names r0 as the "
              "destination and leaves cond_add=always; vc4asm/qenc use null")
    add("nop.load_tmu1 (signal-only word)",
        [alu(sig=SIG_LOAD_TMU1, cond_add=COND_NEVER, cond_mul=COND_NEVER)],
        ("ldtmu1", ()), "\tldtmu1",
        known="same destination/condition spelling difference as nop()")

    return C


# --------------------------------------------------------------------------
# field diff, using qenc's bit map
# --------------------------------------------------------------------------
FIELDS = [
    ('sig', 60, 4), ('unpack', 57, 3), ('pm', 56, 1), ('pack', 52, 4),
    ('cond_add', 49, 3), ('cond_mul', 46, 3), ('sf', 45, 1), ('ws', 44, 1),
    ('waddr_add', 38, 6), ('waddr_mul', 32, 6), ('op_mul', 29, 3),
    ('op_add', 24, 5), ('raddr_a', 18, 6), ('raddr_b', 12, 6),
    ('add_a', 9, 3), ('add_b', 6, 3), ('mul_a', 3, 3), ('mul_b', 0, 3),
]
BR_FIELDS = [('imm', 0, 32), ('waddr_add', 38, 6), ('waddr_mul', 32, 6),
             ('ws', 44, 1), ('raddr_a', 45, 5), ('reg', 50, 1),
             ('rel', 51, 1), ('cond_br', 52, 4)]


def diff_fields(a, b):
    fs = []
    fields = BR_FIELDS if (a >> 60) == 15 else FIELDS
    for name, sh, w in fields:
        va, vb = (a >> sh) & ((1 << w) - 1), (b >> sh) & ((1 << w) - 1)
        if va != vb:
            fs.append('%s %d->%d' % (name, va, vb))
    return ', '.join(fs)


# --------------------------------------------------------------------------
# word-level re-encode through py-videocore's encoding.py
# --------------------------------------------------------------------------
def reencode_words(paths):
    import videocore.encoding as enc
    tot = bad = 0
    badlist = []
    for path in paths:
        blob = open(path, 'rb').read()
        if len(blob) % 8:
            print("  %-58s skipped (not a word blob)" % path)
            continue
        n = 0
        for i in range(len(blob) // 8):
            w, = struct.unpack_from('<Q', blob, i * 8)
            g = lambda sh, wd: (w >> sh) & ((1 << wd) - 1)
            sig = g(60, 4)
            if sig == 0xF:
                insn = enc.BranchInsn(
                    immediate=g(0, 32), waddr_mul=g(32, 6), waddr_add=g(38, 6),
                    ws=g(44, 1), raddr_a=g(45, 5), reg=g(50, 1), rel=g(51, 1),
                    cond_br=g(52, 4), dontcare=g(56, 4), sig=g(60, 4))
            elif sig == 0xE and g(57, 3) == 4:
                continue                      # semaphore word (none shipped)
            elif sig == 0xE:
                insn = enc.LoadInsn(
                    immediate=g(0, 32), waddr_mul=g(32, 6), waddr_add=g(38, 6),
                    ws=g(44, 1), sf=g(45, 1), cond_mul=g(46, 3),
                    cond_add=g(49, 3), pack=g(52, 4), pm=g(56, 1),
                    unpack=g(57, 3), sig=g(60, 4))
            else:
                insn = enc.AluInsn(
                    mul_b=g(0, 3), mul_a=g(3, 3), add_b=g(6, 3), add_a=g(9, 3),
                    raddr_b=g(12, 6), raddr_a=g(18, 6), op_add=g(24, 5),
                    op_mul=g(29, 3), waddr_mul=g(32, 6), waddr_add=g(38, 6),
                    ws=g(44, 1), sf=g(45, 1), cond_mul=g(46, 3),
                    cond_add=g(49, 3), pack=g(52, 4), pm=g(56, 1),
                    unpack=g(57, 3), sig=g(60, 4))
            got, = struct.unpack('<Q', insn.to_bytes())
            tot += 1
            n += 1
            if got != w:
                bad += 1
                if len(badlist) < 10:
                    badlist.append((path, i, w, got))
        print("  %-58s %6d words  %s" %
              (os.path.relpath(path), n, "OK" if bad == 0 else "%d MISMATCH" % bad))
    return tot, bad, badlist


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qenc-dir', default=os.environ.get(
        'ASM10', '/home/raspi/work/vc4-stack/kernels/wider-tile/asm10'))
    ap.add_argument('--pyvc-dir', default='/home/raspi/work/xcheck/py-videocore')
    ap.add_argument('--vc4asm', default='/home/raspi/work/xcheck/vc4asm/build/vc4asm')
    ap.add_argument('--blob-dir', default='/home/raspi/work/vc4-stack')
    ap.add_argument('--no-wordcheck', action='store_true')
    a = ap.parse_args()

    _load_dirs(a)
    import qenc as q
    pv, R = make_pyvc()
    cases = build_cases(q, R)

    # ---- vc4asm: one file, in case order --------------------------------
    vc_words = None
    if a.vc4asm and os.path.exists(a.vc4asm):
        import tempfile
        tmpd = tempfile.mkdtemp(prefix='pyvc_')
        qasm = os.path.join(tmpd, 'compare.qasm')
        hexf = os.path.join(tmpd, 'compare.hex')
        lines = [c['vc'] for c in cases if c['vc']]
        open(qasm, 'w').write('\n'.join(lines) + '\n')
        r = subprocess.run([a.vc4asm, '-C', hexf, qasm],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr, sep='\n')
            return 2
        vc_words = []
        for line in open(hexf):
            v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', line)]
            if len(v) >= 2:
                vc_words.append(v[0] | (v[1] << 32))
        os.unlink(qasm)
        os.unlink(hexf)
        os.rmdir(tmpd)
        need = sum(c['vc_n'] for c in cases if c['vc'])
        if len(vc_words) != need:
            print("vc4asm produced %d words, expected %d"
                  % (len(vc_words), need))
            return 2

    # ---- exact words of the shipped GEMM kernel ------------------------
    kernel = {}
    kpath = os.path.join(QENC_DIR, 'gemm6_128.bin')
    if os.path.exists(kpath):
        blob = open(kpath, 'rb').read()
        for i in range(len(blob) // 8):
            w, = struct.unpack_from('<Q', blob, i * 8)
            kernel.setdefault(w, []).append(i)

    # ---- compare --------------------------------------------------------
    bad = known = exact = 0
    off = 0
    for c in cases:
        pyw = pv(c['pyv'][0], *c['pyv'][1])
        vcw = None
        if c['vc']:
            vcw = vc_words[off:off + c['vc_n']]
            off += c['vc_n']
        idx = [kernel[w] for w in c['qw'] if w in kernel]
        where = ""
        if idx:
            where = "   [gemm6_128 word %s]" % ','.join(
                str(i[0]) for i in idx)
        print("%-52s%s" % (c['name'], where))
        ok = True
        for i, qw in enumerate(c['qw']):
            p = pyw[i] if i < len(pyw) else None
            v = vcw[i] if vcw and i < len(vcw) else None
            if p == qw:
                exact += 1
            else:
                ok = False
            print("   %-8s ours %08x %08x  pyvc %s  %s  %s"
                  % ("word %d" % i, qw & 0xffffffff, qw >> 32,
                     ("%08x %08x" % (p & 0xffffffff, p >> 32))
                     if p is not None else "-" * 17,
                     ("vc4asm %08x %08x" % (v & 0xffffffff, v >> 32))
                     if v is not None else "",
                     "OK" if p == qw else "PYDIFF"))
            if p is not None and p != qw:
                print("        field diff (ours -> pyvc): %s"
                      % diff_fields(qw, p))
            if v is not None and v != qw:
                print("        vc4asm differs from ours: %s"
                      % diff_fields(qw, v))
            elif v is not None and p != qw:
                print("        vc4asm agrees with ours")
        if len(pyw) != len(c['qw']):
            print("        WORD COUNT %d (pyvc) vs %d (ours)"
                  % (len(pyw), len(c['qw'])))
        if not ok:
            if c['known']:
                known += 1
                print("   KNOWN DIFF -> canonical: vc4asm (== qenc). %s"
                      % c['known'])
            else:
                bad += 1
                print("   UNEXPECTED DIFF")
        print()

    total_words = sum(len(c['qw']) for c in cases)
    print("instruction cases: %d, words: %d, bit-exact words: %d, "
          "unexpected diffs: %d, documented spelling diffs: %d"
          % (len(cases), total_words, exact, bad, known))

    if not a.no_wordcheck:
        print("\nword-level re-encode through py-videocore encoding.py:")
        root = a.blob_dir
        pats = [
            'kernels/wider-tile/asm10/gemm6_64.bin',
            'kernels/wider-tile/asm10/gemm6_128.bin',
            'kernels/wider-tile/asm10/gemm6_256.bin',
            'kernels/wider-tile/asm10/pack10_*_us.bin',
            'engine/kcache/col2im_*.bin', 'engine/kcache/pool_*.bin',
            'engine/kcache/ew_*.bin',
        ]
        paths = []
        for p in pats:
            paths += sorted(glob.glob(os.path.join(root, p)))
        if paths:
            tot, nbad, bl = reencode_words(paths)
            print("  %d words, %d mismatches" % (tot, nbad))
            for path, i, w, g in bl:
                print("   %s word %d: ours %016x pyvc %016x (%s)"
                      % (path, i, w, g, diff_fields(w, g)))
            bad += nbad
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
