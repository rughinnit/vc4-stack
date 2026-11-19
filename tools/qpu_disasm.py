#!/usr/bin/env python3
"""Disassemble a raw VC4 QPU kernel blob and flag unknown/illegal encodings.

Input is what our harnesses emit: little-endian 64-bit instruction words with no
header (see qenc.encode_to_bin / gen.Prog.write).

The field map below is written out from the Broadcom/Anholt constants in
vc4-gpu-tools include/vc4_qpu_defines.h (QPU_*_SHIFT / QPU_*_MASK), not from our
encoder, so this tool can catch encoder bugs.  Opcode names follow the same
header.  Notes on ambiguous encodings are in docs/XCHECK.md.

Usage:
    qpu_disasm.py BLOB [--first N] [--count N] [--no-listing]
                        [--allow-unknown] [--json]

Exit status: 0 if every word decodes, 1 if an unknown opcode/illegal field is
found (unless --allow-unknown), 2 on I/O or argument errors.
"""
import argparse
import json
import struct
import sys

# --- authoritative field map (vc4_qpu_defines.h) -------------------------
SIG_SHIFT, SIG_MASK = 60, 0xF
UNPACK_SHIFT, UNPACK_MASK = 57, 0x7
PM = 1 << 56                      # pack field means PACK_MUL / R4 packing
PACK_SHIFT, PACK_MASK = 52, 0xF
COND_ADD_SHIFT, COND_MUL_SHIFT = 49, 46
SF_SHIFT, WS_SHIFT = 45, 44
WADDR_ADD_SHIFT, WADDR_MUL_SHIFT = 38, 32
OP_MUL_SHIFT, OP_ADD_SHIFT = 29, 24
RADDR_A_SHIFT, RADDR_B_SHIFT = 18, 12
ADD_A_SHIFT, ADD_B_SHIFT, MUL_A_SHIFT, MUL_B_SHIFT = 9, 6, 3, 0

# enum qpu_sig
SIG_NAMES = {
    0: "brk", 1: "", 2: "switch", 3: "end", 4: "wait_score",
    5: "unlock_score", 6: "thread_switch", 7: "coverage_load",
    8: "color_load", 9: "color_load_end", 10: "load_tmu0",
    11: "load_tmu1", 12: "alpha_mask_load", 13: "small_imm",
    14: "load_imm", 15: "branch",
}
# enum qpu_add_op (9..11 and 25..29 are undefined)
ADD_NAMES = {
    0: "nop", 1: "fadd", 2: "fsub", 3: "fmin", 4: "fmax", 5: "fminabs",
    6: "fmaxabs", 7: "ftoi", 8: "itof", 12: "add", 13: "sub", 14: "shr",
    15: "asr", 16: "ror", 17: "shl", 18: "min", 19: "max", 20: "and",
    21: "or", 22: "xor", 23: "not", 24: "clz", 30: "v8adds", 31: "v8subs",
}
MUL_NAMES = {
    0: "nop", 1: "fmul", 2: "mul24", 3: "v8muld", 4: "v8min",
    5: "v8max", 6: "v8adds", 7: "v8subs",
}
COND_NAMES = ["never", "always", "zs", "zc", "ns", "nc", "cs", "cc"]
BRANCH_COND_NAMES = ["allz", "allnz", "anyz", "anynz", "alln", "allnn",
                     "anyn", "anynn", "allc", "allnc", "anyc", "anync",
                     "cc12", "cc13", "cc14", "always"]
# enum qpu_pack_mul: 1, 2 and 6..15 are not valid MUL pack values
MUL_PACK_VALID = {0, 3, 4, 5, 6, 7}
PACK_A_NAMES = ["", "16a", "16b", "8888", "8a", "8b", "8c", "8d",
                "32s", "16as", "16bs", "8888s", "8as", "8bs", "8cs", "8ds"]
UNPACK_NAMES = ["", "16a", "16b", "8d_rep", "8a", "8b", "8c", "8d"]

SPECIAL_READ_A = {
    32: "unif", 35: "vary", 38: "elem_num", 39: "nop",
    41: "x_pix", 42: "ms_flags", 48: "vpm", 49: "vpm_ld_busy",
    50: "vpm_ld_wait", 51: "mutex",
}
SPECIAL_READ_B = {
    32: "unif", 35: "vary", 38: "qpu_num", 39: "nop",
    41: "y_pix", 42: "rev_flag", 48: "vpm", 49: "vpm_st_busy",
    50: "vpm_st_wait", 51: "mutex",
}
SPECIAL_WRITE_A = {
    32: "r0", 33: "r1", 34: "r2", 35: "r3", 36: "tmu_noswap", 37: "r5quad",
    38: "irq", 39: "-", 40: "unif_addr", 41: "x_pix", 42: "ms_flags",
    43: "stencil", 44: "tlbz", 45: "tlbm", 46: "tlbc", 47: "tlbam",
    48: "vpm", 49: "vw_setup", 50: "vw_addr", 51: "mutex",
    52: "recip", 53: "recipsqrt", 54: "exp", 55: "log",
    56: "t0s", 57: "t0t", 58: "t0r", 59: "t0b",
    60: "t1s", 61: "t1t", 62: "t1r", 63: "t1b",
}
SPECIAL_WRITE_B = dict(SPECIAL_WRITE_A)
SPECIAL_WRITE_B[37] = "r5rep"
SPECIAL_WRITE_B[40] = "unif_addr_rel"
SPECIAL_WRITE_B[41] = "y_pix"
SPECIAL_WRITE_B[49] = "vr_setup"
SPECIAL_WRITE_B[50] = "vr_addr"

# small immediates 48..63: rotate codes. 48 = rotate by r5, 49..63 = 1..15.
ROTATE_MIN = 48


def f32(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


def small_imm(rb):
    if rb <= 15:
        return str(rb)
    if rb <= 31:
        return str(-16 + (rb - 16))
    if rb <= 39:
        return "%g" % float(1 << (rb - 32))
    if rb <= 47:
        return "%g" % (1.0 / (1 << (48 - rb)))
    return None  # 48..63: rotate code, not a constant in this context


def src(mux, rb, sig):
    if mux <= 5:
        return "r%d" % mux
    if mux == 6:
        return None  # raddr_a
    if mux == 7:
        if sig == 13:
            s = small_imm(rb)
            return s if s is not None else "<rot %d>" % (rb - 48)
        return None  # raddr_b
    return "<mux%d>" % mux


def disasm(w, ann=None):
    """Return (text, flags) for one 64-bit word. flags is a list of strings."""
    flags = []
    sig = (w >> SIG_SHIFT) & SIG_MASK
    if sig == 15:  # branch
        cond = (w >> 52) & 0xF
        rel = (w >> 51) & 1
        reg = (w >> 50) & 1
        ra5 = (w >> 45) & 0x1F
        off = w & 0xFFFFFFFF
        if off & 0x80000000:
            off -= 1 << 32
        tgt = "r:%d" % ra5 if reg else "-"
        txt = "br%s.%s %s, (pc+4)%+d" % (
            "r" if rel else "a", BRANCH_COND_NAMES[cond],
            tgt, off // 8)
        if not rel:
            flags.append("absolute branch (word 51 = 0)")
        return txt, flags

    if sig == 14:  # load immediate
        mode = (w >> 57) & 0x7
        ws = (w >> WS_SHIFT) & 1
        wa = (w >> WADDR_ADD_SHIFT) & 0x3F
        wb = (w >> WADDR_MUL_SHIFT) & 0x3F
        val = w & 0xFFFFFFFF
        if mode != 0:
            flags.append("load_imm mode %d (not LOAD_IMM_32)" % mode)
        dst = SPECIAL_WRITE_A if ws == 0 else SPECIAL_WRITE_B
        parts = []
        if wa != 39:
            parts.append(dst.get(wa, "wa%d" % wa))
        if wb != 39:
            parts.append(dst.get(wb, "wb%d" % wb))
        return "ldi %s, 0x%08x (%g)" % (
            "/".join(parts) if parts else "-", val, f32(val)), flags

    addop = (w >> OP_ADD_SHIFT) & 0x1F
    mulop = (w >> OP_MUL_SHIFT) & 0x7
    ra = (w >> RADDR_A_SHIFT) & 0x3F
    rb = (w >> RADDR_B_SHIFT) & 0x3F
    adda = (w >> ADD_A_SHIFT) & 0x7
    addb = (w >> ADD_B_SHIFT) & 0x7
    mula = (w >> MUL_A_SHIFT) & 0x7
    mulb = (w >> MUL_B_SHIFT) & 0x7
    ws = (w >> WS_SHIFT) & 1
    cond_add = (w >> COND_ADD_SHIFT) & 0x7
    cond_mul = (w >> COND_MUL_SHIFT) & 0x7
    sf = (w >> SF_SHIFT) & 1
    wa = (w >> WADDR_ADD_SHIFT) & 0x3F
    wb = (w >> WADDR_MUL_SHIFT) & 0x3F
    pack = (w >> PACK_SHIFT) & PACK_MASK
    pm = (w >> 56) & 1
    unpack = (w >> UNPACK_SHIFT) & UNPACK_MASK

    if addop not in ADD_NAMES:
        flags.append("unknown add opcode %d" % addop)
    if mulop not in MUL_NAMES:
        flags.append("unknown mul opcode %d" % mulop)
    if pm and pack not in MUL_PACK_VALID:
        flags.append("illegal mul pack %d" % pack)
    if unpack and (mula == 7 or mulb == 7):
        flags.append("unpack on a B-file MUL operand")

    # rotate: sig=small_imm + mul B field in 48..63.  Hardware ignores the
    # mul-B data mux in this form; it is the rotate amount.
    rot = None
    if sig == 13 and rb >= ROTATE_MIN and mulop != 0:
        rot = rb - ROTATE_MIN
        if mula > 5:
            flags.append("rotate source is not an accumulator")

    def reg(mux, is_a):
        if mux <= 5:
            return "r%d" % mux
        if mux == 6:
            return SPECIAL_READ_A.get(ra, "ra%d" % ra)
        if mux == 7:
            if sig == 13:
                s = small_imm(rb)
                return s if s is not None else "<rot>"
            return SPECIAL_READ_B.get(rb, "rb%d" % rb)
        return "<mux%d>" % mux

    def dst(num, is_add):
        # QPU_WS: add writes B when ws=1; mul writes A when ws=1.
        bank_a = (ws == 0) if is_add else (ws == 1)
        tbl = SPECIAL_WRITE_A if bank_a else SPECIAL_WRITE_B
        if num in tbl:
            return tbl[num]
        return ("ra%d" if bank_a else "rb%d") % num

    parts = []
    if addop != 0:
        sfx = "" if cond_add == 1 else "." + COND_NAMES[cond_add]
        if sf:
            sfx += ".setf"
        parts.append("%s%s %s, %s, %s" % (
            ADD_NAMES.get(addop, "addop%d" % addop), sfx, dst(wa, True),
            reg(adda, True), reg(addb, False)))
        if pack:
            parts[-1] += " .%s" % PACK_A_NAMES[pack]
        if unpack:
            parts[-1] += " (unpack .%s)" % UNPACK_NAMES[unpack]
    if mulop != 0:
        sfx = "" if cond_mul == 1 else "." + COND_NAMES[cond_mul]
        if sf:
            sfx += ".setf"
        two = mulop in (3, 4, 5, 6, 7)
        op = "%s%s %s, %s" % (MUL_NAMES.get(mulop, "mulop%d" % mulop),
                              sfx, dst(wb, False), reg(mula, True))
        if two and not (sig == 13 and rb >= ROTATE_MIN):
            op += ", " + reg(mulb, False)
        if rot is not None:
            op += "   <rotate-%s %d>" % ("r5" if rot == 0 else "right", rot)
        if pm and pack:
            op += " .%s" % (PACK_A_NAMES[pack])
        parts.append(op)

    if not parts:
        txt = "nop"
    else:
        txt = "; ".join(parts)
    if sig not in (1,):
        txt += "   [sig %s]" % (SIG_NAMES.get(sig, "?%d" % sig))
    return txt, flags


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blob")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--no-listing", action="store_true")
    ap.add_argument("--allow-unknown", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    try:
        data = open(a.blob, "rb").read()
    except OSError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if len(data) % 8:
        print("error: %s is %d bytes, not a multiple of 8" % (a.blob, len(data)),
              file=sys.stderr)
        return 2
    n = len(data) // 8
    first = a.first
    count = n - first if a.count is None else a.count
    if first < 0 or first > n or count < 0:
        print("error: bad range", file=sys.stderr)
        return 2
    count = min(count, n - first)

    unknown = 0
    rot = 0
    bad = []
    for i in range(first, first + count):
        w = struct.unpack_from("<Q", data, i * 8)[0]
        txt, flags = disasm(w)
        hard = [f for f in flags if "unknown" in f or "illegal" in f
                or "not a multiple" in f or "not LOAD_IMM" in f]
        if "<rot" in txt:
            rot += 1
        if hard:
            unknown += 1
            bad.append((i, w, hard))
        if not a.no_listing:
            mark = "  !! " + "; ".join(hard) if hard else ""
            print("%4d: %08x %08x  %s%s" % (
                i, w & 0xFFFFFFFF, w >> 32, txt, mark))

    summary = {
        "blob": a.blob,
        "words": count,
        "first": first,
        "rotate_words": rot,
        "unknown_words": unknown,
        "unknown": [{"index": i, "word": "0x%016x" % w, "why": f}
                    for i, w, f in bad],
    }
    if a.json:
        print(json.dumps(summary, indent=2))
    else:
        print("-- %s: %d words, %d rotate, %d unknown/illegal"
              % (a.blob, count, rot, unknown), file=sys.stderr)
        for i, w, f in bad:
            print("   word %d 0x%016x: %s" % (i, w, "; ".join(f)),
                  file=sys.stderr)
    if unknown and not a.allow_unknown:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
