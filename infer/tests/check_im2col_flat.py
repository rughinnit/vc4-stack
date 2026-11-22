#!/usr/bin/env python3
"""check_im2col_flat.py -- the flat-panel im2col schedule against the current panel.

No GPU.  Two things are checked, for a table of shapes that includes widths
that are not multiples of 4 or 16, single-row outputs, stride 2, 5x5 and
dilation 2:

1. **The block schedule.**  `build_im2col_nhwc_flat` splits the flat output
   plane into 16-pixel blocks, four 4-pixel groups each, and walks a group
   counter with a row-wrap add.  This test re-runs that arithmetic in Python
   (the uniforms the host would pass, the same add/counter/branch sequence)
   and asserts, for every (pixel, tap, channel block): the VDR lane address
   is the pixel the flat index names, the panel destination is the panel's
   own [k][16-pixel] slot, and the VDR/VDW row fields walk four rows per
   group.  It also splits the blocks over several QPUs and checks the union
   is every real group exactly once.

2. **The values.**  It builds the activation panel the flat schedule writes
   from the padded NHWC image, builds the panel the current row-padded
   schedule writes, and compares them for every real output pixel: both must
   carry the same K values.  The ratio of the panel row counts is printed; it
   is the row padding the flat block removes.

The generator is also called for each shape to check the blob fits the
2048-word code slot and to report the instruction delta against the current
kernel.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import kgen                                                     # noqa: E402

SLOT = 2048
MASK = 0xFFFFFFFF


def check_control_words():
    """The VDR/VDW field values the kernel is built from.

    `dec_vdw` names the fields as the encoder does, which is inverted against
    the hardware (bits 23..29 are the output rows, 16..22 the words per row,
    as the proven C store's 4-row x 16-word word shows).  Assert the bits, not
    the names: four VPM rows loaded, 16 rows by 4 words stored, 48-byte gap.
    """
    import vpmwords as V
    vdr = V.vdr_basic(0, 1, numrows=4, rowlen=16)
    assert ((vdr >> 16) & 0xF) == 4, "VDR numrows"
    assert ((vdr >> 20) & 0xF) == 0, "VDR rowlen (16 is encoded as 0)"
    assert ((vdr >> 12) & 0xF) == 1, "VDR vpitch"
    vdw = V.vdw_basic(0, 4, 16, horizontal=0)
    assert ((vdw >> 23) & 0x7F) == 16, "VDW output rows"
    assert ((vdw >> 16) & 0x7F) == 4, "VDW words per row"
    assert ((vdw >> 14) & 1) == 0, "VDW must transpose"
    assert (V.vdw_stride(48) & 0xFFFF) == 48, "VDW gap"
    print("control words: VDR 4 rows x 16 words, VDW 16 rows x 4 words, "
          "transposing, 48 B gap")


def al(n, a):
    return (n + a - 1) // a * a


def geom(W, H, KH, KW, stride, pad, dil):
    effh = (KH - 1) * dil + 1
    effw = (KW - 1) * dil + 1
    wo = (W + 2 * pad - effw) // stride + 1
    ho = (H + 2 * pad - effh) // stride + 1
    return wo, ho


def padded(img, pad):
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)))


def constants(W, H, Cin, KH, KW, stride, pad, dil):
    wo, ho = geom(W, H, KH, KW, stride, pad, dil)
    Wf = al(wo, 4)
    Wg = al(wo, 16)
    GPR = Wf // 4
    M = al(ho * Wf, 16)
    K = Cin * KH * KW
    nk = al(K, 16) + 16                 # the bias block is part of the panel
    C4 = Cin * 4
    row_pitch = (W + 2 * pad) * Cin * 4
    return dict(wo=wo, ho=ho, Wf=Wf, Wg=Wg, GPR=GPR, M=M, K=K, nk=nk,
                C4=C4, C4d=C4 * dil, pitch=stride * C4,
                gadv=4 * stride * C4,
                row_pitch=row_pitch, radv=stride * row_pitch,
                rowadj=stride * row_pitch - GPR * 4 * stride * C4,
                blocks=M // 16, groups=ho * GPR)


def run_schedule(Cin, KH, KW, stride, pad, dil, g, nq):
    """Re-run the host uniforms and the kernel's group arithmetic.

    Returns {(flat_pixel, tap, cb): (src, dst, vdr_row, vdw_row, block, q, gq)}.
    """
    out = {}
    seen = np.zeros(g["groups"], dtype=int)
    per = g["blocks"] // nq
    assert per * nq == g["blocks"], "blocks %d not divisible by nq %d" % (
        g["blocks"], nq)
    for q in range(nq):
        b0 = q * per
        g0 = 4 * b0
        y0, x0g = divmod(g0, g["GPR"])
        src = (y0 * g["radv"] + x0g * 4 * stride * g["C4"]) & MASK
        gcnt = g["GPR"] - x0g
        gr = min(4 * per, g["groups"] - g0)
        gidx = g0
        for b in range(b0, b0 + per):
            dstx = b * 64 * g["nk"]
            for gq in range(4):
                if gr == 0:
                    break
                y, x4 = divmod(gidx, g["GPR"])
                x = x4 * 4
                assert src == (y * g["radv"] + x * stride * g["C4"]) & MASK, \
                    "q%d b%d g%d: group base %#x" % (q, b, gq, src)
                for s in range(KH * KW):
                    ky, kx = divmod(s, KW)
                    # u12 is the kx step and carries the dilation, like the
                    # current kernel's C4; ky steps the row pitch per dilated row
                    seg = (src + ky * dil * g["row_pitch"] + kx * g["C4d"]) & MASK
                    for cb in range(Cin // 16):
                        dst = dstx + cb * 1024 + gq * 16
                        row = 16 * q + 4 * gq
                        for lane in range(4):
                            p = gidx * 4 + lane
                            tsrc = (seg + lane * g["pitch"] + cb * 64) & MASK
                            key = (p, s, cb)
                            assert key not in out, "pixel written twice"
                            out[key] = (tsrc, dst, row, row, b, q, gq)
                seen[gidx] += 1
                src = (src + g["gadv"]) & MASK
                gcnt -= 1
                if gcnt == 0:
                    src = (src + g["rowadj"]) & MASK
                    gcnt = g["GPR"]
                gidx += 1
                gr -= 1
    assert (seen == 1).all(), "a group was not scheduled exactly once"
    want = g["ho"] * g["Wf"] * g["K"] // 16      # one entry per 16 channels
    assert len(out) == want, \
        "scheduled %d blocks, expected %d" % (len(out), want)
    return out


def flat_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g):
    p = padded(img, pad)
    panel = np.full((g["M"], g["K"]), np.nan, dtype=np.float32)
    for fp in range(g["ho"] * g["Wf"]):
        y, x = divmod(fp, g["Wf"])
        if x >= g["wo"]:
            continue                    # a pad pixel; its reads may leave the row
        for s in range(KH * KW):
            ky, kx = divmod(s, KW)
            panel[fp, s * Cin:(s + 1) * Cin] = \
                p[y * stride + ky * dil, x * stride + kx * dil, :]
    return panel


def current_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g):
    p = padded(img, pad)
    panel = np.full((g["ho"] * g["Wg"], g["K"]), np.nan, dtype=np.float32)
    for y in range(g["ho"]):
        for x in range(g["wo"]):
            row = y * g["Wg"] + x
            for s in range(KH * KW):
                ky, kx = divmod(s, KW)
                panel[row, s * Cin:(s + 1) * Cin] = \
                    p[y * stride + ky * dil, x * stride + kx * dil, :]
    return panel


def probe(Cin, KH, KW, W, H, stride, pad, dil):
    g = constants(W, H, Cin, KH, KW, stride, pad, dil)
    rng = np.random.default_rng(0xC0FFEE + W * 131 + H)
    img = rng.standard_normal((H, W, Cin)).astype(np.float32)
    p = padded(img, pad)

    nqs = [q for q in (12, 6, 4, 3, 2, 1) if g["blocks"] % q == 0]
    for nq in nqs[:2] + [1]:
        sched = run_schedule(Cin, KH, KW, stride, pad, dil, g, nq)
        for (fp, s, cb), (tsrc, dst, vr, wr, b, q, gq) in sched.items():
            y, x = divmod(fp, g["Wf"])
            ky, kx = divmod(s, KW)
            base = ((y * stride + ky * dil) * (W + 2 * pad) +
                    (x * stride + kx * dil)) * Cin * 4
            assert tsrc == (base + cb * 64) & MASK, \
                "pixel %d tap %d cb %d: lane address %#x, expected %#x" % (
                    fp, s, cb, tsrc, (base + cb * 64) & MASK)
            assert dst == b * 64 * g["nk"] + cb * 1024 + gq * 16, \
                "pixel %d tap %d cb %d: panel destination" % (fp, s, cb)
            assert vr == 16 * q + 4 * gq and wr == vr, "VPM row fields"
            assert vr + 3 <= 63, "VPM row %d out of the window" % (vr + 3)

    fp = flat_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g)
    cp = current_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g)
    for y in range(g["ho"]):
        for x in range(g["wo"]):
            if not np.array_equal(fp[y * g["Wf"] + x], cp[y * g["Wg"] + x]):
                raise AssertionError(
                    "shape %dx%dx%d k%d s%d: pixel (%d,%d) differs" %
                    (W, H, Cin, KH, stride, y, x))
    return g


def words(cin, kh, kw, flat):
    kind = kgen.build_im2col_nhwc_flat if flat else kgen.build_im2col_nhwc
    fd, path = tempfile.mkstemp(prefix="check_im2col_", suffix=".bin")
    os.close(fd)
    try:
        ws = kind(cin, kh, kw).write(path)
    finally:
        os.unlink(path)
    return len(ws)


SHAPES = [
    # (Cin, KH, KW, W, H, stride, pad, dil)
    (16, 3, 3, 160, 160, 1, 1, 1),
    (64, 3, 3, 20, 20, 1, 1, 1),        # op 32 -> 20
    (32, 3, 3, 10, 10, 1, 1, 1),        # 16 -> 12
    (16, 3, 3, 5, 5, 1, 1, 1),          # 16 -> 8, the head's narrow layers
    (16, 3, 3, 13, 7, 1, 1, 1),         # not a multiple of 4 or 16
    (16, 3, 3, 7, 13, 1, 1, 1),         # and transposed
    (16, 3, 3, 3, 1, 1, 1, 1),          # one output row
    (16, 3, 3, 4, 3, 1, 1, 1),          # ho*Wf not a multiple of 16
    (16, 3, 3, 20, 20, 2, 1, 1),        # stride 2
    (16, 3, 3, 20, 20, 1, 0, 1),        # no padding
    (16, 5, 5, 20, 20, 1, 2, 1),        # 5x5: the largest code blob
    (16, 3, 3, 20, 20, 1, 2, 2),        # dilation 2
    (32, 1, 1, 40, 40, 1, 0, 1),        # 1x1
]


def main():
    bad = 0
    check_control_words()
    print("%-26s %7s %7s %6s %7s %7s %7s" %
          ("shape", "M flat", "M row", "ratio", "words", "cur", "delta"))
    for (Cin, KH, KW, W, H, stride, pad, dil) in SHAPES:
        wc = words(Cin, KH, KW, False)
        wf = words(Cin, KH, KW, True)
        try:
            g = probe(Cin, KH, KW, W, H, stride, pad, dil)
        except AssertionError as e:
            print("FAIL %dx%dx%d k%d s%d p%d d%d: %s" %
                  (W, H, Cin, KH, stride, pad, dil, e))
            bad += 1
            continue
        if wf > SLOT:
            print("FAIL %dx%dx%d: flat blob %d words > %d" %
                  (W, H, Cin, wf, SLOT))
            bad += 1
        print("%-26s %7d %7d %6.3f %7d %7d %+7d" %
              ("%dx%dx%d k%dx%d s%d p%d d%d" % (W, H, Cin, KH, KW, stride,
                                                pad, dil),
               g["M"], g["ho"] * g["Wg"], g["M"] / (g["ho"] * g["Wg"]),
               wf, wc, wf - wc))
    print()
    print("%d shape(s), %d failure(s)" % (len(SHAPES), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
