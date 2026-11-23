#!/usr/bin/env python3
"""check_im2col_flat.py -- the flat panel's VPM rows and the panel they make.

No GPU.  The flat panel (`kgen.build_im2col_nhwc_flat2`) keeps the row
kernel's 16 x 16 transposing store with a zero stride and fills its 16 VPM
rows from four 4-row VDRs at the pixel pitch.  This test checks, for a table
of shapes that includes widths that are not multiples of 4 or 16, single-row
outputs, stride 2, 5x5 and dilation 2:

1. **The block schedule.**  The flat output plane is `ho` rows of
   `align4(wo)` pixels, split into 16-pixel blocks of four 4-pixel groups.  A
   running group counter walks a group per step and adds the row adjustment
   at the wrap.  This test re-runs that arithmetic in Python -- the uniforms
   the host would pass, the same step/wrap sequence, the same real-group
   count that ends the kernel -- and asserts, for every (pixel, tap, channel
   block), that the VDR lane address is the pixel the flat index names, that
   the group's four rows land on VPM rows `16*q + 4*g + i`, and that the four
   groups of a block cover the QPU's 16-row window exactly once.

2. **The VPM rows.**  Each VDR row is read from the padded image at the
   address the kernel computes, so the model's VPM rows are the words the
   hardware would hold: row `16*q + 4*g + i` must carry the 16 channels of
   pixel `4*g + i` of the block.  That is the same content the row-padded
   kernel (one 16-row VDR per block) puts there.

   The group addresses are accumulated the way the kernel does it (a per-tap
   step, and a 64-byte advance per channel block that the next tap's step
   carries back), so a wrong step constant is a wrong address.

3. **The values.**  The 16 x 16 transposing store is applied to the modelled
   VPM rows (memory row = channel, memory word = pixel) and the resulting
   panel is compared with the row-padded panel for every real output pixel:
   both must carry the same K values per pixel.  The ratio of the two panel
   row counts is printed; it is the row padding the flat block removes.

4. **The blob's VPM words.**  The control words are generator constants, not
   model ones, so the literals in the emitted blob are checked too: the store
   must be the row kernel's own 16 x 16 transposing word with a zero stride,
   the load must be four VPM rows, the channel-block back-step must be there,
   and neither refused shape (a 4-word store, a 48-byte store gap) may appear.

The generator is also called for each shape to check the blob fits the
2048-word code slot and to report the instruction count against the row
kernel it has to beat.
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
QPU = 4                    # the im2col launch's QPU count (16 VPM rows each)


def check_control_words():
    """The VDR/VDW field values the kernel is built from.

    The store must be the row kernel's own word: 16 k rows x 16 pixel words,
    transposing, zero memory stride, at VPM row 16*q.  The load is the only
    new shape: four pixels per group at the source pitch, four VPM rows per
    load, group g at VPM row 16*q + 4*g.
    """
    import vpmwords as V
    base = V.vdr_basic(0, 1, numrows=4, rowlen=16)
    assert ((base >> 16) & 0xF) == 4, "VDR numrows"
    assert ((base >> 20) & 0xF) == 0, "VDR rowlen (16 is encoded as 0)"
    assert ((base >> 12) & 0xF) == 1, "VDR vpitch"
    for q in range(QPU):
        for g in range(4):
            d = V.dec_vdr(base + (q << 8) + 64 * g)
            assert d["row"] == 16 * q + 4 * g, \
                "VDR row field %d, expected %d" % (d["row"], 16 * q + 4 * g)
            assert d["numrows"] == 4 and d["vpitch"] == 1
    w = V.vdw_basic(0, 16, 16, horizontal=0)
    d = V.dec_vdw(w)
    assert d["units"] == 16, "VDW output rows"
    assert d["depth"] == 16, "VDW words per row"
    assert d["horizontal"] == 0, "VDW must transpose"
    assert V.vdw_stride(0) == 0xC0010000, "VDW gap / block mode"
    for q in range(QPU):
        assert V.dec_vdw(w + (q << 11))["row"] == 16 * q
    print("control words: VDR 4 rows x 16 words at row 16q+4g, "
          "VDW 16 rows x 16 words transposing, 0 stride (the row kernel's)")


def blob_immediates(words_):
    """Every 32-bit literal the blob loads (the VPM control words among them)."""
    return {w & 0xFFFFFFFF for w in words_ if ((w >> 60) & 0xF) == 14}


def check_blob_words():
    """The blob must carry the row kernel's store and only the 4-row load.

    The control-word constants are built in the generator, not in the model,
    so pin the ones that failed on silicon: the transposing 16 x 16 store with
    a zero stride must be the same literal the row kernel loads, the load must
    be four VPM rows at 16 words, and neither refused field combination (a
    4-word-wide store, or a 48-byte store gap) may appear.
    """
    import vpmwords as V
    import tempfile

    def emitted(kind, cin, kh, kw):
        fd, path = tempfile.mkstemp(prefix="check_words_", suffix=".bin")
        os.close(fd)
        try:
            return kind(cin, kh, kw).write(path)
        finally:
            os.unlink(path)

    def immediates(kind, cin, kh, kw):
        return blob_immediates(emitted(kind, cin, kh, kw))

    def dst_advances(words_):
        """Words that add the 1024-byte k-block step to a register.

        The destination's step is the one register the generator owns that the
        schedule model cannot see: it must appear once per tap (inside the
        channel loop, where each iteration runs it), or every tap's tile lands
        on the block's first k rows and the rest of the panel stays unwritten.
        """
        reg = None
        for w in words_:
            if ((w >> 60) & 0xF) == 14 and (w & 0xFFFFFFFF) == 1024:
                reg = (w >> 38) & 0x3F
        assert reg is not None, "no 1024-byte k-block step in the blob"
        n = 0
        for w in words_:
            if ((w >> 60) & 0xF) in (14, 15):
                continue                    # literal or branch, no ALU
            if ((w >> 24) & 0x1F) != 12:    # ADD_ADD
                continue
            if ((w >> 12) & 0x3F) == reg and ((w >> 6) & 7) == 7:  # breg
                n += 1
        return n

    store = V.vdw_basic(0, 16, 16, horizontal=0)
    for cin, kh, kw in ((16, 3, 3), (16, 5, 5), (64, 3, 3)):
        flat = immediates(kgen.build_im2col_nhwc_flat2, cin, kh, kw)
        row = immediates(kgen.build_im2col_nhwc, cin, kh, kw)
        assert store in flat and store in row, \
            "store word %#x: flat2 %s, row %s" % (
                store, store in flat, store in row)
        assert V.vdw_stride(0) in flat and V.vdw_stride(0) in row
        assert V.vdr_basic(0, 1, numrows=4, rowlen=16) in flat, \
            "the flat load must be four VPM rows"
        assert (V.vdr_stride(0) & ~0xFFF) in flat
        for refused in (V.vdw_basic(0, 16, 4, horizontal=0), V.vdw_stride(48)):
            assert refused not in flat, \
                "refused VPM word %#x in the blob" % refused
        if cin // 16 > 1:
            # the channel loop leaves the group address one block on, so the
            # tap step has to carry Cin*4 back; the constant is a literal
            assert cin * 4 in flat, \
                "no %d-byte channel-block back-step literal (Cin %d)" % (
                    cin * 4, cin)
        nadv = dst_advances(emitted(kgen.build_im2col_nhwc_flat2, cin, kh, kw))
        assert nadv == kh * kw, \
            "%dx%d k%d: %d destination advances, expected one per tap (%d)" % (
                cin, kh, kw, nadv, kh * kw)
    print("blob words: store %#x is the row kernel's 16 x 16 transposing "
          "store with a zero stride, the load is four VPM rows, and neither "
          "refused shape (4-word store, 48 B gap) appears" % store)


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
    GPR = Wf // 4
    M = al(ho * Wf, 16)
    K = Cin * KH * KW
    nk = al(K, 16) + 16                 # the bias block is part of the panel
    C4 = Cin * 4
    row_pitch = (W + 2 * pad) * Cin * 4
    return dict(wo=wo, ho=ho, Wf=Wf, Wg=al(wo, 16), GPR=GPR, M=M, K=K, nk=nk,
                dst_step=64 * nk,
                C4=C4, C4d=C4 * dil, pitch=stride * C4,
                gadv=4 * stride * C4,
                row_pitch=row_pitch, radv=stride * row_pitch,
                rowadj=stride * row_pitch - GPR * 4 * stride * C4,
                blocks=M // 16, groups=ho * GPR)


def kstep(src, gcnt, g):
    """One group advance: the kernel's step, wrap add and counter reload."""
    src = (src + g["gadv"]) & MASK
    gcnt -= 1
    if gcnt == 0:
        src = (src + g["rowadj"]) & MASK
        gcnt = g["GPR"]
    return src, gcnt


def walk(nq, g):
    """The four group sources of every block, in the kernel's order.

    The cursor is the block's first group.  Three steps reach the block's
    four groups and a fourth reaches the next block's first group, so a block
    still advances the walk four groups; the real-group count ends the QPU
    once a block leaves none.
    """
    per = g["blocks"] // nq
    assert per * nq == g["blocks"], "blocks %d not divisible by nq %d" % (
        g["blocks"], nq)
    out = []
    for q in range(nq):
        b0 = q * per
        g0 = 4 * b0
        y0, x0g = divmod(g0, g["GPR"])
        cur = (y0 * g["radv"] + x0g * g["gadv"]) & MASK
        gcnt = g["GPR"] - x0g
        gr = min(4 * per, g["groups"] - g0)
        blocks = []
        for b in range(b0, b0 + per):
            src = [cur]
            for _ in range(3):
                s, gcnt = kstep(src[-1], gcnt, g)
                src.append(s)
            blocks.append((b, g0 + 4 * (b - b0), src))
            cur, gcnt = kstep(src[-1], gcnt, g)
            gr -= 4
            if gr <= 0:
                break
        assert len(blocks) == per, "q%d ran %d of %d blocks" % (
            q, len(blocks), per)
        out.append((q, blocks))
    seen = [gi for _, bl in out for _, gi0, _ in bl for gi in
            range(gi0, gi0 + 4)]
    assert len(seen) == len(set(seen)), "a group was scheduled twice"
    real = sorted(gi for gi in seen if gi < g["groups"])
    assert real == list(range(g["groups"])), \
        "real groups covered %d..%d of %d" % (
            len(real), real[-1] if real else -1, g["groups"])
    return out


def flat_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g, nq):
    """The panel the flat kernel writes, built from its VPM rows.

    Every VDR row is read from the padded image at the byte address the
    kernel computes, so the 16 x 16 tile below is the VPM content, and the
    store is the hardware's transpose: memory row = channel, words = pixels.

    The four group addresses are *accumulated* the way the kernel does it -- a
    per-tap step (the pixel step within a kernel row, the dilated row step
    less the KW-1 pixels already covered at a new one) and a 64-byte advance
    per channel block that the next tap's step has to carry back -- so a wrong
    step constant or a missing back-step shows up as a wrong address, not as
    the same wrong offset on both sides.  The destination is accumulated the
    same way, one k block per (tap, channel block), because it is the other
    register the generator owns: a missing advance would land every tap's tile
    on the block's first 1024 bytes and leave the rest of the panel unwritten.

    The memory layout is the GEMM's k-major A: word (block, k, pixel) sits at
    ``block*16*nk + k*16 + pixel`` floats, i.e. k rows of 64 bytes.
    """
    p = padded(img, pad)
    buf = p.astype(np.float32).tobytes()
    nlim = len(buf)
    out = np.full(g["blocks"] * 16 * g["nk"], np.nan, dtype=np.float32)
    ncb = Cin // 16
    adj = Cin * 4 if ncb > 1 else 0
    stx = g["C4d"] - adj
    sty = dil * g["row_pitch"] - (KW - 1) * g["C4d"] - adj
    for q, blocks in walk(nq, g):
        for (b, gidx0, src) in blocks:
            qs = list(src)
            dst = b * g["dst_step"] // 4       # float index of k row 0
            for s in range(KH * KW):
                ky, kx = divmod(s, KW)
                if s:
                    step = stx if kx else sty
                    qs = [(v + step) & MASK for v in qs]
                for cb in range(ncb):
                    vpm = np.full((16, 16), np.nan, dtype=np.float32)
                    vrow = set()
                    for k in range(4):
                        base = qs[k]
                        if ncb > 1:      # the channel loop's own advance
                            qs[k] = (qs[k] + 64) & MASK
                        for i in range(4):
                            a = (base + i * g["pitch"]) & MASK
                            row = 16 * q + 4 * k + i
                            assert row not in vrow, "VPM row %d twice" % row
                            vrow.add(row)
                            assert 16 * q <= row < 16 * q + 16
                            # the lane the flat index names
                            fp = gidx0 * 4 + 4 * k + i
                            y, x = divmod(fp, g["Wf"])
                            real = y < g["ho"] and x < g["wo"]
                            if real:
                                want = (((y * stride + ky * dil) *
                                         (W + 2 * pad) +
                                         (x * stride + kx * dil)) * Cin * 4
                                        + cb * 64) & MASK
                                assert a == want, \
                                    "block %d tap %d cb %d g%d i%d: lane " \
                                    "address %#x, expected %#x" % (
                                        b, s, cb, k, i, a, want)
                            if a + 64 > nlim:
                                continue           # a pad lane past the plane
                            vpm[4 * k + i] = np.frombuffer(
                                buf, dtype="<f4", count=16, offset=a)
                    assert vrow == set(range(16 * q, 16 * q + 16)), \
                        "block %d tap %d cb %d: VPM rows %s" % (
                            b, s, cb, sorted(vrow))
                    # the 16 x 16 transposing store: k row w is a memory
                    # row of 16 pixel words, and the k index must be the one
                    # the destination register holds
                    assert dst == b * g["dst_step"] // 4 + \
                        (s * Cin + cb * 16) * 16, \
                        "block %d tap %d cb %d: destination %d, expected %d" \
                        % (b, s, cb, dst,
                           b * g["dst_step"] // 4 + (s * Cin + cb * 16) * 16)
                    out[dst:dst + 256] = vpm.T.reshape(-1)
                    dst += 256
    # back to the logical [M][K] panel the GEMM's rows name
    return out.reshape(g["blocks"], g["nk"], 16).transpose(0, 2, 1) \
              .reshape(g["blocks"] * 16, g["nk"])[:, :g["K"]]


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

    nqs = [q for q in (QPU, 3, 2, 1) if g["blocks"] % q == 0]
    for nq in nqs[:2] + [1]:
        fp = flat_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g, nq)
        cp = current_panel(img, W, H, Cin, KH, KW, stride, pad, dil, g)
        for y in range(g["ho"]):
            for x in range(g["wo"]):
                a = fp[y * g["Wf"] + x]
                b = cp[y * g["Wg"] + x]
                if not np.array_equal(a, b):
                    bad = np.flatnonzero(a != b)
                    raise AssertionError(
                        "shape %dx%dx%d k%d s%d p%d d%d nq%d: pixel (%d,%d) "
                        "differs at k %s (%r vs %r)" % (
                            W, H, Cin, KH, stride, pad, dil, nq, y, x,
                            bad[:4].tolist(), a[bad[0]], b[bad[0]]))
    return g


def words(cin, kh, kw, kind):
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
    (16, 3, 3, 20, 20, 1, 1, 1),        # op 16 -> 20, GPR 5
    (16, 3, 3, 60, 60, 1, 1, 1),        # op 60 -> 60, 225 blocks
]


def main():
    bad = 0
    check_control_words()
    check_blob_words()
    print("%-24s %7s %7s %6s %7s %7s %+7s" %
          ("shape", "M flat", "M row", "ratio", "w flat2", "w row", "delta"))
    tot = 0
    for (Cin, KH, KW, W, H, stride, pad, dil) in SHAPES:
        wf = words(Cin, KH, KW, kgen.build_im2col_nhwc_flat2)
        wc = words(Cin, KH, KW, kgen.build_im2col_nhwc)
        try:
            g = probe(Cin, KH, KW, W, H, stride, pad, dil)
        except AssertionError as e:
            print("FAIL %dx%dx%d k%d s%d p%d d%d: %s" %
                  (W, H, Cin, KH, stride, pad, dil, e))
            bad += 1
            continue
        if wf > SLOT:
            print("FAIL %dx%dx%d: flat2 blob %d words > %d" %
                  (W, H, Cin, wf, SLOT))
            bad += 1
            continue
        tot += 1
        print("%-24s %7d %7d %6.3f %7d %7d %+7d" %
              ("%dx%dx%d k%dx%d s%d p%d d%d" % (W, H, Cin, KH, KW, stride,
                                                pad, dil),
               g["M"], g["ho"] * g["Wg"],
               g["M"] / (g["ho"] * g["Wg"]), wf, wc, wf - wc))
    print()
    print("rows per 16-pixel block: 16 panel rows, one 16x16 store "
          "(four 4-row VDRs, 16 VPM rows %d*q..%d*q+15)" % (16, 16))
    print("%d shape(s), %d failure(s)" % (len(SHAPES), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
