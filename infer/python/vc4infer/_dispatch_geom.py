"""Geometry / detection family handlers (docs/OPS-GEOMETRY.md).

This is the third family dispatch table for ``_onnxrun.Runner``.  It covers
the operators a detection or segmentation graph needs beyond convolution:

    ConvTranspose, Col2Im, AffineGrid, GridSample, RoiAlign, MaxRoiPool,
    NonMaxSuppression, Det, Einsum, DeformConv

Every handler is a composition of element-verified kernels -- ``vc4_copy2d_x``
(every layout move and every indexed gather), ``vc4_elementwise`` (add/mul and
the exact comparisons/selects), ``vc4_unary`` (floor/ceil/absolute and the
affine combination), ``vc4_matmul`` (the contraction and the reduction) and
``vc4_fill2d`` (a zero scatter buffer / a padded panel).  ``infer/src/`` is
untouched; ``vc4_gather`` stays gated.

The ARM never computes an activation value.  It performs exactly the class of
work the existing families already allow:

  * integer index maps -- ONNX's own coordinate / block / tap arithmetic on
    *shapes and attributes*, never on tensor values (Col2Im, ConvTranspose,
    AffineGrid, Einsum, the parameter layouts of RoiAlign/MaxRoiPool);
  * a read-back of *index* data that the QPU produced -- the floor of a grid
    coordinate, the floor of a ROI sample, the rank of a score -- which is the
    same class of read-back ``ArgMax``, ``TopK`` and ``Unique`` perform;
  * constants (normalised coordinate planes, tap offsets, masks) uploaded once
    per shape and reused.

Everything derived from *tensor* values -- the fractional weights, the
reflection fold, the IoU, the score rank, the sampling coordinates -- is
computed on the QPUs.

Two structural points:

* a *scatter* with overlapping destinations cannot be a QPU store (two stores
  to one word are a race), so Col2Im/ConvTranspose accumulate as one zeroed
  tap buffer + one vector add per tap -- the same fixed tap order
  ``docs/NUMERICS.md`` records for the C col2im kernel;
* a gather with *computed* addresses is one ``vc4_copy2d_x`` per 16 words
  whose source step is constant, and a single word otherwise.  Every gather
  below batches its index map into constant-step runs first, so the launch
  count is O(words/16) in the best case and O(words) for a fully random map.
  The counts in docs/OPS-GEOMETRY.md are measured, not estimated.
"""
import numpy as np
import onnx

from . import _binding as vc4infer
from ._errors import UnsupportedNode
from ._dispatch_extra import _Base, _ReduceMixin, _attr, _prod


def _iwords(name, t):
    """A constant int64/int32 index tensor as a python list of ints."""
    return [int(v) for v in np.asarray(t).ravel()]


def _align(n, a):
    return (int(n) + a - 1) // a * a


def _ceildiv(a, b):
    return -((-int(a)) // int(b))


# The daemon validates every *literal* uniform word against the windows a
# device address can name: >= 0xC0000000 (the GPU bus), the V3D register block
# at 0x3FC00000, and the arena's physical range.  A float parameter is passed
# as its bit pattern, so 1.5f (0x3FC00000) is exactly V3D_PHYS_BASE and a
# unary launch carrying it is *denied* (measured: every GridSample case whose
# size gives (dim-1)/2 == 1.5 failed with "unary: device I/O error (LAUNCH:
# denied by daemon policy)").  A parameter in one of those windows therefore
# goes in as an uploaded constant and a separate add instead of a uniform.
_FORBIDDEN_BITS = ((0xC0000000, 0x100000000), (0x3FC00000, 0x3FC40000))


def _bits(v):
    return int(np.float32(v).view(np.int32)) & 0xFFFFFFFF


def _risky_uniform(v):
    b = _bits(v)
    return any(lo <= b < hi for lo, hi in _FORBIDDEN_BITS)


class _Geom(_Base):
    """Shared geometry primitives: views, strided stores, tap scatter, gathers."""

    # ------------------------------------------------------------- buffers
    def view(self, t, off_words=0, words=None):
        """A Buf covering ``words`` words of ``t`` starting at ``off_words``."""
        b = t.buf if hasattr(t, "buf") else t
        off = int(off_words) * 4
        nb = (int(words) * 4) if words is not None else (b.bytes - off)
        if nb <= 0:
            raise UnsupportedNode(self.node.name if self.node else "",
                                  self.node.op_type if self.node else "",
                                  "empty buffer view")
        return vc4infer.Buf(b.off + off, nb)

    def view_t(self, t, off_words, shape):
        """A Tensor over a sub-range of another tensor's buffer.

        Deliberately *not* registered as garbage: ``Pool.release`` works on the
        arena offset of whatever it is handed, so releasing a view would free a
        block in the middle of its parent and the next allocation would hand
        out memory that is still live.  The parent owns the storage and the
        view lives inside one node.
        """
        n = _prod(shape)
        return _Tensor()(self.view(t, off_words, n), tuple(shape))

    def zero(self, shape):
        """A fully zeroed activation buffer (one fill2d launch)."""
        return self.alloc_fill(shape, 0.0)

    # ------------------------------------------------------------- copies
    def stride_take(self, src, dst, n, step, off=0, doff=0):
        """dst[doff + i] = src[off + i*step] for ``n`` words (laned gather)."""
        n = int(n)
        if n > 0:
            self._laned_run(src.buf, dst.buf, int(off), int(doff), n, int(step))

    def stride_store(self, src, dst, n, step, off=0, doff=0):
        """dst[doff + i*step] = src[off + i] -- one-word stores, 16 per vector.

        ``vc4_copy2d_x`` with ``nlane=1, depth=1`` writes one word per vector
        and advances the destination by ``dvec``; that is the only shape in the
        library that writes a *strided* destination, and it is what puts an
        interleaved (x, y) grid together without a host arithmetic pass.
        """
        n = int(n)
        if n <= 0:
            return
        full = n // 16
        if full:
            self.r._x(self.view(src, off), self.view(dst, doff), full, 16,
                      64, 4 * step * 16, 4, 4 * step, 4, 1, 1, 64, 64, 1, 1)
        rem = n % 16
        if rem:
            self.r._x(self.view(src, off + full * 16),
                      self.view(dst, doff + full * 16 * step), 1, rem, 0, 0,
                      4, 4 * step, 4, 1, 1, 64, 64, 1, 1)

    # -------------------------------------------------------------- tap scatter
    def tap_rect(self, n_blocks, out_n, stride, off):
        """Block indices whose position ``i*stride + off`` is inside [0, out_n)."""
        if out_n <= 0:
            return 0, -1
        i0 = max(0, _ceildiv(-int(off), int(stride)))
        i1 = min(int(n_blocks) - 1, (int(out_n) - 1 - int(off)) // int(stride))
        return i0, i1

    def col_scatter(self, out, col, src_base, row_step, tap_step, plane_step,
                    dst_base, planes, kh, kw, nb_h, nb_w, H, W, sh, sw, dh,
                    dw, ph, pw):
        """out[plane][y][x] += col[tap][iy][ix], one tap at a time.

        A copy2d store is 16 *contiguous* words, so an interleaved destination
        (lane step ``sw`` > 1) is written with ``depth=1`` one-word stores --
        one launch per (plane, tap), with the tap's block rows spread over the
        QPUs.  The accumulation is a fresh zeroed tap buffer plus one vector
        add, in fixed (ky, kx) order (docs/NUMERICS.md).
        """
        n = int(out.size)
        acc = out
        for t in range(int(kh) * int(kw)):
            ky, kx = divmod(t, int(kw))
            oy = ky * int(dh) - int(ph)
            ox = kx * int(dw) - int(pw)
            i0, i1 = self.tap_rect(nb_h, H, sh, oy)
            j0, j1 = self.tap_rect(nb_w, W, sw, ox)
            if i1 < i0 or j1 < j0:
                continue
            z = self.zero((n,))
            sy = i0 * sh + oy
            sx = j0 * sw + ox
            for p in range(int(planes)):
                self.r._x(
                    self.view(col, src_base + p * plane_step + t * tap_step +
                              i0 * row_step + j0),
                    self.view(z, dst_base + p * H * W + sy * W + sx),
                    i1 - i0 + 1, j1 - j0 + 1, 4 * row_step, 4 * sh * W,
                    4, 4 * sw, 4, 1, 1, 64, 64, 1, 1)
            acc = self.add(acc, z)
        return self.tag(acc, tuple(out.shape))

    # -------------------------------------------------------------- gathers
    def gather_words(self, src, maps, total, shape=None):
        """One zeroed buffer per index map (a -1 entry stays zero)."""
        outs = []
        for m in maps:
            b = self.zero(shape or (total,))
            self.word_gather(src.buf, b.buf, m, total)
            outs.append(b)
        return outs

    def float_words(self, t, shape):
        """A QPU-produced float buffer that holds integers -> int64 indices.

        The values are exact integers (they come from UOP_FLOOR/CEIL), so the
        conversion is a representation change, not arithmetic: the same class
        of read-back ``ArgMax`` performs on its int32 scan result.
        """
        a = np.asarray(self.r.download(t), np.float32).reshape(shape)
        return np.rint(a).astype(np.int64)

    def tile_plane(self, src, dst, planes, n, doff=0):
        """dst[doff + p*n ...] = src[0:n] for every p (a broadcast row copy)."""
        if planes > 0 and n > 0:
            self.lane_runs(self.view(src), self.view(dst, doff), int(planes),
                           int(n), 0, 4 * int(n), 4)

    def affine_t(self, a, p0, p1, shape=None):
        """affine with a uniform-parameter safety net (see _risky_uniform)."""
        if not _risky_uniform(p0) and not _risky_uniform(p1):
            return self.affine(a, p0, p1, shape)
        t = self.mul(a, self.const_like(("g_aff", p0), a, p0))
        return self.add(t, self.const_like(("g_off", p1), a, p1), shape)

    def maxmin2(self, a, b, want_max=True):
        return _Base.maxmin(self, a, b, want_max)

    # -------------------------------------------------------------- matmul
    def gemm_rect(self, a, m, k, b, n, a_const_padded=None):
        """[m][k] x [k][n] -> [m][n] with both operands padded on the QPUs.

        ``vc4_matmul`` pads a non-aligned operand by copying it *with the ARM*
        over the cacheable arena mapping, which docs/OPS-NORM.md measures as a
        hard failure over a QPU-produced operand.  Here the padding is a QPU
        lane copy into a zeroed panel, so an operand the QPU just wrote is
        safe.  ``a_const_padded`` is an already-aligned constant A (the weight
        matrix), used to skip the copy.
        """
        mp, kp, np_ = _align(m, 8), _align(k, 16), _align(n, 64)
        a_buf = a_const_padded
        if a_buf is None:
            a_buf = self.zero((mp, kp))
            if m > 0 and k > 0:
                self.r._runs(a.buf, a_buf.buf, m, k, 4 * k, 4 * kp)
        b_buf = self.zero((kp, np_))
        if k > 0 and n > 0:
            self.r._runs(b.buf, b_buf.buf, k, n, 4 * n, 4 * np_)
        c = self.alloc((mp, np_))
        self.r.eng.matmul(a_buf.buf if hasattr(a_buf, "buf") else a_buf,
                          b_buf.buf, c.buf, mp, np_, kp)
        out = self.alloc((m, n))
        if m > 0 and n > 0:
            self.r._runs(c.buf, out.buf, m, n, 4 * np_, 4 * n)
        return out

    def permute(self, t, in_shape, perm, out_shape):
        """Arbitrary axis permutation as one flat index map (QPU lane copies)."""
        in_shape = [int(d) for d in in_shape]
        perm = list(perm)
        if perm == list(range(len(perm))):
            return self.tag(t, out_shape)
        idx = _perm_index(in_shape, perm)
        out = self.alloc(out_shape)
        self.word_gather(t.buf, out.buf, idx, _prod(out_shape))
        return out


def _perm_index(in_shape, perm):
    """Flat source index per flat destination index for an axis permutation."""
    n = _prod([in_shape[p] for p in perm])
    strides = [0] * len(in_shape)
    acc = 1
    for k in range(len(in_shape) - 1, -1, -1):
        strides[k] = acc
        acc *= in_shape[k]
    coords = np.unravel_index(np.arange(n, dtype=np.int64),
                              tuple(in_shape[p] for p in perm))
    idx = np.zeros(n, np.int64)
    for k, p in enumerate(perm):
        idx += coords[k] * strides[p]
    return idx


def _Tensor():
    from ._onnxrun import Tensor
    return Tensor


# =====================================================================
# Col2Im / ConvTranspose: the scatter family
# =====================================================================
class Col2ImOp(_Geom):
    """Col2Im: k^2 zeroed tap buffers, one vector add each, fixed tap order.

    ONNX defines the operator as an accumulation of overlapping blocks, and two
    QPU stores to one word are a race, so the composition is
    ``out += scatter(tap)`` with one zeroed scatter buffer and one elementwise
    add per tap -- the same fixed (ky, kx) order ``docs/NUMERICS.md`` records
    for the engine's C col2im.  A tap's destination positions form a
    *sublattice* (stride ``sw``/``sh``), which is exactly a copy2d lane stride.
    """

    def run(self):
        node = self.node
        data = self.ten(node.input[0])
        image = _iwords(node.input[1], self.host_or_fail(node.input[1]))
        block = _iwords(node.input[2], self.host_or_fail(node.input[2]))
        if len(image) != 4:
            self.fail("image_shape must be 4-D [N, C, H, W] (got %d)"
                      % len(image))
        if len(block) != 2:
            self.fail("block_shape must be 2-D (got %d)" % len(block))
        N, C, H, W = (int(v) for v in image)
        kh, kw = (int(v) for v in block)
        dil = [int(v) for v in _attr(node, "dilations", [1, 1])]
        pads = [int(v) for v in _attr(node, "pads", [0, 0, 0, 0])]
        st = [int(v) for v in _attr(node, "strides", [1, 1])]
        if len(dil) != 2 or len(pads) != 4 or len(st) != 2:
            self.fail("dilations/pads/strides must be 2/4/2 long")
        dh, dw = dil
        sh, sw = st
        ph, pw, ph2, pw2 = pads[0], pads[1], pads[2], pads[3]
        nb_h = (H + ph + ph2 - dh * (kh - 1) - 1) // sh + 1
        nb_w = (W + pw + pw2 - dw * (kw - 1) - 1) // sw + 1
        if nb_h < 1 or nb_w < 1:
            self.fail("pads/kernel/stride give %dx%d sliding blocks" %
                      (nb_h, nb_w))
        shp = tuple(int(d) for d in data.shape)
        if len(shp) != 3:
            self.fail("input must be [N, C*kh*kw, L] (got %s)" % (shp,))
        if shp[0] != N or shp[1] != C * kh * kw:
            self.fail("input %s does not match image_shape %s and block %s"
                      % (shp, (N, C, H, W), (kh, kw)))
        L = shp[2]
        if L != nb_h * nb_w:
            self.fail("L=%d but image_shape/block_shape imply %d blocks"
                      % (L, nb_h * nb_w))
        out = self.zero((N, C, H, W))
        for n in range(N):
            out = self.col_scatter(
                out, data, n * C * kh * kw * L, nb_w, L, kh * kw * L,
                n * C * H * W, C, kh, kw, nb_h, nb_w, H, W, sh, sw, dh, dw,
                ph, pw)
        return out


class ConvTransposeOp(_Geom):
    """ConvTranspose: one W^T GEMM per (batch, group), then the same scatter.

    Method (stated as the gate contract asks): **col2im + GEMM**, not a zero-insertion
    forward conv.  ``cols = W^T @ X`` has one row per (output channel, tap) and
    exactly the operator's own MAC count
    ``N*Cin*Hin*Win*Cout/g*kh*kw``; a zero-insertion forward conv would run
    ``N*Cout*Hout*Wout*Cin*kh*kw`` MACs, i.e. ``sh*sw`` times as many
    (4x for stride 2, 9x for stride 3) over a mostly-zero input.  The scatter
    is then the same k^2-tap accumulate Col2Im uses, so the two operators share
    one verified path.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        shape = tuple(int(d) for d in x.shape)
        if len(shape) != 4:
            self.fail("only 4-D NCHW (got %s)" % (shape,))
        N, Cin, Hin, Win = shape
        wsrc = self.r.host.get(node.input[1]) if len(node.input) > 1 else None
        if wsrc is None:
            self.fail("the weight must be a graph constant (its transpose is "
                      "the GEMM operand; a runtime weight would need a QPU "
                      "transpose of a constant-layout matrix)")
        w = np.asarray(wsrc, np.float32)
        if w.ndim != 4:
            self.fail("weight must be 4-D [Cin, Cout/g, kh, kw]")
        Cin_w, Coutg, kh, kw = (int(d) for d in w.shape)
        group = int(_attr(node, "group", 1))
        if Cin_w != Cin or Cin % group or Coutg * group < 1:
            self.fail("weight %s does not match X %s with group=%d"
                      % (w.shape, shape, group))
        Cout = Coutg * group
        st = [int(v) for v in _attr(node, "strides", [1, 1])]
        dil = [int(v) for v in _attr(node, "dilations", [1, 1])]
        sh, sw = st
        dh, dw = dil
        opad = [int(v) for v in _attr(node, "output_padding", [0, 0])]
        oh_pad, ow_pad = opad[0], opad[1]
        auto = _attr(node, "auto_pad", "NOTSET")
        pads = _attr(node, "pads", None)
        oshape = _attr(node, "output_shape", None)
        if pads is not None and len(pads) != 4:
            self.fail("pads must be 4 long")
        kshape = _attr(node, "kernel_shape", None)
        if kshape is not None and [int(v) for v in kshape] != [kh, kw]:
            self.fail("kernel_shape %s != weight %s" % (kshape, (kh, kw)))
        if pads is None and auto not in ("SAME_UPPER", "SAME_LOWER"):
            pads = [0, 0, 0, 0]
        if pads is None:
            if oshape is None:
                Hout, Wout = Hin * sh, Win * sw
            else:
                Hout, Wout = int(oshape[0]), int(oshape[1])
            tot_h = sh * (Hin - 1) + oh_pad + (kh - 1) * dh + 1 - Hout
            tot_w = sw * (Win - 1) + ow_pad + (kw - 1) * dw + 1 - Wout
            if auto == "SAME_UPPER":
                p1h, p1w = tot_h // 2, tot_w // 2
            else:
                p1h, p1w = tot_h - tot_h // 2, tot_w - tot_w // 2
            pads = [p1h, p1w, tot_h - p1h, tot_w - p1w]
        else:
            pads = [int(v) for v in pads]
            if oshape is not None:
                Hout, Wout = int(oshape[0]), int(oshape[1])
            else:
                Hout = sh * (Hin - 1) + oh_pad + (kh - 1) * dh + 1 \
                    - pads[0] - pads[2]
                Wout = sw * (Win - 1) + ow_pad + (kw - 1) * dw + 1 \
                    - pads[1] - pads[3]
        if Hout < 1 or Wout < 1:
            self.fail("output %dx%d is empty" % (Hout, Wout))
        ph, pw = pads[0], pads[1]
        Cin_g = Cin // group
        L = Hin * Win
        mp, kp, np_ = _align(Coutg * kh * kw, 8), _align(Cin_g, 16), \
            _align(L, 64)
        out = self.zero((N, Cout, Hout, Wout))
        for g in range(group):
            # [Coutg*kh*kw][Cin_g] from the group's weight block, transposed:
            # a *constant* layout change (the weight is a graph constant)
            wm = np.zeros((mp, kp), np.float32)
            for c in range(Coutg):
                # ONNX ConvTranspose: W is [C_in, C_out/group, kh, kw]; group g
                # owns input channels [g*Cin_g, (g+1)*Cin_g) and output
                # channels [g*Coutg, (g+1)*Coutg)
                blk = w[g * Cin_g:(g + 1) * Cin_g, c, :, :] \
                    .reshape(Cin_g, kh * kw)
                wm[c * kh * kw:(c + 1) * kh * kw, :Cin_g] = blk.T
            a = self.const(("convt_w", id(node), g), wm)
            for n in range(N):
                b = self.zero((kp, np_))
                self.r._runs(self.view(x, (n * Cin + g * Cin_g) * L),
                             b.buf, Cin_g, L, 4 * L, 4 * np_)
                c = self.alloc((mp, np_))
                self.r.eng.matmul(a.buf, b.buf, c.buf, mp, np_, kp)
                out = self.col_scatter(
                    out, c, 0, Win, np_, kh * kw * np_,
                    (n * Cout + g * Coutg) * Hout * Wout, Coutg, kh, kw,
                    Hin, Win, Hout, Wout, sh, sw, dh, dw, ph, pw)
        if len(node.input) > 2 and node.input[2]:
            bvec = self.ten(node.input[2])
            if int(bvec.shape[0]) != Cout:
                self.fail("bias length %d != Cout %d"
                          % (int(bvec.shape[0]), Cout))
            full = self.zero((N, Cout, Hout * Wout))
            for n in range(N):
                self.lane_runs(bvec.buf, self.view(full, n * Cout * Hout *
                                                   Wout), Cout, Hout * Wout,
                               4, 4 * Hout * Wout, 0)
            out = self.add(out, full, (N, Cout, Hout, Wout))
        return out


# =====================================================================
# AffineGrid
# =====================================================================
class AffineGridOp(_Geom):
    """AffineGrid: two scalar combinations per batch, then an interleaved store.

    ``x = t00*xs + t01*ys + t02`` (and the y row) is a theta broadcast, two
    multiplies and two adds per output coordinate plane; the *only* unusual
    part is the output layout ``[N][H][W][2]``, which a one-word strided store
    writes directly (``stride_store``), so the grid never touches the ARM.
    """

    def run(self):
        node = self.node
        theta = self.ten(node.input[0])
        size = _iwords(node.input[1], self.host_or_fail(node.input[1]))
        if len(size) != 4:
            self.fail("only 4-D sizes (got %d)" % len(size))
        N, C, H, W = (int(v) for v in size)
        if int(theta.shape[0]) != N or _prod(theta.shape[1:]) != 6:
            self.fail("theta must be [N, 2, 3] (got %s)" % (theta.shape,))
        ac = int(_attr(node, "align_corners", 0))
        if H < 1 or W < 1:
            self.fail("empty grid")
        xs = np.empty(W, np.float64)
        ys = np.empty(H, np.float64)
        if ac:
            xs[:] = 2.0 * np.arange(W) / (W - 1) if W > 1 else 0.0
            xs -= 1.0 if W > 1 else 0.0
            ys[:] = 2.0 * np.arange(H) / (H - 1) if H > 1 else 0.0
            ys -= 1.0 if H > 1 else 0.0
        else:
            xs[:] = (2.0 * np.arange(W) + 1.0) / W - 1.0
            ys[:] = (2.0 * np.arange(H) + 1.0) / H - 1.0
        xc = self.const(("affine_x", H, W, ac), np.tile(xs, H).astype(np.float32))
        yc = self.const(("affine_y", H, W, ac), np.repeat(ys, W).astype(np.float32))
        P = H * W
        out = self.alloc((N, H, W, 2))
        for n in range(N):
            bx = [self.alloc((P,)) for _ in range(6)]
            for k in range(6):
                # a scalar read from the theta tensor is a lane copy, never a
                # host value: the ARM never sees it
                self.lane_runs(self.view(theta, n * 6 + k), bx[k].buf, 1, P,
                               0, 0, 0)
            gx = self.add(self.mul(bx[0], xc), self.mul(bx[1], yc))
            gx = self.add(gx, bx[2])
            gy = self.add(self.mul(bx[3], xc), self.mul(bx[4], yc))
            gy = self.add(gy, bx[5])
            self.stride_store(gx.buf, out.buf, P, 2, 0, n * 2 * P)
            self.stride_store(gy.buf, out.buf, P, 2, 0, n * 2 * P + 1)
        return out


# =====================================================================
# GridSample
# =====================================================================
class GridSampleOp(_Geom):
    """GridSample: QPU denormalise/reflect/floor, index read-back, 4 tap copies.

    The gather has computed addresses, and ``vc4_gather`` is still gated, so
    every tap is one ``vc4_copy2d_x`` per constant-step run of its index map
    (one per 16 words when the grid is locally regular).  The *coordinates*,
    the reflection fold and the four bilinear weights are QPU arithmetic; the
    ARM only turns the QPU's integral coordinates into addresses.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        grid = self.ten(node.input[1])
        if len(x.shape) != 4:
            self.fail("only 4-D input (got %s)" % (x.shape,))
        N, C, H, W = (int(d) for d in x.shape)
        if len(grid.shape) != 4 or int(grid.shape[0]) != N or \
                int(grid.shape[3]) != 2:
            self.fail("grid must be [N, Ho, Wo, 2] (got %s)" % (grid.shape,))
        Ho, Wo = int(grid.shape[1]), int(grid.shape[2])
        mode = str(_attr(node, "mode", "linear"))
        if mode == "bilinear":
            mode = "linear"
        if mode not in ("linear", "nearest"):
            self.fail("mode=%r is refused (the engine composes bilinear and "
                      "nearest; cubic needs a 4x4 polynomial per axis)" % mode)
        pad = str(_attr(node, "padding_mode", "zeros"))
        if pad not in ("zeros", "border", "reflection"):
            self.fail("padding_mode=%r" % pad)
        ac = int(_attr(node, "align_corners", 0))
        nmode = _attr(node, "nearest_mode", None)
        P = Ho * Wo
        if P == 0 or C == 0 or N == 0:
            return self.alloc((N, C, Ho, Wo))
        gx = self.alloc((N, P))
        gy = self.alloc((N, P))
        for n in range(N):
            self.stride_take(grid, gx, P, 2, n * 2 * P, n * P)
            self.stride_take(grid, gy, P, 2, n * 2 * P + 1, n * P)
        # denormalise: align_corners -> (g+1)/2*(d-1), else ((g+1)*d-1)/2
        if ac:
            ax_ = (W - 1) / 2.0 if W > 1 else 0.0
            ay_ = (H - 1) / 2.0 if H > 1 else 0.0
        else:
            ax_, ay_ = W / 2.0, H / 2.0
        bx_, by_ = (W - 1) / 2.0, (H - 1) / 2.0
        xc = self.affine_t(gx, ax_, bx_)
        yc = self.affine_t(gy, ay_, by_)
        if pad == "border":
            lo = self.const(("gs_lo", N, P, 0.0), np.zeros((N, P), np.float32))
            xhi = self.const(("gs_hi_w", W), np.full((N, P), W - 1.0, np.float32))
            yhi = self.const(("gs_hi_h", H), np.full((N, P), H - 1.0, np.float32))
            xc = self.maxmin2(self.maxmin2(xc, lo), xhi, False)
            yc = self.maxmin2(self.maxmin2(yc, lo), yhi, False)
        elif pad == "reflection":
            xc = self._reflect(xc, N, P, 0.0 if ac else -0.5,
                               (W - 1.0) if ac else (W - 0.5), "rx")
            yc = self._reflect(yc, N, P, 0.0 if ac else -0.5,
                               (H - 1.0) if ac else (H - 0.5), "ry")
        if mode == "nearest":
            xr = self._round_mode(xc, nmode)
            yr = self._round_mode(yc, nmode)
            yi = self.float_words(yr, (N, P))
            xi = self.float_words(xr, (N, P))
            maps = self._tap_maps(yi, xi, N, C, H, W, P, [(0, 0)], pad)
            taps = self.gather_words(x, maps, N * C * P, (N, C, P))
            return self.tag(taps[0], (N, C, Ho, Wo))
        xf = self.unary(xc, vc4infer.UOP_FLOOR)
        yf = self.unary(yc, vc4infer.UOP_FLOOR)
        fx = self.sub(xc, xf)
        fy = self.sub(yc, yf)
        yi = self.float_words(yf, (N, P))
        xi = self.float_words(xf, (N, P))
        maps = self._tap_maps(yi, xi, N, C, H, W, P,
                              [(0, 0), (0, 1), (1, 0), (1, 1)], pad)
        t00, t01, t10, t11 = self.gather_words(x, maps, N * C * P, (N, C, P))
        hx = self.affine(fx, -1.0, 1.0)
        hy = self.affine(fy, -1.0, 1.0)
        w = [self.mul(hx, hy), self.mul(fx, hy), self.mul(hx, fy),
             self.mul(fx, fy)]
        wf = []
        for k in range(4):
            full = self.alloc((N, C, P))
            for n in range(N):
                self.tile_plane(self.view(w[k], n * P), full, C, P,
                                n * C * P)
            wf.append(full)
        acc = self.mul(wf[0], t00)
        acc = self.add(acc, self.mul(wf[1], t01))
        acc = self.add(acc, self.mul(wf[2], t10))
        acc = self.add(acc, self.mul(wf[3], t11))
        return self.tag(acc, (N, C, Ho, Wo))

    def _round_mode(self, t, nmode):
        """The nearest index: round-to-nearest-even unless nearest_mode says."""
        if nmode is None:
            # the GridSample schema this engine targets (onnx 1.17, opset 22)
            # has no nearest_mode, and onnx.reference / torch round to even
            return self.unary(t, vc4infer.UOP_ROUND)
        if nmode == "floor":
            return self.unary(t, vc4infer.UOP_FLOOR)
        if nmode == "ceil":
            return self.unary(t, vc4infer.UOP_CEIL)
        if nmode == "round_prefer_ceil":
            return self.unary(self.affine(t, 1.0, 0.5), vc4infer.UOP_FLOOR)
        if nmode == "round_prefer_floor":
            return self.unary(self.affine(t, 1.0, -0.5), vc4infer.UOP_CEIL)
        self.fail("nearest_mode=%r" % nmode)

    def _reflect(self, t, N, P, lo, hi, key):
        """ONNX/Torch reflection fold, entirely on the QPUs.

        ``n = floor(dx / rng)``, ``r = dx - n*rng`` and the parity of n selects
        ``lo + r`` or ``hi - r``; parity is an exact bit comparison of the
        integer-valued n.
        """
        lof = self.const((key, "lo", lo), np.full((N, P), lo, np.float32))
        hif = self.const((key, "hi", hi), np.full((N, P), hi, np.float32))
        inv = self.const((key, "inv", hi - lo),
                         np.full((N, P), 1.0 / (hi - lo), np.float32))
        below = self.cmp(vc4infer.OP_CMP_LT, t, lof)
        above = self.cmp(vc4infer.OP_CMP_GT, t, hif)
        res = t
        for mask, base, sgn in ((below, lof, 1.0), (above, hif, -1.0)):
            dx0 = self.sub(base, t) if sgn > 0 else self.sub(t, base)
            q = self.mul(dx0, inv)
            n = self.unary(q, vc4infer.UOP_FLOOR)
            r = self.sub(dx0, self.mul(n, self.constx(N, P, hi - lo, key)))
            two = self.constx(N, P, 2.0, key)
            par = self.sub(n, self.mul(two, self.unary(
                self.mul(n, self.constx(N, P, 0.5, key)), vc4infer.UOP_FLOOR)))
            even = self.cmp(vc4infer.OP_CMP_EQ, par,
                            self.constx(N, P, 0.0, key))
            if sgn > 0:
                v = self.select(even, self.add(lof, r), self.sub(hif, r))
            else:
                v = self.select(even, self.sub(hif, r), self.add(lof, r))
            res = self.select(mask, v, res)
        return res

    def constx(self, N, P, v, key):
        return self.const((key, "c", v), np.full((N, P), v, np.float32))

    def _tap_maps(self, yi, xi, N, C, H, W, P, taps, pad):
        """Flat word maps for the taps; -1 leaves the zeroed buffer alone."""
        maps = []
        for dy, dx in taps:
            y = yi + dy
            x = xi + dx
            if pad == "zeros":
                ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
            else:
                ok = np.ones_like(y, bool)
            y = np.clip(y, 0, H - 1)
            x = np.clip(x, 0, W - 1)
            # [N][C][P]: the same plane map replicated over the channels
            plane = (y * W + x).reshape(N, 1, P)
            base = (np.arange(N, dtype=np.int64) * C * H * W).reshape(N, 1, 1)
            m = base + np.arange(C, dtype=np.int64).reshape(1, C, 1) * H * W \
                + plane
            if pad == "zeros":
                m = np.where(ok.reshape(N, 1, P), m, -1)
            maps.append(m.ravel())
        return maps


# =====================================================================
# RoiAlign / MaxRoiPool
# =====================================================================
class RoiAlignOp(_Geom, _ReduceMixin):
    """RoiAlign: the sampling coordinates are QPU arithmetic, the taps a gather.

    Per ROI the four scalars (start_w, start_h, bin_w, bin_h) come from the ROI
    tensor with QPU multiplies; every sample position is
    ``start + bin * (p + (s + 0.5)/sr)`` with a constant inner factor, so one
    broadcast plus one multiply and one add per axis places all
    ``oh*ow*sr*sr`` samples.  The bilinear taps are then the same four
    ``copy2d`` gathers GridSample uses, the weights are QPU products, and the
    bin reduction is one GEMM (avg) or one exact max chain (max).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 4-D NCHW (got %s)" % (x.shape,))
        N, C, H, W = (int(d) for d in x.shape)
        rois = self.ten(node.input[1])
        R = int(rois.shape[0])
        if int(rois.shape[1]) != 4:
            self.fail("rois must be [num_rois, 4] (got %s)" % (rois.shape,))
        bidx = np.asarray(self.host_or_fail(node.input[2]), np.int64).ravel()
        if bidx.size != R:
            self.fail("batch_indices has %d entries for %d rois"
                      % (bidx.size, R))
        if R == 0:
            return self.alloc((0, C, 1, 1))
        if bidx.min() < 0 or bidx.max() >= N:
            self.fail("batch index %d outside [0, %d)"
                      % (int(bidx.max()), N))
        oh = int(_attr(node, "output_height", 1))
        ow = int(_attr(node, "output_width", 1))
        sr = int(_attr(node, "sampling_ratio", 0))
        scale = float(_attr(node, "spatial_scale", 1.0))
        mode = str(_attr(node, "mode", "avg")).lower()
        ctm = str(_attr(node, "coordinate_transformation_mode",
                        "half_pixel")).lower()
        if ctm not in ("half_pixel", "output_half_pixel"):
            self.fail("coordinate_transformation_mode=%r" % ctm)
        if mode not in ("avg", "max"):
            self.fail("mode=%r" % mode)
        if oh < 1 or ow < 1:
            self.fail("output %dx%d is empty" % (oh, ow))
        if sr < 0:
            self.fail("sampling_ratio=%d" % sr)
        if sr == 0:
            # adaptive: ceil(roi_height/oh) samples per bin, a per-ROI count
            self.fail("sampling_ratio=0 (adaptive) needs a data-dependent "
                      "sample count per bin; give a positive sampling_ratio")
        S = oh * ow * sr * sr
        half = ctm == "half_pixel"
        off = 0.5 if half else 0.0
        # ---- ROI scalars, QPU
        x1 = self.alloc((R,)); y1 = self.alloc((R,))
        x2 = self.alloc((R,)); y2 = self.alloc((R,))
        self.stride_take(rois, x1, R, 4, 0)
        self.stride_take(rois, y1, R, 4, 1)
        self.stride_take(rois, x2, R, 4, 2)
        self.stride_take(rois, y2, R, 4, 3)
        sw = self.affine_t(x1, scale, -off)
        sh_ = self.affine_t(y1, scale, -off)
        ew = self.affine_t(x2, scale, -off)
        eh = self.affine_t(y2, scale, -off)
        rw = self.sub(ew, sw)
        rh = self.sub(eh, sh_)
        if not half:
            one = self.const(("roi_one", R), np.ones(R, np.float32))
            rw = self.maxmin2(rw, one)
            rh = self.maxmin2(rh, one)
        bw = self.affine(rw, 1.0 / ow, 0.0)
        bh = self.affine(rh, 1.0 / oh, 0.0)
        # ---- sample coordinates
        cy, cx = _roi_factors(oh, ow, sr)
        cw = self.const(("roi_cw", oh, ow, sr), cx)
        ch = self.const(("roi_ch", oh, ow, sr), cy)
        xs = self.alloc((R, S))
        ys = self.alloc((R, S))
        for r in range(R):
            for scalar, base, dst, cc in ((bw, sw, xs, cw), (bh, sh_, ys, ch)):
                t = self.alloc((S,))
                self.lane_runs(self.view(scalar, r), t.buf, 1, S, 0, 0, 0)
                b = self.alloc((S,))
                self.lane_runs(self.view(base, r), b.buf, 1, S, 0, 0, 0)
                acc = self.add(self.mul(t, cc), b)
                self.tile_plane(acc, dst, 1, S, r * S)
        # ---- clamp + guard, QPU
        zero = self.const(("roi_zero", R, S), np.zeros((R, S), np.float32))
        xhi = self.const(("roi_xhi", W), np.full((R, S), W - 1.0, np.float32))
        yhi = self.const(("roi_yhi", H), np.full((R, S), H - 1.0, np.float32))
        # the reference zeroes a sample whose *unclamped* coordinate is outside
        # [-1, dim]; that is a range test on the coordinate, so it is a QPU
        # comparison and the 0/1 result multiplies the four weights
        xlo_ok = self.cmp(vc4infer.OP_CMP_GE, xs,
                          self.const(("roi_m1w", R * S),
                                     np.full((R, S), -1.0, np.float32)))
        ylo_ok = self.cmp(vc4infer.OP_CMP_GE, ys,
                          self.const(("roi_m1h", R * S),
                                     np.full((R, S), -1.0, np.float32)))
        xhi_ok = self.cmp(vc4infer.OP_CMP_LE, xs,
                          self.const(("roi_W", R * S),
                                     np.full((R, S), float(W), np.float32)))
        yhi_ok = self.cmp(vc4infer.OP_CMP_LE, ys,
                          self.const(("roi_H", R * S),
                                     np.full((R, S), float(H), np.float32)))
        inside = self.mul(self.mul(xlo_ok, ylo_ok), self.mul(xhi_ok, yhi_ok))
        xc = self.maxmin2(self.maxmin2(xs, zero), xhi, False)
        yc = self.maxmin2(self.maxmin2(ys, zero), yhi, False)
        xf = self.unary(xc, vc4infer.UOP_FLOOR)
        yf = self.unary(yc, vc4infer.UOP_FLOOR)
        lx = self.sub(xc, xf)
        ly = self.sub(yc, yf)
        xi = self.float_words(xf, (R, S))
        yi = self.float_words(yf, (R, S))
        base = ((bidx.reshape(R, 1, 1) * C +
                 np.arange(C).reshape(1, C, 1)) * H * W)
        maps = []
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            yy = np.clip(yi + dy, 0, H - 1).reshape(R, 1, S)
            xx = np.clip(xi + dx, 0, W - 1).reshape(R, 1, S)
            maps.append((base + yy * W + xx).ravel())
        taps = self.gather_words(x, maps, R * C * S, (R, C, S))
        hx = self.affine(lx, -1.0, 1.0)
        hy = self.affine(ly, -1.0, 1.0)
        w = [self.mul(hx, hy), self.mul(lx, hy), self.mul(hx, ly),
             self.mul(lx, ly)]
        wf = []
        for k in range(4):
            full = self.alloc((R, C, S))
            for r in range(R):
                self.tile_plane(self.view(w[k], r * S), full, C, S, r * C * S)
            wf.append(full)
        infull = self.alloc((R, C, S))
        for r in range(R):
            self.tile_plane(self.view(inside, r * S), infull, C, S,
                            r * C * S)
        for k in range(4):
            wf[k] = self.mul(wf[k], infull)
        acc = self.mul(wf[0], taps[0])
        acc = self.add(acc, self.mul(wf[1], taps[1]))
        acc = self.add(acc, self.mul(wf[2], taps[2]))
        acc = self.add(acc, self.mul(wf[3], taps[3]))
        rows = R * C * oh * ow
        D = sr * sr
        if mode == "avg":
            red = self.sum_last_aligned(acc, rows, D)
            red = self.affine(red, 1.0 / D, 0.0)
        else:
            red = self.max_last(self.tag(acc, (rows, D)), rows, D, "max")
        return self.tag(red, (R, C, oh, ow))


def _roi_factors(oh, ow, sr):
    """The constant per-sample factors (y, x), in [oh][ow][sr][sr] order.

    ``y = start_h + bin_h*(ph + (sy+0.5)/sr)`` and
    ``x = start_w + bin_w*(pw + (sx+0.5)/sr)``: the only sample-dependent
    quantity in each coordinate is an inner factor, so a broadcast and one
    multiply/add place every sample on the QPU.
    """
    ph, pw, sy, sx = np.indices((oh, ow, sr, sr))
    cy = (ph + (sy + 0.5) / sr).astype(np.float32)
    cx = (pw + (sx + 0.5) / sr).astype(np.float32)
    return np.ascontiguousarray(cy).ravel(), np.ascontiguousarray(cx).ravel()


class MaxRoiPoolOp(_Geom, _ReduceMixin):
    """MaxRoiPool: integer ROI bins on the QPU, one window copy + a max chain.

    torchvision's geometry is ``round(x*scale)`` with ties away from zero, then
    ``floor``/``ceil`` of the bin edges; all of it is QPU arithmetic on the ROI
    tensor, and the ARM only turns the resulting integral edges into the
    window map.  A window is one ``copy2d`` per channel plane and the max is
    the exact comparison chain (a NaN operand is carried like numpy's).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 4-D NCHW (got %s)" % (x.shape,))
        N, C, H, W = (int(d) for d in x.shape)
        rois = self.ten(node.input[1])
        R = int(rois.shape[0])
        if int(rois.shape[1]) != 5:
            self.fail("rois must be [num_rois, 5] (got %s)" % (rois.shape,))
        pooled = _attr(node, "pooled_shape", None)
        if pooled is None or len(pooled) != 2:
            self.fail("pooled_shape must be 2-D")
        ph, pw = int(pooled[0]), int(pooled[1])
        scale = float(_attr(node, "spatial_scale", 1.0))
        if R == 0 or ph < 1 or pw < 1:
            return self.alloc((R, C, ph, pw))
        bi = self.alloc((R,)); x1 = self.alloc((R,)); y1 = self.alloc((R,))
        x2 = self.alloc((R,)); y2 = self.alloc((R,))
        for dst, off in ((bi, 0), (x1, 1), (y1, 2), (x2, 3), (y2, 4)):
            self.stride_take(rois, dst, R, 5, off)
        biv = np.rint(np.asarray(self.r.download(self.affine(
            bi, 1.0, 0.0)), np.float32)).astype(np.int64)
        if biv.size and (biv.min() < 0 or biv.max() >= N):
            self.fail("batch index %d outside [0, %d)"
                      % (int(biv.max()), N))
        r1 = self._round_away(x1, R, scale)
        r2 = self._round_away(x2, R, scale)
        r3 = self._round_away(y1, R, scale)
        r4 = self._round_away(y2, R, scale)
        lox, hix = self._bins(r1, r2, pw)
        loy, hiy = self._bins(r3, r4, ph)
        out = self.alloc((R, C, ph, pw))
        for r in range(R):
            for i in range(ph):
                h0 = max(0, min(H, int(loy[r * ph + i])))
                h1 = max(0, min(H, int(hiy[r * ph + i])))
                hh = max(h1 - h0, 1)
                for j in range(pw):
                    w0 = max(0, min(W, int(lox[r * pw + j])))
                    w1 = max(0, min(W, int(hix[r * pw + j])))
                    ww = max(w1 - w0, 1)
                    win = self.alloc((C, hh, ww))
                    for c in range(C):
                        self.r._runs(self.view(x, ((int(biv[r]) * C + c) * H
                                                   + h0) * W + w0),
                                     self.view(win, c * hh * ww), hh, ww,
                                     4 * W, 4 * ww)
                    col = self.max_last(self.tag(win, (C * hh, ww)),
                                        C * hh, ww, "max")
                    val = self.max_last(self.tag(col, (C, hh)), C, hh, "max")
                    self.stride_store(val, out, C, ph * pw, 0,
                                      (r * C) * ph * pw + i * pw + j)
        return out

    def _bins(self, first, last, nbin):
        """floor(i*bin + start) / ceil((i+1)*bin + start) per (roi, bin)."""
        height = self.maxmin2(self.add(self.sub(last, first),
                                      self.const(("mr_one", int(first.size)),
                                                 np.ones(int(first.size),
                                                         np.float32))),
                             self.const(("mr_one2", int(first.size)),
                                        np.ones(int(first.size), np.float32)))
        binv = self.affine_t(height, 1.0 / nbin, 0.0)
        R = int(first.size)
        lo = self.alloc((R * nbin,))
        hi = self.alloc((R * nbin,))
        for r in range(R):
            t = self.const(("mr_t", nbin), np.arange(nbin, dtype=np.float32))
            t1 = self.const(("mr_t1", nbin),
                            (np.arange(nbin, dtype=np.float32) + 1.0))
            b = self.alloc((nbin,))
            self.lane_runs(self.view(binv, r), b.buf, 1, nbin, 0, 0, 0)
            st = self.alloc((nbin,))
            self.lane_runs(self.view(first, r), st.buf, 1, nbin, 0, 0, 0)
            self.r._runs(self.unary(self.add(self.mul(b, t), st),
                                    vc4infer.UOP_FLOOR).buf,
                         self.view(lo, r * nbin), 1, nbin, 0, 0)
            self.r._runs(self.unary(self.add(self.mul(b, t1), st),
                                    vc4infer.UOP_CEIL).buf,
                         self.view(hi, r * nbin), 1, nbin, 0, 0)
        return (self.float_words(lo, (R, nbin)).reshape(-1),
                self.float_words(hi, (R, nbin)).reshape(-1))

    def _round_away(self, t, n, scale):
        """round(x*scale) with ties away from zero, on the QPU."""
        a = self.affine_t(t, scale, 0.0)
        m = self.unary(a, vc4infer.UOP_ABS)
        f = self.unary(self.affine(m, 1.0, 0.5), vc4infer.UOP_FLOOR)
        s = self.unary(a, vc4infer.UOP_SIGN)
        return self.mul(s, f)


# =====================================================================
# Det
# =====================================================================
class DetOp(_Geom):
    """Det: the Leibniz expansion, one constant-stride gather per term.

    For an ``M x M`` matrix ``det = sum_pi sign(pi) prod_i a[i][pi(i)]``.  Each
    permutation's diagonal is one laned gather (the batch stride is constant,
    so one launch per factor) and the terms accumulate in two exact multiply/
    add chains, positive and negative, so the fp32 result is a
    deterministic fixed-order sum.  Cost is ``M! * 2M`` launches: 6 permutations
    for a 3x3, 24 for a 4x4 -- which is why ``M > 4`` is refused by name.
    """
    MAX_M = 4

    def run(self):
        import itertools
        node = self.node
        x = self.ten(node.input[0])
        shape = tuple(int(d) for d in x.shape)
        if len(shape) < 2 or shape[-1] != shape[-2]:
            self.fail("the last two dims must be square (got %s)" % (shape,))
        M = int(shape[-1])
        if M < 1:
            self.fail("empty matrix")
        if M == 1:
            return self.tag(x, shape[:-2])
        if M > self.MAX_M:
            self.fail("%dx%d needs %d! = %d permutation terms (%d launches); "
                      "the expansion is bounded at %d"
                      % (M, M, M, _fact(M), _fact(M) * 2 * M, self.MAX_M))
        batch = _prod(shape[:-2]) or 1
        pos = None
        neg = None
        for perm in itertools.permutations(range(M)):
            term = None
            for i in range(M):
                t = self.alloc((batch,))
                self.stride_take(x, t, batch, M * M, i * M + perm[i])
                term = t if term is None else self.mul(term, t)
            if _parity(perm):
                neg = term if neg is None else self.add(neg, term)
            else:
                pos = term if pos is None else self.add(pos, term)
        if neg is None:
            res = pos
        elif pos is None:
            res = self.affine(neg, -1.0, 0.0)
        else:
            res = self.sub(pos, neg)
        return self.tag(res, shape[:-2])


def _fact(n):
    v = 1
    for k in range(2, n + 1):
        v *= k
    return v


def _parity(perm):
    """1 for an odd permutation (a negative term)."""
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return inv & 1


# =====================================================================
# Einsum
# =====================================================================
class EinsumOp(_Geom):
    """Einsum lowered to permutations + one GEMM per batch, not a new kernel.

    An equation with at most two operands and no repeated label inside one
    operand is exactly a batched contraction:
    ``out[batch, freeA, freeB] = A[batch, freeA, K] @ B[batch, K, freeB]``.
    Labels that appear in one operand and nowhere else are summed out first
    (one GEMM against a ones column), permutations are flat index maps, and the
    contraction is ``vc4_matmul`` with both operands padded on the QPUs.  Any
    other equation shape is refused by name: an inner diagonal, an ellipsis, or
    three or more operands.
    """

    def run(self):
        node = self.node
        eq = str(_attr(node, "equation", "") or "").replace(" ", "")
        if not eq:
            self.fail("equation is missing")
        if "..." in eq:
            self.fail("equation %r uses an ellipsis; the lowering needs "
                      "explicit labels" % eq)
        if "->" in eq:
            lhs, out = eq.split("->")
        else:
            lhs, out = eq, None
        terms = lhs.split(",")
        if len(terms) > 2:
            self.fail("equation %r has %d operands; only one or two are "
                      "lowered" % (eq, len(terms)))
        for t in terms:
            if not t or any(not c.isalpha() for c in t):
                self.fail("bad label in term %r" % t)
            if len(set(t)) != len(t):
                self.fail("term %r repeats a label (a diagonal); the "
                          "lowering is a contraction, not a trace" % t)
        ins = [self.ten(node.input[i]) for i in range(len(terms))]
        shapes = [tuple(int(d) for d in t.shape) for t in ins]
        dims = {}
        for t, sh in zip(terms, shapes):
            if len(t) != len(sh):
                self.fail("term %r has %d labels for shape %s"
                          % (t, len(t), sh))
            for c, d in zip(t, sh):
                if c in dims and dims[c] != int(d):
                    self.fail("label %r is %d and %d" % (c, dims[c], int(d)))
                dims[c] = int(d)
        if out is None:
            cnt = {}
            for t in terms:
                for c in t:
                    cnt[c] = cnt.get(c, 0) + 1
            out = "".join(sorted(c for c, n in cnt.items() if n == 1))
        for c in out:
            if c not in dims:
                self.fail("output label %r does not appear in the inputs" % c)
        if len(set(out)) != len(out):
            self.fail("output %r repeats a label" % out)
        if len(terms) == 1:
            x, term = self._reduce_out(ins[0], shapes[0], terms[0],
                                       set(out))
            res = self.permute(x, list(x.shape),
                               [list(term).index(c) for c in out],
                               [dims[c] for c in out]) if term != list(out) \
                else x
            return res
        return self._contract(terms, ins, shapes, dims, out)

    def _reduce_out(self, x, shape, term, keep):
        """Sum every axis whose label is not in ``keep``."""
        shape = list(shape)
        term = list(term)
        for i in range(len(term) - 1, -1, -1):
            if term[i] in keep:
                continue
            T, P, D, I = self.ax_last(self.tag(x, tuple(shape)), i)
            s = self.sum_last_aligned(T, P * I, D)
            del shape[i]
            del term[i]
            x = self.tag(s, tuple(shape))
        return x, term

    def _contract(self, terms, ins, shapes, dims, out):
        a, b = terms
        sa, sb = list(shapes[0]), list(shapes[1])
        # 1. sum the labels that appear in one operand only and not in the out
        ia = self._reduce_out(ins[0], sa, a, set(out) | set(b))
        A0, sa = ia[0], list(ia[0].shape)
        a = "".join(ia[1])
        ib = self._reduce_out(ins[1], sb, b, set(out) | set(a))
        B0, sb = ib[0], list(ib[0].shape)
        b = "".join(ib[1])
        ins = [A0, B0]
        batch = [c for c in a if c in b and c in out]
        freeA = [c for c in a if c not in b]
        freeB = [c for c in b if c not in a]
        cont = [c for c in a if c in b and c not in out]
        for c in freeA + freeB:
            if c not in out:
                self.fail("label %r survives the contraction but is not in "
                          "the output" % c)
        if set(batch) | set(freeA) | set(freeB) != set(out):
            self.fail("the lowering covers %r but the output is %r"
                      % ("".join(batch + freeA + freeB), out))
        if cont:
            A = self._perm_to(a, ins[0], batch + freeA + cont)
            B = self._perm_to(b, ins[1], batch + cont + freeB)
        else:
            # no contracted label: an outer product is a K=1 contraction, so
            # the free-axis layout is already the matmul operand ([M][1] and
            # [1][N] share the [M] / [N] memory) and no ones column is needed
            A = self._perm_to(a, ins[0], batch + freeA)
            B = self._perm_to(b, ins[1], batch + freeB)
        Bn = _prod([dims[c] for c in batch]) or 1
        M = _prod([dims[c] for c in freeA]) or 1
        K = _prod([dims[c] for c in cont]) if cont else 1
        Nn = _prod([dims[c] for c in freeB]) or 1
        if M * K * Nn == 0:
            self.fail("an empty axis in the contraction")
        oshape = [dims[c] for c in out]
        res = self.alloc((Bn, M, Nn))
        for t in range(Bn):
            at = self.view_t(A, t * M * K, (M, K))
            bt = self.view_t(B, t * K * Nn, (K, Nn))
            c = self.gemm_rect(at, M, K, bt, Nn)
            self.r._runs(c.buf, self.view(res, t * M * Nn), M, Nn,
                         4 * Nn, 4 * Nn)
        cur = batch + freeA + freeB
        res = self.tag(res, tuple(dims[c] for c in cur))
        if cur == list(out):
            return res
        perm = [cur.index(c) for c in out]
        return self.permute(res, [dims[c] for c in cur], perm, oshape)

    def _perm_to(self, term, t, order):
        """Transpose ``t`` (labels ``term``) into the label order ``order``."""
        shape = list(t.shape)
        if list(term) == list(order):
            return t
        perm = [list(term).index(c) for c in order]
        return self.permute(t, shape, perm, [shape[p] for p in perm])


# =====================================================================
# DeformConv
# =====================================================================
class DeformConvOp(_Geom):
    """DeformConv: offset sampling is a bilinear im2col, then one GEMM.

    ``sample = bilinear(X, p0 + offset)`` per (offset group, tap) is the same
    four-tap QPU gather GridSample uses, with the offsets added to a constant
    base grid on the QPUs; the mask is one multiply, and the (Cin*kh*kw) x
    (Cout) contraction is one ``vc4_matmul`` per (batch, group).  Its MAC count
    is the operator's own ``N*Cout*Ho*Wo*Cin*kh*kw`` (plus the 4-tap
    interpolation, 4x that in tap reads).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 4-D NCHW (got %s)" % (x.shape,))
        N, Cin, H, W = (int(d) for d in x.shape)
        wsrc = self.r.host.get(node.input[1]) if len(node.input) > 1 else None
        if wsrc is None:
            self.fail("the weight must be a graph constant (its GEMM layout "
                      "is built on the host; it is constant data)")
        w = np.asarray(wsrc, np.float32)
        if w.ndim != 4:
            self.fail("weight must be 4-D [Cout, Cin/g, kh, kw]")
        Cout, Cing, kh, kw = (int(d) for d in w.shape)
        group = int(_attr(node, "group", 1))
        og = int(_attr(node, "offset_group", 1))
        dil = [int(v) for v in _attr(node, "dilations", [1, 1])]
        st = [int(v) for v in _attr(node, "strides", [1, 1])]
        pads = [int(v) for v in _attr(node, "pads", [0, 0, 0, 0])]
        dh, dw = dil
        sh, sw = st
        ph, pw = pads[0], pads[1]
        if Cin != Cing * group or Cout % group:
            self.fail("weight %s is inconsistent with group=%d for Cin=%d"
                      % (w.shape, group, Cin))
        if Cin % og:
            self.fail("Cin=%d is not divisible by offset_group=%d"
                      % (Cin, og))
        off = self.ten(node.input[2])
        if len(off.shape) != 4 or int(off.shape[0]) != N or \
                int(off.shape[1]) != og * kh * kw * 2:
            self.fail("offset must be [N, offset_group*kh*kw*2, Ho, Wo] "
                      "(got %s, offset_group=%d, kernel %dx%d)"
                      % (off.shape, og, kh, kw))
        oh, ow = int(off.shape[2]), int(off.shape[3])
        P = oh * ow
        mask = None
        if len(node.input) > 4 and node.input[4]:
            mask = self.ten(node.input[4])
            if int(mask.shape[1]) != og * kh * kw or \
                    int(mask.shape[2]) != oh or int(mask.shape[3]) != ow:
                self.fail("mask must be [N, offset_group*kh*kw, Ho, Wo] "
                          "(got %s)" % (mask.shape,))
        K = Cin * kh * kw
        mp, kp, np_ = _align(Cout, 8), _align(K, 16), _align(P, 64)
        icg, ocg = Cing, Cout // group
        icg_off = Cin // og
        cols = self.zero((N, K, np_))
        for o in range(og):
            for t in range(kh * kw):
                ky, kx = divmod(t, kw)
                cy = self.const(("dc_by", ky, oh, ow, ph, sh, dh),
                                _dc_base(oh, ow, -ph + ky * dh, sh, True))
                cx = self.const(("dc_bx", kx, oh, ow, pw, sw, dw),
                                _dc_base(oh, ow, -pw + kx * dw, sw, False))
                ch = (o * kh * kw + t) * 2
                mplane = None
                if mask is not None:
                    mplane = []
                for n in range(N):
                    ys = self.add(self.plane(off, n, ch, oh, ow), cy)
                    xs = self.add(self.plane(off, n, ch + 1, oh, ow), cx)
                    yf = self.unary(ys, vc4infer.UOP_FLOOR)
                    xf = self.unary(xs, vc4infer.UOP_FLOOR)
                    ly = self.sub(ys, yf)
                    lx = self.sub(xs, xf)
                    yy = np.rint(np.asarray(self.r.download(yf),
                                            np.float32)).astype(np.int64)
                    xx = np.rint(np.asarray(self.r.download(xf),
                                            np.float32)).astype(np.int64)
                    hy = self.affine(ly, -1.0, 1.0)
                    hx = self.affine(lx, -1.0, 1.0)
                    wt = [self.mul(hx, hy), self.mul(lx, hy),
                          self.mul(hx, ly), self.mul(lx, ly)]
                    idx = []
                    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        ty = yy + dy
                        tx = xx + dx
                        ok = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
                        m = np.clip(ty, 0, H - 1) * W + \
                            np.clip(tx, 0, W - 1)
                        idx.append(np.where(ok, m, -1))
                    if mask is not None:
                        mplane.append(self.plane(mask, n,
                                                 o * kh * kw + t, oh, ow))
                    for ic in range(o * icg_off, (o + 1) * icg_off):
                        base = (n * Cin + ic) * H * W
                        taps = []
                        for k in range(4):
                            b = self.zero((P,))
                            src = np.where(idx[k] >= 0, base + idx[k], -1)
                            self.word_gather(x.buf, b.buf, src, P)
                            taps.append(b)
                        acc = self.mul(wt[0], taps[0])
                        acc = self.add(acc, self.mul(wt[1], taps[1]))
                        acc = self.add(acc, self.mul(wt[2], taps[2]))
                        acc = self.add(acc, self.mul(wt[3], taps[3]))
                        if mask is not None:
                            acc = self.mul(acc, mplane[n])
                        self.r._runs(acc.buf, self.view(
                            cols, n * K * np_ + (ic * kh * kw + t) * np_),
                            1, P, 0, 0)
        # ---- the contraction, per batch and group
        out = self.alloc((N, Cout, oh, ow))
        for g in range(group):
            wm = np.zeros((mp, kp), np.float32)
            wm[:ocg, :icg * kh * kw] = w[g * ocg:(g + 1) * ocg,
                                          :, :, :].reshape(ocg, icg * kh * kw)
            a = self.const(("dc_w", id(node), g), wm)
            for n in range(N):
                bmat = self.zero((kp, np_))
                self.r._runs(self.view(cols, n * K * np_ +
                                       g * icg * kh * kw * np_),
                             bmat.buf, icg * kh * kw, P, 4 * np_, 4 * np_)
                c = self.alloc((mp, np_))
                self.r.eng.matmul(a.buf, bmat.buf, c.buf, mp, np_, kp)
                self.r._runs(c.buf, self.view(out, (n * Cout + g * ocg) * P),
                             ocg, P, 4 * np_, 4 * P)
        if len(node.input) > 3 and node.input[3]:
            bv = self.ten(node.input[3])
            if int(bv.shape[0]) != Cout:
                self.fail("bias length %d != Cout %d"
                          % (int(bv.shape[0]), Cout))
            full = self.zero((N, Cout, P))
            for n in range(N):
                self.lane_runs(bv.buf, self.view(full, n * Cout * P), Cout,
                               P, 4, 4 * P, 0)
            out = self.add(out, full, (N, Cout, oh, ow))
        return out

    def plane(self, t, n, ch, h, w):
        """Channel ``ch`` of batch ``n`` as a compact [h*w] tensor."""
        P = h * w
        return self.view_t(t, (n * t.shape[1] + ch) * P, (P,))


def _dc_base(h, w, start, step, vertical):
    """The constant sampling base grid for one kernel tap."""
    if vertical:
        v = (start + step * np.arange(h)).astype(np.float32)
        return np.repeat(v, w).astype(np.float32)
    v = (start + step * np.arange(w)).astype(np.float32)
    return np.tile(v, h).astype(np.float32)


# =====================================================================
# NonMaxSuppression
# =====================================================================
class NonMaxSuppressionOp(_Geom, _ReduceMixin):
    """NonMaxSuppression: ranks and the suppression closure, all on the QPU.

    The greedy loop is replaced by its closed form:

    * **rank** every candidate by score with two exact comparison matrices and
      two GEMM row sums --
      ``rank[i] = #{s[j] > s[i]} + #{j < i: s[j] == s[i]}``, the reference's
      own tie order (equal scores keep the lower index first);
    * the **IoU matrix** is built once per batch with QPU min/max/multiply;
    * the **suppression graph** ``A[i][j] = iou[i][j] > thr and rank[j] <
      rank[i] and cand[j]`` is a DAG (every edge goes to a strictly earlier
      box), and greedy suppression keeps exactly the boxes with no outgoing
      path: every path ends at a box nothing earlier suppresses.  So
      ``keep0 = cand and not (rowsum((I + A)^N) > 1)``, computed with
      ``ceil(log2 N)`` boolean matrix squarings;
    * ``max_output_boxes_per_class`` is one more GEMM row sum over the decided
      keep vector.

    The ARM reads back only *index* data -- the ranks and the 0/1 keep vector
    (the ``ArgMax``/``NonZero`` class of read-back) -- and turns them into the
    int64 [K][3] list; no suppression decision is taken on the host.  There is
    no bitonic sort on the QPU: ranking by pairwise comparison is O(N^2)
    elementwise work in a fixed number of launches, where a sort network would
    be O(log^2 N) launches of the same O(N^2) work.
    """
    MAX_BOXES = 512

    def run(self):
        node = self.node
        boxes = self.ten(node.input[0])
        scores = self.ten(node.input[1])
        if len(boxes.shape) != 3 or int(boxes.shape[2]) != 4:
            self.fail("boxes must be [num_batches, num_boxes, 4] (got %s)"
                      % (boxes.shape,))
        B, N = int(boxes.shape[0]), int(boxes.shape[1])
        if len(scores.shape) != 3 or int(scores.shape[0]) != B:
            self.fail("scores must be [num_batches, num_classes, num_boxes]"
                      " (got %s)" % (scores.shape,))
        C = int(scores.shape[1])
        if int(scores.shape[2]) != N:
            self.fail("scores has %d boxes for %d boxes of boxes"
                      % (int(scores.shape[2]), N))
        center = int(_attr(node, "center_point_box", 0))
        if center not in (0, 1):
            self.fail("center_point_box=%d" % center)
        if N > self.MAX_BOXES:
            self.fail("num_boxes=%d: the rank/IoU/closure formulation keeps "
                      "~10 N^2 matrices live (4 B each at N=512); the bound is "
                      "%d" % (N, self.MAX_BOXES))
        # an empty tensor is how ONNX spells "use the default" for an optional
        # scalar input: max_output_boxes_per_class then defaults to 0 (no
        # output) and iou_threshold to 0.0
        def scalar(name, default):
            if len(node.input) <= name or not node.input[name]:
                return default
            a = np.asarray(self.host_or_fail(node.input[name])).ravel()
            return default if a.size == 0 else a[0]

        max_out = int(scalar(2, 0))
        iou_thr = float(scalar(3, 0.0))
        score_thr = None
        if len(node.input) > 4 and node.input[4]:
            a = np.asarray(self.host_or_fail(node.input[4])).ravel()
            score_thr = float(a[0]) if a.size else None
        if max_out <= 0 or N == 0 or C == 0 or B == 0:
            return np.zeros((0, 3), np.int64)
        keep_rows = []
        for b in range(B):
            iou = self._iou(boxes, b, N, center)
            for c in range(C):
                plane = self.view_t(scores, (b * C + c) * N, (N,))
                rank, cand = self._rank(plane, N, score_thr)
                kept = self._predicate(iou, rank, cand, N, max_out, iou_thr)
                order = [i for i in np.argsort(rank, kind="stable")
                         if kept[i]]
                for i in order[:max_out]:
                    keep_rows.append((b, c, int(i)))
        if not keep_rows:
            return np.zeros((0, 3), np.int64)
        return np.asarray(keep_rows, np.int64)

    # -------------------------------------------------------------- pieces
    def _iou(self, boxes, b, N, center):
        """The full N x N IoU matrix for one batch, on the QPUs."""
        P = self.view_t(boxes, b * N * 4, (N * 4,))
        c0 = self.alloc((N,)); c1 = self.alloc((N,))
        c2 = self.alloc((N,)); c3 = self.alloc((N,))
        for dst, k in ((c0, 0), (c1, 1), (c2, 2), (c3, 3)):
            self.stride_take(P, dst, N, 4, k)
        if center:
            half2 = self.const(("nms_half", N), np.full(N, 0.5, np.float32))
            xlo = self.sub(c0, self.mul(c2, half2))
            xhi = self.add(c0, self.mul(c2, half2))
            ylo = self.sub(c1, self.mul(c3, half2))
            yhi = self.add(c1, self.mul(c3, half2))
        else:
            xlo = self.maxmin2(c0, c2, False)
            xhi = self.maxmin2(c0, c2, True)
            ylo = self.maxmin2(c1, c3, False)
            yhi = self.maxmin2(c1, c3, True)
        area = self.mul(self.sub(xhi, xlo), self.sub(yhi, ylo))
        zero = self._const_mat("nms_zero", N, 0.0)
        xi, xj = self._pair(xlo, N)
        Xi, Xj = self._pair(xhi, N)
        yi, yj = self._pair(ylo, N)
        Yi, Yj = self._pair(yhi, N)
        iw = self.maxmin2(self.sub(self.maxmin2(Xi, Xj, False),
                                   self.maxmin2(xi, xj, True)), zero)
        ih = self.maxmin2(self.sub(self.maxmin2(Yi, Yj, False),
                                   self.maxmin2(yi, yj, True)), zero)
        inter = self.mul(iw, ih)
        ai, aj = self._pair(area, N)
        union = self.sub(self.add(ai, aj), inter)
        safe = self.maxmin2(union, self.const(("nms_eps", N * N),
                                             np.full((N, N), 1e-30,
                                                     np.float32)))
        return self.mul(inter, self.unary(safe, vc4infer.UOP_RECIPROCAL))

    def _pair(self, v, N):
        """``(A[i][j] = v[j], B[i][j] = v[i])`` as two [N][N] planes."""
        if N * N == 0:
            self.fail("empty box set")
        return (self.tile_rows(v, N, N), self.bcast_rows(v, N, N))

    def _const_mat(self, key, N, value):
        return self.const((key, N), np.full((N, N), value, np.float32))

    def _rank(self, s, N, score_thr):
        """QPU score rank (reference tie order) and the candidate mask."""
        one = self.const(("nms_one", N), np.ones(N, np.float32))
        if score_thr is None:
            cand = self.affine(one, 1.0, 0.0)
        else:
            cand = self.cmp(vc4infer.OP_CMP_GT, s,
                            self.const(("nms_sthr", N),
                                       np.full(N, score_thr, np.float32)))
        # _pair returns (row i = v[j], row i = v[i]): gt[i][j] = s[j] > s[i]
        sj, si = self._pair(s, N)
        gt = self.mul(self.cmp(vc4infer.OP_CMP_GT, sj, si),
                      self.tile_rows(cand, N, N))
        lower = self.const(("nms_lower", N),
                           np.tril(np.ones((N, N), np.float32), -1))
        eq = self.mul(self.mul(self.cmp(vc4infer.OP_CMP_EQ, sj, si), lower),
                      self.tile_rows(cand, N, N))
        r = self.add(self.sum_last_aligned(gt, N, N),
                     self.sum_last_aligned(eq, N, N))
        rank = self.float_words(r, (N,)).reshape(-1)
        cnd = np.rint(np.asarray(self.r.download(cand),
                                 np.float32)).astype(np.int64)
        return rank, cnd

    def _predicate(self, iou, rank, cand, N, max_out, iou_thr):
        """keep[i] from the suppression closure; only the 0/1 vector is read back.

        ``A[i][j] = (iou[i][j] > thr) and rank[j] < rank[i] and cand[j]`` is the
        suppression graph (every edge goes to a strictly earlier box, so it is a
        DAG).  Greedy suppression keeps ``i`` iff no *kept* earlier box
        suppresses it, and because every path in a DAG ends at a box nothing
        earlier suppresses, that is exactly ``i`` has no outgoing path of
        length >= 1.  So ``keep = cand and not (rowsum((I + A)^N) > 1)``:
        ``log2(N)`` boolean matrix squarings, no sequential pass and no
        per-box read-back.
        """
        order = np.argsort(rank, kind="stable")
        pos = np.empty(N, np.int64)
        pos[order] = np.arange(N)
        earlier = (pos.reshape(1, N) < pos.reshape(N, 1)).astype(np.float32)
        W = self.upload_const(earlier, (N, N))
        sup = self.mul(self.cmp(vc4infer.OP_CMP_GT, iou,
                                self.const(("nms_ithr", N * N),
                                           np.full((N, N), iou_thr,
                                                   np.float32))),
                       self.mul(W, self.tile_rows(
                           self.upload_const(cand.astype(np.float32), (N,)),
                           N, N)))
        M = self.add(sup, self.const(("nms_eye", N),
                                     np.eye(N, dtype=np.float32)))
        for _ in range(int(np.ceil(np.log2(max(N, 2))))):
            M = self.gemm_rect(M, N, N, M, N)
        rs = self.sum_last_aligned(M, N, N)
        keep0 = self.mul(self.upload_const(cand.astype(np.float32), (N,)),
                         self.cmp(vc4infer.OP_CMP_LE, rs,
                                  self.const(("nms_one", N),
                                             np.ones(N, np.float32))))
        before = self.sum_last_aligned(
            self.mul(self.tile_rows(keep0, N, N), W), N, N)
        keep = self.mul(keep0, self.cmp(
            vc4infer.OP_CMP_LT, before,
            self.const(("nms_mo", N), np.full(N, float(max_out),
                                              np.float32))))
        return (np.rint(np.asarray(self.r.download(keep), np.float32))
                .astype(np.int64) != 0)


# =====================================================================
# dispatch table
# =====================================================================
def _mk(cls):
    def run(r, node):
        return cls(r, node).run()
    return run


HANDLERS = {
    "NonMaxSuppression": _mk(NonMaxSuppressionOp),
    "Det": _mk(DetOp),
    "Einsum": _mk(EinsumOp),
    "DeformConv": _mk(DeformConvOp),
    "RoiAlign": _mk(RoiAlignOp),
    "MaxRoiPool": _mk(MaxRoiPoolOp),
    "Col2Im": _mk(Col2ImOp),
    "ConvTranspose": _mk(ConvTransposeOp),
    "AffineGrid": _mk(AffineGridOp),
    "GridSample": _mk(GridSampleOp),
}

OVERRIDES = ()
