"""Runner handlers for pooling, normalisation, softmax variants and losses.

``docs/OPS-NORM.md`` is the reference for this file: what every operator is
composed from, which ONNX form is exact, and which form is refused with a
name.  Every handler is a composition of *element-verified* engine kernels:

    vc4_elementwise     Add / Mul / ReluBwd (out = b if a > 0) and the exact
                        fp32 comparisons (kgen._ew_cmp)
    vc4_unary           Abs / Exp / Log / Sqrt / Reciprocal / Affine
    vc4_matmul          every reduction and every window sum: a sum along an
                        axis is one GEMM against a constant matrix (ones for a
                        sum, a band for an LRN channel window)
    vc4_copy2d_x        every layout move: axis-to-last, a row broadcast, one
                        pooling window tap, the MaxUnpool scatter
    vc4_batchnorm_infer inference batch normalisation
    vc4_layernorm_fwd   last-axis (LayerNorm) and plane (InstanceNorm,
                        GroupNorm) standardisation
    vc4_softmax         last-axis softmax: the fused kernel for <= 16 classes
                        and rows % 8 == 0, a reduce/exp/div composition
                        otherwise

``vc4_reduce`` / ``vc4_cumsum`` / ``vc4_select3`` are NOT used: the first two
wedge the QPU on this board and the third returns wrong values
(docs/OPS-REDUCE.md sec. 5).  ``vc4_gather`` is not used either (it wedges the
QPU), which is why ``MaxUnpool`` routes values with ``copy2d_x`` single-word
stores instead of a gather table.

The ARM never touches an activation value.  It only

  * builds integer geometry: pooling window bounds and the per-output valid
    sample counts (integers from the attributes and the input shape), the LRN
    band indices, the MaxUnpool destination offsets (from the int64 indices),
  * builds *constants*: pooling divisors, ones/zeros gamma tiles, the one-hot
    / gather-weight mask a loss multiplies the activations by,
  * reads indices back (ArgMax-class work: the ``Hardmax`` one-hot and the
    int64 ``labels``/``target`` of a loss), and
  * validation reads (``download_check``: a non-finite or out-of-range input is
    refused by name, exactly like ``ReduceLogSumExp`` does).

Layout choice for channel-axis statistics (stated once, used everywhere):
activations are compact NCHW.  A statistic along the channel axis is taken
after the channel axis is made *innermost* on the QPUs (``ax_last`` -> one
``copy2d_x`` per plane block), so the reduction is a row reduction and the
result keeps the ONNX output order with no inverse pass.  InstanceNorm and
GroupNorm need no layout move at all: ``[N][C][H*W]`` already has one channel
plane per row (InstanceNorm) or one contiguous group of planes per row
(GroupNorm).
"""
import numpy as np
import onnx

from . import _binding as vc4infer
from ._dispatch_extra import (_Base, _ReduceMixin, _Tensor, _attr,
                              _bcast_index, _prod)
from ._errors import UnsupportedNode

# The operators of the main dispatch table this module owns.  ``_onnxrun``
# consults the family tables first for these names, so one implementation of
# Softmax exists (the main table keeps the old fast path as
# ``Runner._softmax`` only for reference; see docs/OPS-NORM.md).
OVERRIDES = frozenset(("Softmax",))


# =====================================================================
# small helpers
# =====================================================================
def _opset(model):
    """The ai.onnx opset the model declares (default 1)."""
    v = None
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            v = int(o.version)
    return v if v else 1


def _vec_attr(node, name, n, default):
    v = _attr(node, name, default)
    if v is None:
        return list(default)
    v = [int(x) for x in v]
    if len(v) == 1 and n != 1:
        v = v * n
    if len(v) != n:
        raise UnsupportedNode(node.name, node.op_type,
                              "%s has %d values (want %d)"
                              % (name, len(v), n))
    return v


class _NormBase(_Base, _ReduceMixin):
    """Shared helpers: shape checks, channel vectors, row operations."""

    # ------------------------------------------------------------ shapes
    def rank_at_least(self, x, n):
        if len(x.shape) < n:
            self.fail("needs a rank >= %d input (got %s)"
                      % (n, tuple(x.shape)))

    def channel(self, name, C, what):
        """A compact [C] fp32 tensor (constant uploaded once, or the operand).

        Batchnorm/InstanceNorm/LayerNorm parameters are per-channel (or
        per-column) vectors.  A graph constant is uploaded once and cached in
        the runner's panel table, exactly like a packed conv weight.
        """
        v = self.r.values.get(name)
        if isinstance(v, _Tensor()):
            if v.size != C:
                self.fail("%s %s has %d elements (want %d)"
                          % (what, name, v.size, C))
            if v.layout != "nchw":
                self.fail("%s %s is not a compact fp32 vector" % (what, name))
            return v
        arr = np.asarray(self.host_or_fail(name), np.float32)
        if arr.size != C:
            self.fail("%s %s has %d elements (want %d)"
                      % (what, name, arr.size, C))
        return self.const(("chan", name, C),
                          np.ascontiguousarray(arr.ravel()))

    def bcast_channel(self, name, shape, C, what):
        """[C] -> `shape` = [N][C][D] on the QPUs (one laned copy map)."""
        src = self.channel(name, C, what)
        idx = _bcast_index((1, C, 1), tuple(shape))
        out = self.alloc(shape)
        self.copy_runs(src.buf, out.buf, idx, _prod(shape))
        return out

    # ------------------------------------------------------- row helpers
    def rows_of(self, x, axis, opset13, op):
        """x as rows x D over `axis`, and the way back.

        opset >= 13: the axis itself is reduced per (outer, inner) position
        (``ax_last``, a QPU layout copy when the axis is not last).
        opset < 13: ONNX coerces the input to 2-D
        ``[prod(dims[:axis]), prod(dims[axis:])]`` first; that is the memory
        order of a contiguous tensor, so no copy is needed.
        """
        shape = [int(d) for d in x.shape]
        if opset13:
            T, P, D, I = self.ax_last(x, axis)
            return T, P * I, D, (lambda t: self.ax_back(t, P, D, I, shape))
        rows = _prod(shape[:axis])
        D = _prod(shape[axis:])
        return x, rows, D, (lambda t: self.tag(t, shape))

    def bcast_rows(self, vec, rows, D):
        """[rows] -> [rows][D]: row r is vec[r] repeated D times.

        One lane copy per row block: the source advances one word per row
        (srow=4) and every destination word of the row reads it (slane=0).
        """
        out = self.alloc((rows, D))
        self.lane_runs(vec.buf, out.buf, rows, D, 4, D * 4, 0)
        return out

    def row_mul(self, T, vec, rows, D):
        return self.mul(T, self.bcast_rows(vec, rows, D))

    def row_sub(self, T, vec, rows, D):
        """T - vec, broadcast over the row (vec is never an activation)."""
        return self.add(T, self.affine(self.bcast_rows(vec, rows, D), -1.0,
                                       0.0))

    def row_sum(self, T, rows, D):
        return self.sum_last(T, rows, D)

    # ------------------------------------------------ the GEMM operand rule
    # vc4_matmul pads an A operand whose M is not 8-aligned or whose K is not
    # 16-aligned by copying it with the ARM (infer/src/vc4_gemm.c), and this
    # arena mapping is cacheable, so that copy can read a line the QPU has
    # already overwritten (measured 2026-11-20: a 16x4x16 GEMM returned
    # another buffer's bytes in its first rows, run to run).  Every reduction
    # here pads on the QPUs instead (copy2d into a zeroed, aligned panel), so
    # the library never takes that path with activation data.
    def sum_last(self, T, rows, D):
        """sum over the innermost axis: one GEMM against a ones column."""
        if D <= 0:
            self.fail("reducing an empty axis")
        if D % 16 == 0 and rows % 8 == 0:
            ones = self.const(("ones", D), np.ones((D, 16), np.float32))
            res = self.alloc((rows, 16))
            self.r.eng.matmul(T.buf, ones.buf, res.buf, rows, 16, D)
        else:
            Dp = (D + 15) & ~15
            M = (rows + 7) & ~7
            A = self.alloc_fill((M, Dp), 0.0)
            self.lane_runs(T.buf, A.buf, rows, D, D * 4, Dp * 4, 4)
            ones = self.const(("ones", Dp), np.ones((Dp, 16), np.float32))
            res = self.alloc((M, 16))
            self.r.eng.matmul(A.buf, ones.buf, res.buf, M, 16, Dp)
        return self.slice_col(res, 0, rows, 16)

    def gemm_rows(self, T, rows, D, band):
        """[rows][D] x band[D][D] -> [rows][D], padded on the QPUs.

        ``band`` is the host D x D matrix; it is zero-padded to
        ``[align16(D)][align16(D)]`` and uploaded once (a constant, like a
        packed weight panel).
        """
        Dp = (D + 15) & ~15
        M = (rows + 7) & ~7
        full = np.zeros((Dp, Dp), np.float32)
        full[:D, :D] = np.asarray(band, np.float32)
        key = ("band", D, float(full.sum()), float(np.abs(full).sum()))
        kmat = self.const(key, np.ascontiguousarray(full))
        A = self.alloc_fill((M, Dp), 0.0)
        self.lane_runs(T.buf, A.buf, rows, D, D * 4, Dp * 4, 4)
        res = self.alloc((M, Dp))
        self.r.eng.matmul(A.buf, kmat.buf, res.buf, M, Dp, Dp)
        out = self.alloc((rows, D))
        self.lane_runs(res.buf, out.buf, rows, D, Dp * 4, D * 4, 4)
        return out

    def reduce_axis(self, t, ax, fn):
        """Reduce one axis (made innermost) and retag the logical shape."""
        cur = [int(d) for d in t.shape]
        T, P, D, I = self.ax_last(t, ax)
        if T is not t:
            self.tmp(T)
        res = fn(T, P * I, D)
        return self.tag(res, cur[:ax] + cur[ax + 1:])

    def row_max(self, T, rows, D):
        return self.max_last(T, rows, D, "max")

    def stable_softmax(self, T, rows, D):
        """exp(x - max) / sum exp(x - max), exact and overflow-free."""
        m = self.row_max(T, rows, D)
        e = self.unary(self.row_sub(T, m, rows, D), vc4infer.UOP_EXP)
        s = self.row_sum(e, rows, D)
        self.download_check(s, lambda a: bool(np.min(a) > 0),
                            "Softmax: the exp sum is not positive")
        r = self.unary(s, vc4infer.UOP_RECIPROCAL)
        return self.row_mul(e, r, rows, D)

    def all_finite(self, t, what, allow_neginf=False, allow_inf=False,
                   max_abs=None):
        """Validation read: refuse the inputs this composition cannot answer.

        ``vc4_elementwise(VC4_OP_MUL)`` is *not* IEEE for a non-finite
        operand: measured on this board, ``inf * 0.25`` (and ``inf * 1``)
        return NaN and an overflowing product returns NaN instead of +-inf,
        while ADD and the unary Affine kernel are exact.  Every composition
        here multiplies activations, so a non-finite activation is refused by
        name rather than answered with a NaN ONNX does not have.  The pure
        max/argmax scans (GlobalMaxPool, Hardmax) pass ``allow_inf=True``:
        their comparisons are IEEE-exact for +-inf.
        """
        a = self.r.download(t)
        if not a.size:
            return a
        if np.isnan(a).any():
            self.fail("%s: a NaN input is refused (the compositions use "
                      "IEEE comparisons and multiplication, which do not "
                      "propagate NaN the way numpy's max/argmax do)" % what)
        if allow_neginf:
            if np.isposinf(a).any():
                self.fail("%s: a +inf input is refused (the output is NaN "
                          "for every element of the row in ONNX and the "
                          "engine cannot produce it exactly)" % what)
        elif allow_inf:
            pass
        elif not np.isfinite(a).all():
            self.fail("%s: a non-finite input is refused: the composition "
                      "multiplies activations and the elementwise MUL kernel "
                      "returns NaN for an inf operand (measured), so +-inf "
                      "would not be the ONNX answer" % what)
        if max_abs is not None and np.abs(a).max() > max_abs:
            self.fail("%s: |x| up to %g exceeds the %g bound this "
                      "composition squares (x*x would overflow fp32 and the "
                      "elementwise MUL returns NaN for an overflow)"
                      % (what, float(np.abs(a).max()), max_abs))
        return a


# =====================================================================
# Dropout
# =====================================================================
class Dropout(_NormBase):
    """Inference Dropout is the identity (ONNX: Y = X, mask = all ones)."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        train = 0
        if len(node.input) > 2 and node.input[2]:
            arr = self.host(node.input[2])
            if arr is None or np.asarray(arr).size != 1:
                self.fail("training_mode is not a scalar constant")
            train = int(np.asarray(arr).ravel()[0])
        if train:
            self.fail("training-mode Dropout (training_mode=%d) is "
                      "unimplemented: this engine implements the ONNX "
                      "inference definition only (Y = X, mask = all ones); "
                      "a training pass needs a device RNG" % train)
        out = [x]
        for name in node.output[1:]:
            if name:
                out.append(np.ones(tuple(x.shape), bool))
        return out


# =====================================================================
# BatchNormalization (inference)
# =====================================================================
class BatchNormalization(_NormBase):
    """y = (x - mean) / sqrt(var + eps) * scale + B, on vc4_batchnorm_infer.

    The epsilon is inside the square root, as ONNX defines it; the kernel
    computes ``istd = rsqrt(var + eps)`` and then ``(x - mean) * istd``.  No
    epsilon is added anywhere else.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        self.rank_at_least(x, 2)
        eps = float(_attr(node, "epsilon", 1e-5))
        train = int(_attr(node, "training_mode", 0))
        extra = [o for o in node.output[1:] if o]
        if train or extra:
            self.fail("training_mode=%d: this engine implements the ONNX "
                      "inference form only (y = (x-mean)/sqrt(var+eps)"
                      "*scale+B).  Training mode updates the running "
                      "statistics and needs the extra outputs, which is a "
                      "different kernel (vc4_batchnorm_fwd, training only)"
                      % train)
        N, C = int(x.shape[0]), int(x.shape[1])
        HW = _prod(x.shape[2:])
        if HW <= 0 or C <= 0:
            self.fail("empty input %s" % (tuple(x.shape),))
        self.all_finite(x, "BatchNormalization")
        g = self.channel(node.input[1], C, "scale")
        b = self.channel(node.input[2], C, "B")
        m = self.channel(node.input[3], C, "input_mean")
        v = self.channel(node.input[4], C, "input_var")
        out = self.alloc(x.shape)
        self.r.eng.batchnorm_infer(x.buf, g.buf, b.buf, m.buf, v.buf, out.buf,
                                   N, C, HW, eps)
        return out


# =====================================================================
# InstanceNormalization / GroupNormalization / LayerNormalization
# =====================================================================
class _Standardize(_NormBase):
    """y = (x - mean)/sqrt(var + eps) * scale + B over a row of the input."""

    def standardize(self, x, rows, cols, eps, what):
        if rows <= 0 or cols <= 0:
            self.fail("%s: empty input %s" % (what, tuple(x.shape)))
        self.all_finite(x, what)
        ones = self.const(("ones_col", cols), np.ones(cols, np.float32))
        zero = self.const(("zero_col", cols), np.zeros(cols, np.float32))
        y = self.alloc(x.shape)
        mean = self.alloc((rows,))
        istd = self.alloc((rows,))
        self.r.eng.layernorm_fwd(x.buf, ones.buf, zero.buf, y.buf, mean.buf,
                                 istd.buf, rows, cols, eps)
        return y

    def channel_affine(self, y, shape, C, scale_name, bias_name):
        N = int(shape[0])
        HW = _prod(shape[2:])
        s = self.bcast_channel(scale_name, (N, C, HW), C, "scale")
        b = self.bcast_channel(bias_name, (N, C, HW), C, "B")
        y = self.tag(y, (N, C, HW))
        out = self.add(self.mul(y, s), b)
        return self.tag(out, shape)


class InstanceNormalization(_Standardize):
    """Per (n, c) plane: the row is one channel plane of [N][C][H*W]."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        self.rank_at_least(x, 3)
        eps = float(_attr(node, "epsilon", 1e-5))
        N, C = int(x.shape[0]), int(x.shape[1])
        HW = _prod(x.shape[2:])
        y = self.standardize(x, N * C, HW, eps, "InstanceNormalization")
        return self.channel_affine(y, tuple(x.shape), C, node.input[1],
                                   node.input[2])


class GroupNormalization(_Standardize):
    """Per (n, group): a group is ``C/G`` contiguous channel planes."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        self.rank_at_least(x, 3)
        eps = float(_attr(node, "epsilon", 1e-5))
        if int(_attr(node, "stash_type", 1)) != 1:
            self.fail("stash_type=%d: only fp32 accumulation (stash_type=1) "
                      "is implemented" % int(_attr(node, "stash_type", 1)))
        G = int(_attr(node, "num_groups", 1))
        N, C = int(x.shape[0]), int(x.shape[1])
        if G < 1 or C % G:
            self.fail("num_groups=%d does not divide C=%d" % (G, C))
        HW = _prod(x.shape[2:])
        y = self.standardize(x, N * G, (C // G) * HW, eps,
                             "GroupNormalization")
        return self.channel_affine(y, tuple(x.shape), C, node.input[1],
                                   node.input[2])


class LayerNormalization(_Standardize):
    """Standardise over the axes [axis..rank-1]; Scale/B broadcast over them."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        self.rank_at_least(x, 1)
        rank = len(x.shape)
        if int(_attr(node, "stash_type", 1)) != 1:
            self.fail("stash_type=%d: only fp32 accumulation (stash_type=1) "
                      "is implemented" % int(_attr(node, "stash_type", 1)))
        axis = int(_attr(node, "axis", -1))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        eps = float(_attr(node, "epsilon", 1e-5))
        shape = [int(d) for d in x.shape]
        rows = _prod(shape[:axis])
        cols = _prod(shape[axis:])
        if rows <= 0 or cols <= 0:
            self.fail("empty input %s" % (tuple(shape),))
        trailing = tuple(shape[axis:])
        gamma = self.tile_param(node.input[1], rows, trailing, cols)
        if len(node.input) > 2 and node.input[2]:
            beta = self.tile_param(node.input[2], rows, trailing, cols)
        else:
            beta = self.const(("zero_col", cols),
                              np.zeros(cols, np.float32)).buf
        y = self.alloc(x.shape)
        mean = self.alloc((rows,))
        istd = self.alloc((rows,))
        self.r.eng.layernorm_fwd(x.buf, gamma, beta, y.buf, mean.buf,
                                 istd.buf, rows, cols, eps)
        outs = [y]
        # ONNX: Mean / InvStdDev have shape X.shape[:axis] + [1]*(rank-axis)
        red_shape = tuple(shape[:axis] + [1] * (rank - axis))
        for i in (1, 2):
            if len(node.output) > i and node.output[i]:
                src = mean if i == 1 else istd
                outs.append(self.tag(src, red_shape))
        return outs

    def tile_param(self, name, rows, trailing, cols):
        """Scale/B as the flat per-column vector (they are constant per row)."""
        v = self.r.values.get(name)
        if isinstance(v, _Tensor()):
            if tuple(v.shape) == trailing:
                return v.buf
            t = self.bcast_t(v, (rows,) + trailing)
            return self.r._buf_at(t, 0)
        arr = np.asarray(self.host_or_fail(name), np.float32)
        try:
            full = np.broadcast_to(arr, trailing)
        except ValueError:
            self.fail("Scale/B shape %s does not broadcast to the normalised "
                      "axes %s" % (tuple(arr.shape), trailing))
        flat = np.ascontiguousarray(full.ravel())
        if flat.size != cols:
            self.fail("Scale/B flattens to %d values (want %d)"
                      % (flat.size, cols))
        return self.const(("tile", name, cols), flat).buf


# =====================================================================
# MeanVarianceNormalization
# =====================================================================
class MeanVarianceNormalization(_NormBase):
    """(X - mean) / (sqrt(var) + 1e-9) over `axes` (ONNX function body).

    The epsilon is added *after* the square root, as the ONNX function body
    for MeanVarianceNormalization-13 defines it -- not inside it.
    """

    EPS = 1e-9

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axes = _attr(node, "axes", None)
        if axes is None or len(axes) == 0:
            # the schema default; onnx.reference 1.17's function body reduces
            # every axis when the attribute is absent (see docs/OPS-NORM.md)
            axes = [0, 2, 3] if rank == 4 else list(range(rank))
        self.all_finite(x, "MeanVarianceNormalization", max_abs=1e19)
        axes = sorted({(int(a) + rank if int(a) < 0 else int(a))
                       for a in axes})
        for a in axes:
            if a < 0 or a >= rank:
                self.fail("axes %s for rank %d" % (list(axes), rank))
        count = 1
        for a in axes:
            count *= int(x.shape[a])
        if count <= 0:
            self.fail("an empty axis in %s" % (list(axes),))
        shape = [int(d) for d in x.shape]
        keep = tuple(1 if i in axes else d for i, d in enumerate(shape))
        def mean(t):
            r = t
            for a in reversed(axes):
                r = self.reduce_axis(r, a, self.sum_last)
            return self.affine(r, 1.0 / count, 0.0)

        m = mean(x)                                   # E[X]
        ex2 = self.mul(m, m)                          # E[X]^2
        x2 = self.mul(x, x)
        m2 = mean(x2)                                 # E[X^2]
        var = self.sub(m2, ex2)
        std = self.unary(var, vc4infer.UOP_SQRT)
        std = self.affine(std, 1.0, self.EPS)         # sqrt(var) + eps
        mb = self.bcast_t(self.tag(m, keep), tuple(shape))
        sb = self.bcast_t(self.tag(std, keep), tuple(shape))
        # (X - mean) * (1 / (sqrt(var) + eps))
        rcp = self.unary(sb, vc4infer.UOP_RECIPROCAL)
        return self.mul(self.sub(x, mb), rcp)


# =====================================================================
# LRN
# =====================================================================
class LRN(_NormBase):
    """y = x / (bias + alpha/size * sum_{window} x^2)^beta, channel window.

    The channel axis is made innermost (``ax_last``) so the window sum is one
    GEMM against a constant band matrix; no activation value is transposed by
    the ARM.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        self.rank_at_least(x, 3)
        size = int(_attr(node, "size", 0))
        if size < 1:
            self.fail("size=%d" % size)
        alpha = float(_attr(node, "alpha", 1e-4))
        beta = float(_attr(node, "beta", 0.75))
        bias = float(_attr(node, "bias", 1.0))
        if bias <= 0.0:
            self.fail("bias=%g: (bias + alpha/size*sum)^beta needs a positive "
                      "base for a non-integer beta" % bias)
        self.all_finite(x, "LRN", max_abs=1e19)
        shape = [int(d) for d in x.shape]
        C = shape[1]
        T, P, D, I = self.ax_last(x, 1)
        if T is not x:
            self.tmp(T)
        rows = P * I
        c1 = (size - 1) // 2
        c2 = (size - 1 + 1) // 2 + 1
        band = np.zeros((D, D), np.float32)
        for c in range(D):
            band[c, max(0, c - c1):min(D, c + c2)] = 1.0
        sq = self.mul(T, T)
        s = self.gemm_rows(sq, rows, D, band)
        t = self.affine(s, alpha / size, bias)
        if beta == 1.0:
            denom = t
        else:
            lg = self.unary(t, vc4infer.UOP_LOG)
            denom = self.unary(self.affine(lg, beta, 0.0), vc4infer.UOP_EXP)
        y = self.mul(T, self.unary(denom, vc4infer.UOP_RECIPROCAL))
        return self.ax_back(y, P, D, I, tuple(shape))


# =====================================================================
# Global pooling
# =====================================================================
class GlobalMaxPool(_NormBase):
    """max over the last two axes, one [N*C][H*W] row per channel plane."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 2-D spatial pooling is wired (input rank %d)"
                      % len(x.shape))
        N, C, H, W = (int(d) for d in x.shape)
        if H * W <= 0:
            self.fail("empty spatial extent %dx%d" % (H, W))
        self.all_finite(x, "GlobalMaxPool", allow_inf=True)
        m = self.row_max(self.tag(x, (N * C, H * W)), N * C, H * W)
        return self.tag(m, (N, C, 1, 1))


class GlobalLpPool(_NormBase):
    """(sum |x|^p)^(1/p) over the last two axes, p in {1, 2}."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 2-D spatial pooling is wired (input rank %d)"
                      % len(x.shape))
        p = int(_attr(node, "p", 2))
        N, C, H, W = (int(d) for d in x.shape)
        if H * W <= 0:
            self.fail("empty spatial extent %dx%d" % (H, W))
        self.all_finite(x, "GlobalLpPool", max_abs=1e19 if p == 2 else None)
        if p == 1:
            t = self.unary(x, vc4infer.UOP_ABS)
        elif p == 2:
            t = self.mul(x, x)
        else:
            self.fail("p=%d: only p=1 and p=2 are exact (any other p needs a "
                      "pow on a possibly zero base)" % p)
        s = self.row_sum(self.tag(t, (N * C, H * W)), N * C, H * W)
        if p == 2:
            s = self.unary(s, vc4infer.UOP_SQRT)
        return self.tag(s, (N, C, 1, 1))


# =====================================================================
# AveragePool / LpPool
# =====================================================================
class _PoolGeom(object):
    """Everything about a pooling window that is integer geometry."""

    def __init__(self, node, shape, fail):
        rank = len(shape)
        if rank != 4:
            fail("only 2-D spatial pooling is wired (input rank %d)" % rank)
        self.N, self.C, self.H, self.W = (int(d) for d in shape)
        ks = _attr(node, "kernel_shape", None)
        if not ks or len(ks) != 2:
            fail("kernel_shape %s" % (ks,))
        self.kh, self.kw = int(ks[0]), int(ks[1])
        self.sh, self.sw = _vec_attr(node, "strides", 2, [1, 1])
        self.dh, self.dw = _vec_attr(node, "dilations", 2, [1, 1])
        self.ceil = int(_attr(node, "ceil_mode", 0))
        if min(self.kh, self.kw, self.sh, self.sw, self.dh, self.dw) < 1:
            fail("kernel/stride/dilation %s" % ((self.kh, self.kw, self.sh,
                                                 self.sw, self.dh, self.dw),))
        auto = _attr(node, "auto_pad", "NOTSET") or "NOTSET"
        if auto not in ("NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"):
            fail("auto_pad=%s" % auto)
        H, W = self.H, self.W
        if auto != "NOTSET":
            if self.ceil:
                fail("ceil_mode with auto_pad=%s (ONNX forbids it)" % auto)
            if auto == "VALID":
                pt = pl = pb = pr = 0
            else:
                oh = (H - 1) // self.sh + 1
                ow = (W - 1) // self.sw + 1
                ph = (oh - 1) * self.sh + self.kh - H
                pw = (ow - 1) * self.sw + self.kw - W
                if auto == "SAME_UPPER":
                    pt, pb = ph // 2, ph - ph // 2
                    pl, pr = pw // 2, pw - pw // 2
                else:
                    pt, pb = ph - ph // 2, ph // 2
                    pl, pr = pw - pw // 2, pw // 2
        else:
            pads = _attr(node, "pads", [0, 0, 0, 0])
            if pads is None:
                pads = [0, 0, 0, 0]
            if len(pads) != 4:
                fail("pads %s (want 4 values)" % (list(pads),))
            pt, pl, pb, pr = (int(v) for v in pads)
            if min(pt, pl, pb, pr) < 0:
                fail("negative pads %s" % ([pt, pl, pb, pr],))

        def dim_out(inp, p0, p1, k, s, d):
            v = (inp + p0 + p1 - d * (k - 1) - 1) / float(s) + 1.0
            return int(np.ceil(v)) if self.ceil else int(np.floor(v))

        self.Ho = dim_out(H, pt, pb, self.kh, self.sh, self.dh)
        self.Wo = dim_out(W, pl, pr, self.kw, self.sw, self.dw)
        if self.Ho <= 0 or self.Wo <= 0:
            fail("output %dx%d is empty" % (self.Ho, self.Wo))
        # ceil_mode's implicit extra pad, split exactly as ONNX does
        for dim in (0, 1):
            if not self.ceil:
                continue
            d = self.dh if dim == 0 else self.dw
            k = self.kh if dim == 0 else self.kw
            s = self.sh if dim == 0 else self.sw
            out = self.Ho if dim == 0 else self.Wo
            inp = H if dim == 0 else W
            p0 = pt if dim == 0 else pl
            p1 = pb if dim == 0 else pr
            win = (k - 1) * d + 1
            extra = (out - 1) * s + win - inp - p0 - p1
            if extra > 0:
                p0 += extra // 2
                p1 += extra - extra // 2
            if dim == 0:
                pt, pb = p0, p1
            else:
                pl, pr = p0, p1
        self.pt, self.pl, self.pb, self.pr = pt, pl, pb, pr

    def window(self):
        """The number of *taps* in the window (k*k, dilation included).

        ONNX's count_include_pad divides by every element of the window, and a
        dilated window has `kernel_shape` taps per dimension, not
        `dilation*(kernel-1)+1` positions.
        """
        return self.kh * self.kw

    def spans(self, i, j):
        """Output rows/cols whose tap (i, j) reads inside the input."""
        g = self
        oh0 = max(0, -((-(g.pt - i * g.dh)) // g.sh))
        oh1 = min(g.Ho, -((-(g.H + g.pt - i * g.dh)) // g.sh))
        ow0 = max(0, -((-(g.pl - j * g.dw)) // g.sw))
        ow1 = min(g.Wo, -((-(g.W + g.pl - j * g.dw)) // g.sw))
        if oh0 >= oh1 or ow0 >= ow1:
            return None
        return oh0, oh1, ow0, ow1

    def valid_counts(self):
        """The number of in-bounds taps per output row and column."""
        g = self
        rows = np.zeros(g.Ho, np.float32)
        cols = np.zeros(g.Wo, np.float32)
        for oh in range(g.Ho):
            n = 0
            for i in range(g.kh):
                y = oh * g.sh - g.pt + i * g.dh
                if 0 <= y < g.H:
                    n += 1
            rows[oh] = n
        for ow in range(g.Wo):
            n = 0
            for j in range(g.kw):
                x = ow * g.sw - g.pl + j * g.dw
                if 0 <= x < g.W:
                    n += 1
            cols[ow] = n
        return rows, cols


class _WindowPool(_NormBase):
    """A windowed sum over the spatial axes, one QPU lane copy per tap."""

    def window_sum(self, x, geom):
        n, C, Ho, Wo = geom.N, geom.C, geom.Ho, geom.Wo
        shape = (n, C, Ho, Wo)
        acc = self.alloc_fill(shape, 0.0)
        for i in range(geom.kh):
            for j in range(geom.kw):
                sp = geom.spans(i, j)
                if sp is None:
                    continue
                oh0, oh1, ow0, ow1 = sp
                rows, cols = oh1 - oh0, ow1 - ow0
                src_row = oh0 * geom.sh - geom.pt + i * geom.dh
                src_col = ow0 * geom.sw - geom.pl + j * geom.dw
                # the window rectangle is a sub-block, so the rest of the
                # tap buffer is a zero (the accumulator is a sum of taps)
                tap = self.alloc_fill(shape, 0.0)
                for p in range(n * C):
                    self.lane_runs(
                        self.r._buf_at(
                            x, (p * geom.H + src_row) * geom.W * 4
                            + src_col * 4),
                        self.r._buf_at(tap,
                                       (p * Ho + oh0) * Wo * 4 + ow0 * 4),
                        rows, cols, geom.sh * geom.W * 4, Wo * 4,
                        geom.sw * 4)
                acc = self.add(acc, tap)
        return acc

    def divisor(self, geom):
        """The per-output-pixel divisor as a [1][1][Ho][Wo] constant."""
        r, c = geom.valid_counts()
        d = np.outer(r, c).astype(np.float32).reshape(1, 1, geom.Ho, geom.Wo)
        if np.any(d == 0):
            self.fail("an output window lies entirely in the padding (the "
                      "ONNX average of an empty window is undefined)")
        key = ("pool_div", geom.Ho, geom.Wo, geom.pt, geom.pl, geom.kh,
               geom.kw, geom.sh, geom.sw, geom.dh, geom.dw, float(d.sum()))
        return self.const(key, np.ascontiguousarray(d))


class AveragePool(_WindowPool):
    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        g = _PoolGeom(node, tuple(x.shape), self.fail)
        cip = int(_attr(node, "count_include_pad", 0))
        r, c = g.valid_counts()
        uniform = cip or (r.min() == r.max() and c.min() == c.max())
        if not uniform:
            # a per-pixel divisor has to be a multiply, which is not IEEE for
            # a non-finite activation (see all_finite)
            self.all_finite(x, "AveragePool")
        acc = self.window_sum(x, g)
        if uniform:
            # every tap counts, or no border: the divisor is a scalar, so the
            # divide is one affine (IEEE-exact, including an inf window sum)
            d = float(g.window()) if cip else float(r[0] * c[0])
            return self.affine(acc, 1.0 / d, 0.0)
        db = self.bcast_t(self.divisor(g), tuple(acc.shape))
        return self.mul(acc, self.unary(db, vc4infer.UOP_RECIPROCAL))


class LpPool(_WindowPool):
    """(sum over the window of |x|^p) ^ (1/p), p in {1, 2}.

    ONNX defines no ``count_include_pad`` for LpPool and the reference pads
    with zero, so the padding contributes nothing to the sum and there is no
    divisor at all -- the window sum *is* the LpPool sum (docs/OPS-NORM.md).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        g = _PoolGeom(node, tuple(x.shape), self.fail)
        p = int(_attr(node, "p", 2))
        self.all_finite(x, "LpPool", max_abs=1e19 if p == 2 else None)
        if p == 1:
            t = self.unary(x, vc4infer.UOP_ABS)
        elif p == 2:
            t = self.mul(x, x)
        else:
            self.fail("p=%d: only p=1 and p=2 are exact (any other p needs a "
                      "pow on a possibly zero base)" % p)
        acc = self.window_sum(t, g)
        if p == 2:
            return self.unary(acc, vc4infer.UOP_SQRT)
        return acc


# =====================================================================
# MaxUnpool
# =====================================================================
class MaxUnpool(_NormBase):
    """Zero the output, then store each value at its flat index.

    ONNX (and onnx.reference) flatten the *whole* output buffer and write
    ``Y[I[k]] = X[k]`` in input order.  That is a scatter with a
    data-dependent destination, so the destination offsets are read back
    (int64 index data, the same class of work ArgMax already does) and the
    values are routed with ``copy2d_x`` runs: consecutive indices whose
    destination step is constant become one laned store, an arbitrary index
    costs one single-word store.  No value reaches the ARM.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("only 2-D spatial unpooling is wired (input rank %d)"
                      % len(x.shape))
        ks = _attr(node, "kernel_shape", None)
        if not ks or len(ks) != 2:
            self.fail("kernel_shape %s" % (ks,))
        kh, kw = int(ks[0]), int(ks[1])
        strides = _vec_attr(node, "strides", 2, [1, 1])
        pads = _attr(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
        if len(pads) != 4:
            self.fail("pads %s" % (list(pads),))
        pt, pl, pb, pr = (int(v) for v in pads)
        N, C, Xh, Xw = (int(d) for d in x.shape)
        inf = [N, C,
               (Xh - 1) * strides[0] - (pt + pb) + kh,
               (Xw - 1) * strides[1] - (pl + pr) + kw]
        if min(inf) <= 0:
            self.fail("inferred output shape %s is empty" % (inf,))
        out_shape = list(inf)
        if len(node.input) > 2 and node.input[2]:
            asked = [int(v) for v in
                     np.asarray(self.host_or_fail(node.input[2])).ravel()]
            if asked != out_shape:
                self.fail("output_shape %s differs from the inferred %s; the "
                          "reference writes into the inferred buffer first, "
                          "so a larger requested shape is a different "
                          "semantic that is not wired" % (asked, out_shape))
        total = _prod(inf)
        idx = np.asarray(self.host_or_fail(node.input[1]), np.int64).ravel()
        if idx.size != x.size:
            self.fail("indices has %d values for %d input values"
                      % (idx.size, x.size))
        if idx.size and (idx.min() < 0 or idx.max() >= total):
            self.fail("indices span [%d, %d] for an output of %d elements"
                      % (int(idx.min()), int(idx.max()), total))
        if np.unique(idx).size != idx.size:
            self.fail("duplicate indices: the ONNX result is the last write "
                      "in input order, and two QPU stores to one word are a "
                      "race")
        out = self.alloc_fill(tuple(inf), 0.0)
        k = 0
        n = int(idx.size)
        while k < n:
            step = int(idx[k + 1] - idx[k]) if k + 1 < n else 1
            j = k
            while j + 1 < n and int(idx[j + 1] - idx[j]) == step:
                j += 1
            if j > k and step > 0:
                self.lane_runs(self.r._buf_at(x, k * 4),
                               self.r._buf_at(out, int(idx[k]) * 4),
                               j - k + 1, 1, 4, step * 4, 4)
                k = j + 1
                continue
            # a step of 0 (refused above) or < 0 is not a lane stride of the
            # store: one single-word store, in input order
            self.r._runs(self.r._buf_at(x, k * 4),
                         self.r._buf_at(out, int(idx[k]) * 4), 1, 1, 0, 0)
            k += 1
        return self.tag(out, tuple(inf))


# =====================================================================
# Softmax / LogSoftmax / Hardmax
# =====================================================================
def _axis_of(node, rank, default):
    axis = int(_attr(node, "axis", default))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= max(rank, 1):
        raise UnsupportedNode(node.name, node.op_type,
                              "axis %d for rank %d"
                              % (int(_attr(node, "axis", default)), rank))
    return axis


class Softmax(_NormBase):
    """Softmax over `axis`, both opset conventions.

    opset >= 13: per-position over the axis (the default axis is -1).
    opset < 13: the input is coerced to 2-D [prod(before), prod(from)] first
    (the default axis is 1), so the classes are the flattened tail.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        opset13 = _opset(self.r.model) >= 13
        axis = _axis_of(node, rank, -1 if opset13 else 1)
        T, rows, D, back = self.rows_of(x, axis, opset13, "Softmax")
        if rows <= 0 or D <= 0:
            self.fail("empty input %s" % (tuple(x.shape),))
        if D == 1:
            return back(self.upload_const(np.ones((rows, 1), np.float32)))
        a = self.all_finite(T, "Softmax")
        if D <= 16 and rows % 8 == 0:
            # the fused kernel subtracts the row maximum internally: stable
            # for any finite input, one launch
            out = self.alloc((rows, D))
            self.r.eng.softmax_fwd(T.buf, out.buf, rows, D)
            return back(out)
        if a.size and float(np.max(np.abs(a))) <= 80.0:
            out = self.alloc((rows, D))
            self.r.eng.softmax(T.buf, out.buf, rows, D)
            return back(out)
        return back(self.stable_softmax(T, rows, D))


class LogSoftmax(_NormBase):
    """x - log(sum exp(x)) over `axis`, both opset conventions."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        opset13 = _opset(self.r.model) >= 13
        axis = _axis_of(node, rank, -1 if opset13 else 1)
        T, rows, D, back = self.rows_of(x, axis, opset13, "LogSoftmax")
        if rows <= 0 or D <= 0:
            self.fail("empty input %s" % (tuple(x.shape),))
        a = self.all_finite(T, "LogSoftmax")
        if a.size and float(np.max(np.abs(a))) <= 80.0:
            e = self.unary(T, vc4infer.UOP_EXP)
            s = self.row_sum(e, rows, D)
            self.download_check(s, lambda v: bool(np.min(v) > 0),
                                "LogSoftmax: the exp sum is not positive")
            lg = self.unary(s, vc4infer.UOP_LOG)
            return back(self.row_sub(T, lg, rows, D))
        # out of the direct exponential's range: shift by the row maximum
        m = self.row_max(T, rows, D)
        e = self.unary(self.row_sub(T, m, rows, D), vc4infer.UOP_EXP)
        s = self.row_sum(e, rows, D)
        self.download_check(s, lambda v: bool(np.min(v) > 0),
                            "LogSoftmax: the exp sum is not positive")
        lg = self.add(m, self.unary(s, vc4infer.UOP_LOG))
        return back(self.row_sub(T, lg, rows, D))


class Hardmax(_NormBase):
    """One-hot at the first maximum over `axis`, both opset conventions."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        opset13 = _opset(self.r.model) >= 13
        axis = _axis_of(node, rank, -1 if opset13 else 1)
        shape = [int(d) for d in x.shape]
        T, rows, D, back = self.rows_of(x, axis, opset13, "Hardmax")
        if D <= 0 or rows <= 0:
            self.fail("empty input %s" % (tuple(x.shape),))
        self.all_finite(T, "Hardmax", allow_inf=True)
        from ._dispatch_extra import ArgReduction
        idx = ArgReduction(self.r, node)._scan_impl(T, rows, D, True, False)
        mask = np.zeros((rows, D), np.float32)
        mask[np.arange(rows), idx.reshape(-1)] = 1.0
        return back(self.upload_const(mask))


# =====================================================================
# losses
# =====================================================================
class _Loss(_NormBase):
    """Shared: gather-weight mask, per-position loss, reduction."""

    def labels(self, name, n, C, what):
        arr = np.asarray(self.host_or_fail(name), np.int64).ravel()
        if arr.size != n:
            self.fail("%s %s has %d values (want %d)"
                      % (what, name, arr.size, n))
        return arr

    def weight(self, node, C):
        if len(node.input) > 2 and node.input[2]:
            w = np.asarray(self.host_or_fail(node.input[2]),
                           np.float32).ravel()
            if w.size != C:
                self.fail("weight has %d values (want %d)" % (w.size, C))
            return w
        return None

    def gather(self, tgt, N, D, C, ign, w, what):
        """(gather_weight, keep) and the negated one-hot mask [N*D][C]."""
        keep = np.ones((N, D), bool)
        if ign is not None:
            keep = tgt != ign
        bad = keep & ((tgt < 0) | (tgt >= C))
        if bad.any():
            self.fail("%s: target %d is outside [0, %d) and is not "
                      "ignore_index" % (what, int(tgt[bad].ravel()[0]), C))
        g = np.ones((N, D), np.float32) if w is None else \
            w[np.clip(tgt, 0, C - 1)]
        g = np.where(keep, g, 0.0).astype(np.float32)
        mask = np.zeros((N * D, C), np.float32)
        for n in range(N):
            for d in range(D):
                if keep[n, d]:
                    mask[n * D + d, int(tgt[n, d])] = -g[n, d]
        return g, keep, mask

    def reduce(self, loss_rows, rows, target_shape, reduction, g, keep):
        if reduction == "none":
            return [self.tag(loss_rows, tuple(target_shape))]
        if reduction not in ("mean", "sum"):
            self.fail("reduction=%r (none, mean, sum)" % (reduction,))
        total = self.sum_last(self.tag(loss_rows, (1, rows)), 1, rows)
        if reduction == "sum":
            return [self.tag(total, ())]
        if (g != np.ones_like(g)).any() or not keep.all():
            div = float(g.sum())
        else:
            div = float(rows)
        if div == 0.0:
            self.fail("reduction=mean over an empty (all ignore_index) set")
        return [self.tag(self.affine(total, 1.0 / div, 0.0), ())]


class NegativeLogLikelihoodLoss(_Loss):
    """-input[n, target[n, d], d], weighted, one GEMM per class sum."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        shape = [int(d) for d in x.shape]
        if len(shape) < 2:
            self.fail("input rank %d" % len(shape))
        N, C = shape[0], shape[1]
        D = _prod(shape[2:])
        if C <= 0 or D <= 0:
            self.fail("empty input %s" % (tuple(shape),))
        target_shape = list(shape[:1]) + shape[2:]
        self.all_finite(x, "NegativeLogLikelihoodLoss")
        tgt = self.labels(node.input[1], N * D, C, "target").reshape(N, D)
        ign = _attr(node, "ignore_index", None)
        w = self.weight(node, C)
        g, keep, mask = self.gather(tgt, N, D, C, ign, w,
                                    "NegativeLogLikelihoodLoss")
        T, P, Dc, I = self.ax_last(x, 1)
        if T is not x:
            self.tmp(T)
        rows = P * I
        # the labels are (usually) model constants, so the mask is a
        # constant pattern exactly like OneHot's; it is *not* cached across
        # runs because a runtime target would change it
        m = self.upload_const(mask)
        prod = self.mul(T, m)
        loss = self.row_sum(prod, rows, C)
        red = _attr(node, "reduction", "mean") or "mean"
        return self.reduce(loss, rows, target_shape, red, g, keep)


class SoftmaxCrossEntropyLoss(_Loss):
    """-log_softmax(scores)[label], weighted, with the optional log_prob."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        shape = [int(d) for d in x.shape]
        if len(shape) < 2:
            self.fail("input rank %d" % len(shape))
        N, C = shape[0], shape[1]
        D = _prod(shape[2:])
        if C <= 0 or D <= 0:
            self.fail("empty input %s" % (tuple(shape),))
        target_shape = list(shape[:1]) + shape[2:]
        self.all_finite(x, "SoftmaxCrossEntropyLoss")
        tgt = self.labels(node.input[1], N * D, C, "labels").reshape(N, D)
        ign = _attr(node, "ignore_index", None)
        w = self.weight(node, C)
        g, keep, mask = self.gather(tgt, N, D, C, ign, w,
                                    "SoftmaxCrossEntropyLoss")
        T, P, Dc, I = self.ax_last(x, 1)
        if T is not x:
            self.tmp(T)
        rows = P * I
        lp = self.logsoftmax_rows(T, rows, C)
        m = self.upload_const(mask)
        loss = self.row_sum(self.mul(lp, m), rows, C)
        red = _attr(node, "reduction", "mean") or "mean"
        outs = self.reduce(loss, rows, target_shape, red, g, keep)
        if len(node.output) > 1 and node.output[1]:
            outs.append(self.ax_back(lp, P, C, I, tuple(shape)))
        return outs

    def logsoftmax_rows(self, T, rows, D):
        self.download_check(T, lambda v: bool(np.max(np.abs(v)) <= 80.0),
                            "SoftmaxCrossEntropyLoss: scores outside the "
                            "documented direct-exponential range |x| <= 80")
        e = self.unary(T, vc4infer.UOP_EXP)
        s = self.row_sum(e, rows, D)
        self.download_check(s, lambda v: bool(np.min(v) > 0),
                            "SoftmaxCrossEntropyLoss: the exp sum is not "
                            "positive")
        lg = self.unary(s, vc4infer.UOP_LOG)
        return self.row_sub(T, lg, rows, D)


# =====================================================================
# dispatch table
# =====================================================================
def _mk(cls):
    def run(r, node):
        return cls(r, node).run()
    return run


HANDLERS = {
    "Dropout": _mk(Dropout),
    "BatchNormalization": _mk(BatchNormalization),
    "InstanceNormalization": _mk(InstanceNormalization),
    "GroupNormalization": _mk(GroupNormalization),
    "LayerNormalization": _mk(LayerNormalization),
    "MeanVarianceNormalization": _mk(MeanVarianceNormalization),
    "LRN": _mk(LRN),
    "GlobalMaxPool": _mk(GlobalMaxPool),
    "GlobalLpPool": _mk(GlobalLpPool),
    "AveragePool": _mk(AveragePool),
    "LpPool": _mk(LpPool),
    "MaxUnpool": _mk(MaxUnpool),
    "Softmax": _mk(Softmax),
    "LogSoftmax": _mk(LogSoftmax),
    "Hardmax": _mk(Hardmax),
    "SoftmaxCrossEntropyLoss": _mk(SoftmaxCrossEntropyLoss),
    "NegativeLogLikelihoodLoss": _mk(NegativeLogLikelihoodLoss),
}
