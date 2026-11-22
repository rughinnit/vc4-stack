"""Runner handlers for the reduction / comparison / shape-and-arith family.

This module is a second dispatch table for ``_onnxrun.Runner``.  It exists as
its own file so the operator families it covers (docs/OPS-REDUCE.md) stay in
one place.  Every handler is a composition of element-verified engine kernels:

    vc4_elementwise  Add/Mul/Relu/ReluBwd plus the exact fp32 comparison and
                     NaN/inf predicate bodies (kgen._ew_cmp)
    vc4_unary        Abs/Exp/Log/Sqrt/Affine/Sign
    vc4_matmul       the reduction engine: a sum over an axis is one GEMM with
                     a constant ones row, a prefix sum is one GEMM with a
                     constant triangular matrix
    vc4_copy2d*      every layout move (axis-to-last, broadcast, expand, tile,
                     pad, depth/space, the cumsum shift)
    vc4_fill2d       the Pad constant fill

Three kernels that look like the obvious primitives are NOT used, because they
hang or mis-answer on this board (measured 2026-11-20, docs/OPS-REDUCE.md):
``vc4_reduce`` (wedge), ``vc4_cumsum`` (wedge) and ``vc4_select3`` (wrong
result).  The compositions here replace them with verified kernels only.

The ARM never touches an activation value.  It only
  * builds integer index maps and geometry (broadcast/repeat/pad/tril maps),
  * uploads *constants* (a triangle mask, an eye matrix, a range, a one-hot
    pattern, the ones/triangular reduction matrices),
  * reads the arg-scan indices back for ``ArgMax``/``ArgMin`` and
    converts them to int64 (the class of work ``Shape`` and ``Gather`` already
    do), and
  * converts a 0.0/1.0 mask tensor to the bool dtype ONNX declares, which is
    a dtype conversion like ``Cast``.

Every handler works on compact NCHW (a pitched NHWC tensor is converted first
by a QPU copy) and returns a compact NCHW ``Tensor``, or a host numpy array
for the int64/bool results ONNX defines that way.
"""
import numpy as np
import onnx

from . import _binding as vc4infer
from ._errors import UnsupportedNode


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            t = a.type
            if t == onnx.AttributeProto.INT:
                return a.i
            if t == onnx.AttributeProto.FLOAT:
                return a.f
            if t == onnx.AttributeProto.STRING:
                return a.s.decode()
            if t == onnx.AttributeProto.INTS:
                return list(a.ints)
            if t == onnx.AttributeProto.FLOATS:
                return list(a.floats)
            if t == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _Tensor():
    from ._onnxrun import Tensor
    return Tensor


class _Base:
    """Shared helpers.  ``r`` is the Runner, ``node`` the ONNX node."""

    def __init__(self, r, node):
        self.r = r
        self.node = node

    def fail(self, detail):
        raise UnsupportedNode(self.node.name, self.node.op_type, detail)

    # ---------------------------------------------------------- values
    def ten(self, name, what="input"):
        """An fp32 activation tensor in compact NCHW."""
        from ._onnxrun import Tensor
        v = self.r.values.get(name)
        if not isinstance(v, Tensor):
            self.fail("%s %s is not an fp32 activation tensor" % (what, name))
        if v.layout == "nhwc":
            return self.r._to_nchw_buf(v)       # registered by _alloc_t
        return v

    def host(self, name):
        """The host copy of a graph *constant* (None for a runtime tensor).

        Constants are in ``Runner.host`` by name; a value the graph computes
        at run time is an arena tensor and is downloaded instead.
        """
        from ._onnxrun import Tensor
        v = self.r.host.get(name)
        if v is not None:
            return np.asarray(v)
        v = self.r.values.get(name)
        if isinstance(v, Tensor):
            return self.r.download(v)
        return None if v is None else np.asarray(v)

    def host_or_fail(self, name):
        v = self.host(name)
        if v is None:
            self.fail("input %s is not a constant or a host tensor" % name)
        return v

    def is_tensor(self, name):
        from ._onnxrun import Tensor
        return isinstance(self.r.values.get(name), Tensor)

    # ------------------------------------------------------- allocation
    def tmp(self, t):
        """Register a buffer as dead once this node finishes."""
        return self.r._tmp(t)

    def alloc(self, shape, layout="nchw"):
        return self.r._alloc_t(tuple(int(d) for d in shape), layout)

    def tag(self, buf, shape):
        t = _Tensor()(buf.buf if hasattr(buf, "buf") else buf,
                      tuple(int(d) for d in shape), "nchw")
        return self.tmp(t)

    def upload_const(self, arr, shape=None):
        """A *constant* placed in the arena (mask, eye, range, one-hot)."""
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if shape is not None:
            a = a.reshape(tuple(int(d) for d in shape))
        return self.tmp(self.r.upload(a))

    def const(self, key, arr):
        """A plan-time constant, cached like the runner's packed panels.

        Deliberately *not* registered as per-node garbage: it is reused by
        every run of the same Runner, exactly like a packed weight panel.
        """
        t = self.r._panel_get(key)
        if t is None:
            t = self.r._panel_alloc(
                key, lambda: self.r.upload(
                    np.ascontiguousarray(arr, dtype=np.float32)))
            self.r._panels[key] = t
        return t

    def const_like(self, key, t, value):
        """A constant tensor of t's shape (a full-size broadcast operand)."""
        return self.const((key, tuple(t.shape)),
                          np.full(tuple(t.shape), value, np.float32))

    # ----------------------------------------------------------- copies
    def fill(self, buf, n, value):
        """out[0:n] = value: one fill2d launch (the kernel stores 4 vectors
        per row, so a row is 64 words; the caller must have allocated the
        rounded-up size, see :meth:`alloc_fill`)."""
        rows = (int(n) + 63) // 64
        if rows:
            self.r.eng.fill2d(buf, rows, 4, 256, 64, float(value))

    def alloc_fill(self, shape, value):
        """A constant-filled activation buffer (fill2d rounds to 64 words)."""
        n = _prod(shape)
        words = ((n + 63) // 64) * 64
        t = self.r._alloc_t(tuple(int(d) for d in shape),
                            extra=words - n)
        self.fill(t.buf, n, value)
        return t

    def lane_runs(self, src, dst, rows, n, srow, drow, slane):
        """rows x n words: destination contiguous, source laned by ``slane``.

        Two copy2d_x calls at most (whole 16-word vectors, then the tail), so
        any lane stride -- including 0, a broadcast -- costs O(1) launches per
        row block.
        """
        n = int(n)
        rows = int(rows)
        if n <= 0 or rows <= 0:
            return
        if slane == 0:
            # broadcast one source word over the whole row: one 1-word store
            # per word, all reading the same source word (svec = 0)
            self.r._x(src, dst, rows, n, srow, drow, 0, 4, 4, 1, 1, 64, 64,
                      1, 1)
            return
        nv, rem = divmod(n, 16)
        if nv:
            units = 4 if nv % 4 == 0 else (2 if nv % 2 == 0 else 1)
            self.r._x(src, dst, rows, nv, srow, drow, 16 * slane, 64, slane,
                      16, 1, 64, 64, units, 16)
        if rem:
            self.r._x(self.r._buf_at(src, nv * 16 * slane),
                      self.r._buf_at(dst, nv * 16 * 4), rows, 1, srow, drow,
                      64, 64, slane, rem, 1, 64, 64, 1, rem)

    def copy_runs(self, src, dst, idx, total):
        """Copy through a flat source-index map, batching constant strides.

        ``idx`` holds word indices into ``src`` (-1 = leave the destination
        alone).  Consecutive entries with a constant step -- 0 for a broadcast,
        k for a strided gather -- become ONE lane copy, so a broadcast row, a
        padded row and a tiled row are a couple of launches, not one per
        element.  A negative step falls (correctly) back to single words.
        """
        k, n = 0, int(total)
        idx = np.asarray(idx, np.int64).ravel()
        while k < n:
            if idx[k] < 0:
                k += 1
                continue
            step = int(idx[k + 1] - idx[k]) if k + 1 < n else 1
            j = k
            while j + 1 < n and idx[j + 1] >= 0 and \
                    int(idx[j + 1] - idx[j]) == step:
                j += 1
            if j > k and step >= 1:
                # lane_runs, not Runner._runs: the latter assumes a 4-byte
                # lane stride for its tail offset, and a strided run is the
                # whole point here
                self.lane_runs(self.r._buf_at(src, int(idx[k]) * 4),
                               self.r._buf_at(dst, k * 4), 1, j - k + 1, 0,
                               0, 4 * step)
            elif j > k and step == 0:
                self.lane_runs(self.r._buf_at(src, int(idx[k]) * 4),
                               self.r._buf_at(dst, k * 4), 1, j - k + 1, 0,
                               0, 0)
            else:
                self.r._runs(self.r._buf_at(src, int(idx[k]) * 4),
                             self.r._buf_at(dst, k * 4), 1, 1, 0, 0)
                j = k
            k = j + 1

    def rowcopy(self, src, dst, run, mapping):
        """dst row k = src row mapping[k], ``run`` words each."""
        run = int(run)
        m = list(mapping)
        k = 0
        while k < len(m):
            step = m[k + 1] - m[k] if k + 1 < len(m) else 1
            j = k
            while j + 1 < len(m) and m[j + 1] - m[j] == step:
                j += 1
            if step >= 0:
                self.r._runs(self.r._buf_at(src, m[k] * run * 4),
                             self.r._buf_at(dst, k * run * 4), j - k + 1,
                             run, step * run * 4, run * 4)
            else:
                for q in range(k, j + 1):
                    self.r._runs(self.r._buf_at(src, m[q] * run * 4),
                                 self.r._buf_at(dst, q * run * 4), 1, run,
                                 0, 0)
            k = j + 1

    # ---------------------------------------------------------- layouts
    def ax_last(self, x, ax):
        """(T, P, D, I) with T[p*I + i][d] = x[p][d][i], on the QPUs.

        The reduction axis becomes the innermost one, and the destination row
        index is (p, i) -- exactly the ONNX output order when the axis is
        consumed, so a reduction needs no inverse pass.  An axis that is
        already innermost needs no copy at all (I == 1).
        """
        shape = [int(d) for d in x.shape]
        P = _prod(shape[:ax])
        D = shape[ax]
        I = _prod(shape[ax + 1:])
        if I == 1:
            return x, P, D, I
        out = self.alloc((P * I, D))
        for p in range(P):
            # dest rows are (i), each one vector of D words; the lanes are d,
            # whose source stride is I words, while i steps one word
            self.lane_runs(self.r._buf_at(x, p * D * I * 4),
                           self.r._buf_at(out, p * I * D * 4),
                           I, D, 4, D * 4, I * 4)
        return out, P, D, I

    def ax_back(self, T, P, D, I, shape):
        """Inverse of :meth:`ax_last`: dest[p][d][i] from T[p*I+i][d]."""
        if I == 1:
            return self.tag(T, shape)
        out = self.alloc(shape)
        for p in range(P):
            self.lane_runs(self.r._buf_at(T, p * I * D * 4),
                           self.r._buf_at(out, p * D * I * 4),
                           D, I, 4, I * 4, D * 4)
        return out

    def slice_col(self, T, k, outer, D):
        """Column k of [outer][D] as a compact [outer] tensor (lane copy)."""
        out = self.alloc((outer,))
        self.lane_runs(self.r._buf_at(T, k * 4), out.buf, 1, outer, 0, 0,
                       D * 4)
        return out

    # ------------------------------------------------------ elementwise
    def ew(self, a, b, op, shape=None):
        out = self.r.pool.alloc(a.size * 4 + 64)
        self.r.eng.elementwise(op, a.buf, out, a.size, b=b.buf)
        return self.tmp(_Tensor()(out, tuple(shape or a.shape), "nchw"))

    def ew1(self, a, op, shape=None):
        out = self.r.pool.alloc(a.size * 4 + 64)
        self.r.eng.elementwise(op, a.buf, out, a.size)
        return self.tmp(_Tensor()(out, tuple(shape or a.shape), "nchw"))

    def add(self, a, b, shape=None):
        return self.ew(a, b, vc4infer.OP_ADD, shape)

    def mul(self, a, b, shape=None):
        return self.ew(a, b, vc4infer.OP_MUL, shape)

    def sub(self, a, b, shape=None):
        """a - b on the QPUs: a + (-1)*b."""
        return self.add(a, self.affine(b, -1.0, 0.0), shape)

    def affine(self, a, p0, p1, shape=None):
        t = self.tmp(self.r._uop_on(a, vc4infer.UOP_AFFINE, float(p0),
                                    float(p1), 0.0))
        return self.tag(t, shape) if shape else t

    def unary(self, a, uop, p0=0.0, p1=0.0, shape=None):
        t = self.tmp(self.r._uop_on(a, uop, p0, p1, 0.0))
        return self.tag(t, shape) if shape else t

    def cmp(self, op, a, b):
        """Exact fp32 comparison -> a 0.0/1.0 tensor (bit-exact)."""
        return self.ew(a, b, op)

    def select_masks(self, m, a, b):
        """m ? a : b for a 0/1 mask (exact: two muls and an add)."""
        return self.add(self.mul(m, a), self.mul(self.affine(m, -1.0, 1.0), b))

    # ------------------------------------------------------ exact bit select
    def select(self, m, a, b):
        """m ? a : b for a 0/1 mask: bit-exact for every fp32 operand.

        ``select_masks`` (m*a + (1-m)*b) is wrong when the *unselected*
        operand is +-inf, because 0*inf is NaN on the QPU FPU (measured,
        docs/OPS-NORM.md).  ``ReluBwd`` masks with an integer AND on the
        operand bits (kgen OP_RELU_BWD), so both halves are copies and the ADD
        only ever adds a true zero.
        """
        hi = self.ew(self.affine(m, 1.0, -0.5), a, vc4infer.OP_RELU_BWD)
        lo = self.ew(self.affine(m, -1.0, 0.5), b, vc4infer.OP_RELU_BWD)
        return self.add(hi, lo)

    def maxmin(self, a, b, want_max):
        """numpy maximum/minimum: an infinite operand is copied, a NaN wins.

        ``m = cmp(a, b)`` is 0 when *either* operand is NaN, so an ``a`` that
        is NaN is forced into the selected slot by the ``IsNaN(a)`` term: the
        two masks cannot both be 1, because a NaN comparison is always false.
        """
        m = self.cmp(vc4infer.OP_CMP_GE if want_max else vc4infer.OP_CMP_LE,
                     a, b)
        return self.select(self.add(m, self.ew1(a, vc4infer.OP_ISNAN)), a, b)

    def shape_of(self, name):
        from ._onnxrun import Tensor
        v = self.r.values.get(name)
        if isinstance(v, Tensor):
            return tuple(int(d) for d in v.shape)
        return tuple(int(d) for d in np.asarray(self.host_or_fail(name)).shape)

    # ------------------------------------------------- indexed word movement
    # A copy2d vector reads 16 words from 16 independently addressed lanes and
    # stores them as 16 *contiguous* words.  So a gather (destination
    # contiguous, source arbitrary) is expressible run by run, while a scatter
    # (destination arbitrary) is expressible only when the destination words
    # are contiguous.  Every word map below is batched by a constant index
    # step: a gather of n words with a constant step costs ceil(n/16) launches
    # and a step of 0 costs one; a scatter costs n/16 launches when the
    # destination is consecutive and one launch per word otherwise.
    MAX_LANE_STEP = 200000

    def _laned_run(self, src, dst, s0, d0, n, step):
        """dst[d0 + d] = src[s0 + d*step] for n words."""
        n = int(n)
        if n <= 0:
            return
        if step == 1:
            self.r._runs(self.r._buf_at(src, s0 * 4),
                         self.r._buf_at(dst, d0 * 4), 1, n, 0, 0)
            return
        if step == 0:
            # one source word broadcast over the whole run: one launch
            self.r._x(self.r._buf_at(src, s0 * 4),
                      self.r._buf_at(dst, d0 * 4), 1, n, 0, 0, 0, 4, 4, 1, 1,
                      64, 64, 1, 1)
            return
        if step < 0 or step > self.MAX_LANE_STEP:
            for v in range(n):
                self.r._runs(self.r._buf_at(src, (s0 + v * step) * 4),
                             self.r._buf_at(dst, (d0 + v) * 4), 1, 1, 0, 0)
            return
        v = 0
        while v < n:
            depth = 16 if n - v >= 16 else n - v
            self.r._x(self.r._buf_at(src, (s0 + v * step) * 4),
                      self.r._buf_at(dst, (d0 + v) * 4), 1, 1, 0, 0, 64, 64,
                      4 * step, depth, 1, 64, 64, 1, depth)
            v += depth

    def word_gather(self, src, dst, idx, total):
        """dst word d = src word idx[d]; idx[d] < 0 leaves the word alone."""
        idx = np.asarray(idx, np.int64).ravel()
        n = int(total)
        k = 0
        while k < n:
            s0 = int(idx[k])
            if s0 < 0:
                k += 1
                continue
            step = int(idx[k + 1] - s0) if (k + 1 < n and idx[k + 1] >= 0) \
                else 0
            j = k
            while j + 1 < n and idx[j + 1] >= 0 and \
                    int(idx[j + 1] - idx[j]) == step:
                j += 1
            self._laned_run(src, dst, s0, k, j - k + 1, step)
            k = j + 1

    def word_scatter(self, src, dst, didx, total):
        """dst word didx[d] = src word d; didx[d] < 0 skips d."""
        didx = np.asarray(didx, np.int64).ravel()
        n = int(total)
        k = 0
        while k < n:
            d0 = int(didx[k])
            if d0 < 0:
                k += 1
                continue
            step = int(didx[k + 1] - d0) if (k + 1 < n and didx[k + 1] >= 0) \
                else 0
            j = k
            while j + 1 < n and didx[j + 1] >= 0 and \
                    int(didx[j + 1] - didx[j]) == step:
                j += 1
            if step == 1:
                self.r._runs(self.r._buf_at(src, k * 4),
                             self.r._buf_at(dst, d0 * 4), 1, j - k + 1, 0, 0)
            else:
                for q in range(k, j + 1):
                    self.r._runs(self.r._buf_at(src, q * 4),
                                 self.r._buf_at(dst, int(didx[q]) * 4), 1, 1,
                                 0, 0)
            k = j + 1

    def sum_last_aligned(self, T, rows, D):
        """sum over the innermost axis, with the GEMM operands padded on the QPU.

        ``_ReduceMixin.sum_last`` hands ``vc4_matmul`` an A whose M is not
        8-aligned or whose K is not 16-aligned; the library then pads by
        copying that operand *with the ARM*, and on this cacheable arena
        mapping that copy can read a line the QPU has already overwritten
        (measured: a 16x4x16 GEMM returned another buffer's bytes in its first
        rows).  Any composition over a QPU-*produced* operand must pad on the
        QPUs instead, which is what this does.
        """
        if D <= 0:
            self.fail("reducing an empty axis")
        Dp = (D + 15) & ~15
        M = (rows + 7) & ~7
        ones = self.const(("ones", Dp), np.ones((Dp, 16), np.float32))
        A = self.alloc_fill((M, Dp), 0.0)
        self.lane_runs(T.buf, A.buf, rows, D, D * 4, Dp * 4, 4)
        res = self.alloc((M, 16))
        self.r.eng.matmul(A.buf, ones.buf, res.buf, M, 16, Dp)
        return self.slice_col(res, 0, rows, 16)

    def tile_rows(self, row, rows, D):
        """[D] -> [rows][D]: every row is the same D-word block (srow 0)."""
        out = self.alloc((rows, D))
        self.lane_runs(row.buf, out.buf, rows, D, 0, D * 4, 4)
        return out

    def bcast_rows(self, vec, rows, D):
        """[rows] -> [rows][D]: row r is vec[r] repeated D times, one launch."""
        out = self.alloc((rows, D))
        self.lane_runs(vec.buf, out.buf, rows, D, 4, D * 4, 0)
        return out

    def rows_scatter(self, src, dst, run, mapping):
        """dst row mapping[k] = src row k, ``run`` words each."""
        m = [int(v) for v in mapping]
        k = 0
        while k < len(m):
            step = (m[k + 1] - m[k]) if k + 1 < len(m) else 0
            j = k
            while j + 1 < len(m) and m[j + 1] - m[j] == step:
                j += 1
            if step == 1:
                self.r._runs(self.r._buf_at(src, k * run * 4),
                             self.r._buf_at(dst, m[k] * run * 4), j - k + 1,
                             run, run * 4, run * 4)
            else:
                for q in range(k, j + 1):
                    self.r._runs(self.r._buf_at(src, q * run * 4),
                                 self.r._buf_at(dst, m[q] * run * 4), 1, run,
                                 0, 0)
            k = j + 1

    # ------------------------------------------------------ broadcasting
    def bcast(self, name, shape):
        """`name` as a device tensor of ``shape`` (broadcast if needed)."""
        shape = tuple(int(d) for d in shape)
        if self.is_tensor(name):
            t = self.ten(name)
            if tuple(t.shape) == shape:
                return t
            return self.bcast_t(t, shape)
        arr = np.asarray(self.host_or_fail(name), np.float32)
        try:
            full = np.broadcast_to(arr, shape)
        except ValueError:
            self.fail("operand %s (%s) does not broadcast to %s"
                      % (name, arr.shape, shape))
        return self.upload_const(np.ascontiguousarray(full))

    def bcast_t(self, t, shape):
        """Broadcast a compact NCHW tensor with QPU lane copies."""
        shape = tuple(int(d) for d in shape)
        idx = _bcast_index(tuple(t.shape), shape)
        out = self.alloc(shape)
        self.copy_runs(t.buf, out.buf, idx, _prod(shape))
        return out

    # ---------------------------------------------------------- results
    def bool_of(self, t):
        """0.0/1.0 tensor -> the bool array ONNX declares (a dtype cast)."""
        return (self.r.download(t) != 0)

    def idx_of(self, t, shape):
        """int32 indices in an fp32 arena buffer -> int64 numpy."""
        raw = self.r.eng.download(t.buf, t.size * 4)
        return np.asarray(raw, np.float32).view(np.int32).reshape(shape) \
            .astype(np.int64)

    def download_check(self, t, ok, msg):
        """Refuse a value-dependent kernel domain violation, naming it."""
        a = self.r.download(t)
        if a.size and not ok(a):
            self.fail("%s (measured range [%r, %r])"
                      % (msg, float(np.min(a)), float(np.max(a))))
        return a


def _bcast_index(in_shape, out_shape):
    """Flat source index per flat destination index, for a broadcast."""
    n = _prod(out_shape)
    ish = [1] * (len(out_shape) - len(in_shape)) + list(in_shape)
    strides = []
    acc = 1
    for d in reversed(ish):
        strides.append(acc)
        acc *= d
    strides.reverse()
    idx = np.zeros(n, np.int64)
    mult = 1
    for k in range(len(out_shape) - 1, -1, -1):
        idx += (np.arange(n, dtype=np.int64) // mult % out_shape[k]) \
            % ish[k] * strides[k]
        mult *= out_shape[k]
    return idx


# =====================================================================
# the reduction engine: vc4_matmul with constant operand matrices
# =====================================================================
class _ReduceMixin:

    def gemm_last(self, T, outer, D, mat):
        """[outer][D] x mat[D][D] -> [outer][D] on the QPU (one GEMM)."""
        out = self.alloc((outer, D))
        self.r.eng.matmul(T.buf, mat.buf, out.buf, outer, D, D)
        return out

    def sum_last(self, T, outer, D):
        """sum over the innermost axis: one GEMM with a ones column block."""
        if D == 0:
            self.fail("reducing an empty axis")
        ones = self.const(("ones", D), np.ones((D, 16), np.float32))
        res = self.alloc((outer, 16))
        self.r.eng.matmul(T.buf, ones.buf, res.buf, outer, 16, D)
        return self.slice_col(res, 0, outer, 16)

    def max_last(self, T, outer, D, op):
        """max/min over the innermost axis: an exact masked chain."""
        if D == 0:
            self.fail("reducing an empty axis")
        acc = self.slice_col(T, 0, outer, D)
        cmp = vc4infer.OP_CMP_GE if op == "max" else vc4infer.OP_CMP_LE
        for k in range(1, D):
            v = self.slice_col(T, k, outer, D)
            m = self.cmp(cmp, acc, v)
            acc = self.add(self.mul(m, acc),
                           self.mul(self.affine(m, -1.0, 1.0), v))
        return acc

    def prod_last(self, T, outer, D):
        acc = self.slice_col(T, 0, outer, D)
        for k in range(1, D):
            acc = self.mul(acc, self.slice_col(T, k, outer, D))
        return acc

    def tri_last(self, T, outer, D, rev):
        """Inclusive prefix sum along the innermost axis, one GEMM.

        out[k] = sum_{j <= k} T[j]  (forward)  or  sum_{j >= k} T[j]  (rev),
        i.e. one matrix whose (j, k) entry is the indicator, so the whole scan
        is a single GEMM against a constant matrix.
        """
        ones = np.ones((D, D), np.float32)
        m = np.tril(ones) if rev else np.triu(ones)
        tri = self.const(("tri", D, bool(rev)), np.ascontiguousarray(m))
        return self.gemm_last(T, outer, D, tri)


# =====================================================================
# reductions
# =====================================================================
class Reduction(_Base, _ReduceMixin):

    def axes(self, rank):
        node = self.node
        axes = _attr(node, "axes", None)
        if len(node.input) > 1 and node.input[1]:
            axes = np.asarray(self.host_or_fail(node.input[1])).ravel().tolist()
        noop = int(_attr(node, "noop_with_empty_axes", 0))
        if axes is None or len(axes) == 0:
            if noop:
                return None
            return sorted(range(rank), reverse=True)
        out = []
        for a in axes:
            a = int(a)
            if a < 0:
                a += rank
            if a < 0 or a >= rank:
                self.fail("axis %d out of range for rank %d" % (a, rank))
            if a not in out:
                out.append(a)
        return sorted(out, reverse=True)

    def one(self, t, ax, fn):
        """Reduce axis `ax` (made innermost) and retag the logical shape."""
        cur = [int(d) for d in t.shape]
        T, P, D, I = self.ax_last(t, ax)
        if T is not t:
            self.tmp(T)
        res = fn(T, P * I, D)
        return self.tag(res, cur[:ax] + cur[ax + 1:])

    def run(self):
        node = self.node
        op = node.op_type
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axes = self.axes(rank)
        keepdims = int(_attr(node, "keepdims", 1))
        if axes is None:
            return x
        if keepdims:
            shape = [1 if i in axes else int(d)
                     for i, d in enumerate(x.shape)]
        else:
            shape = [int(d) for i, d in enumerate(x.shape) if i not in axes]
        count = 1
        for i in axes:
            count *= int(x.shape[i])

        if op in ("ReduceSum", "ReduceMean"):
            t = x
            for ax in axes:
                t = self.one(t, ax, self.sum_last)
            if op == "ReduceMean":
                if count == 0:
                    self.fail("ReduceMean of an empty axis")
                t = self.affine(t, 1.0 / count, 0.0)
            return self.tag(t, shape)

        if op in ("ReduceMax", "ReduceMin"):
            which = "max" if op == "ReduceMax" else "min"
            t = x
            for ax in axes:
                t = self.one(t, ax,
                             lambda T, o, d: self.max_last(T, o, d, which))
            return self.tag(t, shape)

        if op == "ReduceProd":
            t = x
            for ax in axes:
                t = self.one(t, ax, self.prod_last)
            return self.tag(t, shape)

        # the composed family: transform, then sum
        if op == "ReduceL1":
            t = self.unary(x, vc4infer.UOP_ABS)
        elif op in ("ReduceL2", "ReduceSumSquare"):
            t = self.mul(x, x)
        elif op == "ReduceLogSumExp":
            self.download_check(
                x, lambda a: bool(np.isfinite(a).all()
                                  and np.max(np.abs(a)) <= 88.0),
                "ReduceLogSumExp: exp is outside the kernel's documented "
                "range |x| <= 88")
            t = self.unary(x, vc4infer.UOP_EXP)
        else:
            t = x
        for ax in axes:
            t = self.one(t, ax, self.sum_last)
        if op == "ReduceL2":
            t = self.unary(t, vc4infer.UOP_SQRT)
        elif op in ("ReduceLogSum", "ReduceLogSumExp"):
            self.download_check(t, lambda a: bool(np.min(a) > 0),
                                "%s: log of a non-positive sum" % op)
            t = self.unary(t, vc4infer.UOP_LOG)
        return self.tag(t, shape)


class ArgReduction(_Base):

    def run(self):
        node = self.node
        op = node.op_type
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = int(_attr(node, "axis", 0))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d out of range for rank %d" % (axis, rank))
        keepdims = int(_attr(node, "keepdims", 1))
        last = int(_attr(node, "select_last_index", 0))
        T, P, D, I = self.ax_last(x, axis)
        self.tmp(T)
        outer = P * I
        if D == 0:
            self.fail("arg reduce of an empty axis")
        idx = self._scan_impl(T, outer, D, op == "ArgMax", last)
        shape = [int(d) for d in x.shape]
        if keepdims:
            shape = [1 if i == axis else d for i, d in enumerate(shape)]
        else:
            shape = [d for i, d in enumerate(shape) if i != axis]
        return idx.reshape(shape)

    def _scan_impl(self, T, outer, D, is_max, last):
        cmp = (vc4infer.OP_CMP_GE if last else vc4infer.OP_CMP_GT) if is_max \
            else (vc4infer.OP_CMP_LE if last else vc4infer.OP_CMP_LT)
        best = self.slice_col(T, 0, outer, D)
        idx = self.upload_const(np.zeros(outer, np.float32))
        for k in range(1, D):
            v = self.slice_col(T, k, outer, D)
            m = self.cmp(cmp, v, best)                 # v beats best?
            best = self.add(self.mul(m, v),
                            self.mul(self.affine(m, -1.0, 1.0), best))
            kk = self.const(("idxk", k), np.full(outer, float(k), np.float32))
            idx = self.add(idx, self.mul(m, self.sub(kk, idx)))
        return self.r.download(idx).astype(np.int64).reshape(outer)


# =====================================================================
# comparisons / predicates / select
# =====================================================================
_CMP = {
    "Equal": vc4infer.OP_CMP_EQ,
    "Greater": vc4infer.OP_CMP_GT,
    "GreaterOrEqual": vc4infer.OP_CMP_GE,
    "Less": vc4infer.OP_CMP_LT,
    "LessOrEqual": vc4infer.OP_CMP_LE,
}


class Compare(_Base):

    def _shape(self, name):
        if self.is_tensor(name):
            return tuple(self.r.values[name].shape)
        return tuple(np.asarray(self.host_or_fail(name)).shape)

    def run(self):
        a_name, b_name = self.node.input[0], self.node.input[1]
        try:
            shape = np.broadcast_shapes(self._shape(a_name),
                                        self._shape(b_name))
        except ValueError:
            self.fail("%s and %s do not broadcast"
                      % (self._shape(a_name), self._shape(b_name)))
        a = self.bcast(a_name, shape)
        b = self.bcast(b_name, shape)
        return self.bool_of(self.cmp(_CMP[self.node.op_type], a, b))


class Predicate(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if node.op_type == "IsNaN":
            return self.bool_of(self.ew1(x, vc4infer.OP_ISNAN))
        neg = int(_attr(node, "detect_negative", 1))
        pos = int(_attr(node, "detect_positive", 1))
        if not neg and not pos:
            return np.zeros(tuple(x.shape), bool)
        inf = self.ew1(x, vc4infer.OP_ISINF)
        if neg and not pos:
            t = self.ew(self.affine(x, -1.0, 0.0), inf, vc4infer.OP_RELU_BWD)
        elif pos and not neg:
            t = self.ew(x, inf, vc4infer.OP_RELU_BWD)
        else:
            t = inf
        return self.bool_of(t)


class Where(_Base):

    def run(self):
        node = self.node
        names = [node.input[0], node.input[1], node.input[2]]
        shapes = []
        for nm in names:
            shapes.append(tuple(self.r.values[nm].shape) if self.is_tensor(nm)
                          else tuple(np.asarray(self.host_or_fail(nm)).shape))
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError:
            self.fail("Where operands %s do not broadcast" % (shapes,))
        c = self.bcast(names[0], shape)
        a = self.bcast(names[1], shape)
        b = self.bcast(names[2], shape)
        return self.select_masks(c, a, b)


# =====================================================================
# CumSum
# =====================================================================
class CumSum(_Base, _ReduceMixin):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        excl = int(_attr(node, "exclusive", 0))
        rev = int(_attr(node, "reverse", 0))
        T, P, D, I = self.ax_last(x, axis)
        self.tmp(T)
        rows = P * I
        if D == 0 or rows == 0:
            return x
        res = self.tri_last(T, rows, D, rev)
        if excl:
            res = self._shift(res, rows, D, rev)
        return self.ax_back(res, P, D, I, x.shape)

    def _shift(self, res, rows, D, rev):
        """exclusive: forward out[k] = incl[k-1], reverse out[k] = revincl[k+1].

        Both are one contiguous run per row into a zero-filled buffer; the
        zero sits at the start (forward) or the end (reverse).
        """
        out = self.alloc_fill((rows, D), 0.0)
        if D > 1:
            if rev:
                self.r._runs(self.r._buf_at(res, 4), out.buf, rows, D - 1,
                             D * 4, D * 4)
            else:
                self.r._runs(res.buf, self.r._buf_at(out, 4), rows, D - 1,
                             D * 4, D * 4)
        return out


# =====================================================================
# shape / arithmetic family
# =====================================================================
class Expand(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        target = tuple(int(v) for v in
                       np.asarray(self.host_or_fail(node.input[1])).ravel())
        if not target:
            return x
        if len(target) < len(x.shape):
            self.fail("Expand target rank %d < input rank %d"
                      % (len(target), len(x.shape)))
        try:
            out_shape = tuple(int(d) for d in np.broadcast_shapes(
                tuple(x.shape), target))
        except ValueError:
            self.fail("Expand %s -> %s" % (tuple(x.shape), target))
        idx = _bcast_index(tuple(x.shape), out_shape)
        out = self.alloc(out_shape)
        self.copy_runs(x.buf, out.buf, idx, _prod(out_shape))
        return out


class Tile(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        reps = [int(v) for v in
                np.asarray(self.host_or_fail(node.input[1])).ravel()]
        if len(reps) != len(x.shape):
            self.fail("Tile repeats rank %d != input rank %d"
                      % (len(reps), len(x.shape)))
        if any(r < 1 for r in reps):
            self.fail("Tile repeats %s must be >= 1" % (reps,))
        out_shape = tuple(int(d) * r for d, r in zip(x.shape, reps))
        idx = _bcast_index(tuple(x.shape), out_shape)
        out = self.alloc(out_shape)
        self.copy_runs(x.buf, out.buf, idx, _prod(out_shape))
        return out


class Pad(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        pads = [int(v) for v in
                np.asarray(self.host_or_fail(node.input[1])).ravel()]
        mode = _attr(node, "mode", "constant") or "constant"
        cval = 0.0
        if len(node.input) > 2 and node.input[2]:
            cval = float(np.asarray(self.host_or_fail(node.input[2])
                                    ).ravel()[0])
        axes = None
        if len(node.input) > 3 and node.input[3]:
            axes = [int(v) for v in
                    np.asarray(self.host_or_fail(node.input[3])).ravel()]
        if axes is None:
            if len(pads) != 2 * rank:
                self.fail("Pad pads has %d values, rank is %d"
                          % (len(pads), rank))
            beg, end = pads[:rank], pads[rank:]
        else:
            if len(pads) != 2 * len(axes):
                self.fail("Pad pads has %d values for %d axes"
                          % (len(pads), len(axes)))
            beg, end = [0] * rank, [0] * rank
            for i, ax in enumerate(axes):
                ax = ax + rank if ax < 0 else ax
                if ax < 0 or ax >= rank:
                    self.fail("Pad axis %d for rank %d" % (ax, rank))
                beg[ax] = pads[i]
                end[ax] = pads[i + len(axes)]
        if any(p < 0 for p in beg + end):
            self.fail("Pad negative pads %s" % (list(beg) + list(end),))
        if not any(beg + end):
            return x
        out_shape = [d + b + e for d, b, e in zip(x.shape, beg, end)]
        if any(d <= 0 for d in out_shape):
            self.fail("Pad output %s is empty" % (out_shape,))
        if mode not in ("constant", "reflect", "edge", "wrap"):
            self.fail("Pad mode=%s (constant, reflect, edge, wrap)" % mode)
        if mode == "reflect":
            for i, (b, e) in enumerate(zip(beg, end)):
                if b >= x.shape[i] or e >= x.shape[i]:
                    self.fail("Pad reflect pad %d/%d on dim %d of %d"
                              % (b, e, i, x.shape[i]))
        idx = _pad_index(tuple(x.shape), tuple(out_shape), beg, mode)
        if mode == "constant":
            out = self.alloc_fill(out_shape, cval)
        else:
            out = self.alloc(out_shape)
        self.copy_runs(x.buf, out.buf, idx, _prod(out_shape))
        return out


def _pad_index(in_shape, out_shape, beg, mode):
    """Flat source index per output element; -1 leaves the fill value."""
    total = _prod(out_shape)
    idx = np.zeros(total, np.int64)
    strides = []
    acc = 1
    for d in reversed(in_shape):
        strides.append(acc)
        acc *= d
    strides.reverse()
    mult = 1
    ok = np.ones(total, bool)
    for k in range(len(out_shape) - 1, -1, -1):
        o = np.arange(total, dtype=np.int64) // mult % out_shape[k]
        o = o - beg[k]
        D = in_shape[k]
        if mode == "constant":
            ok &= (o >= 0) & (o < D)
            o = np.clip(o, 0, D - 1)
        elif mode == "edge":
            o = np.clip(o, 0, D - 1)
        elif mode == "wrap":
            o = o % D
        else:                                          # reflect
            period = max(1, 2 * D - 2)
            o = np.mod(o, period)
            o = np.where(o < D, o, period - o)
        idx += o * strides[k]
        mult *= out_shape[k]
    if mode == "constant":
        return np.where(ok, idx, -1)
    return idx


class Compress(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        cond = np.asarray(self.host_or_fail(node.input[1])).ravel()
        axis = _attr(node, "axis", None)
        if axis is None:
            self.fail("Compress axis=None selects single elements from the "
                      "flattened input; that is a per-element scatter, and "
                      "copy2d destination vectors are 16 contiguous words "
                      "(give an axis)")
        rank = len(x.shape)
        axis = int(axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("Compress axis %d for rank %d" % (axis, rank))
        d = int(x.shape[axis])
        keep = np.nonzero(cond != 0)[0]
        keep = keep[keep < d]
        inner = _prod(x.shape[axis + 1:])
        outer = _prod(x.shape[:axis])
        out_shape = list(x.shape)
        out_shape[axis] = len(keep)
        out = self.alloc(out_shape)
        for o in range(outer):
            self.rowcopy(self.r._buf_at(x, o * d * inner * 4),
                         self.r._buf_at(out, o * len(keep) * inner * 4),
                         inner, [int(v) for v in keep])
        return out


class Pow(_Base):
    """Pow: exact for a small integer exponent, exp(b*log a) otherwise.

    A negative base with a non-integer exponent is NaN in ONNX and outside the
    QPU log kernel's domain, so those shapes are refused rather than answered
    with a number the kernels cannot stand behind.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        exp_name = node.input[1]
        const = self.r.host.get(exp_name)
        if const is not None:
            e = np.asarray(const, np.float64)
            if e.size == 1:
                k = float(e.ravel()[0])
                if k == int(k) and abs(int(k)) <= 4:
                    return self._int_pow(x, int(k))
                if k == 0.5:
                    return self.unary(x, vc4infer.UOP_SQRT)
                if k == -1.0:
                    return self.unary(x, vc4infer.UOP_RECIPROCAL)
        try:
            shape = np.broadcast_shapes(tuple(x.shape), self._shape(exp_name))
        except ValueError:
            self.fail("%s and %s do not broadcast"
                      % (tuple(x.shape), self._shape(exp_name)))
        a = self.bcast(node.input[0], shape)
        b = self.bcast(exp_name, shape)
        self.download_check(a, lambda v: bool(np.min(v) > 0),
                            "Pow of a non-positive base (exp(b*log a) needs "
                            "a > 0; use a constant integer exponent for a "
                            "negative base)")
        lg = self.unary(a, vc4infer.UOP_LOG)
        t = self.mul(b, lg)
        return self.unary(t, vc4infer.UOP_EXP)

    def _shape(self, name):
        if self.is_tensor(name):
            return tuple(self.r.values[name].shape)
        return tuple(np.asarray(self.host_or_fail(name)).shape)

    def _int_pow(self, x, k):
        if k == 1:
            return x
        if k == 0:
            return self.upload_const(np.ones(tuple(x.shape), np.float32))
        n = abs(k)
        acc = x
        for _ in range(n - 1):
            acc = self.mul(acc, x)
        if k < 0:
            return self.unary(acc, vc4infer.UOP_RECIPROCAL)
        return acc


class Mod(_Base):
    """Mod: a - b*floor(a/b) (fmod=0) or a - b*trunc(a/b) (fmod=1).

    q = a * rcp(b), so a zero divisor is inf and the result is NaN; ONNX leaves
    that case undefined for fmod=1 and integer division by zero is an error, so
    the node raises on a constant zero divisor instead of returning one.
    """

    def run(self):
        node = self.node
        fmod = int(_attr(node, "fmod", 0))
        names = [node.input[0], node.input[1]]
        shapes = []
        for nm in names:
            shapes.append(tuple(self.r.values[nm].shape) if self.is_tensor(nm)
                          else tuple(np.asarray(self.host_or_fail(nm)).shape))
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError:
            self.fail("%s do not broadcast" % (shapes,))
        a = self.bcast(names[0], shape)
        b = self.bcast(names[1], shape)
        if not self.is_tensor(names[1]):
            c = np.asarray(self.host_or_fail(names[1]), np.float64)
            if c.size and np.any(c == 0):
                self.fail("Mod by a constant 0")
        zero = self.const(("zero", tuple(shape)),
                          np.zeros(tuple(shape), np.float32))
        q = self.mul(a, self.unary(b, vc4infer.UOP_RECIPROCAL))
        n = self.unary(q, vc4infer.UOP_ROUND)          # nearest integer
        r = self.sub(a, self.mul(b, n))                # |r| <= |b|/2 + eps
        # The reciprocal is 1e-7 off, so `round(a/b)` is wrong only when a/b
        # sits within that of a half integer; the exact remainder is then one
        # of r, r-b, r+b and the test below picks it from a sign comparison,
        # never from a near-integer decision.
        if not fmod:
            # floor(a/b) = n - 1 exactly when r and b have opposite signs
            c = self.cmp(vc4infer.OP_CMP_LT, self.mul(r, b), zero)
            return self.add(r, self.mul(c, b))
        # trunc(a/b): when the rounding went away from zero -- r and a have
        # opposite signs -- the remainder gains |b| in the direction of a
        c = self.cmp(vc4infer.OP_CMP_LT, self.mul(r, a), zero)
        sign_a = self.unary(a, vc4infer.UOP_SIGN)       # +1 / 0 / -1
        step = self.mul(self.unary(b, vc4infer.UOP_ABS), sign_a)
        return self.add(r, self.mul(c, step))


class OneHot(_Base):

    def run(self):
        node = self.node
        idx = np.asarray(self.host_or_fail(node.input[0]), np.int64)
        depth = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        vals = np.asarray(self.host_or_fail(node.input[2]), np.float32).ravel()
        if vals.size != 2:
            self.fail("OneHot values has %d entries (2)" % vals.size)
        if depth <= 0:
            self.fail("OneHot depth %d" % depth)
        axis = int(_attr(node, "axis", -1))
        rank = idx.ndim
        if axis < 0:
            axis += rank + 1
        if axis < 0 or axis > rank:
            self.fail("OneHot axis %d for rank %d" % (axis, rank))
        out_shape = list(idx.shape)
        out_shape.insert(axis, depth)
        mask = np.zeros(out_shape, np.float32)
        inside = (idx >= 0) & (idx < depth)
        np.put_along_axis(mask, np.expand_dims(np.where(inside, idx, 0), axis),
                          np.expand_dims(np.where(inside, 1.0, 0.0), axis),
                          axis=axis)
        m = self.const(("onehot", tuple(out_shape)), mask)
        on, off = float(vals[1]), float(vals[0])
        span = self.add(self.mul(m, self.const_like("onehot_on", m, on - off)),
                        self.const_like("onehot_off", m, off))
        return self.tag(span, tuple(out_shape))


class Trilu(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) < 2:
            self.fail("Trilu needs a rank >= 2 input")
        upper = int(_attr(node, "upper", 1))
        k = 0
        if len(node.input) > 1 and node.input[1]:
            k = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        rows, cols = int(x.shape[-2]), int(x.shape[-1])
        r = np.arange(rows).reshape(rows, 1)
        c = np.arange(cols).reshape(1, cols)
        keep = (c - r >= k) if upper else (c - r <= k)
        mask = np.broadcast_to(keep, tuple(x.shape)).astype(np.float32)
        return self.mul(x, self.upload_const(np.ascontiguousarray(mask)))


class Range(_Base):

    def run(self):
        node = self.node

        def scalar(nm):
            return float(np.asarray(self.host_or_fail(nm)).ravel()[0])

        start, limit, delta = (scalar(node.input[0]), scalar(node.input[1]),
                               scalar(node.input[2]))
        if delta == 0.0:
            self.fail("Range delta is 0")
        n = int(max(np.ceil((limit - start) / delta), 0))
        arr = (start + np.arange(n, dtype=np.float64) * delta).astype(np.float32)
        return self.upload_const(arr)


class EyeLike(_Base):

    def run(self):
        node = self.node
        shape = (tuple(self.r.values[node.input[0]].shape)
                 if self.is_tensor(node.input[0])
                 else tuple(np.asarray(self.host_or_fail(node.input[0])).shape))
        if len(shape) != 2:
            self.fail("EyeLike needs a 2-D input (got %s)" % (shape,))
        k = int(_attr(node, "k", 0))
        rows, cols = int(shape[0]), int(shape[1])
        eye = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            j = i + k
            if 0 <= j < cols:
                eye[i, j] = 1.0
        dt = int(_attr(node, "dtype", 0)) or onnx.TensorProto.FLOAT
        if dt != onnx.TensorProto.FLOAT:
            if dt not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                self.fail("EyeLike dtype %s"
                          % onnx.TensorProto.DataType.Name(dt))
            return eye.astype(np.int64 if dt == onnx.TensorProto.INT64
                              else np.int32)
        return self.upload_const(eye)


class CastLike(_Base):

    def run(self):
        node = self.node
        from ._onnxrun import Tensor
        x = self.r.values[node.input[0]]
        tgt = self.r.values.get(node.input[1])
        to = onnx.TensorProto.FLOAT
        if tgt is not None and not isinstance(tgt, Tensor):
            dt = np.asarray(tgt).dtype
            to = {np.dtype(np.float32): onnx.TensorProto.FLOAT,
                  np.dtype(np.int64): onnx.TensorProto.INT64,
                  np.dtype(np.int32): onnx.TensorProto.INT32,
                  np.dtype(np.bool_): onnx.TensorProto.BOOL}.get(
                      dt, onnx.TensorProto.FLOAT)
        if isinstance(x, Tensor):
            if to != onnx.TensorProto.FLOAT:
                self.fail("CastLike fp32 -> %s needs a conversion kernel"
                          % onnx.TensorProto.DataType.Name(to))
            return x
        arr = np.asarray(x)
        if to == onnx.TensorProto.FLOAT:
            return arr.astype(np.float32)
        return arr.astype({onnx.TensorProto.INT64: np.int64,
                           onnx.TensorProto.INT32: np.int32,
                           onnx.TensorProto.BOOL: bool}[to])


class SpaceToDepth(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 4:
            self.fail("SpaceToDepth needs a 4-D input")
        b = int(_attr(node, "blocksize", 1))
        N, C, H, W = (int(d) for d in x.shape)
        if b < 1 or H % b or W % b:
            self.fail("SpaceToDepth blocksize %d does not divide %dx%d"
                      % (b, H, W))
        Ho, Wo = H // b, W // b
        out = self.alloc((N, C * b * b, Ho, Wo))
        for n in range(N):
            for c in range(C):
                for i in range(b):
                    for j in range(b):
                        # ONNX SpaceToDepth order: (i, j, c), i.e.
                        # oc = (i*blocksize + j)*C + c
                        oc = (i * b + j) * C + c
                        obase = (n * C * b * b + oc) * Ho * Wo
                        for h in range(Ho):
                            self.lane_runs(
                                self.r._buf_at(
                                    x, ((n * C + c) * H + h * b + i) * W * 4
                                    + j * 4),
                                self.r._buf_at(out, (obase + h * Wo) * 4),
                                1, Wo, 0, 0, b * 4)
        return out


class Shrink(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        lambd = float(_attr(node, "lambd", 0.5))
        bias = float(_attr(node, "bias", 0.0))
        lo = self.upload_const(np.full(x.shape, -lambd, np.float32))
        hi = self.upload_const(np.full(x.shape, lambd, np.float32))
        lt = self.cmp(vc4infer.OP_CMP_LT, x, lo)
        gt = self.cmp(vc4infer.OP_CMP_GT, x, hi)
        xb = self.add(x, self.upload_const(np.full(x.shape, bias, np.float32)))
        xn = self.sub(x, self.upload_const(np.full(x.shape, bias, np.float32)))
        return self.add(self.mul(lt, xb), self.mul(gt, xn))


class ThresholdedRelu(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        alpha = float(_attr(node, "alpha", 1.0))
        shifted = self.sub(x, self.const_like("alpha", x, alpha))
        return self.ew(shifted, x, vc4infer.OP_RELU_BWD)


class Celu(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        alpha = float(_attr(node, "alpha", 1.0))
        if alpha == 0.0:
            self.fail("Celu alpha is 0")
        pos = self.ew1(x, vc4infer.OP_RELU)
        e = self.unary(self.affine(x, 1.0 / alpha, 0.0), vc4infer.UOP_EXP)
        e = self.affine(e, alpha, 0.0)             # alpha*exp(x/alpha)
        e = self.sub(e, self.const_like("celu_alpha", e, alpha))
        neg = self.affine(self.ew1(self.affine(e, -1.0, 0.0),
                                   vc4infer.OP_RELU), -1.0, 0.0)
        return self.add(pos, neg)


class PRelu(_Base):

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        slope = self.bcast(node.input[1], x.shape)
        pos = self.cmp(vc4infer.OP_CMP_GT, x, self.upload_const(
            np.zeros(x.shape, np.float32)))
        keep = self.add(pos, self.mul(self.affine(pos, -1.0, 1.0), slope))
        return self.mul(x, keep)


# =====================================================================
# inverse / hyperbolic unary family (docs/OPS-UNARY.md)
# =====================================================================
class Trig(_Base):
    """One QPU kernel per op, each with the domain its body can answer.

    The bodies are range reduction + polynomial (asin/acos/atan/tan) or the
    verified log/exp cores (asinh/acosh/atanh/sinh/cosh); see
    docs/OPS-UNARY.md for the method and the measured accuracy per op.  A
    value the kernel cannot answer is *refused by name* here, never answered
    wrongly - the rule Log/Sqrt/Sin already follow (docs/ONNX.md 12).

    ``Tan`` is the one op whose limit is not a plain interval: near a pole the
    Cody-Waite reduction's absolute error in the reduced r becomes a relative
    error of the answer, so inputs with |cos(x)| below ``TAN_COS_MIN`` are
    refused.  The gate is a domain check on the operand (the same class of
    work as the min/max test Log needs), not a computed result.
    """

    UOP = {
        "Acos": vc4infer.UOP_ACOS, "Acosh": vc4infer.UOP_ACOSH,
        "Asin": vc4infer.UOP_ASIN, "Asinh": vc4infer.UOP_ASINH,
        "Atan": vc4infer.UOP_ATAN, "Atanh": vc4infer.UOP_ATANH,
        "Tan": vc4infer.UOP_TAN, "Sinh": vc4infer.UOP_SINH,
        "Cosh": vc4infer.UOP_COSH,
    }
    # (lo, hi) of the operand; None is unbounded, +-inf is refused everywhere
    RANGE = {
        "Asin": (-1.0, 1.0), "Acos": (-1.0, 1.0), "Atanh": (-1.0, 1.0),
        "Acosh": (1.0, None), "Asinh": (None, None), "Atan": (None, None),
        "Tan": (-1e5, 1e5), "Sinh": (-88.0, 88.0), "Cosh": (-88.0, 88.0),
    }
    # atanh(+-1) is +-inf and atanh(x) is real only for |x| < 1: this op's
    # bounds are open where the others' are closed
    STRICT = ("Atanh",)
    TAN_COS_MIN = 0.05

    def run(self):
        node = self.node
        op = node.op_type
        x = self.ten(node.input[0])
        lo, hi = self.RANGE[op]

        strict = op in self.STRICT

        def in_domain(a):
            if not np.isfinite(a).all():
                return False
            if lo is not None:
                if float(np.min(a)) < lo or \
                        (strict and float(np.min(a)) <= lo):
                    return False
            if hi is not None:
                if float(np.max(a)) > hi or \
                        (strict and float(np.max(a)) >= hi):
                    return False
            if op == "Tan":
                return bool(np.min(np.abs(np.cos(a.astype(np.float64))))
                            >= self.TAN_COS_MIN)
            return True

        if lo is None and hi is None:
            span = "every finite x"
        elif strict:
            span = "%g < x < %g" % (lo, hi)
        else:
            span = "%g <= x <= %s" % (lo, "+inf" if hi is None else "%g" % hi)
        if op == "Asin" or op == "Acos":
            span = "-1 <= x <= 1"
        if op == "Tan":
            span = "|x| <= 1e5 and |cos(x)| >= %g" % self.TAN_COS_MIN
        self.download_check(
            x, in_domain,
            "%s: the QPU kernel's verified domain is %s (a value outside it "
            "is refused rather than answered)" % (op, span))
        return self.unary(x, self.UOP[op])


# =====================================================================
# variadic elementwise, conversions, normalisation and the gather family
# (docs/OPS-GAPS.md)
# =====================================================================
class Variadic(_Base):
    """Sum / Mean / Max / Min: one broadcast elementwise pass per operand.

    Max/Min use :meth:`_Base.select`, an exact *bit* select, so an infinite
    operand is *copied* rather than multiplied by zero (``0*inf`` is NaN on the
    QPU FPU, which is why the older ``m*a + (1-m)*b`` form is not used here).
    A NaN operand is refused: numpy's ``maximum``/``minimum`` propagate it and
    the FPU cannot return a NaN at all (measured, docs/OPS-GAPS.md 5.1).  Sum
    and Mean refuse every non-finite operand for the same measured reason.
    """

    def run(self):
        node = self.node
        op = node.op_type
        names = [n for n in node.input if n]
        if not names:
            self.fail("%s has no inputs" % op)
        shapes = [self.shape_of(n) for n in names]
        try:
            shape = tuple(int(d) for d in np.broadcast_shapes(*shapes))
        except ValueError:
            self.fail("operands %s do not broadcast" % (shapes,))
        vals = [self.bcast(n, shape) for n in names]
        for v in vals:
            if op in ("Sum", "Mean"):
                # measured on this FPU: fadd maps a NaN operand to +-inf and
                # (+inf) + (-inf) to -inf, so a non-finite sum is not the IEEE
                # one and is refused rather than answered
                self.download_check(
                    v, lambda a: bool(np.isfinite(a).all()),
                    "%s: a non-finite operand is refused (the QPU fadd maps a "
                    "NaN operand to +-inf and (+inf)+(-inf) to -inf; measured)"
                    % op)
            else:
                # the Max/Min combine step is an ADD of one bit-copied zero,
                # which is exact for +-inf -- but no composition here can
                # *produce* a NaN, so a NaN operand is refused
                self.download_check(
                    v, lambda a: bool(not np.isnan(a).any()),
                    "%s: a NaN operand is refused (ONNX/numpy propagate NaN "
                    "and the QPU fadd turns any NaN into +-inf, so the result "
                    "could not be produced; +-inf are exact here)" % op)
        if op in ("Sum", "Mean"):
            acc = vals[0]
            for v in vals[1:]:
                acc = self.add(acc, v)
            if op == "Mean":
                acc = self.affine(acc, 1.0 / len(vals), 0.0)
            return self.tag(acc, shape)
        acc = vals[0]
        for v in vals[1:]:
            acc = self.maxmin(acc, v, op == "Max")
        return self.tag(acc, shape)


class DivOp(_Base):
    """Div: a constant scalar divisor folds to a multiply; else a*(1/b).

    The reciprocal kernel is a sign-split bit trick (max rel 5.7e-8) and the
    product is the correctly-rounded MUL, so the composition is ~2 ulp -- the
    accuracy class the folded-constant path already has.  The MUL kernel
    returns NaN for an infinite operand (measured, docs/OPS-NORM.md), so a
    non-finite operand and a zero divisor are refused by name rather than
    answered with a value ONNX does not have.
    """

    def run(self):
        node = self.node
        from ._onnxrun import Tensor
        a_v = self.r.values.get(node.input[0])
        b_v = self.r.values.get(node.input[1])
        if not isinstance(a_v, Tensor) and not isinstance(b_v, Tensor):
            return np.floor_divide(np.asarray(a_v), np.asarray(b_v)) \
                .astype(np.int64)
        c = self.r.host.get(node.input[1])
        if c is not None and np.asarray(c).size == 1:
            x = self.ten(node.input[0])
            k = float(np.asarray(c).ravel()[0])
            if k == 0.0:
                self.fail("Div by a constant 0")
            return self.affine(x, 1.0 / k, 0.0)
        shapes = [self.shape_of(n) for n in (node.input[0], node.input[1])]
        try:
            shape = tuple(int(d) for d in np.broadcast_shapes(*shapes))
        except ValueError:
            self.fail("divisor shape %s does not broadcast to %s"
                      % (shapes[1], shapes[0]))
        a = self.bcast(node.input[0], shape)
        b = self.bcast(node.input[1], shape)
        for t, what in ((a, "dividend"), (b, "divisor")):
            self.download_check(
                t, lambda v: bool(np.isfinite(v).all()),
                "Div: a non-finite %s is refused (a*(1/b) needs the "
                "correctly-rounded MUL, which returns NaN for an infinite "
                "operand; measured)" % what)
        self.download_check(b, lambda v: bool(np.all(v != 0)),
                            "Div: a zero divisor is refused (0/0 and x/0 are "
                            "inf/NaN in ONNX and the reciprocal kernel has no "
                            "value at 0)")
        return self.mul(a, self.unary(b, vc4infer.UOP_RECIPROCAL))


class CastOp(_Base):
    """Cast: a view for fp32, a truncating QPU conversion for the int types.

    ``Cast`` to INT32/INT64 truncates toward zero, which is ``floor(x)`` for
    x >= 0 and ``ceil(x)`` for x < 0 -- two exact unary kernels and one exact
    bit select.  The resulting integer-valued fp32 arena buffer is then
    *converted* (a dtype change, the boundary work ``ArgMax``/bool outputs
    already do; no arithmetic on a value).
    """

    def run(self):
        node = self.node
        from ._onnxrun import Tensor
        to = int(_attr(node, "to", onnx.TensorProto.FLOAT))
        v = self.r.values[node.input[0]]
        name = onnx.TensorProto.DataType.Name(to)
        if not isinstance(v, Tensor):
            arr = np.asarray(v)
            if to == onnx.TensorProto.FLOAT:
                return arr.astype(np.float32)
            if to == onnx.TensorProto.DOUBLE:
                return arr.astype(np.float64)
            if to == onnx.TensorProto.INT64:
                return arr.astype(np.int64)
            if to == onnx.TensorProto.INT32:
                return arr.astype(np.int32)
            if to == onnx.TensorProto.BOOL:
                return arr.astype(bool)
            self.fail("Cast of a host tensor to %s" % name)
        x = self.ten(node.input[0])
        if to == onnx.TensorProto.FLOAT:
            return x
        if to == onnx.TensorProto.DOUBLE:
            return self.r.download(x).astype(np.float64)
        zero = self.const_like("cast_zero", x, 0.0)
        if to in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            self.download_check(
                x, lambda a: bool(np.isfinite(a).all()),
                "Cast of a non-finite value to %s is refused: numpy's "
                "float->int conversion of NaN/inf is undefined (it yields "
                "INT32_MIN on this platform) and the Floor/Ceil kernels do "
                "not carry NaN through to it" % name)
            fl = self.unary(x, vc4infer.UOP_FLOOR)
            ce = self.unary(x, vc4infer.UOP_CEIL)
            tr = self.select(self.cmp(vc4infer.OP_CMP_GE, x, zero), fl, ce)
            a = self.r.download(tr)
            with np.errstate(invalid="ignore"):
                return a.astype(np.int32 if to == onnx.TensorProto.INT32
                                else np.int64)
        if to == onnx.TensorProto.BOOL:
            eq = self.cmp(vc4infer.OP_CMP_EQ, x, zero)
            return self.bool_of(self.affine(eq, -1.0, 1.0))
        self.fail("Cast fp32 -> %s: no conversion kernel (the engine arena is "
                  "fp32 activations and its integer outputs are host index "
                  "tensors); FLOAT, DOUBLE, INT32, INT64 and BOOL are wired"
                  % name)


class LpNormalization(_Base, _ReduceMixin):
    """x / ||x||_p along one axis, p in {1, 2}: one sum, one Sqrt for p=2,
    one Reciprocal and one broadcast MUL."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = int(_attr(node, "axis", -1))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        p = int(_attr(node, "p", 2))
        T, P, D, I = self.ax_last(x, axis)
        self.tmp(T)
        rows = P * I
        if D == 0:
            self.fail("normalising an empty axis")
        if p == 1:
            s = self.sum_last_aligned(self.unary(T, vc4infer.UOP_ABS), rows, D)
        elif p == 2:
            s = self.sum_last_aligned(self.mul(T, T), rows, D)
            s = self.unary(s, vc4infer.UOP_SQRT)
        else:
            self.fail("p=%d (only p=1 and p=2 are wired: ||x||_1 is a sum of "
                      "|x|, ||x||_2 is sqrt(sum x^2))" % p)
        self.download_check(
            s, lambda a: bool(np.all(a > 0)),
            "LpNormalization of a zero-norm slice is refused: ONNX gives "
            "0/0 = NaN there and the reciprocal kernel has no value at 0")
        r = self.unary(s, vc4infer.UOP_RECIPROCAL)
        t = self.mul(T, self.bcast_rows(r, rows, D))
        return self.ax_back(t, P, D, I, x.shape)


# ------------------------------------------------------------ gather family
class GatherOp(_Base):
    """Gather along an axis: one row copy per output row.

    The unit is the contiguous block after ``axis`` (``run`` words); the index
    tensor selects one block per output row, so the operator is ``rowcopy``:
    a constant index step collapses into one launch, an arbitrary index costs
    one.  Negative indices count from the end.
    """

    def run(self):
        node = self.node
        from ._onnxrun import Tensor
        rank_of = None
        if not isinstance(self.r.values.get(node.input[0]), Tensor):
            # an integer (shape) tensor: index arithmetic on the ARM, exactly
            # as the main table did before this handler owned Gather
            d = np.asarray(self.host_or_fail(node.input[0]))
            idx = np.asarray(self.host_or_fail(node.input[1]), np.int64)
            axis = int(_attr(node, "axis", 0))
            if axis < 0:
                axis += d.ndim
            if axis < 0 or axis >= d.ndim:
                self.fail("axis %d for rank %d" % (axis, d.ndim))
            return np.take(d, idx, axis=axis).astype(np.int64)
        x = self.ten(node.input[0])
        idx = np.asarray(self.host_or_fail(node.input[1]), np.int64)
        rank = len(x.shape)
        axis = int(_attr(node, "axis", 0))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        d = int(x.shape[axis])
        if idx.size and (idx.min() < -d or idx.max() >= d):
            self.fail("indices span [%d, %d] for axis %d of size %d"
                      % (int(idx.min()), int(idx.max()), axis, d))
        out_shape = list(x.shape[:axis]) + list(idx.shape) + \
            list(x.shape[axis + 1:])
        inner = _prod(x.shape[axis + 1:])
        outer = _prod(x.shape[:axis])
        k = int(idx.size)
        out = self.alloc(out_shape)
        if k == 0 or inner == 0 or outer == 0:
            return out
        idx = np.where(idx < 0, idx + d, idx).ravel()
        for o in range(outer):
            # rowcopy's map is relative to the src view, which is already
            # the o-th outer block (Compress passes relative rows too)
            self.rowcopy(self.r._buf_at(x, o * d * inner * 4),
                         self.r._buf_at(out, o * k * inner * 4), inner,
                         [int(v) for v in idx])
        return out


class GatherElementsOp(_Base):
    """GatherElements: one arbitrary word pick per output element.

    The axis is moved last on the QPUs, which makes every output row's picks
    local to its own data row, so the operator is one flat index map and
    consecutive picks with a constant stride become one laned copy.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = int(_attr(node, "axis", 0))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        ishape = self.shape_of(node.input[1])
        iv = np.asarray(self.host_or_fail(node.input[1]), np.int64)
        if len(ishape) != rank:
            self.fail("indices rank %d != data rank %d" % (len(ishape), rank))
        d = int(x.shape[axis])
        j = int(ishape[axis])
        for i in range(rank):
            if i != axis and int(ishape[i]) > int(x.shape[i]):
                self.fail("indices dim %d is %d > data dim %d"
                          % (i, ishape[i], x.shape[i]))
        if iv.size and (iv.min() < -d or iv.max() >= d):
            self.fail("indices span [%d, %d] for axis %d of size %d"
                      % (int(iv.min()), int(iv.max()), axis, d))
        P = _prod(x.shape[:axis])
        Ii = _prod(x.shape[axis + 1:])
        if _prod(ishape[axis + 1:]) != Ii:
            self.fail("indices tail %s != data tail %s"
                      % (ishape[axis + 1:], tuple(x.shape[axis + 1:])))
        iv = np.where(iv < 0, iv + d, iv).reshape(P, j, Ii)
        T, Pp, D, Ip = self.ax_last(x, axis)
        self.tmp(T)
        rows = Pp * Ip
        pick = np.ascontiguousarray(iv.transpose(0, 2, 1)).reshape(rows, j)
        src = (np.arange(rows, dtype=np.int64)[:, None] * d + pick).ravel()
        out = self.alloc((rows, j))
        self.word_gather(T.buf, out.buf, src, rows * j)
        return self.ax_back(out, P, j, Ii,
                            list(x.shape[:axis]) + [j]
                            + list(x.shape[axis + 1:]))


class GatherNDOp(_Base):
    """GatherND: every output element is a contiguous run of ``data``.

    ``run = prod(data.shape[batch_dims + last_index_dim:])``, so the operator
    is a row copy over an index map the ARM builds from the integer index
    tensor (the ``MaxUnpool``/``ArgMax`` class of work: indices, not values).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        indices = np.asarray(self.host_or_fail(node.input[1]), np.int64)
        bd = int(_attr(node, "batch_dims", 0))
        dshape = [int(d) for d in x.shape]
        rank = len(dshape)
        if indices.ndim < 1:
            self.fail("indices must have rank >= 1")
        r = int(indices.shape[-1])
        if bd < 0 or bd >= len(indices.shape) or bd + r > rank:
            self.fail("batch_dims %d / last index dim %d for data rank %d"
                      % (bd, r, rank))
        out_shape = [int(d) for d in indices.shape[:-1]] + dshape[bd + r:]
        run = _prod(dshape[bd + r:])
        iv = indices.copy()
        for k in range(r):
            dim = dshape[bd + k]
            sl = iv[..., k]
            if sl.size and (sl.min() < -dim or sl.max() >= dim):
                self.fail("index %d spans [%d, %d] for dim %d of size %d"
                          % (k, int(sl.min()), int(sl.max()), bd + k, dim))
            iv[..., k] = np.where(sl < 0, sl + dim, sl)
        outer = _prod([int(d) for d in indices.shape[:-1]])
        out = self.alloc(out_shape)
        if outer == 0 or run == 0:
            return out
        strides = []
        acc = 1
        for d in reversed(dshape):
            strides.append(acc)
            acc *= d
        strides.reverse()
        mapping = np.zeros(outer, np.int64)
        icoords = [int(d) for d in indices.shape[:-1]]
        if bd and icoords:
            coords = np.stack(np.unravel_index(np.arange(outer), icoords),
                              axis=-1)
            for b in range(bd):
                mapping += coords[:, b] * strides[b]
        flat = iv.reshape(outer, r)
        for k in range(r):
            mapping += flat[:, k] * strides[bd + k]
        self.rowcopy(x.buf, out.buf, run, [int(v) // run for v in mapping])
        return out


class ScatterNDOp(_Base):
    """ScatterND: a row scatter; add/mul go through a dense update buffer.

    ``copy2d`` destination vectors are 16 *contiguous* words, so one
    destination row is one launch and two writes to the same row are a race:
    duplicate index rows are refused (``MaxUnpool`` refuses them for the same
    reason).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        indices = np.asarray(self.host_or_fail(node.input[1]), np.int64)
        updates = self.ten(node.input[2])
        red = _attr(node, "reduction", "none") or "none"
        dshape = [int(d) for d in x.shape]
        rank = len(dshape)
        r = int(indices.shape[-1])
        if r < 1 or r > rank:
            self.fail("last index dim %d for data rank %d" % (r, rank))
        upd_shape = [int(d) for d in indices.shape[:-1]] + dshape[r:]
        out_shape = list(dshape)
        run = _prod(dshape[r:])
        k = _prod([int(d) for d in indices.shape[:-1]])
        if list(updates.shape) != upd_shape:
            self.fail("updates shape %s != indices shape %s + data tail %s"
                      % (tuple(updates.shape), tuple(indices.shape[:-1]),
                         dshape[r:]))
        iv = indices.copy()
        for c in range(r):
            dim = dshape[c]
            sl = iv[..., c]
            if sl.size and (sl.min() < -dim or sl.max() >= dim):
                self.fail("index %d out of range for dim %d of size %d"
                          % (c, c, dim))
            iv[..., c] = np.where(sl < 0, sl + dim, sl)
        strides = []
        acc = 1
        for d in reversed(dshape):
            strides.append(acc)
            acc *= d
        strides.reverse()
        flat = iv.reshape(max(k, 0), r)
        mapping = np.zeros(max(k, 0), np.int64)
        for c in range(r):
            mapping += flat[:, c] * strides[c]
        mapping //= run
        if len(set(int(v) for v in mapping)) != mapping.size:
            self.fail("duplicate index rows: the ONNX result is the last "
                      "write in update order and two QPU stores to one row "
                      "are a race")
        if red not in ("none", "add", "mul"):
            self.fail("reduction=%s (none, add and mul are wired; max/min "
                      "need a per-word comparison pass)" % red)
        if red == "none":
            out = self.alloc(out_shape)
            self.r._runs(x.buf, out.buf, 1, x.size, 0, 0)
            self.rows_scatter(updates.buf, out.buf, run,
                              [int(v) for v in mapping])
            return self.tag(out, out_shape)
        base = self.alloc_fill(out_shape, 0.0 if red == "add" else 1.0)
        self.rows_scatter(updates.buf, base.buf, run,
                          [int(v) for v in mapping])
        op = vc4infer.OP_ADD if red == "add" else vc4infer.OP_MUL
        return self.ew(x, base, op, out_shape)


class ScatterElementsOp(_Base):
    """ScatterElements: one word written per update element, axis last.

    The ONNX output has the *data* shape; the update elements are walked in
    indices order, so the destination word is ``row*D + idx`` with the axis
    moved innermost, and the update values are walked in the same order by the
    same layout copy.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = int(_attr(node, "axis", 0))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        ishape = self.shape_of(node.input[1])
        iv = np.asarray(self.host_or_fail(node.input[1]), np.int64)
        upd = self.ten(node.input[2])
        red = _attr(node, "reduction", "none") or "none"
        if tuple(upd.shape) != tuple(ishape):
            self.fail("updates shape %s != indices shape %s"
                      % (tuple(upd.shape), tuple(ishape)))
        if len(ishape) != rank:
            self.fail("indices rank %d != data rank %d" % (len(ishape), rank))
        d = int(x.shape[axis])
        if iv.size and (iv.min() < -d or iv.max() >= d):
            self.fail("indices span [%d, %d] for axis %d of size %d"
                      % (int(iv.min()), int(iv.max()), axis, d))
        P = _prod(x.shape[:axis])
        Ii = _prod(x.shape[axis + 1:])
        if _prod(ishape[axis + 1:]) != Ii:
            self.fail("indices tail %s != data tail %s"
                      % (ishape[axis + 1:], tuple(x.shape[axis + 1:])))
        j = int(ishape[axis])
        iv = np.where(iv < 0, iv + d, iv).reshape(P, j, Ii)
        pick = np.ascontiguousarray(iv.transpose(0, 2, 1)).reshape(P * Ii, j)
        pos = (np.arange(P * Ii, dtype=np.int64)[:, None] * d + pick).ravel()
        if len(set(int(v) for v in pos)) != pos.size:
            self.fail("duplicate destinations: two QPU stores to one word are "
                      "a race")
        if red not in ("none", "add", "mul"):
            self.fail("reduction=%s (none, add and mul are wired)" % red)
        T, _, _, _ = self.ax_last(x, axis)
        self.tmp(T)
        U, _, _, _ = self.ax_last(upd, axis)
        self.tmp(U)
        if red == "none":
            out = self.alloc((P * Ii, d))
            self.r._runs(T.buf, out.buf, 1, P * Ii * d, 0, 0)
            self.word_scatter(U.buf, out.buf, pos, P * Ii * j)
        else:
            base = self.alloc_fill((P * Ii, d), 0.0 if red == "add" else 1.0)
            self.word_scatter(U.buf, base.buf, pos, P * Ii * j)
            op = vc4infer.OP_ADD if red == "add" else vc4infer.OP_MUL
            out = self.ew(T, base, op, (P * Ii, d))
        return self.ax_back(out, P, d, Ii, x.shape)


class NonZeroOp(_Base):
    """NonZero: the QPU produces the 0/1 mask, the ARM the int64 coordinates.

    ``x != 0`` is one exact bit comparison (NaN counts as non-zero, -0.0 as
    zero, which is what ONNX and numpy do).  The mask is then *coordinate*
    data -- the same class of read-back ``ArgMax`` does -- and ``argwhere``
    turns it into the [rank][nnz] int64 output.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        zero = self.const_like("nonzero_zero", x, 0.0)
        m = self.affine(self.cmp(vc4infer.OP_CMP_EQ, x, zero), -1.0, 1.0)
        pos = np.argwhere(self.r.download(m) != 0)
        return np.ascontiguousarray(pos.T.astype(np.int64))


class ReverseSequenceOp(_Base):
    """Reverse the first ``sequence_lens[b]`` steps of each batch row."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        shape = [int(d) for d in x.shape]
        lens = np.asarray(self.host_or_fail(node.input[1]), np.int64).ravel()
        ba = int(_attr(node, "batch_axis", 1))
        ta = int(_attr(node, "time_axis", 0))
        ba = ba + rank if ba < 0 else ba
        ta = ta + rank if ta < 0 else ta
        if ba < 0 or ba >= rank or ta < 0 or ta >= rank or ba == ta:
            self.fail("batch_axis %d / time_axis %d for rank %d"
                      % (_attr(node, "batch_axis", 1),
                         _attr(node, "time_axis", 0), rank))
        if lens.size != shape[ba]:
            self.fail("sequence_lens has %d entries for batch %d"
                      % (lens.size, shape[ba]))
        m = np.arange(_prod(shape), dtype=np.int64).reshape(shape)
        order = [ba, ta] + [i for i in range(rank) if i not in (ba, ta)]
        v = np.transpose(m, order).copy()
        steps = shape[ta]
        for b in range(len(lens)):
            span = int(min(max(lens[b], 0), steps))
            if span > 1:
                v[b, :span] = v[b, :span][::-1].copy()
        back = [0] * rank
        for i, ax in enumerate(order):
            back[ax] = i
        src = np.ascontiguousarray(np.transpose(v, back)).reshape(-1)
        out = self.alloc(shape)
        self.word_gather(x.buf, out.buf, src, _prod(shape))
        return out


class CenterCropPadOp(_Base):
    """CenterCropPad: a flat index map with -1 in the padded region.

    The map is the reference evaluator's own crop/pad slice pair applied to an
    index array, so the centring rule (the odd extra element is dropped or
    added on the right) is the oracle's, not a re-derivation.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        target = [int(v) for v in
                  np.asarray(self.host_or_fail(node.input[1])).ravel()]
        if len(node.input) > 2 and node.input[2]:
            axes = [int(v) for v in
                    np.asarray(self.host_or_fail(node.input[2])).ravel()]
        else:
            axes = list(range(rank))
        axes = [a + rank if a < 0 else a for a in axes]
        if len(target) != len(axes):
            self.fail("shape has %d entries for %d axes"
                      % (len(target), len(axes)))
        for a in axes:
            if a < 0 or a >= rank:
                self.fail("axis %d for rank %d" % (a, rank))
        in_shape = [int(d) for d in x.shape]
        new_shape = list(in_shape)
        crop = [slice(0, d) for d in in_shape]
        pad = [slice(0, d) for d in in_shape]
        for a, sh in zip(axes, target):
            if sh < 0:
                self.fail("negative target extent %d on axis %d" % (sh, a))
            if sh == in_shape[a]:
                continue
            new_shape[a] = sh
            if sh < in_shape[a]:
                off = (in_shape[a] - sh) // 2
                crop[a] = slice(off, off + sh)
            else:
                off = (sh - in_shape[a]) // 2
                pad[a] = slice(off, off + in_shape[a])
        m = np.full(new_shape, -1, np.int64)
        src = np.arange(_prod(in_shape), dtype=np.int64).reshape(in_shape)
        m[tuple(pad)] = src[tuple(crop)]
        out = self.alloc_fill(tuple(new_shape), 0.0)
        self.word_gather(x.buf, out.buf, m.reshape(-1), _prod(new_shape))
        return out


class DepthToSpaceOp(_Base):
    """DepthToSpace: a fixed element-level layout map (DCR and CRD).

    The map is derived from the ONNX definition (the same reshape/transpose
    the reference evaluator applies), so it is exact by construction and does
    not depend on any stride pattern: a destination word reads one source
    word, and consecutive destination words with a constant source step become
    one laned copy of 16.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        b = int(_attr(node, "blocksize", 1))
        mode = _attr(node, "mode", "DCR") or "DCR"
        if len(x.shape) != 4:
            self.fail("DepthToSpace needs a 4-D input (got %s)"
                      % (tuple(x.shape),))
        N, C, H, W = (int(d) for d in x.shape)
        if b < 1 or C % (b * b):
            self.fail("blocksize %d does not divide %d channels" % (b, C))
        if mode not in ("DCR", "CRD"):
            self.fail("mode=%s (DCR, CRD)" % mode)
        idx = np.arange(N * C * H * W, dtype=np.int64).reshape(N, C, H, W)
        c = C // (b * b)
        if mode == "DCR":
            t = idx.reshape(N, b, b, c, H, W).transpose(0, 3, 4, 1, 5, 2)
        else:
            t = idx.reshape(N, c, b, b, H, W).transpose(0, 1, 4, 2, 5, 3)
        src = np.ascontiguousarray(t).reshape(-1)
        out = self.alloc((N, c, H * b, W * b))
        self.word_gather(x.buf, out.buf, src, out.size)
        return out


class TopKOp(_Base):
    """TopK: k selection passes, each one an ArgMax-class scan.

    Pass t takes the largest (or smallest) remaining value per row, records
    the value (an exact bit copy) and the index, then masks that element out
    by writing a constant +-inf over it -- a one-word byte store per row, no
    arithmetic.  Values are ordered as ONNX requires; indices are the raw
    int32 the scan produced (the ``ArgMax`` read-back).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        k = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        axis = int(_attr(node, "axis", -1))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            self.fail("axis %d for rank %d" % (axis, rank))
        largest = int(_attr(node, "largest", 1))
        d = int(x.shape[axis])
        if k < 0 or k > d:
            self.fail("k=%d for an axis of %d" % (k, d))
        self.download_check(
            x, lambda a: bool(np.isfinite(a).all()),
            "TopK: a non-finite input is refused (the mask value is +-inf and "
            "every NaN comparison is false, so a NaN would be ranked "
            "arbitrarily instead of the way numpy orders it)")
        T, P, D, I = self.ax_last(x, axis)
        self.tmp(T)
        rows = P * I
        work = self.alloc((rows, D))
        self.r._runs(T.buf, work.buf, 1, rows * D, 0, 0)
        out_shape = list(x.shape)
        out_shape[axis] = k
        vals = self.alloc((rows, k))
        picks = np.zeros((rows, k), np.int64)
        inf = self.const(("topk_mask", rows),
                         np.full(rows, -np.inf if largest else np.inf,
                                 np.float32))
        for t in range(k):
            best, idx = self._scan(work, rows, D, bool(largest))
            # column t of the [rows][k] value buffer: dest word r*k + t
            self.word_scatter(best.buf, vals.buf,
                              np.arange(rows, dtype=np.int64) * k + t, rows)
            iw = np.asarray(self.r.download(idx)).astype(np.int64).reshape(rows)
            picks[:, t] = iw
            self.word_scatter(inf.buf, work.buf,
                              np.arange(rows, dtype=np.int64) * D + iw, rows)
        outv = self.ax_back(self.tag(vals, (rows, k)), P, k, I, out_shape)
        pi = picks.reshape(P, I, k).transpose(0, 2, 1).reshape(out_shape)
        return [self.tag(outv, out_shape), pi.astype(np.int64)]

    def _scan(self, T, rows, D, largest):
        """(best value, index) per row: the ArgMax scan with an exact select."""
        cmp = vc4infer.OP_CMP_GT if largest else vc4infer.OP_CMP_LT
        best = self.slice_col(T, 0, rows, D)
        idx = self.upload_const(np.zeros(rows, np.float32))
        for c in range(1, D):
            v = self.slice_col(T, c, rows, D)
            m = self.cmp(cmp, v, best)
            best = self.select(m, v, best)
            idx = self.select(m, self.const(("topk_c", c, rows),
                                            np.full(rows, float(c),
                                                    np.float32)), idx)
        return best, idx


class UniqueOp(_Base, _ReduceMixin):
    """Unique: pairwise equality/order matrices on the QPU, coordinates on the
    ARM.

    Every slice along ``axis`` (or every element for ``axis=None``) is a unit.
    Two units are equal iff the exact bit comparison says so for all their
    words, and one is smaller iff its first differing word is smaller; the QPU
    produces those two 0/1 matrices and the ARM only turns them into
    coordinates -- the same class of read-back ``ArgMax`` performs.  ``Y``
    itself is a row copy of the device buffer, never a host value.
    """

    MAX_UNITS = 256
    MAX_CELLS = 65536

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        rank = len(x.shape)
        axis = _attr(node, "axis", None)
        srt = int(_attr(node, "sorted", 1))
        n_out = max(len([o for o in node.output if o]), 1)
        self.download_check(
            x, lambda a: bool(not np.isnan(a).any()),
            "Unique: NaN is refused (numpy gives every NaN its own place at "
            "the end of the sorted order and the pairwise composition cannot "
            "reproduce that order)")
        if axis is None:
            units = _prod(x.shape)
            run = 1
            Tt = self.tag(x, (units,))
            layout = None
        else:
            axis = int(axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                self.fail("axis %d for rank %d" % (axis, rank))
            T, P, D, I = self.ax_last(x, axis)
            self.tmp(T)
            # the units are T's *columns* (the axis index), so the pairwise
            # work runs on the transpose [D][P*I] in every case
            Tt = self.ax_last(T, 0)[0]
            self.tmp(Tt)
            units = D
            run = P * I
            layout = (P, I)
        if units > self.MAX_UNITS or units * run > self.MAX_CELLS:
            self.fail("Unique over %d units of %d words (%d cells) exceeds the "
                      "bound of this pairwise composition (%d units, %d "
                      "cells); split the tensor"
                      % (units, run, units * run, self.MAX_UNITS,
                         self.MAX_CELLS))
        if units == 0:
            return list((self.alloc([0]), np.zeros(0, np.int64),
                         np.zeros(0, np.int64),
                         np.zeros(0, np.int64))[:n_out])
        eq, lt = self._matrices(Tt, units, run)
        first = [i for i in range(units) if not any(eq[i][j]
                                                    for j in range(i))]
        pos = {}
        for n, f in enumerate(first):
            pos[f] = n
        inv = np.zeros(units, np.int64)
        for i in range(units):
            hit = [j for j in range(units) if eq[i][j]]
            j = hit[0] if hit else i
            inv[i] = pos.get(j, pos.get(i, 0))
        if srt:
            order = sorted(range(len(first)),
                           key=lambda p: sum(lt[first[q]][first[p]]
                                             for q in range(len(first))))
        else:
            order = list(range(len(first)))
        remap = np.zeros(len(first), np.int64)
        for n, o in enumerate(order):
            remap[o] = n
        inv = np.asarray([remap[v] for v in inv], np.int64)
        counts = np.zeros(len(first), np.int64)
        for v in inv:
            counts[v] += 1
        Y = self.alloc((len(first), run))
        self.rowcopy(Tt.buf, Y.buf, run, [first[o] for o in order])
        if layout is None:
            y = self.tag(Y, (len(first),))
        else:
            P, I = layout
            # Y is [unique][P*I]; ax_back wants [P*I][unique]
            Yt = self.ax_last(Y, 0)[0]
            self.tmp(Yt)
            y = self.ax_back(Yt, P, len(first), I,
                             list(x.shape[:axis]) + [len(first)]
                             + list(x.shape[axis + 1:]))
        idxs = np.asarray([first[o] for o in order], np.int64)
        if axis is None:
            inv = inv.reshape(x.shape)
        return list((y, idxs, inv, counts)[:n_out])

    def _matrices(self, Tt, units, run):
        """(equality, less-than) matrices as host nested 0/1 lists."""
        eq = [[False] * units for _ in range(units)]
        lt = [[False] * units for _ in range(units)]
        full = self.const(("unique_run", run),
                          np.full(units, float(run), np.float32))
        for i in range(units):
            row = self.tag(self.r._buf_at(Tt, i * run * 4), (run,))
            tile = self.tile_rows(row, units, run)       # [units][run]
            e = self.sum_last_aligned(
                self.cmp(vc4infer.OP_CMP_EQ, tile, Tt), units, run)
            eq[i] = (np.asarray(self.r.download(
                self.cmp(vc4infer.OP_CMP_EQ, e, full))) != 0).tolist()
            less = None
            pre = None
            for w in range(run):
                aw = self.tag(self.r._buf_at(row, w * 4), (1,))
                a = self.bcast_t(aw, (units,))
                b = self._column(Tt, w, units, run)
                cl = self.cmp(vc4infer.OP_CMP_LT, a, b)
                cq = self.cmp(vc4infer.OP_CMP_EQ, a, b)
                if less is None:
                    less, pre = cl, cq
                else:
                    less = self.add(less, self.mul(pre, cl))
                    pre = self.mul(pre, cq)
            lt[i] = (np.asarray(self.r.download(less)) != 0).tolist()
        return eq, lt

    def _column(self, Tt, w, units, run):
        """Column w of [units][run] as a compact [units] tensor."""
        out = self.alloc((units,))
        self._laned_run(Tt.buf, out.buf, w, 0, units, run)
        return out

# =====================================================================
# dispatch table
# =====================================================================
_REDUCE = ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
           "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
           "ReduceSumSquare")


def _mk(cls):
    def run(r, node):
        return cls(r, node).run()
    return run


HANDLERS = {}
for _op in _REDUCE:
    HANDLERS[_op] = _mk(Reduction)
for _op in _CMP:
    HANDLERS[_op] = _mk(Compare)
for _op in ("IsNaN", "IsInf"):
    HANDLERS[_op] = _mk(Predicate)
HANDLERS.update({
    "Sum": _mk(Variadic),
    "Mean": _mk(Variadic),
    "Max": _mk(Variadic),
    "Min": _mk(Variadic),
    "Div": _mk(DivOp),
    "Cast": _mk(CastOp),
    "LpNormalization": _mk(LpNormalization),
    "Gather": _mk(GatherOp),
    "GatherElements": _mk(GatherElementsOp),
    "GatherND": _mk(GatherNDOp),
    "ScatterElements": _mk(ScatterElementsOp),
    "ScatterND": _mk(ScatterNDOp),
    "NonZero": _mk(NonZeroOp),
    "TopK": _mk(TopKOp),
    "Unique": _mk(UniqueOp),
    "ReverseSequence": _mk(ReverseSequenceOp),
    "CenterCropPad": _mk(CenterCropPadOp),
    "DepthToSpace": _mk(DepthToSpaceOp),
    "Where": _mk(Where),
    "ArgMax": _mk(ArgReduction),
    "ArgMin": _mk(ArgReduction),
    "CumSum": _mk(CumSum),
    "Expand": _mk(Expand),
    "Tile": _mk(Tile),
    "Pad": _mk(Pad),
    "Compress": _mk(Compress),
    "OneHot": _mk(OneHot),
    "Pow": _mk(Pow),
    "Mod": _mk(Mod),
    "Trilu": _mk(Trilu),
    "Range": _mk(Range),
    "EyeLike": _mk(EyeLike),
    "CastLike": _mk(CastLike),
    "SpaceToDepth": _mk(SpaceToDepth),
    "Shrink": _mk(Shrink),
    "ThresholdedRelu": _mk(ThresholdedRelu),
    "Celu": _mk(Celu),
    "PRelu": _mk(PRelu),
})
for _op in Trig.UOP:
    HANDLERS[_op] = _mk(Trig)

# operators the main table also knows: this family owns them now (the main
# Div only folds a constant scalar divisor, Cast only views fp32 buffers and
# Gather only indexes host shape tensors)
OVERRIDES = ("Div", "Cast", "Gather")
