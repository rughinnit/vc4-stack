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
        t = self.r._panels.get(key)
        if t is None:
            t = self.r.upload(np.ascontiguousarray(arr, dtype=np.float32))
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


class DepthToSpace(_Base):

    def run(self):
        node = self.node
        b = int(_attr(node, "blocksize", 1))
        x = self.ten(node.input[0])
        self.fail("DepthToSpace needs an element-level scatter: the output row "
                  "[ow*b + j] alternates between %d input channels, and a "
                  "copy2d destination vector is 16 contiguous words, so no "
                  "lane stride expresses it.  The gather kernel that would "
                  "(vc4_gather) is refused on this board because it wedges "
                  "the QPU; SpaceToDepth, its inverse, does run." % b)


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
    "DepthToSpace": _mk(DepthToSpace),
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
