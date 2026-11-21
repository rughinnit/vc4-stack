"""Runner handlers for the signal, quantise and random operators.

``docs/OPS-RNN.md`` is the measurement record for this file (the recurrent /
random / signal / quantise block).  Contents:

    QuantizeLinear          affine + round + clamp on the QPUs; the int8/uint8
                            array is a *storage* dtype at the boundary
    DynamicQuantizeLinear   the same, with scale and zero point reduced on the
                            QPUs (a 16-wide max/min tree)
    DFT                     one GEMM against a constant cos/sin matrix
    STFT                    framing (copy2d) + the DFT core
    HannWindow(+Hamming, Blackman)
                            QPU index ramp + QPU SIN/COS: the window table is
                            produced by the device on every run
    MelWeightMatrix         QPU triangle arithmetic over parameter-derived
                            index ramps (the bins are index metadata)
    RandomUniform(+Like)    VC4_UOP_RNG, a counter-based device RNG
    RandomNormal(+Like)     two draws + Box-Muller on the QPUs
    Bernoulli               one draw + an exact fp32 compare
    Multinomial             exp/normalise/cumsum GEMM + one draw per sample

Nothing here uses a host RNG, and no two activation values are arithmetically
combined on the ARM.  The ARM builds integer geometry (frame offsets, the
triangular/ones GEMM matrices), constant panels derived from *parameter*
tensors (scales, zero points, windows, the DFT twiddle matrix), and converts
device results to the declared ONNX dtype at the boundary (int8/uint8/int64),
which is a dtype conversion, not arithmetic.
"""
import hashlib
import math

import numpy as np
import onnx

from . import _binding as vc4infer
from ._dispatch_extra import _Base, _ReduceMixin, _Tensor, _attr, _prod
from ._errors import UnsupportedNode


def _Tn():
    """The runner's Tensor class (imported lazily: _onnxrun imports this
    module while it is building its dispatch table)."""
    from ._onnxrun import Tensor
    return Tensor

_NP = {onnx.TensorProto.FLOAT: np.float32,
       onnx.TensorProto.UINT8: np.uint8,
       onnx.TensorProto.INT8: np.int8,
       onnx.TensorProto.INT32: np.int32,
       onnx.TensorProto.INT64: np.int64}


def _al(n, a):
    return (int(n) + a - 1) // a * a


class _Sig(_Base, _ReduceMixin):
    """Shared helpers: parameter constants, reductions, the device RNG."""

    # ---------------------------------------------------------- small utils
    def reshaped(self, t, shape):
        """A view of t's buffer with another shape.

        Never re-registers the buffer: every tensor this module produces is
        already in the runner's per-node garbage list, and registering it
        twice would release the same arena range twice.
        """
        return _Tn()(t.buf, tuple(int(d) for d in shape))

    def param_vec(self, name, what="scale"):
        """A quantisation/window *parameter* as a host float32 vector.

        A graph initialiser is already on the host; a runtime tensor is
        downloaded once and then used as a constant panel.  These are
        parameters (scales, zero points, windows), never activations.
        """
        v = self.r.host.get(name)
        if v is None:
            from ._onnxrun import Tensor
            t = self.r.values.get(name)
            if not isinstance(t, Tensor):
                self.fail("%s %s is neither a constant nor an fp32 tensor"
                          % (what, name))
            v = self.r.download(t)
        a = np.ascontiguousarray(v, np.float32).ravel()
        if a.size == 0:
            self.fail("%s %s is empty" % (what, name))
        return a

    def param_raw(self, name):
        """The host bytes of a (small) parameter, dtype preserved."""
        from ._onnxrun import Tensor
        v = self.r.host.get(name)
        if v is None:
            t = self.r.values.get(name)
            v = self.r.download(t) if isinstance(t, Tensor) else t
        return np.asarray(v)

    # ------------------------------------------------- GEMM over QPU data
    def _mm(self, T, rows, D, mat):
        """out = T[rows][D] @ mat[D][N]; returns (out, row pitch).

        T is QPU-produced, so an A operand the GEMM would have to pad is
        zero-padded and copied by ``copy2d_x`` on the QPUs: ``vc4_matmul``
        pads with the ARM, which can read a stale cacheable line
        (docs/OPS-NORM.md, ``sum_last_aligned``).
        """
        N = int(mat.shape[1])
        Mp, Kp, Np = _al(rows, 8), _al(D, 16), _al(N, 64)
        if Mp == rows and Kp == D and Np == N:
            m = self.const(("mm", D, N, mat.tobytes()), mat)
            out = self.alloc((rows, N))
            self.r.eng.matmul(T.buf, m.buf, out.buf, rows, N, D)
            return out, N
        A = self.alloc_fill((Mp, Kp), 0.0)
        self.lane_runs(T.buf, A.buf, rows, D, D * 4, Kp * 4, 4)
        m = self.const(("mm", D, N, mat.tobytes()),
                       np.ascontiguousarray(np.pad(
                           mat.astype(np.float32),
                           ((0, Kp - D), (0, Np - N)))))
        out = self.alloc((Mp, Np))
        self.r.eng.matmul(A.buf, m.buf, out.buf, Mp, Np, Kp)
        return out, Np

    def row_sum(self, T, rows, D, ncol=16):
        """Row sums as a [rows] tensor (one GEMM, one lane copy)."""
        if D <= 0:
            self.fail("reducing an empty axis")
        out, pitch = self._mm(T, rows, D, np.ones((D, ncol), np.float32))
        return self.slice_col(out, 0, rows, pitch)

    def row_tri(self, T, rows, D):
        """Inclusive prefix sum along the innermost axis (one GEMM)."""
        out, pitch = self._mm(T, rows, D,
                              np.triu(np.ones((D, D), np.float32)))
        if pitch == D:
            return out
        c = self.alloc((rows, D))
        self.lane_runs(out.buf, c.buf, rows, D, pitch * 4, D * 4, 4)
        return c

    # ---------------------------------------------------------- reduction
    def max_all(self, t, want_max=True):
        """A full-tensor max (or min) as a 1-element tensor.

        ``vc4_reduce`` is not used (it wedges the QPU, docs/OPS-REDUCE.md) and
        the per-column chain the pooling handlers use costs 4 launches per
        element.  This is a 16-wide tree: every level is one ``max_last`` over
        16 columns (64 launches) whatever the row count, so the whole
        reduction is O(64 log16 N) launches.
        """
        n = int(t.size)
        if n <= 0:
            self.fail("cannot reduce an empty tensor")
        src = t
        while n > 16:
            rows = _al(n, 16)
            if rows != n:
                pad = self.alloc_fill((rows,), -np.inf if want_max else np.inf)
                self.lane_runs(src.buf, pad.buf, 1, n, 0, 0, 4)
                src = pad
            out = self.alloc((rows // 16,))
            self._max_last(src, rows // 16, 16, want_max, out)
            src = out
            n = rows // 16
        out = self.alloc((1,))
        self._max_last(src, 1, n, want_max, out)
        return out

    def _max_last(self, T, outer, D, want_max, out):
        """Row max/min over the innermost axis of an [outer][D] tensor."""
        acc = self.slice_col(T, 0, outer, D)
        cmp = vc4infer.OP_CMP_GE if want_max else vc4infer.OP_CMP_LE
        for k in range(1, D):
            v = self.slice_col(T, k, outer, D)
            m = self.cmp(cmp, acc, v)
            acc = self.add(self.mul(m, acc),
                           self.mul(self.affine(m, -1.0, 1.0), v))
        self.lane_runs(acc.buf, out.buf, 1, outer, 0, 0, 4)
        return out

    def add_const(self, t, c):
        """t + c where c may be any finite constant.

        A kernel parameter whose value is at or below -2.0 has a bit pattern
        vc4d refuses as a possible GPU bus address, so a large negative offset
        is applied as ``-(-t + |c|)``: two sign flips and one positive add,
        all exact for the integer-valued tensors this is used on.
        """
        c = float(c)
        if c == 0.0:
            return t
        if c > -2.0:
            return self.affine(t, 1.0, c)
        return self.affine(self.affine(self.affine(t, -1.0, 0.0), 1.0, -c),
                           -1.0, 0.0)

    def clamp_zero(self, t, zero, want_max):
        """max(t, 0) / min(t, 0) for equal-shaped tensors (an exact select)."""
        m = self.cmp(vc4infer.OP_CMP_GE if want_max else vc4infer.OP_CMP_LE,
                     t, zero)
        return self.select(m, t, zero)

    def twiddle(self, D, N, K, C, inverse):
        """The padded [K2][2K] constant: exp(-2 pi i j k / N), j < D."""
        rows = D if C == 1 else 2 * D
        key = ("dft", self.node.name, D, N, K, C, bool(inverse))
        cache = self.r.__dict__.setdefault("_dft_np", {})
        if key in cache:
            return self.const(key, cache[key])
        j = np.arange(D)[:, None]
        k = np.arange(K)[None, :]
        th = 2.0 * np.pi * j * k / float(N)
        if inverse:
            th = -th
        co, si = np.cos(th), np.sin(th)
        b = np.zeros((rows, 2 * K), np.float32)
        b[:D, 0::2] = co
        b[:D, 1::2] = -si
        if C == 2:
            b[D:, 0::2] = si
            b[D:, 1::2] = co
        if inverse:
            b /= float(N)
        m = np.ascontiguousarray(np.pad(
            b, ((0, _al(rows, 16) - rows), (0, _al(2 * K, 64) - 2 * K))))
        cache[key] = m
        return self.const(key, m)

    # ---------------------------------------------------------- device RNG
    def rng_seed(self):
        """The stream seed: the ``seed`` attribute, else a hash of the node
        name (a seedless graph is then reproducible, not random)."""
        s = _attr(self.node, "seed", None)
        if s is not None:
            return int(abs(float(s))) % (1 << 24)
        h = hashlib.sha256(("%s|%s" % (self.node.name or "",
                                       self.node.op_type)).encode()).digest()
        return int.from_bytes(h[:4], "little") % (1 << 24)

    def rng_call(self):
        """A per-node launch counter: each invocation advances the stream by
        2^20 counter positions, so two draws in one model are independent and
        a repeated run of the same graph is reproducible."""
        d = getattr(self.r, "_rng_calls", None)
        if d is None:
            d = {}
            self.r._rng_calls = d
        key = self.node.name or self.node.op_type
        c = d.get(key, 0)
        d[key] = (c + 1) & 0xFFF
        return c

    def rng_key(self, call):
        """The 32-bit stream key for (seed, call).

        A key schedule, not a draw: one integer per launch, derived from the
        operator's seed and its launch counter.  Every element value is then
        produced by the QPU hash (``kops.b_rng``).
        """
        return (self.rng_seed() * 0x9E3779B1 + int(call) * 0x85EBCA77) \
            & 0xFFFFFFFF

    def rng_uniform(self, n, call):
        """``n`` uniforms in [0,1) from the QPU (one launch)."""
        n = int(n)
        cb = self.const(("rngcount", n),
                        np.arange(n, dtype=np.uint32).view(np.float32))
        buf = self.r.pool.alloc(n * 4 + 64)
        key = self.rng_key(call)
        self.r.eng.unary(vc4infer.UOP_RNG, cb.buf, buf, n,
                         float(key & 0xFFFF), float(key >> 16), 0.0)
        return self.r._tmp(_Tn()(buf, (n,)))


# =====================================================================
# QuantizeLinear / DynamicQuantizeLinear
# =====================================================================
class QuantizeLinear(_Sig):
    """y = saturate(round(x / scale) + zero_point), int8/uint8 *storage*.

    There is no packed-byte MAC on this GPU: the arena is fp32 and every
    kernel reads fp32 words.  This operator stores 4x fewer bytes at the graph
    boundary and nothing more -- a memory/IO lever, not a speed one, with no
    compute saved anywhere downstream.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        shape = tuple(int(d) for d in x.shape)
        if not int(_attr(node, "saturate", 1)):
            self.fail("saturate=0 (opset 19 float saturation) is not "
                      "implemented: it needs a float output range")
        if _attr(node, "block_size", None):
            self.fail("block_size=%d: block quantisation is not implemented"
                      % int(_attr(node, "block_size")))
        scale = self.param_vec(node.input[1], "y_scale")
        zp = None
        dtype = onnx.TensorProto.UINT8
        if len(node.input) > 2 and node.input[2]:
            zarr = self.param_raw(node.input[2])
            if zarr.dtype == np.dtype(np.int8):
                dtype = onnx.TensorProto.INT8
            zp = zarr.astype(np.float32).ravel()
        qmin, qmax = (0.0, 255.0) if dtype == onnx.TensorProto.UINT8 \
            else (-128.0, 127.0)
        axis = int(_attr(node, "axis", 1))
        rank = len(shape)
        if scale.size == 1:
            t = self.affine(x, float(np.float32(1.0) / scale[0]), 0.0)
            if zp is not None:
                t = self.add_const(t, float(zp[0]))
        else:
            if axis < 0:
                axis += rank
            if not (0 <= axis < rank) or scale.size != shape[axis]:
                self.fail("y_scale has %d values for axis %d of %s"
                          % (scale.size, axis, shape))
            if zp is not None and zp.size not in (1, scale.size):
                self.fail("y_zero_point has %d values for %d scales"
                          % (zp.size, scale.size))
            bshape = [1] * rank
            bshape[axis] = scale.size
            t = self.mul(x, self.const(
                ("qz_inv", node.name),
                np.ascontiguousarray(np.broadcast_to(
                    (np.float32(1.0) / scale).reshape(bshape), shape))))
            if zp is not None:
                z = zp if zp.size == scale.size else np.repeat(zp, scale.size)
                t = self.add(t, self.const(
                    ("qz_zp", node.name),
                    np.ascontiguousarray(np.broadcast_to(
                        z.reshape(bshape), shape))))
        t = self.unary(t, vc4infer.UOP_ROUND)
        if qmin < -2.0:
            # UOP_CLIP takes its bounds as kernel uniforms, and a value at or
            # below -2.0 has a bit pattern vc4d refuses as a possible bus
            # address: saturate int8 as clip(x + 128, 0, 255) - 128, which is
            # exact for the already-integral values
            t = self.add_const(t, 128.0)
            t = self.unary(t, vc4infer.UOP_CLIP, 0.0, 255.0)
            t = self.add_const(t, -128.0)
        else:
            t = self.unary(t, vc4infer.UOP_CLIP, qmin, qmax)
        return self.r.download(t).astype(_NP[dtype])


class DynamicQuantizeLinear(_Sig):
    """x -> (uint8 y, fp32 y_scale, uint8 y_zero_point), all on the QPUs.

    scale = (max(0, max x) - min(0, min x)) / 255 (1 when that is 0),
    zp = rint(clamp(-min / scale, 0, 255)),
    y = clamp(rint(x / scale) + zp, 0, 255).
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if x.size == 0:
            self.fail("empty input")
        flat = self.reshaped(x, (int(x.size),))
        zero = self.const(("dq_zero",), np.zeros(1, np.float32))
        hi = self.clamp_zero(self.max_all(flat, True), zero, True)
        neg = self.max_all(self.affine(flat, -1.0, 0.0), True)
        lo = self.clamp_zero(self.affine(neg, -1.0, 0.0), zero, False)
        span = self.sub(hi, lo)
        pos = self.cmp(vc4infer.OP_CMP_GT, span, zero)
        # ONNX: scale = 1/255 when max == min, (max-min)/255 otherwise
        scale = self.select(
            pos,
            self.mul(span, self.const(("dq_255",),
                                      np.full(1, np.float32(1.0 / 255.0)))),
            self.const(("dq_inv255",),
                       np.full(1, np.float32(1.0 / 255.0))))
        inv = self.unary(scale, vc4infer.UOP_RECIPROCAL)
        zf = self.mul(self.affine(lo, -1.0, 0.0), inv)
        zf = self.unary(zf, vc4infer.UOP_CLIP, 0.0, 255.0)
        zf = self.unary(zf, vc4infer.UOP_ROUND)
        # the two scalars parametrise the kernels below; what the graph gets
        # back for y_scale/y_zero_point is the device tensors themselves
        sv = float(self.r.download(scale)[0])
        zv = float(self.r.download(zf)[0])
        t = self.add_const(self.affine(x, 1.0 / sv, 0.0), zv)
        t = self.unary(t, vc4infer.UOP_ROUND)
        t = self.unary(t, vc4infer.UOP_CLIP, 0.0, 255.0)
        y = self.r.download(t).astype(np.uint8)
        return [y, np.asarray(sv, np.float32), np.asarray(int(zv), np.uint8)]


# =====================================================================
# DFT
# =====================================================================
class DFT(_Sig):
    """One GEMM against a constant twiddle matrix.

    The output column order is (k, real/imag) interleaved *by construction*
    (column 2k is Re X[k], column 2k+1 is Im X[k]), so the GEMM result is
    already the ONNX layout: no interleave pass is needed.  Any transform axis
    works; the twiddle matrix is the only host-built constant.
    """

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        shape = [int(d) for d in x.shape]
        if len(shape) < 2:
            self.fail("needs rank >= 2 (got %s)" % (tuple(shape),))
        C = shape[-1]
        if C not in (1, 2):
            self.fail("the last axis must be the complex pair (1 or 2 wide); "
                      "got %d" % C)
        axis = _attr(node, "axis", None)
        if len(node.input) > 2 and node.input[2]:
            axis = int(np.asarray(self.host_or_fail(node.input[2])).ravel()[0])
        axis = int(-2 if axis is None else axis) % len(shape)
        if axis >= len(shape) - 1:
            self.fail("axis %d is the complex axis" % axis)
        N = shape[axis]
        if len(node.input) > 1 and node.input[1]:
            N = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        if N < 1:
            self.fail("dft_length %d" % N)
        inverse = bool(int(_attr(node, "inverse", 0)))
        onesided = bool(int(_attr(node, "onesided", 0)))
        K = N // 2 + 1 if onesided else N
        P = _prod(shape[:axis])
        D = shape[axis]
        I0 = _prod(shape[axis + 1:]) // C
        if D < 1:
            self.fail("the transform axis is empty")
        if P * K * I0 > 20000:
            self.fail("a %dx%dx%d transform is refused: when the transform "
                      "axis is not the second-to-last one the output needs "
                      "one 2-word copy per complex pair (%d here, the limit "
                      "is 20000)" % (P, K, I0, P * K * I0))
        return self.core(x, P, D, I0, C, N, K, inverse)

    def core(self, x, P, D, I0, C, N, K, inverse):
        """x: [P][D][I0][C] -> [P][K][I0][2] on the QPUs.

        ``np.fft.fft(x, n=N)`` truncates when N < D and zero-pads when N > D;
        zero samples contribute nothing, so only the first min(D, N) samples
        take part in the sum.
        """
        J = min(D, N)          # truncate (N < D); zero samples add nothing
        I = I0 * C
        R = P * I0
        K2 = (J if C == 1 else 2 * J)
        Mp, Kp, Np = _al(R, 8), _al(K2, 16), _al(2 * K, 64)
        A = self.alloc_fill((Mp, Kp), 0.0)
        for r in range(R):
            p, i = divmod(r, I0)
            base = (p * D * I + i * C) * 4      # D is the *stored* length
            self.lane_runs(self.r._buf_at(x, base),
                           self.r._buf_at(A, r * Kp * 4), 1, J, 0, 0, I * 4)
            if C == 2:
                self.lane_runs(self.r._buf_at(x, base + 4),
                               self.r._buf_at(A, (r * Kp + J) * 4), 1, J,
                               0, 0, I * 4)
        W = self.twiddle(J, N, K, C, inverse)
        out = self.alloc((Mp, Np))
        self.r.eng.matmul(A.buf, W.buf, out.buf, Mp, Np, Kp)
        if I0 == 1:
            o = self.alloc((P, K, 2))
            self.lane_runs(out.buf, o.buf, P, 2 * K, Np * 4, 2 * K * 4, 4)
            return o
        o = self.alloc((P, K, I0, 2))
        for p in range(P):
            for k in range(K):
                for i in range(I0):
                    src = self.r._buf_at(out, ((p * I0 + i) * Np + 2 * k) * 4)
                    dst = self.r._buf_at(o, (((p * K + k) * I0 + i) * 2) * 4)
                    self.r._runs(src, dst, 1, 2, 0, 0)
        return o

# =====================================================================
# STFT
# =====================================================================
class STFT(_Sig):
    """framing (copy2d) + the DFT core, the window as a constant panel."""

    def run(self):
        node = self.node
        sig = self.ten(node.input[0])
        shape = [int(d) for d in sig.shape]
        if len(shape) != 3:
            self.fail("needs [batch][length][1|2] (got %s)" % (tuple(shape),))
        B, L, C = shape
        if C not in (1, 2):
            self.fail("the last axis must be 1 or 2 (got %d)" % C)
        step = int(np.asarray(self.host_or_fail(node.input[1])).ravel()[0])
        if step < 1:
            self.fail("frame_step %d" % step)
        win = self.param_vec(node.input[2], "window") \
            if len(node.input) > 2 and node.input[2] else None
        flen = int(win.size) if win is not None else L
        if flen < 1 or flen > L:
            self.fail("frame_length %d with signal length %d" % (flen, L))
        F = 1 + (L - flen) // step
        if F < 1:
            self.fail("no complete frame: length %d, frame %d, step %d"
                      % (L, flen, step))
        onesided = int(_attr(node, "onesided", 1))
        K = flen // 2 + 1 if onesided else flen
        T = self.alloc((B * F, flen, C))
        for b in range(B):
            for f in range(F):
                off = (b * L + f * step) * C
                self.r._runs(self.r._buf_at(sig, off * 4),
                             self.r._buf_at(T, ((b * F + f) * flen) * C * 4),
                             1, flen * C, 0, 0)
        if win is not None:
            tile = np.tile(np.repeat(win, C), B * F).reshape(B * F, flen, C)
            T = self.mul(T, self.const(
                ("stftwin", node.name, B, F, flen, C),
                np.ascontiguousarray(tile, np.float32)))
        o = DFT.core(self, T, B * F, flen, 1, C, flen, K, False)
        return self.reshaped(o, (B, F, K, 2))


# =====================================================================
# the random family
# =====================================================================
class RandomUniform(_Sig):
    def run(self):
        node = self.node
        if node.op_type.endswith("Like"):
            shape = tuple(int(d) for d in self.ten(node.input[0]).shape)
        else:
            shape = tuple(int(d) for d in (_attr(node, "shape", []) or []))
        dtype = int(_attr(node, "dtype", onnx.TensorProto.FLOAT))
        if dtype != onnx.TensorProto.FLOAT:
            self.fail("dtype=%s: only float32 draws are produced (the "
                      "runner boundary is fp32)"
                      % onnx.TensorProto.DataType.Name(dtype))
        n = _prod(shape)
        if n <= 0:
            self.fail("shape %s has no elements" % (shape,))
        lo = float(_attr(node, "low", 0.0))
        hi = float(_attr(node, "high", 1.0))
        if hi < lo:
            self.fail("high %g < low %g" % (hi, lo))
        u = self.rng_uniform(n, self.rng_call())
        if (lo, hi) != (0.0, 1.0):
            u = self.add_const(self.affine(u, hi - lo, 0.0), lo)
        return self.reshaped(u, shape)


class RandomNormal(_Sig):
    """Box-Muller on the QPUs from two counter-based uniforms."""

    def run(self):
        node = self.node
        if node.op_type.endswith("Like"):
            shape = tuple(int(d) for d in self.ten(node.input[0]).shape)
        else:
            shape = tuple(int(d) for d in (_attr(node, "shape", []) or []))
        dtype = int(_attr(node, "dtype", onnx.TensorProto.FLOAT))
        if dtype != onnx.TensorProto.FLOAT:
            self.fail("dtype=%s: only float32 draws are produced"
                      % onnx.TensorProto.DataType.Name(dtype))
        n = _prod(shape)
        if n <= 0:
            self.fail("shape %s has no elements" % (shape,))
        mean = float(_attr(node, "mean", 0.0))
        scale = float(_attr(node, "scale", 1.0))
        if scale < 0:
            self.fail("scale %g is negative" % scale)
        c = self.rng_call()
        u1 = self.rng_uniform(n, c)
        u2 = self.rng_uniform(n, (c + 1) & 0xFFF)
        eps = 2.0 ** -25
        # a uniform remapped into (0,1): log() stays finite and the interval
        # shrinks by 2^-24 of its length, far below its own 2^-24 resolution
        u1 = self.affine(u1, 1.0 - 2 * eps, eps)
        # -2 ln u as 2 ln(1/u): no uniform parameter may be at or below -2.0
        # (that bit pattern looks like a GPU bus address to vc4d), and the 2
        # folds into the final scale instead
        r = self.unary(self.unary(u1, vc4infer.UOP_RECIPROCAL),
                       vc4infer.UOP_LOG)
        r = self.unary(r, vc4infer.UOP_SQRT)
        th = self.affine(u2, float(2 * math.pi), 0.0)
        z = self.mul(r, self.unary(th, vc4infer.UOP_COS))
        # the Box-Muller radius is sqrt(2 ln(1/u1)) * cos(...): one affine
        # carries the operator's scale, its mean and the sqrt(2)
        z = self.affine(z, scale * math.sqrt(2.0), mean)
        return self.reshaped(z, shape)


class Bernoulli(_Sig):
    def run(self):
        node = self.node
        p = self.ten(node.input[0])
        dtype = int(_attr(node, "dtype", onnx.TensorProto.FLOAT))
        if dtype != onnx.TensorProto.FLOAT:
            self.fail("dtype=%s: only float32 0/1 output is produced"
                      % onnx.TensorProto.DataType.Name(dtype))
        n = int(p.size)
        u = self.rng_uniform(n, self.rng_call())
        return self.reshaped(self.cmp(vc4infer.OP_CMP_LT, u, p), p.shape)


class Multinomial(_Sig):
    """Inverse-CDF sampling; int32/int64 indices are an index output."""

    def run(self):
        node = self.node
        x = self.ten(node.input[0])
        if len(x.shape) != 2:
            self.fail("needs [batch][class] (got %s)" % (tuple(x.shape),))
        Bat, Cn = (int(d) for d in x.shape)
        sample = int(_attr(node, "sample_size", 1))
        if sample < 1:
            self.fail("sample_size %d" % sample)
        dtype = int(_attr(node, "dtype", onnx.TensorProto.INT32))
        if dtype not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            self.fail("dtype=%s: int32/int64 only"
                      % onnx.TensorProto.DataType.Name(dtype))
        if Cn < 1 or Bat < 1:
            self.fail("empty input %s" % (tuple(x.shape),))
        self.download_check(
            x, lambda a: bool(np.max(np.abs(a)) <= 80.0),
            "Multinomial: |input| <= 80 (the direct-exponential log-"
            "probability domain)")
        e = self.unary(x, vc4infer.UOP_EXP)
        s = self.row_sum(e, Bat, Cn)
        inv = self.unary(s, vc4infer.UOP_RECIPROCAL)
        p = self.mul(e, self.bcast_rows(inv, Bat, Cn))
        cdf = self.row_tri(p, Bat, Cn)             # inclusive prefix sum
        # one launch of Bat*sample consecutive counter positions: the stream
        # the 10^6-draw evidence covers.  Sample k is the contiguous block
        # [k*Bat, (k+1)*Bat), i.e. position = k*Bat + b.
        u = self.rng_uniform(Bat * sample, self.rng_call())
        out = np.empty((Bat, sample), np.int64)
        for k in range(sample):
            uk = _Tn()(self.r._buf_at(u, k * Bat * 4), (Bat,))
            # the sampled index is the number of classes with cdf <= u
            m = self.cmp(vc4infer.OP_CMP_LE, cdf,
                         self.bcast_rows(uk, Bat, Cn))
            out[:, k] = np.rint(self.r.download(
                self.row_sum(m, Bat, Cn))).astype(np.int64)
        if np.any(out < 0) or np.any(out >= Cn):
            self.fail("internal: sampled index outside [0,%d)" % Cn)
        return out.astype(_NP[dtype])



# =====================================================================
# window tables and the mel filterbank
# =====================================================================
class _Window(_Sig):
    """HannWindow / HammingWindow / BlackmanWindow, evaluated on the QPUs.

    ONNX "Windows" (HannWindow-17, HammingWindow-17, BlackmanWindow-17): with
    N_1 = size when periodic else size - 1 and n = 0 .. size-1,

        Hann       w[n] = sin^2(pi n / N_1)
        Hamming    w[n] = alpha - beta cos(2 pi n / N_1),
                        alpha = 25/46, beta = 1 - alpha
        Blackman   w[n] = 0.42 - 0.5 cos(2 pi n / N_1)
                           + 0.08 cos(4 pi n / N_1)

    The index ramp n is integer metadata (a plan-time constant panel); the
    angle scaling, the SIN/COS and the coefficient arithmetic are engine
    kernels, so the table is computed by the QPU on every run instead of being
    folded into the plan.  Only fp32 output is wired: ``output_datatype`` may
    name half or double, which the fp32 arena cannot hold.
    """

    def size_and_n1(self):
        node = self.node
        size = int(np.asarray(self.host_or_fail(node.input[0])).ravel()[0])
        if size < 1:
            self.fail("size must be >= 1 (got %d)" % size)
        odt = int(_attr(node, "output_datatype", onnx.TensorProto.FLOAT))
        if odt != onnx.TensorProto.FLOAT:
            self.fail("output_datatype=%s: the arena is fp32 only"
                      % onnx.TensorProto.DataType.Name(odt))
        periodic = int(_attr(node, "periodic", 1))
        n1 = size if periodic else size - 1
        if n1 < 1:
            self.fail("size 1 with periodic=0 leaves N_1 = 0: the window is "
                      "undefined (division by zero)")
        return size, n1

    def ramp(self, size):
        return self.const(("win_ramp", int(size)),
                          np.arange(int(size), dtype=np.float32))


class HannWindow(_Window):
    def run(self):
        size, n1 = self.size_and_n1()
        s = self.unary(self.affine(self.ramp(size), math.pi / n1, 0.0),
                       vc4infer.UOP_SIN)
        return self.mul(s, s)


class HammingWindow(_Window):
    def run(self):
        size, n1 = self.size_and_n1()
        alpha = 25.0 / 46.0
        beta = 1.0 - alpha
        c = self.unary(self.affine(self.ramp(size),
                                   2.0 * math.pi / n1, 0.0),
                       vc4infer.UOP_COS)
        return self.affine(c, -beta, alpha)


class BlackmanWindow(_Window):
    ALPHA = 0.42
    BETA = 0.08

    def run(self):
        size, n1 = self.size_and_n1()
        r = self.ramp(size)
        w = 2.0 * math.pi / n1
        c1 = self.unary(self.affine(r, w, 0.0), vc4infer.UOP_COS)
        c2 = self.unary(self.affine(r, 2.0 * w, 0.0), vc4infer.UOP_COS)
        return self.add(self.affine(c1, -0.5, 0.0),
                        self.affine(c2, self.BETA, self.ALPHA))


class MelWeightMatrix(_Sig):
    """MelWeightMatrix: the filterbank values are QPU arithmetic.

    ONNX MelWeightMatrix-17 (formula spelled out in the operator
    documentation): with mel(f) = 2595 log10(1 + f/700),

        m_i   = mel(low) + i * (mel(high) - mel(low)) / (num_mel_bins + 2)
        f_i   = 700 (10^(m_i/2595) - 1)
        bin_i = floor((dft_length + 1) f_i / sample_rate)

    and band i, with (a, b, c) = (bin_i, bin_i+1, bin_i+2), owns the triangle
    that rises from 0 at a to 1 at b and falls back to 0 at c.  The two
    degenerate cases the reference defines are kept: b == a gives the single
    entry at b the value 1, and c == b has no falling half.

    The bins are *index metadata*: five scalar parameters in, integers out,
    the same class of work as an axis list or a padding map.  The ramps and
    the per-band gathers are index maps, and every value in the table is
    produced by engine kernels -- subtract, reciprocal-multiply (divide),
    exact comparisons, the two clamps and the max of the rising and falling
    halves -- so no tensor arithmetic runs on the ARM.
    """

    def run(self):
        node = self.node
        if len(node.input) < 5:
            self.fail("needs five scalar inputs (got %d)" % len(node.input))
        v = [np.asarray(self.host_or_fail(node.input[i])).ravel()[0]
             for i in range(5)]
        mel, dft, sr = int(v[0]), int(v[1]), int(v[2])
        low, high = float(v[3]), float(v[4])
        odt = int(_attr(node, "output_datatype", onnx.TensorProto.FLOAT))
        if odt != onnx.TensorProto.FLOAT:
            self.fail("output_datatype=%s: the arena is fp32 only"
                      % onnx.TensorProto.DataType.Name(odt))
        if mel < 1 or dft < 2 or sr <= 0:
            self.fail("need num_mel_bins >= 1, dft_length >= 2 and a positive "
                      "sample_rate (got %d, %d, %d)" % (mel, dft, sr))
        bins = dft // 2 + 1
        lo_mel = 2595.0 * np.log10(1.0 + low / 700.0)
        hi_mel = 2595.0 * np.log10(1.0 + high / 700.0)
        step = (hi_mel - lo_mel) / (mel + 2)
        f = 700.0 * (np.power(10.0, (lo_mel + step * np.arange(mel + 2))
                              / 2595.0) - 1.0)
        edge = np.floor((dft + 1) * f / sr).astype(np.int64)
        a = edge[:mel]
        b = edge[1:mel + 1]
        c = edge[2:mel + 2]
        if int(a.min()) < 0 or int(c.max()) > bins:
            self.fail("mel edges [%d..%d] fall outside the %d DFT bins"
                      % (int(a.min()), int(c.max()), bins))
        n = bins * mel
        k = np.arange(n, dtype=np.int64)
        jj = (k // mel).astype(np.float32)
        ii = (k % mel)
        key = (int(mel), int(dft), int(sr), round(low, 6), round(high, 6))
        J = self.const(("mel_j",) + key, jj)
        A = self.const(("mel_a",) + key, a[ii].astype(np.float32))
        B = self.const(("mel_b",) + key, b[ii].astype(np.float32))
        C = self.const(("mel_c",) + key, c[ii].astype(np.float32))
        one = self.const(("mel_one", n), np.ones(n, np.float32))
        zero = self.const(("mel_zero", n), np.zeros(n, np.float32))

        def clamp(t):
            return self.maxmin(self.maxmin(t, one, False), zero, True)

        def rcp(t):
            return self.unary(t, vc4infer.UOP_RECIPROCAL)

        lc = self.sub(B, A)                       # b - a
        lc0 = self.cmp(vc4infer.OP_CMP_EQ, lc, zero)
        lcs = self.add(lc, lc0)                   # >= 1, so the divide is safe
        rise = clamp(self.mul(self.sub(J, A), rcp(lcs)))
        rise = self.mul(rise, self.cmp(vc4infer.OP_CMP_LE, J, B))
        ch = self.sub(C, B)                       # c - b
        chp = self.cmp(vc4infer.OP_CMP_GT, ch, zero)
        chs = self.add(ch, self.cmp(vc4infer.OP_CMP_EQ, ch, zero))
        fall = clamp(self.mul(self.sub(C, J), rcp(chs)))
        fall = self.mul(fall, self.cmp(vc4infer.OP_CMP_GE, J, B))
        fall = self.mul(fall, chp)                # c == b: no falling half
        out = self.maxmin(rise, fall, True)
        spike = self.mul(lc0, self.cmp(vc4infer.OP_CMP_EQ, J, B))
        out = self.maxmin(out, spike, True)       # b == a: the entry at b is 1
        return self.tag(out, (bins, mel))


# =====================================================================
# dispatch table
# =====================================================================
def _mk(cls):
    def run(r, node):
        return cls(r, node).run()
    return run


# operators the main table also knows and this family owns: none
OVERRIDES = ()

HANDLERS = {
    "QuantizeLinear": _mk(QuantizeLinear),
    "DynamicQuantizeLinear": _mk(DynamicQuantizeLinear),
    "DFT": _mk(DFT),
    "STFT": _mk(STFT),
    "RandomUniform": _mk(RandomUniform),
    "RandomUniformLike": _mk(RandomUniform),
    "RandomNormal": _mk(RandomNormal),
    "RandomNormalLike": _mk(RandomNormal),
    "Bernoulli": _mk(Bernoulli),
    "Multinomial": _mk(Multinomial),
    "HannWindow": _mk(HannWindow),
    "HammingWindow": _mk(HammingWindow),
    "BlackmanWindow": _mk(BlackmanWindow),
    "MelWeightMatrix": _mk(MelWeightMatrix),
}
