#!/usr/bin/env python3
"""onnxrun.py -- execute a small ONNX graph on the VideoCore IV QPUs.

Every arithmetic node is dispatched to libvc4infer through its ctypes binding
(``vc4infer.py``).  There is no CPU fallback: a node, attribute combination or
shape the engine cannot do raises ``UnsupportedNode`` naming the node and its
op type.  The only CPU work is byte movement that the engine itself also does
on the ARM (padding, channel gathering, concat, nearest resize, compaction);
those helpers never multiply, add or compare floats.

Supported nodes
    Conv         3x3/1x1, stride 1; stride 2 (3x3 pad 1 or 1x1 pad 0) via a
                 4-phase stride-1 conv; group=1 or depthwise (groups=Cin=Cout)
                 via an expanded block-diagonal weight; channel padding to a
                 multiple of 16 is a zero fill
    Relu, Add, Mul, Sigmoid
    MaxPool      2x2 stride 2 (native kernel) or any NxN/stride/pad (generic kernel)
    Concat       channel axis (NCHW)
    Resize       nearest, asymmetric coordinates (integer or float scale)
    MatMul      2-D
    Gemm         transA=0, alpha=1, beta in {0,1}; bias is a GPU Add
    Softmax      last axis, <= 16 classes, rows padded to 8
    Flatten, Reshape, Identity
    GlobalAveragePool  as a GPU GEMM with a 1/HW column

Layout-only nodes are executed by copying bytes in the arena (the same class
of operation as the engine's own padding and NCHW gather); the docs state this
explicitly.
"""
import os
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[-1]


ENGINE = _first_dir(os.environ.get("VC4INFER_ENGINE"),
                    os.path.join(HERE, "..", "engine"),
                    "/home/raspi/work/vc4-stack/engine")
sys.path.insert(0, os.environ.get("VC4INFER_PY",
                                  os.path.join(ENGINE, "python")))
import vc4infer  # noqa: E402


class UnsupportedNode(RuntimeError):
    pass


class Tensor:
    __slots__ = ("buf", "shape")

    def __init__(self, buf, shape):
        self.buf = buf
        self.shape = tuple(int(s) for s in shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __repr__(self):
        return "Tensor(%s)" % (self.shape,)


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode()
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOATS:
                return list(a.floats)
            if a.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t)
    return default


def _align16(v):
    return (int(v) + 15) & ~15


class Runner:
    def __init__(self, model_path, eng):
        self.path = model_path
        self.eng = eng
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.values = {}
        self.host = {}          # host copy of every initializer / Constant
        self.float_inputs = []
        self.profile = []

        for init in self.graph.initializer:
            self.values[init.name] = self._constant(init.name,
                                                    numpy_helper.to_array(init))
        for inp in self.graph.input:
            if inp.name in self.values:
                continue
            tt = inp.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else -1
                    for d in tt.shape.dim]
            if tt.elem_type == onnx.TensorProto.FLOAT:
                self.float_inputs.append((inp.name, dims))
            else:
                raise UnsupportedNode("graph input %s: only fp32 inputs" % inp.name)

    # ---------------------------------------------------------------- buffers
    def _alloc(self, n):
        return self.eng.alloc(int(n) * 4 + 64)

    def upload(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        buf = self._alloc(arr.size)
        self.eng.upload(arr.ravel(), buf)
        return Tensor(buf, arr.shape)

    def download(self, t):
        x = self.eng.download(t.buf, t.size * 4)
        return np.asarray(x, dtype=np.float32).reshape(t.shape).copy()

    def _constant(self, name, arr):
        arr = np.asarray(arr)
        self.host[name] = arr
        if arr.dtype in (np.float32, np.float64):
            return self.upload(arr.astype(np.float32))
        return arr                      # shape / index tensors stay on the host

    def _get(self, name):
        return self.values[name]

    def _float(self, name):
        v = self.values[name]
        if not isinstance(v, Tensor):
            raise UnsupportedNode("%s is not an fp32 tensor" % name)
        return v

    # ------------------------------------------------------------- utilities
    def _upload_conv_input(self, x, cpad):
        """Zero-pad the channel dimension (byte layout, no arithmetic)."""
        a = self.download(x)
        if a.shape[1] == cpad:
            return x
        p = np.zeros((a.shape[0], cpad) + a.shape[2:], np.float32)
        p[:, :a.shape[1]] = a
        return self.upload(p)

    def _prepad(self, x, pads, value=0.0):
        """pads = [top, left, bottom, right]; returns a new tensor."""
        a = self.download(x)
        p = np.pad(a, ((0, 0), (0, 0), (pads[0], pads[2]),
                       (pads[1], pads[3])), constant_values=value)
        return self.upload(p)

    def _conv_core(self, x, w, b, kh, kw, stride, pad, name="Conv"):
        C = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        hp = H + 2 * pad
        wp = ((W + 2 * pad) + 15) & ~15
        if hp * wp * 4 > 8191:
            raise UnsupportedNode(
                "Conv %s: padded plane %dx%d = %d B > 8191 B "
                "(engine im2col VDR pitch limit)" % (name, hp, wp, hp * wp * 4))
        Cout = w.shape[0]
        out = self._alloc(Cout * ((H + 2 * pad - kh) // stride + 1) *
                          ((W + 2 * pad - kw) // stride + 1))
        ho, wo = self.eng.conv2d(x.buf, w.buf, out, C, H, W, Cout, kh, kw,
                                 bias=b.buf if b is not None else None,
                                 stride=stride, pad=pad)
        return Tensor(out, (1, Cout, ho, wo))

    # ------------------------------------------------------------------ nodes
    def _conv(self, node):
        x = self._float(node.input[0])
        w = self._float(node.input[1])
        b = self._float(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if len(x.shape) != 4:
            raise UnsupportedNode("Conv %s: expected NCHW input" % node.name)
        if x.shape[0] != 1:
            raise UnsupportedNode("Conv %s: batch != 1" % node.name)

        group = int(_attr(node, "group", 1))
        ks = _attr(node, "kernel_shape", list(w.shape[2:]))
        kh, kw = int(ks[0]), int(ks[1])
        strides = _attr(node, "strides", [1, 1])
        sh, sw = int(strides[0]), int(strides[1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        dil = _attr(node, "dilations", [1, 1])
        if list(dil) != [1, 1]:
            raise UnsupportedNode("Conv %s: dilations" % node.name)
        if sh != sw:
            raise UnsupportedNode("Conv %s: non-square stride" % node.name)

        Cin = x.shape[1]
        Cout = w.shape[0]

        # Fail before any large layout copy if the engine's im2col pitch
        # cannot hold the (padded, or for stride 2 phase-split) plane.
        H, W = x.shape[2], x.shape[3]
        if pads[0] == pads[1] == pads[2] == pads[3]:
            p = pads[0]
            if sh == 1:
                hp, wp = H + 2 * p, _align16(W + 2 * p)
            elif sh == 2 and kh == 3:
                hp = (H + 1) // 2 + 2
                wp = _align16(((W + 1) // 2) + 2)
            else:
                hp = wp = 1
            if hp * wp * 4 > 8191:
                raise UnsupportedNode(
                    "Conv %s: padded plane %dx%d = %d B > 8191 B "
                    "(engine im2col VDR pitch limit)" %
                    (node.name, hp, wp, hp * wp * 4))

        wa = self.download(w)
        ba = self.download(b) if b is not None else None

        if group not in (1, Cin):
            raise UnsupportedNode("Conv %s: groups=%d (only 1 or depthwise)"
                                  % (node.name, group))

        # channel padding to the engine's %16 requirement (zeros, layout only)
        cpad = _align16(Cin)
        opad = _align16(Cout)
        xp = self._upload_conv_input(x, cpad)
        if wa.shape[1] != cpad and group == 1:
            wa = np.pad(wa, ((0, 0), (0, cpad - wa.shape[1]), (0, 0), (0, 0)))

        if group == Cin:
            if sh != 1:
                raise UnsupportedNode("Conv %s: stride-2 depthwise" % node.name)
            # expand to a dense, block-diagonal [Cout][Cin][kh][kw] weight; the
            # QPU GEMM then does the same per-channel MACs (zeros on the
            # off-diagonal).  Weight expansion is a byte copy/zero fill.
            dense = np.zeros((opad, cpad, kh, kw), np.float32)
            for c in range(Cout):
                dense[c, c] = wa[c, 0]
            wa = dense
            # the input channels are already cpad
        else:
            wa = np.pad(wa, ((0, opad - Cout), (0, 0), (0, 0), (0, 0))
                        if wa.shape[0] != opad else ((0, 0), (0, 0), (0, 0), (0, 0)))
        if ba is not None and ba.size != opad:
            ba = np.pad(ba, (0, opad - ba.size))
        if ba is None:
            ba = np.zeros(opad, np.float32)
        wb = self.upload(wa)
        bb = self.upload(ba)

        # asymmetric padding is a byte fill on the host, then pad=0
        if pads[0] == pads[1] == pads[2] == pads[3]:
            pad = int(pads[0])
        else:
            xp = self._prepad(xp, pads)
            pad = 0

        # ---- stride 1 -----------------------------------------------------
        if sh == 1:
            t = self._conv_core(xp, wb, bb, kh, kw, 1, pad, node.name)
        # ---- stride 2 -----------------------------------------------------
        elif sh == 2:
            if kh == 1 and kw == 1 and pad == 0:
                # output[i][j] = in[2i][2j]: one phase + 1x1 conv
                a = self.download(xp)
                ph = np.ascontiguousarray(a[:, :, 0::2, 0::2])
                t = self._conv_core(self.upload(ph), wb, bb, 1, 1, 1, 0,
                                    node.name)
            elif kh == 3 and kw == 3 and pad == 1 and \
                    pads == [1, 1, 1, 1]:
                t = self._conv_stride2_3x3(xp, wa, bb, cpad, Cout, opad,
                                           node.name)
            else:
                raise UnsupportedNode(
                    "Conv %s: stride 2 with k=%dx%d pads=%s"
                    % (node.name, kh, kw, pads))
        else:
            raise UnsupportedNode("Conv %s: stride %d" % (node.name, sh))

        if t.shape[1] != Cout:
            a = self.download(t)[:, :Cout]
            t = self.upload(np.ascontiguousarray(a))
        return t

    def _conv_stride2_3x3(self, xp, wa, bb, cpad, Cout, opad, name="Conv"):
        """Conv 3x3 stride 2 pad 1 == 4 phase images + a stride-1 conv.

        Source pixel 2i+ky-1 has row parity a=(ky-1)&1 and phase index
        i+dr; the same holds for columns.  Concatenating the four de-interleaved
        phase images on the channel axis and giving each phase block a 3x3
        weight with the taps that belong to its parity (others zero) reproduces
        the stride-2 conv exactly.  The phase split is a byte gather; the
        MACs are the verified QPU im2col+GEMM.
        """
        a = self.download(xp)
        H, W = a.shape[2], a.shape[3]
        if H % 2 or W % 2:
            a = np.pad(a, ((0, 0), (0, 0), (0, H % 2), (0, W % 2)))
        H2, W2 = a.shape[2] // 2, a.shape[3] // 2
        ph = np.empty((1, 4 * cpad, H2, W2), np.float32)
        for ar in (0, 1):
            for ac in (0, 1):
                blk = (ar * 2 + ac) * cpad
                ph[0, blk:blk + cpad] = a[:, :, ar::2, ac::2]
        wp = np.zeros((opad, 4 * cpad, 3, 3), np.float32)
        # output[i] uses source i-1, i, i+1 (padded); map each tap to the
        # phase-kernel index kh where the stride-1 conv reads P[i+kh-1].
        for ar in (0, 1):
            rows = [(1, 1)] if ar == 0 else [(0, 0), (1, 2)]
            for ac in (0, 1):
                cols = [(1, 1)] if ac == 0 else [(0, 0), (1, 2)]
                blk = (ar * 2 + ac) * cpad
                for kh, ky in rows:
                    for kw, kx in cols:
                        wp[:Cout, blk:blk + cpad, kh, kw] = wa[:Cout, :, ky, kx]
        wb2 = self.upload(wp)
        return self._conv_core(self.upload(np.ascontiguousarray(ph)),
                               wb2, bb, 3, 3, 1, 1, name)

    def _relu(self, node):
        x = self._float(node.input[0])
        out = self._alloc(x.size)
        self.eng.elementwise(vc4infer.OP_RELU, x.buf, out, x.size)
        return Tensor(out, x.shape)

    def _sigmoid(self, node):
        x = self._float(node.input[0])
        out = self._alloc(x.size)
        self.eng.elementwise(vc4infer.OP_SIGMOID, x.buf, out, x.size)
        return Tensor(out, x.shape)

    def _binary(self, node, op):
        a = self._float(node.input[0])
        b = self._float(node.input[1])
        if a.shape != b.shape:
            raise UnsupportedNode("%s %s: broadcast %s vs %s"
                                  % (node.op_type, node.name, a.shape, b.shape))
        out = self._alloc(a.size)
        self.eng.elementwise(op, a.buf, out, a.size, b=buf_of(b))
        return Tensor(out, a.shape)

    def _maxpool(self, node):
        x = self._float(node.input[0])
        if len(x.shape) != 4:
            raise UnsupportedNode("MaxPool %s: expected NCHW" % node.name)
        ks = _attr(node, "kernel_shape")
        strides = _attr(node, "strides", [1, 1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        if _attr(node, "ceil_mode", 0):
            raise UnsupportedNode("MaxPool %s: ceil_mode" % node.name)
        if _attr(node, "dilations", [1, 1]) != [1, 1]:
            raise UnsupportedNode("MaxPool %s: dilations" % node.name)
        N, C, H, W = x.shape
        if N != 1:
            raise UnsupportedNode("MaxPool %s: batch != 1" % node.name)
        n, s = int(ks[0]), int(strides[0])
        if ks[0] != ks[1] or strides[0] != strides[1]:
            raise UnsupportedNode("MaxPool %s: non-square window" % node.name)
        if pads != [0, 0, 0, 0] and pads != [pads[0]] * 4:
            x = self._prepad(x, pads, value=-np.inf)
            pads = [0, 0, 0, 0]
        pad = int(pads[0])
        # the generic QPU kernel covers every window/stride/pad; the 2x2
        # specialisation is not needed and the native entry point has extra
        # alignment restrictions
        ho = (H + 2 * pad - n) // s + 1
        wo = (W + 2 * pad - n) // s + 1
        op = _align16(wo)
        hop = (ho + 3) & ~3
        out = self._alloc(C * hop * op)
        h, w = self.eng.maxpooln(x.buf, out, C, H, W, n, s, pad)
        t = Tensor(out, (1, C, hop, op))
        a = self.download(t)[:, :, :h, :w]
        return self.upload(np.ascontiguousarray(a))

    def _concat(self, node):
        axis = int(_attr(node, "axis", 0))
        ts = [self._float(n) for n in node.input if n]
        shape = ts[0].shape
        if len(shape) != 4 or axis not in (1, -3):
            raise UnsupportedNode("Concat %s: axis %d (only channel axis of NCHW)"
                                  % (node.name, axis))
        for t in ts:
            if t.shape[0] != shape[0] or t.shape[2:] != shape[2:]:
                raise UnsupportedNode("Concat %s: shape mismatch" % node.name)
        a = np.concatenate([self.download(t) for t in ts], axis=1)
        return self.upload(np.ascontiguousarray(a))

    def _resize(self, node):
        x = self._float(node.input[0])
        mode = _attr(node, "mode", "nearest")
        coord = _attr(node, "coordinate_transformation_mode", "half_pixel")
        if mode != "nearest":
            raise UnsupportedNode("Resize %s: mode=%s" % (node.name, mode))
        if coord not in ("asymmetric", "half_pixel"):
            raise UnsupportedNode("Resize %s: coord=%s" % (node.name, coord))
        scales = None
        sizes = None
        if len(node.input) > 2 and node.input[2]:
            v = self.host.get(node.input[2], self.values.get(node.input[2]))
            scales = np.asarray(v, dtype=np.float64)
        if len(node.input) > 3 and node.input[3]:
            v = self.host.get(node.input[3], self.values.get(node.input[3]))
            sizes = np.asarray(v, dtype=np.int64)
        a = self.download(x)
        if sizes is not None:
            Ho, Wo = int(sizes[2]), int(sizes[3])
        elif scales is not None:
            Ho = int(round(a.shape[2] * float(scales[2])))
            Wo = int(round(a.shape[3] * float(scales[3])))
        else:
            raise UnsupportedNode("Resize %s: no scales/sizes" % node.name)
        sy = np.arange(Ho, dtype=np.float64)
        sx = np.arange(Wo, dtype=np.float64)
        if coord == "asymmetric":
            iy = np.floor(sy * (a.shape[2] / Ho)).astype(np.int64)
            ix = np.floor(sx * (a.shape[3] / Wo)).astype(np.int64)
        else:
            iy = np.floor((sy + 0.5) * (a.shape[2] / Ho) - 0.5).astype(np.int64)
            ix = np.floor((sx + 0.5) * (a.shape[3] / Wo) - 0.5).astype(np.int64)
        iy = np.clip(iy, 0, a.shape[2] - 1)
        ix = np.clip(ix, 0, a.shape[3] - 1)
        y = a[:, :, iy][:, :, :, ix]
        return self.upload(np.ascontiguousarray(y))

    def _matmul(self, node):
        a = self._float(node.input[0])
        b = self._float(node.input[1])
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise UnsupportedNode("MatMul %s: only 2-D (got %s x %s)"
                                  % (node.name, a.shape, b.shape))
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise UnsupportedNode("MatMul %s: K mismatch %s x %s"
                                  % (node.name, a.shape, b.shape))
        out = self._alloc(M * N)
        self.eng.matmul(a.buf, b.buf, out, M, N, K)
        return Tensor(out, (M, N))

    def _gemm(self, node):
        a = self._float(node.input[0])
        b = self._float(node.input[1])
        alpha = float(_attr(node, "alpha", 1.0))
        beta = float(_attr(node, "beta", 1.0))
        ta = int(_attr(node, "transA", 0))
        tb = int(_attr(node, "transB", 0))
        if ta != 0 or alpha != 1.0 or beta not in (0.0, 1.0):
            raise UnsupportedNode("Gemm %s: transA=%d alpha=%g beta=%g"
                                  % (node.name, ta, alpha, beta))
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise UnsupportedNode("Gemm %s: only 2-D" % node.name)
        M, K = a.shape
        if tb:
            N, K2 = b.shape
            ba = self.download(b).T.copy()
        else:
            K2, N = b.shape
            ba = self.download(b)
        if K != K2:
            raise UnsupportedNode("Gemm %s: K mismatch" % node.name)
        bb = self.upload(np.ascontiguousarray(ba))
        out = self._alloc(M * N)
        self.eng.matmul(a.buf, bb.buf, out, M, N, K)
        t = Tensor(out, (M, N))
        if beta == 1.0 and len(node.input) > 2 and node.input[2]:
            c = self._float(node.input[2])
            ca = self.download(c)
            if ca.size == N:
                ca = np.broadcast_to(ca.reshape(1, N), (M, N))
            elif ca.shape != (M, N):
                raise UnsupportedNode("Gemm %s: C shape %s" % (node.name, ca.shape))
            cb = self.upload(np.ascontiguousarray(ca))
            o2 = self._alloc(M * N)
            self.eng.elementwise(vc4infer.OP_ADD, t.buf, o2, M * N, b=cb.buf)
            t = Tensor(o2, (M, N))
        return t

    def _softmax(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", -1))
        if axis not in (-1, len(x.shape) - 1):
            raise UnsupportedNode("Softmax %s: axis %d (last only)"
                                  % (node.name, axis))
        c = x.shape[-1]
        if c > 16:
            raise UnsupportedNode("Softmax %s: %d classes (max 16)"
                                  % (node.name, c))
        a = self.download(x).reshape(-1, c)
        rows = a.shape[0]
        rp = (rows + 7) & ~7
        if rp != rows:
            a = np.concatenate([a, np.zeros((rp - rows, c), np.float32)])
        zb = self.upload(a)
        out = self._alloc(rp * c)
        self.eng.softmax_fwd(zb.buf, out, rp, c)
        y = self.download(Tensor(out, (rp, c)))[:rows]
        return self.upload(y.reshape(x.shape))

    def _global_avg_pool(self, node):
        x = self._float(node.input[0])
        if len(x.shape) != 4:
            raise UnsupportedNode("GlobalAveragePool %s: expected NCHW"
                                  % node.name)
        N, C, H, W = x.shape
        if N != 1:
            raise UnsupportedNode("GlobalAveragePool %s: batch != 1" % node.name)
        HW = H * W
        b = np.full((HW, 1), 1.0 / HW, np.float32)
        bb = self.upload(b)
        out = self._alloc(C)
        self.eng.matmul(x.buf, bb.buf, out, C, 1, HW)
        return Tensor(out, (1, C, 1, 1))

    def _reshape(self, node):
        x = self._float(node.input[0])
        target = self.host.get(node.input[1], self.values.get(node.input[1]))
        target = np.asarray(target, dtype=np.int64).tolist()
        out = []
        for i, d in enumerate(target):
            if d == 0:
                out.append(x.shape[i])
            elif d == -1:
                out.append(-1)
            else:
                out.append(int(d))
        if out.count(-1) > 1:
            raise UnsupportedNode("Reshape %s: two -1 dims" % node.name)
        if -1 in out:
            k = out.index(-1)
            known = int(np.prod([d for i, d in enumerate(out) if i != k]))
            if known == 0 or x.size % known:
                raise UnsupportedNode("Reshape %s: %s -> %s" % (node.name, x.shape, out))
            out[k] = x.size // known
        if int(np.prod(out)) != x.size:
            raise UnsupportedNode("Reshape %s: %s -> %s" % (node.name, x.shape, out))
        return Tensor(x.buf, out)

    def _flatten(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", 1))
        if axis < 0:
            axis += len(x.shape)
        before = int(np.prod(x.shape[:axis])) if axis else 1
        after = int(np.prod(x.shape[axis:]))
        return Tensor(x.buf, (before, after))

    def _identity(self, node):
        return self._float(node.input[0])

    def _const_node(self, node):
        v = _attr(node, "value")
        if v is None:
            raise UnsupportedNode("Constant %s: no value" % node.name)
        return self._constant(node.output[0], np.asarray(v))

    # ---------------------------------------------------------------- execute
    def _dispatch(self, node):
        op = node.op_type
        fn = {
            "Conv": self._conv,
            "Relu": self._relu,
            "Sigmoid": self._sigmoid,
            "Add": lambda n: self._binary(n, vc4infer.OP_ADD),
            "Mul": lambda n: self._binary(n, vc4infer.OP_MUL),
            "MaxPool": self._maxpool,
            "Concat": self._concat,
            "Resize": self._resize,
            "MatMul": self._matmul,
            "Gemm": self._gemm,
            "Softmax": self._softmax,
            "GlobalAveragePool": self._global_avg_pool,
            "Reshape": self._reshape,
            "Flatten": self._flatten,
            "Identity": self._identity,
            "Constant": self._const_node,
        }.get(op)
        if fn is None:
            raise UnsupportedNode("node %r: unsupported op %s"
                                  % (node.name or "?", op))
        try:
            return fn(node)
        except vc4infer.Unsupported as e:
            raise UnsupportedNode("node %r (%s): %s" % (node.name, op, e))
        except vc4infer.Error as e:
            raise UnsupportedNode("node %r (%s): engine error: %s"
                                  % (node.name, op, e))

    def run(self, feeds, profile=False):
        self.values = {k: v for k, v in self.values.items()
                       if k in [i.name for i in self.graph.initializer]}
        for name, arr in feeds.items():
            self.values[name] = self.upload(arr)
        self.profile = []
        for node in self.graph.node:
            t0 = time.perf_counter()
            s0 = self.eng.stats
            out = self._dispatch(node)
            s1 = self.eng.stats
            dt = (time.perf_counter() - t0) * 1e3
            if profile:
                gpu = (s1.pack_ns - s0.pack_ns + s1.kernel_ns - s0.kernel_ns) / 1e6
                self.profile.append({
                    "name": node.name or node.op_type,
                    "op": node.op_type,
                    "wall_ms": dt,
                    "gpu_ms": gpu,
                    "launches": s1.launches - s0.launches,
                    "flops": s1.flops - s0.flops,
                })
            for i, name in enumerate(node.output):
                if name:
                    self.values[name] = out if len(node.output) == 1 else out
        outs = []
        for o in self.graph.output:
            v = self.values[o.name]
            if not isinstance(v, Tensor):
                raise UnsupportedNode("output %s is not a tensor" % o.name)
            outs.append(self.download(v))
        return outs


def buf_of(t):
    return t.buf


def load_model(path):
    return onnx.load(path)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--input", help=".npy input (else random)")
    ap.add_argument("--out", help="write the first output to this .npy")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cache", default=os.path.join(ENGINE, "kcache"))
    args = ap.parse_args(argv)

    eng = vc4infer.Engine(cache_dir=args.cache)
    try:
        r = Runner(args.model, eng)
        if args.input:
            arr = np.load(args.input).astype(np.float32)
        else:
            name, dims = r.float_inputs[0]
            arr = np.random.default_rng(0).standard_normal(
                [d if d > 0 else 1 for d in dims]).astype(np.float32)
        feed = {r.float_inputs[0][0]: arr}
        for rep in range(args.repeat):
            t0 = time.perf_counter()
            outs = r.run(feed, profile=args.profile)
            dt = (time.perf_counter() - t0) * 1e3
            print("run %d: %.3f ms  output %s" % (rep, dt, outs[0].shape))
        if args.profile:
            print("%-28s %-18s %9s %9s %5s" %
                  ("node", "op", "wall ms", "gpu ms", "launch"))
            for p in r.profile:
                print("%-28s %-18s %9.3f %9.3f %5d" %
                      (p["name"][:28], p["op"], p["wall_ms"], p["gpu_ms"],
                       p["launches"]))
        if args.out:
            np.save(args.out, outs[0])
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
