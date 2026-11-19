"""A numpy float32 ONNX evaluator for the ops YOLOv8n uses.

Used as an independent CPU reference and as a per-node oracle when debugging.
"""
import numpy as np
import onnx
from onnx import numpy_helper

ATTR = {}


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


def conv(x, w, b, pads, strides, group, dil=(1, 1)):
    n, c, h, wd = x.shape
    cout, cg, kh, kw = w.shape
    ph, pw = pads[0], pads[1]
    sh, sw = strides
    dh, dw = int(dil[0]), int(dil[1])
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (ph, pads[2]),
                                       (pw, pads[3])))
    ho = (h + ph + pads[2] - (kh - 1) * dh - 1) // sh + 1
    wo = (wd + pw + pads[3] - (kw - 1) * dw - 1) // sw + 1
    out = np.zeros((n, cout, ho, wo))
    og = cout // group
    for g in range(group):
        xg = xp[:, g * cg:(g + 1) * cg]
        wg = w[g * og:(g + 1) * og].astype(np.float64)
        acc = np.zeros((n, og, ho, wo))
        for ky in range(kh):
            for kx in range(kw):
                y0 = ky * dh
                x0 = kx * dw
                patch = xg[:, :, y0:y0 + sh * ho:sh, x0:x0 + sw * wo:sw]
                acc += np.einsum("nchw,oc->nohw", patch, wg[:, :, ky, kx])
        out[:, g * og:(g + 1) * og] = acc
    if b is not None:
        out += b.astype(np.float64).reshape(1, -1, 1, 1)
    return out


def maxpool(x, k, s, pads):
    n, c, h, w = x.shape
    xp = np.pad(x.astype(np.float64),
                ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
                constant_values=-np.inf)
    ho = (h + pads[0] + pads[2] - k) // s + 1
    wo = (w + pads[1] + pads[3] - k) // s + 1
    out = np.full((n, c, ho, wo), -np.inf)
    for ky in range(k):
        for kx in range(k):
            out = np.maximum(out, xp[:, :, ky:ky + s * ho:s, kx:kx + s * wo:s])
    return out


def softmax(x, axis):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def eval_node(node, ins, vals=None):
    """Evaluate one ONNX node given its input arrays."""
    op = node.op_type
    a = _attr
    if op == "Constant":
        out = np.asarray(a(node, "value"))
    elif op == "Conv":
        out = conv(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                   a(node, "pads", [0, 0, 0, 0]),
                   a(node, "strides", [1, 1]), int(a(node, "group", 1)),
                   a(node, "dilations", [1, 1]))
    elif op == "Relu":
        out = np.maximum(ins[0], 0)
    elif op == "Sigmoid":
        out = 1.0 / (1.0 + np.exp(-ins[0].astype(np.float64)))
    elif op == "Mul":
        out = ins[0] * ins[1]
    elif op == "Add":
        out = ins[0] + ins[1]
    elif op == "Sub":
        out = ins[0] - ins[1]
    elif op == "Div":
        out = ins[0] / ins[1]
    elif op == "MaxPool":
        out = maxpool(ins[0], int(a(node, "kernel_shape")[0]),
                      int(a(node, "strides", [1, 1])[0]),
                      a(node, "pads", [0, 0, 0, 0]))
    elif op == "Concat":
        out = np.concatenate([i for i in ins], axis=int(a(node, "axis", 0)))
    elif op == "Split":
        axis = int(a(node, "axis", 0))
        sizes = None
        if len(ins) > 1 and ins[1] is not None:
            sizes = [int(s) for s in np.asarray(ins[1]).ravel()]
        if sizes is None:
            k = len(node.output)
            sizes = [ins[0].shape[axis] // k] * k
        out = np.split(ins[0], np.cumsum(sizes)[:-1], axis=axis)
    elif op == "Reshape":
        out = ins[0].reshape([int(v) for v in np.asarray(ins[1]).ravel()])
    elif op == "Transpose":
        out = np.transpose(ins[0], a(node, "perm"))
    elif op == "Softmax":
        out = softmax(ins[0].astype(np.float64), int(a(node, "axis", -1)))
    elif op == "Slice":
        st = [int(v) for v in np.asarray(ins[1]).ravel()]
        en = [int(v) for v in np.asarray(ins[2]).ravel()]
        ax = [int(v) for v in np.asarray(ins[3]).ravel()] if len(ins) > 3 and ins[3] is not None else list(range(len(st)))
        sp = [int(v) for v in np.asarray(ins[4]).ravel()] if len(ins) > 4 and ins[4] is not None else [1] * len(st)
        sl = [slice(None)] * ins[0].ndim
        for s0, e0, axx, spp in zip(st, en, ax, sp):
            sl[axx] = slice(s0, e0, spp)
        out = ins[0][tuple(sl)]
    elif op == "Shape":
        out = np.asarray(ins[0].shape, np.int64)
    elif op == "Gather":
        out = np.take(ins[0], np.asarray(ins[1], np.int64),
                      axis=int(a(node, "axis", 0)))
    elif op == "Resize":
        x0 = ins[0].astype(np.float64)
        sizes = ins[3] if len(ins) > 3 else None
        scales = ins[2] if len(ins) > 2 else None
        if sizes is not None:
            Ho, Wo = int(sizes[2]), int(sizes[3])
        else:
            Ho = int(round(x0.shape[2] * float(scales[2])))
            Wo = int(round(x0.shape[3] * float(scales[3])))
        coord = a(node, "coordinate_transformation_mode", "half_pixel")
        mode = a(node, "mode", "nearest")
        h, w = x0.shape[2], x0.shape[3]

        def ax(n_in, n_out):
            s = np.arange(n_out, dtype=np.float64)
            src = (s * (n_in / n_out) if coord == "asymmetric"
                   else (s + 0.5) * (n_in / n_out) - 0.5)
            i0u = np.floor(src).astype(np.int64)
            if mode == "nearest":
                idx = (np.floor(s * (n_in / n_out)).astype(np.int64)
                       if coord == "asymmetric"
                       else np.floor((s + 0.5) * (n_in / n_out) - 0.5)
                       .astype(np.int64))
                return np.clip(idx, 0, n_in - 1), None, None
            return (np.clip(i0u, 0, n_in - 1), np.clip(i0u + 1, 0, n_in - 1),
                    src - i0u)

        iy0, iy1, wy = ax(h, Ho)
        ix0, ix1, wx = ax(w, Wo)
        if mode == "nearest":
            out = x0[:, :, iy0][:, :, :, ix0]
        else:
            a00 = x0[:, :, iy0][:, :, :, ix0]
            a01 = x0[:, :, iy0][:, :, :, ix1]
            a10 = x0[:, :, iy1][:, :, :, ix0]
            a11 = x0[:, :, iy1][:, :, :, ix1]
            wy0, wy1 = (1.0 - wy)[:, None], wy[:, None]
            wx0, wx1 = (1.0 - wx)[None, :], wx[None, :]
            out = (wy0 * wx0 * a00 + wy0 * wx1 * a01 +
                   wy1 * wx0 * a10 + wy1 * wx1 * a11)
    elif op in ("Identity",):
        out = ins[0]
    elif op == "Cast":
        out = ins[0].astype(np.float32 if int(a(node, "to")) == 1 else np.int64)
    elif op in ("Squeeze", "Unsqueeze", "Flatten"):
        out = ins[0]
        if op == "Flatten":
            out = ins[0].reshape(ins[0].shape[0], -1)
    else:
        raise RuntimeError("ref: unsupported op %s" % op)
    return out


def run_model(path, feeds, want=None):
    m = onnx.load(path)
    g = m.graph
    vals = {}
    for init in g.initializer:
        a = numpy_helper.to_array(init)
        vals[init.name] = a.astype(np.float32) if a.dtype == np.float32 else a
    for k, v in feeds.items():
        vals[k] = np.asarray(v, np.float32)
    trace = {}
    for node in g.node:
        op = node.op_type
        ins = [vals.get(i) for i in node.input]
        a = _attr
        if op == "Constant":
            out = np.asarray(a(node, "value"))
        elif op == "Conv":
            out = conv(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                       a(node, "pads", [0, 0, 0, 0]),
                       a(node, "strides", [1, 1]), int(a(node, "group", 1)),
                       a(node, "dilations", [1, 1]))
        elif op == "Relu":
            out = np.maximum(ins[0], 0)
        elif op == "Sigmoid":
            out = 1.0 / (1.0 + np.exp(-ins[0].astype(np.float64)))
        elif op == "Mul":
            out = ins[0] * ins[1]
        elif op == "Add":
            out = ins[0] + ins[1]
        elif op == "Sub":
            out = ins[0] - ins[1]
        elif op == "Div":
            out = ins[0] / ins[1]
        elif op == "MaxPool":
            out = maxpool(ins[0], int(a(node, "kernel_shape")[0]),
                          int(a(node, "strides", [1, 1])[0]),
                          a(node, "pads", [0, 0, 0, 0]))
        elif op == "Concat":
            out = np.concatenate([i for i in ins], axis=int(a(node, "axis", 0)))
        elif op == "Split":
            axis = int(a(node, "axis", 0))
            sizes = None
            if len(ins) > 1 and ins[1] is not None:
                sizes = [int(s) for s in np.asarray(ins[1]).ravel()]
            if sizes is None:
                k = len(node.output)
                sizes = [ins[0].shape[axis] // k] * k
            out = np.split(ins[0], np.cumsum(sizes)[:-1], axis=axis)
        elif op == "Reshape":
            out = ins[0].reshape([int(v) for v in np.asarray(ins[1]).ravel()])
        elif op == "Transpose":
            out = np.transpose(ins[0], a(node, "perm"))
        elif op == "Softmax":
            out = softmax(ins[0].astype(np.float64), int(a(node, "axis", -1)))
        elif op == "Slice":
            st = [int(v) for v in np.asarray(ins[1]).ravel()]
            en = [int(v) for v in np.asarray(ins[2]).ravel()]
            ax = [int(v) for v in np.asarray(ins[3]).ravel()] if len(ins) > 3 and ins[3] is not None else list(range(len(st)))
            sp = [int(v) for v in np.asarray(ins[4]).ravel()] if len(ins) > 4 and ins[4] is not None else [1] * len(st)
            sl = [slice(None)] * ins[0].ndim
            for s, e, axx, spp in zip(st, en, ax, sp):
                sl[axx] = slice(s, e, spp)
            out = ins[0][tuple(sl)]
        elif op == "Shape":
            out = np.asarray(ins[0].shape, np.int64)
        elif op == "Gather":
            out = np.take(ins[0], np.asarray(ins[1], np.int64),
                          axis=int(a(node, "axis", 0)))
        elif op == "Resize":
            x0 = ins[0].astype(np.float64)
            sizes = ins[3] if len(ins) > 3 else None
            scales = ins[2] if len(ins) > 2 else None
            if sizes is not None:
                Ho, Wo = int(sizes[2]), int(sizes[3])
            else:
                Ho = int(round(x0.shape[2] * float(scales[2])))
                Wo = int(round(x0.shape[3] * float(scales[3])))
            coord = a(node, "coordinate_transformation_mode", "half_pixel")
            mode = a(node, "mode", "nearest")
            h, w = x0.shape[2], x0.shape[3]

            def ax(n_in, n_out):
                s = np.arange(n_out, dtype=np.float64)
                src = (s * (n_in / n_out) if coord == "asymmetric"
                       else (s + 0.5) * (n_in / n_out) - 0.5)
                i0u = np.floor(src).astype(np.int64)
                if mode == "nearest":
                    idx = (np.floor(s * (n_in / n_out)).astype(np.int64)
                           if coord == "asymmetric"
                           else np.floor((s + 0.5) * (n_in / n_out) - 0.5)
                           .astype(np.int64))
                    return np.clip(idx, 0, n_in - 1), None, None
                return (np.clip(i0u, 0, n_in - 1),
                        np.clip(i0u + 1, 0, n_in - 1), src - i0u)

            iy0, iy1, wy = ax(h, Ho)
            ix0, ix1, wx = ax(w, Wo)
            if mode == "nearest":
                out = x0[:, :, iy0][:, :, :, ix0]
            else:
                a00 = x0[:, :, iy0][:, :, :, ix0]
                a01 = x0[:, :, iy0][:, :, :, ix1]
                a10 = x0[:, :, iy1][:, :, :, ix0]
                a11 = x0[:, :, iy1][:, :, :, ix1]
                wy0, wy1 = (1.0 - wy)[:, None], wy[:, None]
                wx0, wx1 = (1.0 - wx)[None, :], wx[None, :]
                out = (wy0 * wx0 * a00 + wy0 * wx1 * a01 +
                       wy1 * wx0 * a10 + wy1 * wx1 * a11)
        elif op in ("Identity",):
            out = ins[0]
        elif op == "Cast":
            out = ins[0].astype(np.float32 if int(a(node, "to")) == 1 else np.int64)
        elif op in ("Squeeze", "Unsqueeze", "Flatten"):
            out = ins[0]
            if op == "Flatten":
                out = ins[0].reshape(ins[0].shape[0], -1)
        else:
            raise RuntimeError("ref: unsupported op %s" % op)
        outs = out if isinstance(out, list) else [out]
        for i, name in enumerate(node.output):
            if name:
                v = np.asarray(outs[i], dtype=np.float32) \
                    if np.asarray(outs[i]).dtype.kind == "f" else outs[i]
                vals[name] = v
                if want is not None and name in want:
                    trace[name] = v
    res = [vals[o.name] for o in g.output]
    return (res, trace) if want is not None else res
