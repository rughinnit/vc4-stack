"""aten implementations for the vc4 subset backend.

Every entry point here is registered at ``PrivateUse1`` and is reached only
when torch has decided the op runs on vc4.  The compute is on the QPU; the
Python/ARM side only

  * reshapes/views (metadata), and
  * moves bytes (channel padding, phase images, weight permutes, concat,
    nearest resampling, broadcast expansion, index gathers, compaction) and
    materialises constants,

all through ``torch.ops.torch_vc4.*`` (csrc/torch_vc4.cpp).  There is no CPU
maths and no CPU fallback: a shape or attribute the engine cannot do raises.

Batches run through the engine's batched NHWC conv path; pools/resize/cat
use one launch per image.  Reductions over arbitrary dims are GEMMs
(sum/mean) or a relu/sign max tree (amax/amin) because the engine's generic
``vc4_reduce`` blob is not used by this backend's paths (see COMPLETENESS.md).
Gather/index_select are host word gathers (pure copies); bilinear resize is
the engine's four-gather lowering (`vc4_resize_bilinear_mode`).
"""

import math

import torch

T = torch.ops.torch_vc4

_LIBS = []


def _keep(lib):
    _LIBS.append(lib)
    return lib


__all__ = [
    "register",
    "conv2d",
    "conv2d_backward",
    "channel_sums",
    "fold_bn_into_conv",
]


# --------------------------------------------------------------------- checks

def f32(t, what):
    if t.device.type != "vc4":
        raise RuntimeError(
            "torch_vc4: %s: expected a vc4 tensor, got %s" % (what, t.device))
    if t.dtype != torch.float32:
        raise RuntimeError("torch_vc4: %s: only float32 is supported, got %s"
                           % (what, t.dtype))
    if not t.is_contiguous():
        raise RuntimeError(
            "torch_vc4: %s: needs a contiguous tensor; this backend has no "
            "strided kernels (slicing/permute give metadata-only views that "
            "compute ops reject)" % what)
    return t


def _align16(v):
    return (v + 15) & ~15


def _pair(v):
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise RuntimeError("torch_vc4: expected a pair, got %r" % (v,))
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _norm_dim(d, nd, what="op"):
    d = int(d)
    if d < 0:
        d += nd
    if d < 0 or d >= nd:
        raise RuntimeError("torch_vc4: %s: dim %d is out of range for a %d-D "
                           "tensor" % (what, d, nd))
    return d


def _norm_dims(dims, nd, what="op"):
    if dims is None:
        return list(range(nd))
    if isinstance(dims, int):
        dims = [dims]
    out = []
    for d in dims:
        d = _norm_dim(d, nd, what)
        if d not in out:
            out.append(d)
    return out


# ----------------------------------------------------------- layout helpers

def _permute_dims(t, order):
    """Metadata permutation + materialise (host word gather, no arithmetic)."""
    order = [int(i) for i in order]
    if order == list(range(t.dim())):
        return t
    sizes = [t.shape[i] for i in order]
    strides = [t.stride(i) for i in order]
    v = T.as_strided_alias(t, sizes, strides, t.storage_offset())
    return T.materialize(v)


def _broadcast_view(shape, tshape, tstrides):
    """Plan a right-aligned broadcast: ``(sizes, strides)`` or ``None``.

    Pure shape arithmetic, split out so the broadcast rule is testable
    without a device.  A size-1 axis takes a stride of 0; the caller then
    materialises that view, which is a byte replication, not arithmetic.
    """
    shape = tuple(int(s) for s in shape)
    tshape = tuple(int(s) for s in tshape)
    if tshape == shape:
        return None
    lead = len(shape) - len(tshape)
    if lead < 0:
        raise RuntimeError("torch_vc4: cannot broadcast %s to %s"
                           % (tshape, shape))
    sizes = (1,) * lead + tshape
    strides = (0,) * lead + tuple(int(s) for s in tstrides)
    ns, nst = [], []
    for s, o, st in zip(sizes, shape, strides):
        if s == o:
            ns.append(o)
            nst.append(st)
        elif s == 1:
            ns.append(o)
            nst.append(0)
        else:
            raise RuntimeError(
                "torch_vc4: cannot broadcast %s to %s (dim %d: %d vs %d)"
                % (tshape, shape, len(ns), s, o))
    return ns, nst


def _expand(t, shape):
    """Right-aligned broadcast by materialising a 0-stride view.

    The arithmetic on the result is on the QPU; this is a byte replication
    (the same class of work as contiguous()), not a compute fallback.
    """
    shape = tuple(int(s) for s in shape)
    if tuple(t.shape) == shape:
        return t
    f32(t, "broadcast")
    plan = _broadcast_view(shape, t.shape, t.stride())
    if plan is None:
        return t
    ns, nst = plan
    v = T.as_strided_alias(t, ns, nst, t.storage_offset())
    return T.materialize(v)


def _slice_last(x2, start, step, length):
    """x2[:, start : start+length*step : step] as a contiguous [outer][len]."""
    outer, red = x2.shape
    if length <= 0:
        raise RuntimeError("torch_vc4: empty slice")
    v = T.as_strided_alias(x2, [outer, length], [red, step],
                           x2.storage_offset() + start)
    return T.materialize(v)


def contiguous(t):
    """A contiguous tensor with the same values (byte copy when needed)."""
    f32(t, "contiguous")
    return t if t.is_contiguous() else T.materialize(t)


# ------------------------------------------------------ reduce machinery

def _ones_like_rows(rows, cols, like):
    return torch.ones((rows, cols), dtype=torch.float32, device=like.device)


def _sum_last(x2):
    """[outer][red] -> [outer], a GEMM with a ones vector."""
    outer, red = x2.shape
    if red == 0:
        raise RuntimeError("torch_vc4: reduction over an empty dimension")
    if outer == 1:
        return T.sum_all(x2.view(-1)).view(1)
    ones = _ones_like_rows(red, 1, x2)
    return torch.mm(x2, ones).view(outer)


def _select_max(a, b):
    """Elementwise max(a,b) = b + relu(a-b) (one rounding; ~1 ulp)."""
    d = T.add(a, T.mul_scalar(b, -1.0))
    return T.add(b, torch.relu(d))


def _max_last(x2):
    """[outer][red] -> [outer] max, a balanced relu tree on the QPU.

    Odd levels carry the unpaired last column (and its negation for min) to
    the end; every other comparison is vectorised over the whole row.
    """
    outer, red = x2.shape
    leftovers = []
    cur = x2
    while cur.shape[1] > 1:
        n = cur.shape[1]
        if n % 2:
            leftovers.append(_slice_last(cur, n - 1, 1, 1))
            cur = _slice_last(cur, 0, 1, n - 1)
        h = cur.shape[1] // 2
        a = _slice_last(cur, 0, 2, h)
        b = _slice_last(cur, 1, 2, h)
        cur = _select_max(a, b)
    for t in leftovers:
        cur = _select_max(cur, t)
    return cur.view(outer)


def _min_last(x2):
    return T.mul_scalar(_max_last(T.mul_scalar(x2, -1.0)), -1.0)


def _reduce_plan(shape, dims, keepdim):
    """Plan a dim reduction: ``(order, kept, red, outer, out_shape)``.

    Pure shape arithmetic.  ``order`` puts the reduced axes last so the
    following gather materialises ``[outer][red]``; ``red`` is the product of
    the reduced sizes and ``out_shape`` is the aten result shape.
    """
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    dims = _norm_dims(dims, nd)
    keep = [i for i in range(nd) if i not in dims]
    red = 1
    for i in dims:
        red *= shape[i]
    total = 1
    for s in shape:
        total *= s
    outer = total // red if red else 0
    if keepdim:
        out_shape = tuple(1 if i in dims else shape[i] for i in range(nd))
    else:
        out_shape = tuple(shape[i] for i in keep)
    return keep + dims, tuple(shape[i] for i in keep), red, outer, out_shape


def _dims_reduce(t, dims, keepdim, op):
    """Reduce ``t`` over ``dims`` (sum/mean/max/min) on the QPU.

    The tensor is permuted so the reduced dims are last (a byte gather), then
    flattened to [outer][red]: sums are one GEMM, max/min a relu tree.
    """
    dims = _norm_dims(dims, t.dim())
    order, kept_shape, red, outer, out_shape = _reduce_plan(
        t.shape, dims, keepdim)
    xp = _permute_dims(t, order)
    x2 = xp.view(outer, red)
    if op in ("sum", "mean"):
        r = _sum_last(x2)
        if op == "mean":
            r = T.mul_scalar(r, 1.0 / red)
    elif op == "max":
        r = _max_last(x2)
    elif op == "min":
        r = _min_last(x2)
    else:  # pragma: no cover
        raise RuntimeError("torch_vc4: unknown reduction %r" % (op,))
    if keepdim:
        return r.view(list(out_shape))
    return r.view(kept_shape)


def _expand_reduced(g, shape, dims):
    """Broadcast a reduction gradient back over the reduced dims (0 strides)."""
    nd = len(shape)
    dims = _norm_dims(dims, nd)
    keep = [i for i in range(nd) if i not in dims]
    if tuple(g.shape) != tuple(shape[i] for i in keep):
        g = g.view([shape[i] for i in keep])
    strides = [0] * nd
    gs = list(g.stride())
    for k, i in enumerate(keep):
        strides[i] = gs[k]
    v = T.as_strided_alias(g, list(shape), strides, g.storage_offset())
    return T.materialize(v)


def _sum_to(g, shape):
    """Sum a broadcast gradient back down to ``shape``."""
    if tuple(g.shape) == tuple(shape):
        return g
    f32(g, "broadcast-backward")
    lead = g.dim() - len(shape)
    if lead > 0:
        g = _dims_reduce(g, list(range(lead)), False, "sum")
    dims = [i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1]
    if dims:
        g = _dims_reduce(g, dims, True, "sum")
    return g.reshape(shape)


# ------------------------------------------------------------------- conv2d

def phase_q(pad):
    """Phase-conv padding for the stride-2 decomposition (docs/ONNX.md §6).

    For a stride-2, 3x3 conv with padding ``pad``, tap ``k`` reads input row
    ``2i + k - pad``, i.e. parity ``a = (k - pad) mod 2`` at phase offset
    ``d = (k - pad - a)/2``.  The phase conv therefore needs padding
    ``q = max(0, -min(d))`` so that every read lands inside the phase image.
    """
    ds = []
    for k in (0, 1, 2):
        u = k - pad
        a = u % 2
        ds.append((u - a) // 2)
    return max(0, -min(ds))


def _conv_geom(x, w, b, stride, pad, groups):
    f32(x, "conv2d(input)")
    f32(w, "conv2d(weight)")
    if x.dim() != 4 or w.dim() != 4:
        raise RuntimeError("torch_vc4: conv2d supports 4-D NCHW tensors only, "
                           "got %d-D input and %d-D weight"
                           % (x.dim(), w.dim()))
    N, C, H, W = x.shape
    Cout, Cin, kh, kw = w.shape
    if N < 1:
        raise RuntimeError("torch_vc4: conv2d needs a non-empty batch")
    if groups < 1:
        raise RuntimeError("torch_vc4: conv2d groups must be >= 1, got %d"
                           % groups)
    depthwise = False
    if groups == 1:
        if Cin != C:
            raise RuntimeError("torch_vc4: conv2d: weight has %d input "
                               "channels, input has %d" % (Cin, C))
    elif groups == C:
        if Cin != 1:
            raise RuntimeError(
                "torch_vc4: conv2d with groups == Cin needs a [C][1][k][k] "
                "weight (multiplier 1); got Cin=%d" % Cin)
        if Cout != C:
            raise RuntimeError(
                "torch_vc4: depthwise conv2d needs Cout == Cin, got %d -> %d"
                % (C, Cout))
        depthwise = True
    else:
        # grouped: the weight is [Cout][Cin/groups][k][k]
        if Cin * groups != C:
            raise RuntimeError(
                "torch_vc4: conv2d: weight has %d channels per group and "
                "groups=%d, so it covers %d input channels; the input has %d "
                "(the weight must be [Cout][Cin/groups][k][k])"
                % (Cin, groups, Cin * groups, C))
        if groups > C or C % groups or Cout % groups:
            raise RuntimeError(
                "torch_vc4: conv2d: groups=%d does not divide Cin=%d and "
                "Cout=%d (the weight is [Cin][Cin/groups][k][k] and the bias "
                "is [Cout])" % (groups, C, Cout))
    if kh != kw or kh not in (1, 3):
        raise RuntimeError("torch_vc4: conv2d supports 1x1 and 3x3 kernels, "
                           "got %dx%d" % (kh, kw))
    if stride not in (1, 2):
        raise RuntimeError("torch_vc4: conv2d supports stride 1 and 2, got %d"
                           % stride)
    if kh == 1 and pad != 0:
        raise RuntimeError("torch_vc4: conv2d 1x1 takes padding 0")
    if pad < 0 or pad > kh // 2:
        raise RuntimeError("torch_vc4: conv2d padding %d is outside PyTorch's "
                           "range for a %dx%d kernel" % (pad, kh, kh))
    if b is not None:
        f32(b, "conv2d(bias)")
        if b.dim() != 1 or b.numel() != Cout:
            raise RuntimeError("torch_vc4: conv2d bias must be [%d], got %s"
                               % (Cout, tuple(b.shape)))
    return N, C, H, W, Cout, kh, stride, pad, depthwise


def _padded(w, b, C, Cout, depthwise):
    """Weight/bias in the engine's padded layout (byte copies only)."""
    cpad, opad = _align16(C), _align16(Cout)
    if depthwise:
        wd = T.dense_dw_w(w, cpad, opad)
    elif cpad != C or opad != Cout:
        wd = T.pad_w(w, cpad, opad)
    else:
        wd = w
    bp = b
    if b is not None and opad != Cout:
        bp = T.pad_ch(b, opad)
    return wd, bp, cpad, opad


def _conv2d_nhwc(x, w, b, stride, pad, depthwise, native_dw):
    """Whole-batch conv through the engine's NHWC kernels (stride 1/2).

    depthwise + 3x3/s1/p1 uses the engine's native depthwise kernel; the other
    depthwise cases use its exact dense block-diagonal decomposition (same
    MACs, no host loop).
    """
    N, C, H, W, Cout, k, stride, pad, _ = _conv_geom(
        x, w, b, stride, pad, x.shape[1] if depthwise else 1)
    cpad, opad = _align16(C), _align16(Cout)
    xp = T.nhwc_pack(x, cpad)                     # [N][H][W][cpad]
    if native_dw:
        panel = T.depthwise_panel(w, cpad)
        y = T.depthwise_nhwc(xp, panel, N, cpad, H, W)
    else:
        if depthwise:
            wd = T.dense_dw_w(w, cpad, opad)
        elif cpad != C or opad != Cout:
            wd = T.pad_w(w, cpad, opad)
        else:
            wd = w
        bp = T.pad_ch(b, opad) if (b is not None and opad != Cout) else b
        panel = T.conv_nhwc_pack_w(wd, bp, cpad, opad)
        y = T.conv_nhwc(xp, panel, N, cpad, H, W, opad, k, k, stride, pad, 1)
    ho = y.shape[1]
    wo = (W + 2 * pad - k) // stride + 1
    out = T.nhwc_unpack(y, N, Cout, ho, wo)
    if native_dw and b is not None:
        # vc4_depthwise_nhwc takes no bias; add it on the QPU (broadcast)
        out = _binary(out, b.view(1, Cout, 1, 1), "add")
    return out


def _conv2d_nchw(x, w, b, stride, pad, depthwise):
    """One image through the engine's verified NCHW conv / phase path."""
    N, C, H, W, Cout, k, stride, pad, _ = _conv_geom(
        x, w, b, stride, pad, x.shape[1] if depthwise else 1)
    wd, bp, cpad, opad = _padded(w, b, C, Cout, depthwise)
    x3 = x.view(C, H, W)
    xp = T.pad_ch(x3, cpad) if cpad != C else x3
    if stride == 1:
        y3 = T.conv_raw(xp, wd, bp, pad)
    elif k == 1:
        # stride-2 1x1 == 1x1 over the (0,0) parity phase: exact, no waste
        y3 = T.conv_raw(T.subsample2(xp), wd, bp, 0)
    else:
        q = phase_q(pad)
        ho = (H + 2 * pad - 3) // 2 + 1
        wo = (W + 2 * pad - 3) // 2 + 1
        hp, wp = ho + 2 - 2 * q, wo + 2 - 2 * q
        ph = T.phase_split(xp, hp, wp)
        y3 = T.conv_raw(ph, T.phase_w(wd, cpad, opad, pad, q), bp, q)
    if y3.shape[0] != Cout:
        y3 = T.take_ch(y3, 0, Cout)
    return y3.view(1, Cout, y3.shape[1], y3.shape[2])


def _conv2d_grouped(x, w, b, stride, pad, groups):
    """groups > 1 beyond depthwise: the engine's grouped NHWC kernel.

    ``vc4_conv_nhwc_group`` runs one packed-panel GEMM per group and writes
    each group's channel range straight into the single NHWC output (the GEMM
    C row stride is Cout), so nothing is interleaved on the ARM and the MACs
    are the real grouped MACs.  The per-group panel is k-major with one
    16-channel block per tap, which is where the multiple-of-16 requirement
    comes from; an unsupported shape is refused by name, never split.
    """
    N, C, H, W = x.shape
    Cout, Cin, kh, kw = w.shape
    cg, og = C // groups, Cout // groups
    if Cin != cg:
        raise RuntimeError("torch_vc4: grouped conv2d: weight has %d channels "
                           "per group, input has %d over groups=%d"
                           % (Cin, C, groups))
    if cg % 16 or og % 16:
        raise RuntimeError(
            "torch_vc4: grouped conv2d needs Cin/groups and Cout/groups to be "
            "multiples of 16 (a group's weight panel is one 16-channel block "
            "per tap); got Cin/groups=%d and Cout/groups=%d for groups=%d. "
            "Pad the channels, or run this layer on the CPU."
            % (cg, og, groups))
    if stride * C * 4 > 8191:
        raise RuntimeError(
            "torch_vc4: grouped conv2d: the engine reads the padded image at "
            "a VDR pitch of stride*Cin*4 = %d B (Cin=%d, stride=%d), over its "
            "8191 B field; run this layer at a smaller stride or with fewer "
            "input channels" % (stride * C * 4, C, stride))
    xp = T.nhwc_pack(x, C)                        # [N][H][W][C]
    panels = []
    for g in range(groups):
        wg = w[g * og:(g + 1) * og]
        bg = b[g * og:(g + 1) * og] if b is not None else None
        panels.append(T.conv_nhwc_pack_w(wg.contiguous(), bg, cg, og))
    y = T.conv_nhwc_group(xp, panels, N, C, H, W, Cout, groups, kh, kw,
                          stride, pad, 1)
    ho = y.shape[1]
    wo = (W + 2 * pad - kh) // stride + 1
    return T.nhwc_unpack(y, N, Cout, ho, wo)


def conv2d(x, w, b, stride=1, pad=0, groups=1):
    N, C, H, W, Cout, k, stride, pad, depthwise = _conv_geom(
        x, w, b, stride, pad, groups)
    if groups > 1 and not depthwise:
        return _conv2d_grouped(x, w, b, stride, pad, groups)
    # The engine's batched NHWC kernels take stride 1/2 and a whole batch in
    # one launch; the native 3x3 depthwise kernel covers pad-1/stride-1, the
    # other depthwise cases use its exact block-diagonal decomposition.
    # Re-verified against the engine test (test_nhwc 5) and vs the CPU in
    # tests/test_complete.py; see COMPLETENESS.md item 6 for the exact call.
    native_dw = depthwise and k == 3 and stride == 1 and pad == 1
    if N > 1 or stride == 2 or native_dw:
        return _conv2d_nhwc(x, w, b, stride, pad, depthwise, native_dw)
    return _conv2d_nchw(x, w, b, stride, pad, depthwise)


def _conv2d_backward_single(x, w, dy, stride, pad, groups):
    _, C, H, W, Cout, k, stride, pad, depthwise = _conv_geom(
        x, w, None, stride, pad, groups)
    if groups > 1 and not depthwise:
        raise RuntimeError(
            "torch_vc4: grouped conv2d backward (groups=%d, beyond depthwise) "
            "is not implemented: the engine's dX/dW kernels run one dense "
            "GEMM over the whole channel axis and there is no per-group "
            "binder for them; grouped conv is forward only in this version"
            % groups)
    f32(dy, "conv2d_backward(dy)")
    ho = (H + 2 * pad - k) // stride + 1
    wo = (W + 2 * pad - k) // stride + 1
    if tuple(dy.shape) != (1, Cout, ho, wo):
        raise RuntimeError("torch_vc4: conv2d_backward: dy %s does not match "
                           "%s" % (tuple(dy.shape), (1, Cout, ho, wo)))
    wd, _, cpad, opad = _padded(w, None, C, Cout, depthwise)
    x3 = x.view(C, H, W)
    xp = T.pad_ch(x3, cpad) if cpad != C else x3
    dy3 = dy.view(Cout, ho, wo)
    dyp = T.pad_ch(dy3, opad) if opad != Cout else dy3
    if stride == 1:
        dxp, dwp = T.conv_bwd_raw(xp, wd, dyp, pad)
        dx3 = T.take_ch(dxp, 0, C) if cpad != C else dxp
    elif k == 1:
        dxs, dwp = T.conv_bwd_raw(T.subsample2(xp), wd, dyp, 0)
        dx3 = T.scatter2(dxs, H, W)
        if cpad != C:
            dx3 = T.take_ch(dx3, 0, C)
    else:
        q = phase_q(pad)
        hp, wp = ho + 2 - 2 * q, wo + 2 - 2 * q
        ph = T.phase_split(xp, hp, wp)
        dpp, dwp = T.conv_bwd_raw(ph, T.phase_w(wd, cpad, opad, pad, q),
                                  dyp, q)
        dx3 = T.phase_merge(dpp, cpad, H, W)
        if cpad != C:
            dx3 = T.take_ch(dx3, 0, C)
    if stride == 1 or k == 1:
        dw = T.w_block(dwp, Cout, C, depthwise)
    else:
        dw = T.w_phase_block(dwp, Cout, C, pad, phase_q(pad), depthwise)
    return dx3.view(1, C, H, W), dw


def conv2d_backward(x, w, dy, stride=1, pad=0, groups=1):
    """dX, dW for the forward above, using the engine's dX/dW kernels."""
    f32(x, "conv2d_backward(input)")
    f32(dy, "conv2d_backward(dy)")
    f32(w, "conv2d_backward(weight)")
    N = x.shape[0]
    if N == 1:
        return _conv2d_backward_single(x, w, dy, stride, pad, groups)
    dxs = []
    dw = None
    for n in range(N):
        dxn, dwn = _conv2d_backward_single(
            x[n:n + 1], w, dy[n:n + 1], stride, pad, groups)
        dxs.append(dxn)
        dw = dwn if dw is None else T.add(dw, dwn)
    dx = T.cat_ch([d.view(d.shape[1], d.shape[2], d.shape[3]) for d in dxs])
    return dx.view(N, x.shape[1], x.shape[2], x.shape[3]), dw


def channel_sums(t3):
    """Sum over all but the channel axis (GEMM with a ones vector)."""
    C = t3.shape[0]
    flat = t3.reshape(C, -1)
    ones = torch.ones((flat.shape[1], 1), dtype=torch.float32,
                      device=t3.device)
    return torch.mm(flat, ones).view(C)


# --------------------------------------------------------------- pooling

def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False):
    f32(x, "max_pool2d")
    if x.dim() == 3:
        C, H, W = x.shape
        lead = ()
        batches = [x]
    elif x.dim() == 4:
        N, C, H, W = x.shape
        lead = (N,)
        batches = [x[n] for n in range(N)]
    else:
        raise RuntimeError("torch_vc4: max_pool2d expects 3-D or 4-D input")
    k = _pair(kernel_size)
    s = _pair(stride if stride not in (None, []) else kernel_size)
    p = _pair(padding)
    d = _pair(dilation)
    if k[0] != k[1] or s[0] != s[1] or p[0] != p[1]:
        raise RuntimeError("torch_vc4: max_pool2d needs a square kernel, "
                           "stride and padding")
    if d != (1, 1):
        raise RuntimeError("torch_vc4: max_pool2d dilation != 1 is not "
                           "implemented on the engine")
    if ceil_mode:
        raise RuntimeError("torch_vc4: max_pool2d ceil_mode is not implemented")
    if p[0] > k[0] // 2:
        raise RuntimeError("torch_vc4: max_pool2d padding %d > kernel/2"
                           % p[0])

    def one(x3):
        if k[0] == 2 and s[0] == 2 and p[0] == 0 and H % 8 == 0 and W % 16 == 0:
            return T.maxpool2x2(x3)
        return T.maxpooln(x3, k[0], s[0], p[0])

    ys = [one(b) for b in batches]
    if len(ys) == 1:
        y3 = ys[0]
    else:
        y3 = T.cat_ch([y.view(y.shape[0], y.shape[1], y.shape[2])
                       for y in ys])
    return y3.view(tuple(lead) + (C,) + tuple(y3.shape[1:]))


def max_pool2d_backward(x, dy, k, s, p):
    f32(x, "max_pool2d(backward)")
    f32(dy, "max_pool2d(backward, dy)")
    if not (k == 2 and s == 2 and p == 0):
        raise RuntimeError(
            "torch_vc4: the backward of max_pool2d exists only for the "
            "engine's 2x2/stride-2 kernel (got %dx%d stride %d pad %d); "
            "the generic NxN pool is forward-only" % (k, k, s, p))
    if x.dim() == 3:
        C, H, W = x.shape
        if H % 4 or W % 16:
            raise RuntimeError(
                "torch_vc4: max_pool2d backward needs H%%4==0 and W%%16==0 "
                "(engine task geometry), got %dx%d" % (H, W))
        return T.maxpool2x2_bwd(x, dy)
    N, C, H, W = x.shape
    if H % 4 or W % 16:
        raise RuntimeError(
            "torch_vc4: max_pool2d backward needs H%%4==0 and W%%16==0 "
            "(engine task geometry), got %dx%d" % (H, W))
    dxs = [T.maxpool2x2_bwd(x[n], dy[n]) for n in range(N)]
    y = T.cat_ch(dxs)
    return y.view(N, C, H, W)


def adaptive_matrix(H, W, oh, ow):
    """The adaptive-average matrix A[o][p] = 1/count(window(o) covering p).

    Pure data preparation on the host (PyTorch's floor/ceil window rule); the
    averaging itself is a GEMM on the QPU in both directions.
    """
    a = torch.zeros((oh * ow, H * W), dtype=torch.float32)
    for i in range(oh):
        h0, h1 = (i * H) // oh, -((-(i + 1) * H) // oh)
        for j in range(ow):
            w0, w1 = (j * W) // ow, -((-(j + 1) * W) // ow)
            a[i * ow + j].view(H, W)[h0:h1, w0:w1] = 1.0 / ((h1 - h0) * (w1 - w0))
    return a


def adaptive_avg_pool2d(x, output_size):
    f32(x, "adaptive_avg_pool2d")
    if x.dim() == 3:
        C, H, W = x.shape
        lead = ()
        rows = C
        flat = x.view(C, H * W)
    elif x.dim() == 4:
        N, C, H, W = x.shape
        lead = (N,)
        rows = N * C
        flat = x.view(N * C, H * W)
    else:
        raise RuntimeError("torch_vc4: adaptive_avg_pool2d expects 3-D/4-D")
    oh, ow = _pair(output_size)
    if oh <= 0 or ow <= 0:
        raise RuntimeError("torch_vc4: adaptive_avg_pool2d: empty output size")
    a = adaptive_matrix(H, W, oh, ow)          # [oh*ow][H*W] on the CPU
    avt = a.t().contiguous().to(x.device)      # upload: a byte copy
    y = torch.mm(flat, avt)                    # [rows][oh*ow], QPU GEMM
    return y.view(lead + (C, oh, ow))


# --------------------------------------------------------------- softmax

def _exp_neg(x):
    """exp(x) for x <= 0, from sigmoid(-x): exp(x) = (1-s)/s.

    Only valid for non-positive arguments, which is all softmax needs after
    the max subtraction.  The engine's own exp body is wrong in this build
    (COMPLETENESS.md), so this composite is the device path.
    """
    s = T.sigmoid(T.mul_scalar(x, -1.0))
    one_minus = T.add_scalar(T.mul_scalar(s, -1.0), 1.0)
    r = T.rsqrt(s)
    return T.mul(one_minus, T.mul(r, r))


def softmax(x, dim, half_to_float=False):
    f32(x, "softmax")
    if x.dim() == 0:
        raise RuntimeError("torch_vc4: softmax of a scalar")
    nd = x.dim()
    d = _norm_dim(dim, nd, "softmax")
    perm = list(range(nd))
    perm.append(perm.pop(d))
    last = d == nd - 1
    xp = x if last else _permute_dims(x, perm)
    cols = xp.shape[-1]
    rows = xp.numel() // cols
    if cols == 0 or rows == 0:
        raise RuntimeError("torch_vc4: softmax of an empty tensor")
    x2 = xp.view(rows, cols)
    if cols <= 16:
        y2 = T.softmax(x2)                     # verified engine kernel
    else:
        m = _max_last(x2)
        xm = T.add(x2, T.mul_scalar(_expand(m.view(rows, 1), (rows, cols)),
                                    -1.0))
        e = _exp_neg(xm)
        tot = _sum_last(e)                     # [rows]
        r = T.rsqrt(tot)
        y2 = T.mul(e, _expand(T.mul(r, r).view(rows, 1), (rows, cols)))
    y = y2.view(xp.shape)
    if not last:
        inv = [0] * nd
        for i, p in enumerate(perm):
            inv[p] = i
        y = _permute_dims(y, inv)
    return y


# --------------------------------------------------------------- elementwise

# vc4_uop values (include/vc4infer.h)
(UOP_ABS, UOP_NEG, UOP_IDENTITY, UOP_CLIP, UOP_LEAKYRELU, UOP_FLOOR,
 UOP_CEIL, UOP_ROUND, UOP_SIGN, UOP_EXP, UOP_LOG, UOP_SQRT, UOP_RECIPROCAL,
 UOP_TANH, UOP_ERF, UOP_ELU, UOP_SELU, UOP_SOFTPLUS, UOP_SOFTSIGN, UOP_GELU,
 UOP_GELU_ERF, UOP_HARDSIGMOID, UOP_HARDSWISH, UOP_SIN, UOP_COS) = range(25)

# Unary ops whose QPU body the engine element-verified on silicon.  The
# evidence is vc4-stack docs/ONNX.md section 12.3 (2026-11-20): five shapes
# per op, 0 mismatches over 105,264 elements / 120 runs, at
# 1e-4 + 1e-4*max(|ref|,1).  It covers identity, abs, neg, sign, floor, ceil,
# round, clip, leakyrelu, hardsigmoid, hardswish, exp, log, sqrt, reciprocal,
# tanh, erf, softplus, softsign, elu, selu, gelu (tanh form), sin and cos.
#
# Only the ids with an aten entry point are routed here.  The gate stays so a
# future body that has not been through that protocol cannot be reached by
# accident; sqrt/reciprocal/tanh still use the older composites below, which
# are verified through this backend's own tables.
_VERIFIED_UOPS = {UOP_ABS, UOP_NEG, UOP_SIGN, UOP_CLIP, UOP_LEAKYRELU,
                  UOP_HARDSIGMOID, UOP_HARDSWISH, UOP_FLOOR, UOP_CEIL,
                  UOP_ROUND, UOP_EXP, UOP_LOG, UOP_ERF}

_BROKEN = ("the engine's vc4_unary kernel for this op is present but has not "
           "been element-verified on this board, so it is not routed. No CPU "
           "fallback is used")


def _uop(op, name, x, p0=0.0, p1=0.0, p2=0.0):
    f32(x, name)
    if op not in _VERIFIED_UOPS:
        raise RuntimeError("torch_vc4: %s is unavailable: %s" % (name, _BROKEN))
    return T.unary(x, int(op), float(p0), float(p1), float(p2))


def _unavailable(name, why):
    raise RuntimeError("torch_vc4: %s is unavailable on the vc4 backend: %s"
                       % (name, why))


def unary_abs(x):
    return _uop(UOP_ABS, "abs", x)


def unary_sign(x):
    return _uop(UOP_SIGN, "sign", x)


def unary_clip(x, lo=None, hi=None):
    """clip as relu composites.

    The engine's clip body takes the bounds as *uniform words*; vc4d's launch
    policy rejects a uniform that aliases the GPU bus window, and any float
    with |v| >= 2 and a negative sign (including -inf) has that bit pattern.
    Composing from relu + a host-materialised constant keeps arbitrary bounds
    working (COMPLETENESS.md).
    """
    f32(x, "clip")
    if lo is not None and hi is not None and float(lo) > float(hi):
        raise RuntimeError("torch_vc4: clip: min > max")
    y = x
    if lo is not None:
        y = T.add_scalar(torch.relu(T.add_scalar(y, -float(lo))), float(lo))
    if hi is not None:
        # min(y, hi) = y - relu(y - hi)
        y = T.add(y, T.mul_scalar(
            torch.relu(T.add_scalar(y, -float(hi))), -1.0))
    return y


def unary_leaky_relu(x, slope=0.01):
    """leaky_relu = relu(x) - slope*relu(-x), a device composite."""
    f32(x, "leaky_relu")
    pos = torch.relu(x)
    neg = torch.relu(T.mul_scalar(x, -1.0))
    return T.add(pos, T.mul_scalar(neg, -float(slope)))


def unary_hardsigmoid(x):
    return _uop(UOP_HARDSIGMOID, "hardsigmoid", x, 1.0 / 6.0, 0.5)


def unary_hardswish(x):
    return _uop(UOP_HARDSWISH, "hardswish", x, 1.0 / 6.0, 0.5)


def unary_floor(x):
    """floor: exact over the finite fp32 range.

    |x| >= 2^23 is returned unchanged by the body, which is also why large
    values no longer overflow the int32 conversion.  +-inf and NaN take the
    same pass-through (measured: floor(inf) = inf, floor(nan) = nan).
    """
    return _uop(UOP_FLOOR, "floor", x)


def unary_ceil(x):
    """ceil: exact over the finite fp32 range (see unary_floor)."""
    return _uop(UOP_CEIL, "ceil", x)


def unary_round(x):
    """round: nearest, ties to even, as aten::round defines it."""
    return _uop(UOP_ROUND, "round", x)


def unary_exp(x):
    """exp: 2^(x*log2e) with an exact exponent scale.

    Engine-measured worst case 5.7e-6 relative over [-88, 88].  Finite
    inputs above ln(FLT_MAX) overflow to inf as the fp32 reference does;
    a result that would be denormal is flushed to the smallest normal
    (exp(-100) -> 2^-126).  NaN is not in the measured domain.
    """
    return _uop(UOP_EXP, "exp", x)


def unary_log(x):
    """log: exponent/mantissa split plus an atanh series.

    Engine-measured worst case 4.8e-7 relative.  The measured domain is
    x > 0; the body clamps the argument at 2^-126, so log(0) and negative
    arguments are not the -inf/NaN the IEEE reference returns.
    """
    return _uop(UOP_LOG, "log", x)


def unary_erf(x):
    """erf: A&S 7.1.26 with the exp core, exact +-1 saturation past |x| = 4."""
    return _uop(UOP_ERF, "erf", x)


def sigmoid(x):
    f32(x, "sigmoid")
    return T.sigmoid(x)


def silu(x):
    f32(x, "silu")
    return T.silu(x)


def tanh(x):
    """tanh = 2*sigmoid(2x) - 1 (device composite; engine tanh body is wrong)."""
    f32(x, "tanh")
    if not x.numel():
        return x
    return T.add_scalar(T.mul_scalar(T.sigmoid(T.mul_scalar(x, 2.0)), 2.0),
                        -1.0)


def sqrt(x):
    """sqrt = x * rsqrt(x) (engine rsqrt is the verified one used by BN)."""
    f32(x, "sqrt")
    if not x.numel():
        return x
    return T.mul(x, T.rsqrt(x))


def reciprocal(x):
    """1/x = sign(x) * rsqrt(|x|)^2 (rsqrt is the verified engine kernel)."""
    f32(x, "reciprocal")
    if not x.numel():
        return x
    r = T.rsqrt(T.unary(x, int(UOP_ABS), 0.0, 0.0, 0.0))
    return T.mul(T.unary(x, int(UOP_SIGN), 0.0, 0.0, 0.0), T.mul(r, r))


def gelu(x, approximate="none"):
    """GELU in both aten forms.

    approximate='tanh' is x*sigmoid(2c(x + 0.044715 x^3)), c = sqrt(2/pi),
    a device composite.  approximate='none' is 0.5x(1+erf(x/sqrt(2))), which
    uses the engine's element-verified erf body.
    """
    f32(x, "gelu")
    if approximate not in ("none", "tanh"):
        _unavailable("gelu(approximate=%r)" % (approximate,),
                     "aten defines 'none' and 'tanh'; both are routed, "
                     "anything else is not a GELU form")
    if approximate == "none":
        z = T.mul_scalar(x, 1.0 / math.sqrt(2.0))
        return T.mul_scalar(T.mul(x, T.add_scalar(unary_erf(z), 1.0)), 0.5)
    c = math.sqrt(2.0 / math.pi)
    x3 = T.mul(x, T.mul(x, x))
    z = T.mul_scalar(T.add(x, T.mul_scalar(x3, 0.044715)), 2.0 * c)
    return T.mul(x, T.sigmoid(z))


def unary_raise(name, why):
    def f(x, *a, **k):
        _unavailable(name, why)
    return f


# ------------------------------------------------------------ binary ops

def _binary(a, b, op, alpha=1.0):
    sa = _scalar_of(a)
    sb = _scalar_of(b)
    if sa is not None:
        if op == "add":
            return T.add_scalar(b, sa * alpha)
        if op == "mul":
            return T.mul_scalar(b, sa)
        if op == "sub":
            # scalar - tensor: materialise the scalar as a full tensor
            return T.add_scalar(T.mul_scalar(b, -1.0), sa)
        if op == "div":
            return T.mul_scalar(reciprocal(b), sa)
    if sb is not None:
        if op == "add":
            return T.add_scalar(a, sb * alpha)
        if op == "mul":
            return T.mul_scalar(a, sb)
        if op == "sub":
            return T.add_scalar(a, -sb * alpha)
        if op == "div":
            return T.mul_scalar(a, 1.0 / sb)
    f32(b, op)
    shape = torch.broadcast_shapes(tuple(a.shape), tuple(b.shape))
    aa = _expand(a, shape)
    bb = _expand(b, shape)
    if op == "add":
        if alpha == 1.0:
            return T.add(aa, bb)
        return T.add(aa, T.mul_scalar(bb, alpha))
    if op == "sub":
        return T.add(aa, T.mul_scalar(bb, -alpha))
    if op == "mul":
        return T.mul(aa, bb)
    if op == "div":
        return T.mul(aa, reciprocal(bb))
    raise RuntimeError("torch_vc4: unknown binary op %r" % (op,))


def _add_tensor(a, b, alpha=1):
    return _binary(a, b, "add", float(alpha))


def _sub_tensor(a, b, alpha=1):
    return _binary(a, b, "sub", float(alpha))


def _mul_tensor(a, b):
    return _binary(a, b, "mul")


def _div_tensor(a, b):
    return _binary(a, b, "div")


# ---------------------------------------------------------------- views

def view(x, size):
    f32(x, "view")
    return T.view_alias(x, list(size))


def flatten(x, start_dim=0, end_dim=-1):
    f32(x, "flatten")
    nd = x.dim()
    if nd == 0:
        return x.view(1)
    s = start_dim if start_dim >= 0 else start_dim + nd
    e = end_dim if end_dim >= 0 else end_dim + nd
    if s < 0 or e >= nd or s > e:
        raise RuntimeError("torch_vc4: flatten: bad dims %d..%d of %d"
                           % (start_dim, end_dim, nd))
    shape = list(x.shape)
    merged = 1
    for i in range(s, e + 1):
        merged *= shape[i]
    return x.view(shape[:s] + [merged] + shape[e + 1:])


def as_strided(x, size, stride, storage_offset=None):
    f32(x, "as_strided")
    return T.as_strided_alias(x, list(size), list(stride),
                              None if storage_offset is None
                              else int(storage_offset))


# ---------------------------------------------------------------- concat

def cat(tensors, dim=0):
    if not tensors:
        raise RuntimeError("torch_vc4: cat of an empty list")
    for t in tensors:
        f32(t, "cat")
    t0 = tensors[0]
    nd = t0.dim()
    if nd == 4:
        if dim not in (1, -3):
            raise RuntimeError("torch_vc4: cat is implemented on the channel "
                               "axis of NCHW tensors (dim=1), got dim=%d" % dim)
        N = t0.shape[0]
        for t in tensors:
            if t.shape[0] != N or t.shape[2:] != t0.shape[2:]:
                raise RuntimeError("torch_vc4: cat: shapes %s are not "
                                   "compatible on the channel axis"
                                   % [tuple(t.shape) for t in tensors])
        parts = []
        for n in range(N):
            parts.append(T.cat_ch([t[n].view(t.shape[1], t.shape[2],
                                             t.shape[3]) for t in tensors]))
        y = T.cat_ch(parts) if N > 1 else parts[0]
        C = sum(t.shape[1] for t in tensors)
        return y.view(N, C, t0.shape[2], t0.shape[3])
    if nd == 3 and dim in (0, -3):
        return T.cat_ch(list(tensors))
    raise RuntimeError("torch_vc4: cat: unsupported rank %d / dim %d"
                       % (nd, dim))


# --------------------------------------------------------------- resize

def upsample_nearest2d(x, output_size, scales_h=None, scales_w=None):
    f32(x, "upsample_nearest2d")
    if x.dim() != 4:
        raise RuntimeError("torch_vc4: upsample_nearest2d expects NCHW")
    N, C, H, W = x.shape
    ho, wo = int(output_size[0]), int(output_size[1])
    parts = [T.up_nearest(x[n], ho, wo,
                          float(scales_h) if scales_h else 0.0,
                          float(scales_w) if scales_w else 0.0)
             for n in range(N)]
    y = T.cat_ch(parts) if N > 1 else parts[0]
    return y.view(N, C, ho, wo)


def upsample_bilinear2d(x, output_size, align_corners=False, scales_h=None,
                        scales_w=None):
    """Bilinear resize from the engine's four-gather lowering.

    The engine builds four int32 index maps and four weight vectors from the
    output grid (geometry only) and runs 4 gathers + 4 multiplies + 3 adds on
    the QPUs, so the ARM never touches an activation value.  Both torch
    mappings are supported: align_corners=False is ONNX "half_pixel",
    align_corners=True is "align_corners".  The scale arguments are not
    needed: torch has already resolved output_size from them.
    """
    f32(x, "upsample_bilinear2d")
    if x.dim() != 4:
        raise RuntimeError("torch_vc4: upsample_bilinear2d expects NCHW")
    ho, wo = int(output_size[0]), int(output_size[1])
    if ho <= 0 or wo <= 0:
        raise RuntimeError("torch_vc4: upsample_bilinear2d: empty output "
                           "%dx%d" % (ho, wo))
    return T.up_bilinear(x, ho, wo, bool(align_corners))


def upsample_bilinear2d_vec(x, output_size, align_corners=False,
                           scale_factors=None):
    if output_size is not None:
        return upsample_bilinear2d(x, output_size, align_corners)
    if scale_factors is None:
        raise RuntimeError("torch_vc4: upsample_bilinear2d needs a size or "
                           "scale factors")
    H, W = x.shape[-2], x.shape[-1]
    sh, sw = float(scale_factors[0]), float(scale_factors[1])
    ho = int(math.floor(H * sh))
    wo = int(math.floor(W * sw))
    return upsample_bilinear2d(x, (ho, wo), align_corners)


def upsample_nearest2d_vec(x, output_size, scale_factors):
    if output_size is not None:
        return upsample_nearest2d(x, output_size)
    if scale_factors is None:
        raise RuntimeError("torch_vc4: upsample_nearest2d needs a size or "
                           "scale factors")
    H, W = x.shape[-2], x.shape[-1]
    sh, sw = float(scale_factors[0]), float(scale_factors[1])
    ho = int(math.floor(H * sh))
    wo = int(math.floor(W * sw))
    return upsample_nearest2d(x, (ho, wo), sh, sw)


# --------------------------------------------------------------- batch norm

def _bn_apply(x, mean, var, weight, bias, eps):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, all on device."""
    N, C = x.shape[0], x.shape[1]
    inv = T.rsqrt(T.add_scalar(var.view(C), eps))
    scale = T.mul(inv, weight) if weight is not None else inv
    shift = T.mul(scale, mean.view(C))
    shift = T.mul_scalar(shift, -1.0)
    if bias is not None:
        f32(bias, "batch_norm(bias)")
        shift = T.add(shift, bias)
    sc = _expand(scale.view(1, C, 1, 1), x.shape)
    sh = _expand(shift.view(1, C, 1, 1), x.shape)
    return T.add(T.mul(x, sc), sh)


def _bn_batch_stats(x):
    """Per-channel mean and biased variance of NCHW, on device."""
    N, C, H, W = x.shape
    n = N * H * W
    mean = _dims_reduce(x, [0, 2, 3], True, "sum")
    mean = T.mul_scalar(mean, 1.0 / n)                       # [1,C,1,1]
    d = T.add(x, T.mul_scalar(_expand(mean, x.shape), -1.0))
    var = _dims_reduce(T.mul(d, d), [0, 2, 3], True, "sum")
    var = T.mul_scalar(var, 1.0 / n)
    return mean.view(C), var.view(C), d


def _bn_update(running, batch, momentum):
    newv = T.add(T.mul_scalar(running, 1.0 - momentum),
                 T.mul_scalar(batch, momentum))
    running.copy_(newv)


def _bn_forward(x, weight, bias, mean, var, training, momentum, eps):
    if x.dim() != 4:
        raise RuntimeError("torch_vc4: batch_norm expects NCHW")
    N, C, H, W = x.shape
    if training:
        bmean, bvar, d = _bn_batch_stats(x)
        invstd = T.rsqrt(T.add_scalar(bvar, eps))
        xhat = T.mul(d, _expand(invstd.view(1, C, 1, 1), x.shape))
        y = (T.mul(xhat, _expand(weight.view(1, C, 1, 1), x.shape))
             if weight is not None else xhat)
        if bias is not None:
            y = T.add(y, _expand(bias.view(1, C, 1, 1), x.shape))
        n = N * H * W
        if mean is not None:
            f32(mean, "batch_norm(running_mean)")
            _bn_update(mean.view(C), bmean, momentum)
        if var is not None:
            f32(var, "batch_norm(running_var)")
            unb = T.mul_scalar(bvar, n / (n - 1) if n > 1 else 1.0)
            _bn_update(var.view(C), unb, momentum)
        return y, bmean, invstd
    if mean is None or var is None:
        raise RuntimeError("torch_vc4: batch_norm needs running_mean and "
                           "running_var in inference mode")
    f32(mean, "batch_norm(running_mean)")
    f32(var, "batch_norm(running_var)")
    if mean.numel() != C or var.numel() != C:
        raise RuntimeError("torch_vc4: batch_norm running stats have %d/%d "
                           "elements, expected %d" % (mean.numel(),
                                                      var.numel(), C))
    invstd = T.rsqrt(T.add_scalar(var.view(C), eps))
    xhat = T.add(x, T.mul_scalar(_expand(mean.view(1, C, 1, 1), x.shape), -1.0))
    xhat = T.mul(xhat, _expand(invstd.view(1, C, 1, 1), x.shape))
    y = T.mul(xhat, _expand(weight.view(1, C, 1, 1), x.shape)) \
        if weight is not None else xhat
    if bias is not None:
        y = T.add(y, _expand(bias.view(1, C, 1, 1), x.shape))
    return y, mean.view(C), invstd


def native_batch_norm(x, weight, bias, mean, var, training, momentum, eps):
    f32(x, "batch_norm")
    y, bmean, invstd = _bn_forward(x, weight, bias, mean, var, bool(training),
                                   float(momentum), float(eps))
    return y, bmean, invstd


def batch_norm(x, weight, bias, mean, var, training, momentum, eps,
               cudnn_enabled=False):
    return native_batch_norm(x, weight, bias, mean, var, training, momentum,
                             eps)[0]


# ---------------------------------------------------------------- linear

def linear(x, weight, bias=None):
    f32(x, "linear(input)")
    f32(weight, "linear(weight)")
    if weight.dim() != 2:
        raise RuntimeError("torch_vc4: linear expects a 2-D weight")
    out_f, in_f = weight.shape
    if x.dim() == 0:
        raise RuntimeError("torch_vc4: linear of a scalar")
    if x.shape[-1] != in_f:
        raise RuntimeError("torch_vc4: linear: input %s vs weight %s"
                           % (tuple(x.shape), tuple(weight.shape)))
    lead = tuple(x.shape[:-1])
    x2 = x.view(-1, in_f)
    y = T.mm_bt(x2, weight)          # x @ W^T on the QPU (pads to 16)
    if bias is not None:
        f32(bias, "linear(bias)")
        y = T.add(y, T.bcast_rows(bias, y.shape[0]))
    return y.view(lead + (out_f,))


# --------------------------------------------------------------- reductions

_SUM_CHUNK = 4096


def sum_all(t, dtype=None):
    """Full reduction as GEMM(s) with a host-materialised ones vector.

    One GEMM over the whole tensor would need a 64-padded B panel of
    64*n floats (16 MiB at n = 64 Ki), which does not fit the engine's
    scratch next to a 24 MiB tensor region; the reduction is therefore done
    in 4096-element chunks (one launch each) and summed with GPU adds.
    """
    f32(t, "sum")
    if dtype is not None and dtype != torch.float32:
        raise RuntimeError("torch_vc4: sum(dtype=...) is not supported")
    flat = t.view(-1)
    n = flat.numel()
    if n == 0:
        raise RuntimeError("torch_vc4: sum of an empty tensor")
    if n <= _SUM_CHUNK:
        return T.sum_all(flat).view([])
    acc = None
    for i in range(0, n, _SUM_CHUNK):
        part = T.sum_all(flat[i:i + _SUM_CHUNK])
        acc = part if acc is None else T.add(acc, part)
    return acc.view([])


def mean_all(t, dtype=None):
    f32(t, "mean")
    if t.numel() == 0:
        raise RuntimeError("torch_vc4: mean of an empty tensor")
    return T.mul_scalar(sum_all(t), 1.0 / t.numel())


def sum_dims(t, dim, keepdim=False, dtype=None):
    f32(t, "sum")
    if dtype is not None and dtype != torch.float32:
        raise RuntimeError("torch_vc4: sum(dtype=...) is not supported")
    if dim is None or len(dim) == 0:
        r = sum_all(t)
        return r.view([1] * t.dim()) if keepdim else r
    dims = _norm_dims(dim, t.dim(), "sum")
    return _dims_reduce(t, dims, bool(keepdim), "sum")


def mean_dims(t, dim, keepdim=False, dtype=None):
    f32(t, "mean")
    if dtype is not None and dtype != torch.float32:
        raise RuntimeError("torch_vc4: mean(dtype=...) is not supported")
    if dim is None or len(dim) == 0:
        r = mean_all(t)
        return r.view([1] * t.dim()) if keepdim else r
    dims = _norm_dims(dim, t.dim(), "mean")
    return _dims_reduce(t, dims, bool(keepdim), "mean")


def amax_dims(t, dim=None, keepdim=False):
    f32(t, "amax")
    if dim is None or len(dim) == 0:
        dims = list(range(t.dim()))
    else:
        dims = _norm_dims(dim, t.dim(), "amax")
    return _dims_reduce(t, dims, bool(keepdim), "max")


def amin_dims(t, dim=None, keepdim=False):
    f32(t, "amin")
    if dim is None or len(dim) == 0:
        dims = list(range(t.dim()))
    else:
        dims = _norm_dims(dim, t.dim(), "amin")
    return _dims_reduce(t, dims, bool(keepdim), "min")


def _reshape_idx(r, shape):
    """Reshape an int64 index result (the view kernel is float-only)."""
    if tuple(r.shape) == tuple(shape):
        return r
    out = torch.empty(tuple(shape), dtype=torch.int64, device=r.device)
    out.copy_(r.cpu().reshape(shape))
    return out


def argmax(t, dim=None, keepdim=False):
    f32(t, "argmax")
    nd = t.dim()
    if dim is None:
        flat = t.view(1, t.numel())
        return _reshape_idx(T.argmax_host(flat, 1, t.numel(), True), ())
    d = _norm_dim(dim, nd, "argmax")
    perm = list(range(nd))
    perm.append(perm.pop(d))
    xp = t if d == nd - 1 else _permute_dims(t, perm)
    cols = xp.shape[-1]
    rows = xp.numel() // cols
    r = T.argmax_host(xp.view(rows, cols), rows, cols, True)   # [rows] int64
    keepshape = [t.shape[i] for i in range(nd) if i != d]
    shape = keepshape if not keepdim else \
        [1 if i == d else t.shape[i] for i in range(nd)]
    return _reshape_idx(r, shape)


def argmin(t, dim=None, keepdim=False):
    f32(t, "argmin")
    nd = t.dim()
    if dim is None:
        flat = t.view(1, t.numel())
        return _reshape_idx(T.argmax_host(flat, 1, t.numel(), False), ())
    d = _norm_dim(dim, nd, "argmin")
    perm = list(range(nd))
    perm.append(perm.pop(d))
    xp = t if d == nd - 1 else _permute_dims(t, perm)
    cols = xp.shape[-1]
    rows = xp.numel() // cols
    r = T.argmax_host(xp.view(rows, cols), rows, cols, False)
    shape = ([1 if i == d else t.shape[i] for i in range(nd)] if keepdim
             else [t.shape[i] for i in range(nd) if i != d])
    return _reshape_idx(r, shape)


def max_full(t):
    return amax_dims(t, None, False)


def min_full(t):
    return amin_dims(t, None, False)


def max_dim(t, dim, keepdim=False):
    v = amax_dims(t, [dim], keepdim)
    i = argmax(t, dim, keepdim)
    return v, i


def min_dim(t, dim, keepdim=False):
    v = amin_dims(t, [dim], keepdim)
    i = argmin(t, dim, keepdim)
    return v, i


# -------------------------------------------------------------- indexing

def _index_host(index, what):
    """Indices are bookkeeping: keep them on the host (a byte copy), never on
    the QPU.  The vc4 allocator has no int64 gather, and a 64-bit index tensor
    is never a compute operand."""
    if index.dtype != torch.int64:
        raise RuntimeError("torch_vc4: %s index must be int64, got %s"
                           % (what, index.dtype))
    if index.device.type == "vc4":
        if not index.is_contiguous():
            raise RuntimeError("torch_vc4: %s index must be contiguous"
                               % what)
        index = index.cpu()
    return index.contiguous()


def gather(x, dim, index, sparse_grad=False):
    """out[i][j] = x[...][index[i][j]][...] (exact, host word gather).

    PyTorch's gather semantics: the index shape is the output shape, the
    index size along each non-gathered axis may only shrink the source
    (a prefix is read), and index.select's per-row broadcast is not involved.
    """
    f32(x, "gather")
    index = _index_host(index, "gather")
    nd = x.dim()
    d = _norm_dim(dim, nd, "gather")
    if index.dim() != nd:
        raise RuntimeError("torch_vc4: gather index rank %d != source rank %d"
                           % (index.dim(), nd))
    sizes = list(x.shape)
    for i in range(nd):
        if i == d:
            continue
        if index.shape[i] > x.shape[i]:
            raise RuntimeError(
                "torch_vc4: gather: index dim %d is %d > source %d"
                % (i, index.shape[i], x.shape[i]))
        sizes[i] = index.shape[i]
    if sizes != list(x.shape):
        v = T.as_strided_alias(x, sizes, [x.stride(i) for i in range(nd)],
                               x.storage_offset())
        x = T.materialize(v)
    perm = [d] + [i for i in range(nd) if i != d]
    xp = _permute_dims(x, perm) if perm != list(range(nd)) else x
    ip = index.permute(perm).contiguous() if perm != list(range(nd)) else index
    D = xp.shape[0]
    rest = xp.numel() // D
    B = index.shape[d]
    out2 = T.take(xp.view(D, rest), ip.view(B, rest), rest)
    out = out2.view([B] + [sizes[i] for i in perm[1:]])
    inv = [0] * nd
    for i, p in enumerate(perm):
        inv[p] = i
    if inv != list(range(nd)):
        out = _permute_dims(out, inv)
    return out


def index_select(x, dim, index):
    """out[..., i, ...] = x[..., index[i], ...] (same index for every row)."""
    f32(x, "index_select")
    if index.dim() != 1:
        raise RuntimeError("torch_vc4: index_select expects a 1-D index")
    index = _index_host(index, "index_select")
    nd = x.dim()
    d = _norm_dim(dim, nd, "index_select")
    perm = [d] + [i for i in range(nd) if i != d]
    xp = _permute_dims(x, perm) if perm != list(range(nd)) else x
    D = x.shape[d]
    rest = x.numel() // D
    B = index.numel()
    out2 = T.take(xp.view(D, rest), index, rest)       # row mode
    out = out2.view([B] + [x.shape[i] for i in perm[1:]])
    inv = [0] * nd
    for i, p in enumerate(perm):
        inv[p] = i
    if inv != list(range(nd)):
        out = _permute_dims(out, inv)
    return out


def _scatter_raise(name):
    def f(*a, **k):
        raise RuntimeError(
            "torch_vc4: %s is unsupported on this backend: there is no "
            "scatter kernel (the engine's table-driven vc4_gather only reads, "
            "and it hangs on this board), and a host scatter would be a CPU "
            "fallback. Rewrite the model without scatter or run it on CPU."
            % name)
    return f


def _sort_raise(name):
    def f(*a, **k):
        raise RuntimeError(
            "torch_vc4: %s is unsupported on this backend: there is no QPU "
            "sort/selection kernel in this engine build; a host sort would be "
            "a full CPU fallback, which this backend does not do."
            % name)
    return f


# ------------------------------------------------------------- fold helper

def fold_bn_into_conv(weight, bias, mean, var, gamma, beta, eps=1e-5):
    """Inference-time constant folding: conv(x, w', b') == bn(conv(x, w, b)).

    Parameters are CPU tensors here on purpose: folding is a one-off on the
    *parameters*, not on activations, so it is host arithmetic by design.
    Returns (w', b') as contiguous CPU tensors; upload them to vc4 with .to().
    """
    if weight.dim() != 4:
        raise RuntimeError("torch_vc4: fold_bn_into_conv expects [Cout][Cin]"
                           "[kh][kw]")
    scale = (gamma / torch.sqrt(var + eps)).view(-1, 1, 1, 1)
    w2 = weight * scale
    b2 = (bias - mean) * scale.view(-1) + beta
    return w2.contiguous(), b2.contiguous()


# ------------------------------------------------------------- registration

def _scalar_of(x):
    """The Python value of a scalar operand (Python number or CPU 0-dim)."""
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x) and x.device.type == "cpu" and x.numel() == 1:
        return float(x)
    return None


def register():
    # torch deregisters a Library when it is collected; keep it alive
    lib = _keep(torch.library.Library("aten", "IMPL", "PrivateUse1"))

    def impl(name, fn):
        lib.impl(name, fn)

    impl("convolution_overrideable",
         lambda x, w, b, stride, pad, dil, transposed, out_pad, groups:
         _conv_op(x, w, b, stride, pad, dil, transposed, out_pad, groups))
    impl("convolution_backward_overrideable", convolution_backward)
    impl("max_pool2d", max_pool2d)
    impl("_adaptive_avg_pool2d", adaptive_avg_pool2d)
    impl("adaptive_avg_pool2d", adaptive_avg_pool2d)
    impl("_softmax", softmax)
    impl("sigmoid", sigmoid)
    impl("silu", silu)
    impl("tanh", tanh)
    impl("sqrt", sqrt)
    impl("abs", unary_abs)
    impl("sign", unary_sign)
    impl("clip", unary_clip)
    impl("leaky_relu", unary_leaky_relu)
    impl("hardsigmoid", unary_hardsigmoid)
    impl("hardswish", unary_hardswish)
    impl("gelu", gelu)
    impl("reciprocal", reciprocal)
    impl("exp", unary_exp)
    impl("log", unary_log)
    impl("erf", unary_erf)
    impl("floor", unary_floor)
    impl("ceil", unary_ceil)
    impl("round", unary_round)
    # aten::view/aten::as_strided are registered natively (csrc) -- the
    # Python layer only supplies the autograd Functions for them.
    impl("flatten.using_ints", flatten)
    impl("cat", cat)
    impl("upsample_nearest2d", upsample_nearest2d)
    impl("upsample_nearest2d.vec", upsample_nearest2d_vec)
    impl("upsample_bilinear2d", upsample_bilinear2d)
    impl("upsample_bilinear2d.vec", upsample_bilinear2d_vec)
    impl("native_batch_norm", native_batch_norm)
    impl("batch_norm", batch_norm)
    impl("linear", linear)
    impl("sum", sum_all)
    impl("mean", mean_all)
    impl("sum.dim_IntList", sum_dims)
    impl("mean.dim", mean_dims)
    impl("amax", amax_dims)
    impl("amin", amin_dims)
    impl("argmax", argmax)
    impl("argmin", argmin)
    impl("max", max_full)
    impl("min", min_full)
    impl("max.dim", max_dim)
    impl("min.dim", min_dim)
    impl("gather", gather)
    impl("index_select", index_select)
    def impl_safe(name, fn):
        # the overload name has to exist in this torch build; an unknown one
        # would make importing the backend fail for no good reason
        try:
            lib.impl(name, fn)
        except Exception:
            pass

    for _n in ("scatter", "scatter.value", "scatter_", "scatter_.value",
               "scatter_add", "scatter_add_", "index_add", "index_add_"):
        impl_safe(_n, _scatter_raise(_n))
    for _n in ("sort", "sort.stable", "topk", "kthvalue", "median",
               "median.dim", "unique", "unique_dim"):
        impl_safe(_n, _sort_raise(_n))
    # scalar overloads; the Tensor overloads re-register here so that a
    # scalar operand in the *second* position never reaches a two-tensor
    # kernel
    impl("mul.Scalar", lambda x, s: T.mul_scalar(x, float(s)))
    impl("div.Scalar", lambda x, s: T.mul_scalar(x, 1.0 / float(s)))
    impl("div.Tensor", _div_tensor)
    impl("add.Tensor", _add_tensor)
    impl("mul.Tensor", _mul_tensor)
    impl("sub.Tensor", _sub_tensor)
    impl("add_.Tensor", _add_inplace)
    impl("add.Scalar", lambda x, s, alpha=1: T.add_scalar(x, float(s) * alpha))
    impl("sub.Scalar", lambda x, s, alpha=1:
         T.add_scalar(x, -float(s) * alpha))
    impl("neg", lambda x: T.mul_scalar(x, -1.0))
    impl("pow.Tensor_Scalar", _pow_scalar)
    return lib


def _add_inplace(self, other, alpha=1):
    """p.add_(g, alpha) for the autograd engine's gradient accumulation."""
    out = _add_scalar_alpha(self, other, alpha)
    self.copy_(out)
    return self


def _add_scalar_alpha(a, b, alpha):
    f32(a, "add_")
    shape = torch.broadcast_shapes(tuple(a.shape), tuple(b.shape))
    bb = _expand(b, shape)
    if shape != tuple(a.shape):
        raise RuntimeError("torch_vc4: add_ cannot broadcast into a different "
                           "shape (%s -> %s)" % (tuple(a.shape), shape))
    if alpha == 1:
        return T.add(a, bb)
    return T.add(a, T.mul_scalar(bb, float(alpha)))


def convolution_backward(g, x, w, stride, pad, dil, transposed, out_pad,
                        groups, output_mask):
    """The backward hook torch's own conv autograd calls for custom backends
    (``aten::convolution_backward_overrideable``).
    """
    if transposed:
        raise RuntimeError("torch_vc4: transposed conv backward is not "
                           "implemented")
    if _pair(dil) != (1, 1):
        raise RuntimeError("torch_vc4: conv dilation != 1 is not implemented")
    s, p = _pair(stride), _pair(pad)
    if s[0] != s[1] or p[0] != p[1]:
        raise RuntimeError("torch_vc4: conv needs square stride/padding")
    f32(g, "conv2d_backward(grad)")
    want_x, want_w, want_b = (bool(v) for v in output_mask)
    dx = dw = db = None
    if want_x or want_w:
        dx_, dw_ = conv2d_backward(x, w, g, s[0], p[0], int(groups))
        dx = dx_ if want_x else None
        dw = dw_ if want_w else None
    if want_b:
        db = _dims_reduce(g, [0, 2, 3], False, "sum")
    return dx, dw, db


def _conv_op(x, w, b, stride, pad, dil, transposed, out_pad, groups):
    if transposed:
        raise RuntimeError("torch_vc4: transposed convolution is not "
                           "implemented")
    if any(int(v) for v in out_pad):
        raise RuntimeError("torch_vc4: conv output_padding is not implemented")
    if _pair(dil) != (1, 1):
        raise RuntimeError("torch_vc4: conv dilation != 1 is not implemented "
                           "on the engine")
    s = _pair(stride)
    p = _pair(pad)
    if s[0] != s[1]:
        raise RuntimeError("torch_vc4: conv needs a square stride, got %s"
                           % (tuple(s),))
    if p[0] != p[1]:
        raise RuntimeError("torch_vc4: conv needs symmetric padding, got %s"
                           % (tuple(p),))
    return conv2d(x, w, b, s[0], p[0], int(groups))


def _pow_scalar(x, exponent):
    if float(exponent) != 2.0:
        raise RuntimeError("torch_vc4: pow is implemented for exponent 2 only "
                           "(x*x); got %r" % (exponent,))
    f32(x, "pow")
    return T.mul(x, x)
