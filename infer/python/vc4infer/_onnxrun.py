#!/usr/bin/env python3
"""onnxrun.py -- execute a small ONNX graph on the VideoCore IV QPUs.

Strict mode (the default) dispatches every arithmetic node to libvc4infer
through its ctypes binding (``vc4infer.py``).  There is no CPU fallback: a
node, attribute combination or shape the engine cannot do raises
``UnsupportedNode`` naming the node and its op type.  The only CPU work is byte
movement that the engine itself also does on the ARM (padding, channel
gathering, concat, nearest resize, transposes, compaction) plus integer
bookkeeping (``Shape``/``Gather`` and the int64 shape arithmetic of the head's
dynamic ``Reshape``).  No float multiply, add or divide of activations happens
on the ARM: ``Div`` by a graph constant is a GPU ``Mul`` by the folded
reciprocal, ``Sub`` of a runtime tensor is GPU ``Mul(-1)`` + GPU ``Add``.

Hybrid mode (opt-in, ``mode="hybrid"``) adds a host tier for nodes VC4 cannot
express.  The policy is explicit:

    host_policy="structural" (default)
        host-meta (Shape/Size/Constant and integer shape arithmetic) and
        host-control (If/Loop/Scan) may use ``onnx.reference``.  A numeric
        operator with no VC4 kernel (``host-unwritten``) raises
        ``HostNotAllowed``: the debt stays visible.
    host_policy="any"
        host-unwritten nodes also run in the reference evaluator.  This is
        compatibility mode, not a performance claim.

``unsupported-graph`` (invalid domains/types) is refused under both policies.
Crossing the boundary copies NCHW fp32 tensors between the arena and numpy;
the copy count is part of the profile.  Strict mode is byte-for-byte the old
executor and never consults the reference evaluator.

Two conv paths
    NHWC (default)  ``vc4_conv_nhwc``: QPU im2col over [pixel][channel] rows +
                    QPU GEMM, output [pixels][channel].  The im2col VDR pitch
                    is ``stride*Cin*4`` bytes, not the whole plane, so the
                    8191-byte pitch field no longer limits the map size
                    (Cin <= 2047).  Cin/Cout are zero-padded to 16 channels.
    NCHW (legacy)   ``vc4_conv2d``: kept for shapes the NHWC kernel does not
                    take (non-square kernel or stride, odd padding).

Layout bookkeeping: a 4-D fp32 tensor carries a ``layout`` ("nchw"/"nhwc") and,
for NHWC, the storage ``pitch`` (the conv writes ``align16(wo)`` columns).
Elementwise ops are layout-agnostic (they touch every element); every op that
depends on the channel axis converts on the host first (a byte gather).

Supported nodes
    Conv          NHWC 1x1/3x3/5x5, stride 1/2, dilation 1..8, group=1,
                  grouped (per-group Cin/Cout/K %16) or depthwise (dense
                  block-diagonal weight); NCHW fallback for the rest
    Relu, Sigmoid
    Add, Mul      same shape, or broadcast from a graph constant (host expand)
    MaxPool       2x2 s2 (NCHW) and kxk s1 (NHWC, SPPF)
    Concat        channel axis (4-D) and axis 1/-1 on 3-D
    Split         channel axis (4-D)
    Slice         any axis, int64 starts/ends/axes/steps
    Div, Sub      lowered to GPU Mul / Mul+Add (constant operand for Div)
    Softmax       any axis (host permute to the last, <= 16 classes, rows%8)
    Transpose     host permute
    Resize        nearest (4 nearest_mode values) or linear (GPU Mul/Add),
                  all 6 coordinate_transformation_mode values
    MatMul, Gemm  2-D
    Flatten, Reshape, Squeeze, Unsqueeze, Cast, Identity, Constant
    Shape, Gather int64 host tensors (static shapes)

Batch N is supported for the NHWC conv path, pool, upsample, concat,
elementwise and softmax; the NCHW fallback, Gemm and GlobalAveragePool are
batch 1 and raise otherwise.
"""
import functools
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


def default_cache_dir():
    """A writable kernel-blob cache.

    An installed run has /usr/lib/vc4infer read-only, so the default is
    $VC4INFER_CACHE or ~/.cache/vc4infer/kcache.
    """
    env = os.environ.get("VC4INFER_CACHE")
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "vc4infer", "kcache")


# The ctypes binding is a module of this package, so the executor and the
# public API always share one Engine class (and one library handle).
from . import coverage
from . import _binding as vc4infer                      # noqa: E402
from ._errors import (HostNodeError, HostNotAllowed, NeedsDeviceDispatch,
                      UnsupportedNode)                    # noqa: E402

# The 13 nv4 ops were element-verified on 2026-11-20: 0 mismatches over
# 105,264 kernels elements (five shapes each) and 0 over 10,478 graph
# elements (docs/ONNX.md sec. 12.3).  VC4_UNARY_VERIFIED=0 stages them out
# again (the C non-16 refusal is the matching hard gate).
_UNARY_NV4_VERIFIED = os.environ.get("VC4_UNARY_VERIFIED", "1") != "0"
_UNARY_NV4_STAGED = ("Exp", "Log", "Sqrt", "Reciprocal", "Erf", "Sin", "Cos",
                     "Round", "Softplus", "Elu", "Selu", "Gelu", "Mish")
from ._hybrid import HostEvaluator, outer_refs             # noqa: E402

# Kept for the scripts that print the engine directory.
ENGINE = _first_dir(os.environ.get("VC4INFER_ENGINE"),
                    os.path.join(HERE, "..", "engine"),
                    "/usr/lib/vc4infer")


def _align16(v):
    return (int(v) + 15) & ~15


_EXTRA_HANDLERS = None
_EXTRA_OVERRIDES = frozenset()


def _extra_handlers():
    """The lazy family dispatch tables (imported once, on first use).

    ``_dispatch_extra`` is the reduction / comparison / shape-and-arith family
    (docs/OPS-REDUCE.md), ``_dispatch_norm`` the pooling / normalisation /
    softmax-variant / loss family (docs/OPS-NORM.md).  They are imported
    lazily because they are built on this module's ``Tensor`` and ``Runner``
    helpers; a module-level import would be circular.

    A family module may also *own* an operator the main table can dispatch
    (``OVERRIDES``): ``Softmax`` moved to ``_dispatch_norm`` because the main
    table's handler refused more than 16 classes, and the family handler calls
    the old fast paths for the shapes they cover.
    """
    global _EXTRA_HANDLERS, _EXTRA_OVERRIDES
    if _EXTRA_HANDLERS is None:
        from . import (_dispatch_extra, _dispatch_norm, _dispatch_geom,
                       _dispatch_rnn, _dispatch_sig)
        _EXTRA_HANDLERS = dict(_dispatch_extra.HANDLERS)
        _EXTRA_HANDLERS.update(_dispatch_norm.HANDLERS)
        _EXTRA_HANDLERS.update(_dispatch_geom.HANDLERS)
        _EXTRA_HANDLERS.update(_dispatch_rnn.HANDLERS)
        _EXTRA_HANDLERS.update(_dispatch_sig.HANDLERS)
        _EXTRA_OVERRIDES = (
            frozenset(getattr(_dispatch_norm, "OVERRIDES", ()))
            | frozenset(getattr(_dispatch_extra, "OVERRIDES", ()))
            | frozenset(getattr(_dispatch_geom, "OVERRIDES", ()))
            | frozenset(getattr(_dispatch_rnn, "OVERRIDES", ()))
            | frozenset(getattr(_dispatch_sig, "OVERRIDES", ())))
    return _EXTRA_HANDLERS


def _align(n, a=64):
    return (int(n) + a - 1) // a * a


def _mib(n):
    return "%.1f MiB" % (n / 2 ** 20)


def _align16(n):
    return (int(n) + 15) // 16 * 16


def _tensor_bytes(shape, nhwc=False):
    """Bytes the runner's pool gives a tensor of this ONNX shape.

    Conv outputs are stored NHWC with the channel and width axes padded to 16
    (the QPU kernel's grid), so they are bigger than their element count; every
    other tensor is a plain contiguous fp32 buffer.  The +64 is the allocator's
    per-buffer slack (see Runner._alloc).
    """
    if not shape or any(d is None or int(d) <= 0 for d in shape):
        return 0
    n = 1
    for d in shape:
        n *= int(d)
    if nhwc and len(shape) == 4:
        nb, c, h, w = (int(d) for d in shape)
        return _align(nb * h * _align16(w) * _align16(c) * 4) + 64
    return _align(n * 4) + 64


def _conv_attrs(node):
    a = {x.name: x for x in node.attribute}
    ks = list(a["kernel_shape"].ints) if "kernel_shape" in a else None
    st = list(a["strides"].ints) if "strides" in a else [1, 1]
    pd = list(a["pads"].ints) if "pads" in a else [0, 0, 0, 0]
    dl = list(a["dilations"].ints) if "dilations" in a else [1, 1]
    gp = int(a["group"].i) if "group" in a else 1
    return ks, st, pd, dl, gp


def _is_float_init(t):
    import onnx
    return t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)


def _numel_proto(t):
    n = 1
    for d in t.dims:
        n *= int(d)
    return n


def _init_shape_map(graph):
    """name -> dims for initialisers (shape inference does not cover them)."""
    out = {}
    for init in graph.initializer:
        out[init.name] = [int(d) for d in init.dims]
    return out


def estimate_pool_bytes(model, shapes=None):
    """The graph pool's peak demand in bytes, computed from the graph itself.

    The pool holds three things at once:

      * every float initialiser, uploaded once and kept for the whole run
        (``Runner.values``);
      * the live activations of the moment with the largest fan-out, from an
        ONNX liveness walk (first use .. last use);
      * the packed NHWC weight panels, built on a conv's first use and cached
        in ``Runner._panels`` (never freed), plus the transient copy of a
        weight while its panel is built.

    It is a plan-time *estimate*: good to within a few percent on the selftest
    models, and the pool grows by another segment if it was short (Pool.alloc),
    so an under-estimate costs a reallocation, not a failure.  The point is
    that the pool is sized from the model instead of from the arena with an
    8 MiB floor (G10).
    """
    from . import coverage

    graph = model.graph
    if shapes is None:
        shapes, _ = coverage._shape_map(model)
    init = _init_shape_map(graph)

    def shp(name):
        s = shapes.get(name)
        return s if s else init.get(name)

    perma = 0
    for init_t in graph.initializer:
        if _is_float_init(init_t):
            perma += _align(_numel_proto(init_t) * 4) + 64

    # conv panels + the transient weight upload they need
    panels = 0
    transient = 0
    for n in graph.node:
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        ws = shp(n.input[1])
        if not ws or len(ws) != 4 or any(d <= 0 for d in ws):
            continue
        ks, st, pd, dl, gp = _conv_attrs(n)
        cout, cin = int(ws[0]), int(ws[1]) * gp
        kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(ws[2]), int(ws[3]))
        cpad, opad = _align16(cin), _align16(cout)
        panels += opad * (_align16(cpad * kh * kw) + 16) * 4
        transient = max(transient, _align(opad * cpad * kh * kw * 4) + 64)

    first, last = {}, {}
    for idx, n in enumerate(graph.node):
        for t in n.input:
            if t:
                first.setdefault(t, idx)
        for t in n.output:
            if t:
                last[t] = idx

    live = 0
    peak = perma
    sizes = {}
    for idx, n in enumerate(graph.node):
        for t in n.input:                       # free what dies here
            if t and last.get(t) == idx:
                live -= sizes.pop(t, 0)
        extra = 0
        for t in n.output:
            if not t or t in init:
                continue        # an initialiser is already in `perma`
            b = _tensor_bytes(shapes.get(t), nhwc=(n.op_type == "Conv"))
            if not b:
                continue
            sizes[t] = b
            live += b
        if n.op_type == "Conv" and len(n.input) > 1:
            extra = transient
        peak = max(peak, perma + live + extra)
    return int(peak + panels)


def estimate_pool_floor(model, shapes=None):
    """The bytes the pool *cannot* do without, whatever the liveness says.

    Weights + packed conv panels + the single largest tensor: a model below
    this cannot run in the arena, and this is the number to refuse on.  The
    full estimate (estimate_pool_bytes) is deliberately conservative - it
    assumes every tensor the ONNX graph keeps syntactically live is resident,
    while the runner releases some earlier - so it sizes the pool but does not
    decide refusal on its own.
    """
    from . import coverage

    graph = model.graph
    if shapes is None:
        shapes, _ = coverage._shape_map(model)
    init = _init_shape_map(graph)

    perma = 0
    for t in graph.initializer:
        if _is_float_init(t):
            perma += _align(_numel_proto(t) * 4) + 64
    panels = 0
    biggest = 0
    for n in graph.node:
        out = n.output[0] if n.output else None
        b = _tensor_bytes(shapes.get(out))
        if b > biggest:
            biggest = b
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        ws = shapes.get(n.input[1]) or init.get(n.input[1])
        if not ws or len(ws) != 4 or any(d <= 0 for d in ws):
            continue
        ks, st, pd, dl, gp = _conv_attrs(n)
        cin = int(ws[1]) * gp
        cout = int(ws[0])
        kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(ws[2]), int(ws[3]))
        panels += (_align16(cout) *
                   (_align16(_align16(cin) * kh * kw) + 16) * 4)
    return int(perma + panels + biggest)


def estimate_scratch_bytes(model, shapes=None):
    """Bytes the engine needs *beside* the pool, from the largest node.

    libvc4infer takes its own scratch from the arena above the pool: the NHWC
    conv zero-pads the whole input image into scratch and builds an im2col
    panel there (measured: ~6 MiB at 160x160, ~23 MiB at 320x320 - which is
    where VC4_POOL_RESERVE_MB=24 came from), and an elementwise op with an
    unaligned element count pads its operands in scratch.  Both are reset per
    op, so the requirement is the largest single op, not a sum.
    """
    from . import coverage

    graph = model.graph
    if shapes is None:
        shapes, _ = coverage._shape_map(model)
    init = _init_shape_map(graph)

    def shp(name):
        s = shapes.get(name)
        return s if s else init.get(name)

    conv = 0
    for n in graph.node:
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        xs = shp(n.input[0])
        ws = shp(n.input[1])
        if not xs or not ws or len(xs) != 4 or len(ws) != 4:
            continue
        if any(d <= 0 for d in xs + ws):
            continue
        ks, st, pd, dl, gp = _conv_attrs(n)
        nb, cin, h, w = (int(d) for d in xs)
        cout = int(ws[0])
        kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(ws[2]), int(ws[3]))
        cpad, opad = _align16(cin), _align16(cout)
        ho = (h + 2 * pd[0] - (dl[0] * (kh - 1) + 1)) // st[0] + 1
        wo = (w + 2 * pd[1] - (dl[1] * (kw - 1) + 1)) // st[1] + 1
        if ho <= 0 or wo <= 0:
            continue
        pad_img = nb * cpad * (h + 2 * pd[0]) * (w + 2 * pd[1]) * 4 + 64
        nk = _align16(cpad * kh * kw) + 16
        panel = nb * ho * _align16(wo) * nk * 4
        conv = max(conv, pad_img + panel)
    # An elementwise op whose element count is not a multiple of its group
    # rounds the operand up in scratch (up to three tensor-sized buffers), and
    # every op resets scratch first, so the need is a per-op maximum, not a sum.
    tail = 0
    for n in graph.node:
        t = n.output[0] if n.output else None
        sh = shp(t)
        if not sh or any(int(d) <= 0 for d in sh):
            continue
        n_el = 1
        for d in sh:
            n_el *= int(d)
        if n_el % 256:
            tail = max(tail, 3 * _align(((n_el + 255) // 256) * 256 * 4))
    return int(max(conv, tail) + (256 << 10))


class Pool:
    """A bump/free allocator over one or more engine allocations.

    libvc4infer has no free(), so the runner carves every activation out of
    arena buffers it takes from the engine and recycles them by ONNX liveness.
    ``Buf`` objects are plain (off, bytes) pairs, so a sub-range of the pool is
    a perfectly good engine operand.

    The pool used to be one allocation of ``max(arena - reserve, 8 MiB)``: a
    small client arena therefore produced a fixed 8 MiB pool and then failed
    inside the run with ``graph pool exhausted ... of 8388608 B``, which reads
    like "model too big" (G10).  Now the first segment is sized from the graph
    (estimate_pool_bytes), a short estimate grows by asking the engine for
    another segment, and a request the arena cannot serve names the number
    needed and the number available.
    """

    MIN_SEG = 1 << 20

    def __init__(self, eng, nbytes, reserve=0):
        self.eng = eng
        self.reserve = reserve
        self.segs = []                  # [[base, size, bump], ...]
        self.free = []                  # [(off, size)] engine-global offsets
        self.high = 0                   # high-water of used_bytes
        if not self._add_segment(max(int(nbytes), self.MIN_SEG)):
            raise self._exhausted(int(nbytes))

    # -- segments ----------------------------------------------------------
    def _add_segment(self, want):
        """Ask the engine for another pool segment; 0 if the arena is full."""
        n = _align(max(int(want), 1 << 16))
        while n >= (1 << 16):
            try:
                b = self.eng.alloc(n)
                self.segs.append([int(b.off), n, 0])
                return n
            except vc4infer.Error:
                n //= 2
        return 0

    def _exhausted(self, n):
        return UnsupportedNode(
            "graph pool exhausted: one node needs %d B (%.1f MiB), the pool is "
            "%s in %d segment(s) and the client arena is %s%s: start vc4d with "
            "a larger --arena-mb (or set VC4INFER_ARENA_MB), or lower "
            "VC4_POOL_RESERVE_MB"
            % (n, n / 2 ** 20, _mib(self.size), len(self.segs),
               _mib(self.eng.arena_bytes),
               (" (%.1f MiB held for conv scratch)" % (self.reserve / 2 ** 20))
               if self.reserve else ""))

    def _ensure(self, n):
        for seg in self.segs:
            if seg[2] + n <= seg[1]:
                return seg
        # short estimate: grow by at least the request (halved inside
        # _add_segment if the arena cannot give that much at once)
        self._add_segment(max(n, 4 << 20))
        seg = self.segs[-1]
        if len(self.segs) and seg[1] < n:
            raise self._exhausted(n)
        return seg

    # -- allocation --------------------------------------------------------
    def alloc(self, nbytes):
        n = _align(nbytes)
        for i, (off, sz) in enumerate(self.free):
            if sz >= n:
                if sz == n:
                    self.free.pop(i)
                else:
                    self.free[i] = (off + n, sz - n)
                return vc4infer.Buf(off, int(nbytes))
        seg = self._ensure(n)
        off = seg[0] + seg[2]
        seg[2] += n
        used = self.used_bytes
        if used > self.high:
            self.high = used
        return vc4infer.Buf(off, int(nbytes))

    def release(self, b):
        if b is None:
            return
        off = int(b.off)
        inside = any(seg[0] <= off < seg[0] + seg[1] for seg in self.segs)
        if not inside:
            return
        n = _align(int(b.bytes))
        lo, hi = off, off + n
        merged = []
        for o, sz in self.free:                 # sorted, so one pass coalesces
            if o + sz >= lo and o <= hi:
                lo = min(lo, o)
                hi = max(hi, o + sz)
            else:
                merged.append((o, sz))
        merged.append((lo, hi - lo))
        merged.sort()
        self.free = merged

    def reset(self):
        for seg in self.segs:
            seg[2] = 0
        self.free = []
        self.high = 0

    @property
    def size(self):
        return sum(seg[1] for seg in self.segs)

    @property
    def segments(self):
        return len(self.segs)

    @property
    def used_bytes(self):
        used = sum(seg[2] for seg in self.segs)
        return used - sum(sz for _, sz in self.free)


class Tensor:
    """An fp32 tensor in the arena.

    ``shape`` is always the ONNX (NCHW) logical shape; ``layout`` says how the
    bytes are ordered.  For NHWC the buffer is [n][H][pitch][C] and ``pitch``
    may exceed W (the conv's compact output grid).
    """
    __slots__ = ("buf", "shape", "layout", "pitch")

    def __init__(self, buf, shape, layout="nchw", pitch=None):
        self.buf = buf
        self.shape = tuple(int(s) for s in shape)
        self.layout = layout
        self.pitch = int(pitch) if pitch is not None else \
            (self.shape[3] if layout == "nhwc" else 0)

    @property
    def logical(self):
        return int(np.prod(self.shape))

    @property
    def size(self):
        """floats actually stored in the buffer"""
        if self.layout == "nhwc":
            n, c, h, w = self.shape
            return n * h * self.pitch * c
        return self.logical

    def __repr__(self):
        return "Tensor(%s,%s%s)" % (self.shape, self.layout,
                                    ",pitch=%d" % self.pitch
                                    if self.layout == "nhwc" else "")


def _flat_index(shape, axes, rngs):
    """Flat source index per flat destination index (integer metadata).

    ``axes[k]`` is the source axis that destination axis ``k`` walks and
    ``rngs[k]`` the index values it walks.  A Slice passes identity axes with
    per-axis ranges, a Transpose passes the permutation with full ranges, so
    one helper builds both tables.  This is integer index arithmetic on
    metadata; no tensor value is read or computed here.
    """
    shape = [int(d) for d in shape]
    rank = len(shape)
    stride = [1] * rank
    for k in range(rank - 2, -1, -1):
        stride[k] = stride[k + 1] * shape[k + 1]
    out_shape = [int(len(r)) for r in rngs]
    tab = np.zeros(out_shape, np.int64)
    for k, (ax, r) in enumerate(zip(axes, rngs)):
        # a one-element range still contributes its start offset, so only an
        # *empty* range is skipped (the reshape needs a known size)
        if len(r) > 0 and stride[int(ax)]:
            b = np.asarray(r, np.int64).reshape(
                [(-1 if a == k else 1) for a in range(rank)])
            tab += b * stride[int(ax)]
    return tab.ravel(), out_shape


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


class Runner:
    def __init__(self, model_path, eng, mode="strict", host_policy="device",
                 passes=None):
        if mode not in ("strict", "hybrid"):
            raise ValueError("mode must be 'strict' or 'hybrid'")
        if host_policy not in ("device", "structural", "any"):
            raise ValueError("host_policy must be 'device', 'structural' "
                             "or 'any'")
        self.path = model_path
        self.eng = eng
        self.mode = mode
        self.host_policy = host_policy
        self.model = onnx.load(model_path)
        # Graph optimisation (docs/OPTIMIZER.md).  Every pass is off by
        # default: `passes` / $VC4_OPT selects them, and an empty spec leaves
        # the model, the node indices and the SiLU rewrite exactly as they
        # were.  The rewrite runs before the pool plan, so the plan is sized
        # from the graph that will actually execute.
        self.optimise_report = None
        spec = passes if passes is not None else os.environ.get("VC4_OPT", "")
        if str(spec or "").strip() not in ("", "none"):
            from . import optimizer
            self.optimise_report = optimizer.optimise(self.model, passes=spec)
            if self.optimise_report.changed:
                self.model = self.optimise_report.model
        self.graph = self.model.graph
        self.values = {}
        self.host = {}          # host copy of every initializer / Constant
        self.float_inputs = []
        self.profile = []
        self.backend_report = []      # last run: index/name/op/backend/reason
        self.boundary_uploads = 0     # host numpy -> arena (this run)
        self.boundary_downloads = 0   # arena -> host numpy (this run)
        self.host_bytes = 0           # bytes of model data moved by the ARM
        self.host_sites = []          # [{node,op,site,bytes,count}, ...]
        self._host_site_acc = {}
        self._node_ctx = None
        self._count_host = False
        self._folded = {}             # name -> numpy, plan-time resolved
        self._folded_idx = set()
        self._prepared = False
        self._host_eval = None
        self._coverage_report = None
        self._backends = None
        self._panels = {}       # packed NHWC weight panels, by node name
        # One pool per engine: libvc4infer cannot free, and running several
        # Runner objects on one Engine at the same time is not supported.
        #
        # The pool is sized from *this model's* peak (weights + longest-lived
        # activations + cached conv panels), not from the arena with a fixed
        # reserve and an 8 MiB floor: the floor produced `graph pool exhausted
        # ... of 8388608 B` for a model that was nowhere near too big (G10).
        # The engine's own scratch (the conv padded image + im2col panel,
        # elementwise workspaces) still has to fit beside the pool, so it is
        # estimated too; VC4_POOL_RESERVE_MB overrides that estimate.
        need, self.pool_plan = self._plan_pool_bytes()
        pool = getattr(eng, "_vc4_pool", None)
        if pool is None or pool.size < need:
            pool = Pool(eng, need)
            eng._vc4_pool = pool
        else:
            pool.reset()
        self.pool = pool
        self._garbage = []      # per-node temporaries
        self._consts = set(i.name for i in self.graph.initializer)
        self._last_use = {}

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
        self._find_silu()
        self._collect_fused()

    # ---------------------------------------------------- optimiser markers
    def _collect_fused(self):
        """Read the optimiser's provenance records into dispatch tables.

        A node marked ``fused`` is skipped (its work happens in ``into``); a
        node that carries a rewrite rule (``silu``, ``hardswish``, ``affine``,
        ``activation``) finds what to compute here instead of re-deriving it.
        A model with no markers yields empty tables and the executor behaves
        exactly as it did before the optimiser existed.

        The record lives on one ``doc_string`` line (see
        ``vc4infer.optimizer.ir``); a malformed line is ignored rather than
        failing a run, because a run must never depend on debug metadata.
        """
        self._fused = {}
        self._fused_skip = {}
        try:
            from .optimizer.ir import Node as _IRNode
        except Exception:                       # pragma: no cover
            self._skip = set(getattr(self, "_silu_skip", None) or set())
            self._skip_reason = {}
            return
        for i, n in enumerate(self.graph.node):
            try:
                rec = _IRNode(n, i).fused
            except Exception:                   # pragma: no cover
                continue
            if not rec:
                continue
            rules = rec.get("rules") or {}
            if rules:
                self._fused[i] = rules
            if "fused" in rules:
                self._fused_skip[i] = rules["fused"].get("by", "fused")
        self._skip = set(getattr(self, "_silu_skip", None) or set())
        self._skip_reason = {}
        for i in self._skip:
            self._skip_reason[i] = "fused into the consumer Mul as one SiLU launch"
        for i, by in self._fused_skip.items():
            self._skip.add(i)
            self._skip_reason[i] = "fused into %s (%s)" % (
                (self._fused.get(i) or {}).get("fused", {}).get("into", "?"),
                by)
        # A fused *activation* still owns its output name: downstream nodes and
        # a graph output read the activation's result, not the conv's.  The
        # conv returns the activated buffer, so at the skipped node's turn the
        # name is moved onto that buffer (one owner, so the pool cannot free a
        # reused block twice).
        self._alias = {}
        by_name = {}
        for i, n in enumerate(self.graph.node):
            # the pass records Node.name, which falls back to "<Op>_<index>"
            by_name[n.name or ("%s_%s" % (n.op_type, i))] = i
        for i, by in self._fused_skip.items():
            if by != "activation":
                continue
            into = (self._fused.get(i) or {}).get("fused", {}).get("into")
            j = by_name.get(into)
            if j is None:
                continue
            dst = self.graph.node[i].output[0] if self.graph.node[i].output else ""
            src = self.graph.node[j].output[0] if self.graph.node[j].output else ""
            if dst and src:
                self._alias[i] = (dst, src)

    def _apply_alias(self, idx):
        """Move a fused activation's output name onto the producing buffer."""
        pair = self._alias.get(idx)
        if not pair:
            return
        dst, src = pair
        if src in self.values and dst not in self.values:
            self.values[dst] = self.values.pop(src)

    # ------------------------------------------------------ host-tier policy
    def _fold_metadata(self):
        """Resolve metadata nodes at plan time (device policy).

        ``Shape``/``Size``/``Constant``/``ConstantOfShape`` and the integer
        shape arithmetic feeding them are evaluated once, here, with the real
        static shapes.  Their outputs enter ``self._consts`` and are never
        executed at runtime.  Returns the set of folded node indices.
        """
        if self.mode != "hybrid" or self.host_policy != "device":
            return set()
        if self._prepared:
            return self._folded_idx
        rep = self.coverage_report()
        folded, known = coverage.fold_metadata(self.model, rep)
        self._folded_idx = set(folded)
        for idx in self._folded_idx:
            for name in self.graph.node[idx].output:
                if not name or name not in known:
                    continue
                arr = np.asarray(known[name])
                if arr.dtype.kind == "f":
                    if name not in self.values:
                        self.values[name] = self._constant(name, arr)
                else:
                    self.host[name] = arr
                    self.values[name] = arr
                self._folded[name] = arr
                self._consts.add(name)
        self._prepared = True
        return self._folded_idx

    def prepare(self):
        """Plan-time work required by the policy (idempotent)."""
        self._fold_metadata()
        self.backend_plan()

    def coverage_report(self):
        """Cached static analysis (tiers included) of this model."""
        if self._coverage_report is None:
            self._coverage_report = coverage.analyse_model(self.model)
        return self._coverage_report

    def backend_plan(self):
        """Static per-node backend verdict (no execution)."""
        if self._backends is None:
            rep = self.coverage_report()
            self._backends = coverage.backend_plan(
                rep, mode=self.mode, host_policy=self.host_policy,
                silu_skip=self._silu_skip, silu_src=self._silu_src,
                folded=self._folded_idx)
        return self._backends

    def host_compute_plan(self):
        """Static nodes that would perform runtime host compute."""
        return [n for n in self.backend_plan()
                if n["backend"] in ("host-meta", "host-control",
                                    "host-unwritten")]

    def host_debt(self):
        return [n for n in self.backend_plan()
                if n["backend"] == "host-unwritten"]

    def check_policy(self):
        """Raise for node tiers the current policy refuses.

        device: every runtime node must be VC4; host-unwritten raises
        ``HostNotAllowed``, If/Loop/Scan raise ``NeedsDeviceDispatch``, and a
        metadata node that could not be folded at plan time also raises.
        structural: host-unwritten raises ``HostNotAllowed``.
        unsupported-graph raises under every policy.
        """
        if self.mode != "hybrid":
            return
        self.prepare()
        bad = [n for n in self.backend_plan()
               if n["backend"] == "unsupported-graph"]
        if bad:
            n = bad[0]
            raise UnsupportedNode(n["name"], n["op"], n["reason"])
        if self.host_policy == "device":
            ctl = [n for n in self.backend_plan()
                   if n["backend"] == "host-control"]
            if ctl:
                n = ctl[0]
                raise NeedsDeviceDispatch(
                    n["name"], n["op"],
                    "If/Loop/Scan branch execution on the device")
            for n in self.backend_plan():
                b = n["backend"]
                if b == "host-unwritten":
                    raise HostNotAllowed(n["name"], n["op"], n["reason"],
                                         policy="device")
                if b == "host-meta":
                    raise HostNotAllowed(
                        n["name"], n["op"],
                        "metadata node could not be resolved at plan time "
                        "(%s)" % n["reason"], policy="device")
            return
        if self.host_policy == "structural":
            debt = self.host_debt()
            if debt:
                n = debt[0]
                raise HostNotAllowed(n["name"], n["op"], n["reason"],
                                     policy="structural")

    # ------------------------------------------------- host byte accounting
    def _charge_host(self, site, nbytes):
        if not self._count_host or self._node_ctx is None:
            return
        idx, name, op = self._node_ctx
        key = (idx, name, op, site)
        acc = self._host_site_acc
        e = acc.get(key)
        if e is None:
            e = acc[key] = {"index": idx, "name": name, "op": op,
                            "site": site, "bytes": 0, "count": 0}
        e["bytes"] += int(nbytes)
        e["count"] += 1

    def host_copy_report(self):
        return sorted(self.host_sites, key=lambda e: -e["bytes"])

    def host_compute_nodes(self):
        """Nodes that performed runtime CPU compute in the last run."""
        return [n for n in self.backend_report
                if n["backend"] in ("host-meta", "host-control",
                                    "host-unwritten")]

    # ------------------------------------------------- host execution bits
    def _node_name(self, idx, node):
        return node.name or "%s_%d" % (node.op_type, idx)

    def _promote(self, node):
        """Upload host float intermediates as VC4 tensors (boundary copies)."""
        n = 0
        for name in node.input:
            if not name:
                continue
            v = self.values.get(name)
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                if v.dtype != np.float32:
                    v = v.astype(np.float32)
                self.values[name] = self.upload(np.ascontiguousarray(v))
                n += 1
        self.boundary_uploads += n
        return n

    def _host_feeds(self, node):
        """numpy inputs for one node, including subgraph lexical captures."""
        if self._host_eval is None:
            self._host_eval = HostEvaluator(self.model)
        names = [i for i in node.input if i]
        for extra in self._host_eval.outer_refs(node):
            if extra not in names:
                names.append(extra)
        feeds = {}
        for name in names:
            v = self.values.get(name)
            if v is None:
                raise HostNodeError(node.name, node.op_type,
                                    "input %r has no value at this point"
                                    % name)
            if isinstance(v, Tensor):
                feeds[name] = self.nchw(v)
                self.boundary_downloads += 1
            else:
                feeds[name] = np.asarray(v)
        return feeds

    def _host_node(self, node, idx):
        """Run one node in the numpy reference evaluator; store its outputs."""
        feeds = self._host_feeds(node)
        try:
            outs = self._host_eval.run(node, feeds, idx)
        except HostNodeError:
            raise
        except Exception as e:                       # noqa: BLE001
            raise HostNodeError(node.name, node.op_type,
                                "%s: %s" % (type(e).__name__, e)) from e
        real = [i for i, o in enumerate(node.output) if o]
        for k, pos in enumerate(real):
            arr = outs[k]
            if arr.dtype.kind == "f" and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            self.values[node.output[pos]] = arr
        return len(real)

    def _fallback_allowed(self, tier, node, err):
        if self.mode != "hybrid":
            return False
        if str(err.detail).startswith("engine error") or \
                "graph pool exhausted" in str(err.detail):
            return False
        if self.host_policy == "device":
            return False
        if self.host_policy == "any":
            return True
        return tier == "host-meta"

    # ------------------------------------------------------------- rewrites
    def _plan_pool_bytes(self):
        """Bytes the graph pool must be able to hold, from the graph itself.

        Returns (need, plan).  `need` is the size of the first pool segment:
        the model's estimated peak (weights + live activations + cached conv
        panels), clamped to what the arena can give beside the engine's own
        scratch.  A model that cannot fit at all is refused *here*, at plan
        time, with both numbers - that is the difference between
        "graph pool exhausted ... of 8388608 B" (which looks like a model
        problem) and "needs 30.2 MiB, arena is 24.0 MiB".
        """
        from . import coverage as _cov
        shapes, _types = _cov._shape_map(self.model)
        peak = estimate_pool_bytes(self.model, shapes)
        floor = estimate_pool_floor(self.model, shapes)
        scratch = estimate_scratch_bytes(self.model, shapes)
        env = os.environ.get("VC4_POOL_RESERVE_MB")
        if env is not None:
            try:
                scratch = int(env) << 20
            except ValueError:
                pass
        arena = int(self.eng.arena_bytes)
        avail = max(arena - int(scratch), 0)
        if floor > avail:
            raise UnsupportedNode(
                "graph pool needs at least %s (weights + packed conv panels + "
                "the largest tensor) and this model's peak is about %s, but "
                "the client arena is %s with %s reserved for conv/elementwise "
                "scratch: start vc4d with a larger --arena-mb (or set "
                "VC4INFER_ARENA_MB) - about %s would do - or lower "
                "VC4_POOL_RESERVE_MB"
                % (_mib(floor), _mib(peak), _mib(arena), _mib(scratch),
                   _mib(floor + scratch + (8 << 20))))
        need = min(int(peak), avail) if avail else int(peak)
        need = max(need, 1 << 20)
        plan = {"pool_bytes": need, "estimate_bytes": int(peak),
                "floor_bytes": int(floor), "scratch_bytes": int(scratch),
                "arena_bytes": arena, "clamped": need < int(peak)}
        return need, plan

    def _find_silu(self):
        """Collapse ``Mul(x, Sigmoid(x))`` into one fused SiLU.

        YOLOv8 writes SiLU as two ONNX nodes, so the runner paid two
        elementwise calls (two passes over the tensor, two daemon round trips
        and two sets of launches) for every activation: 949 of the 1145
        launches at 160x160 were these pairs.  The engine already has
        ``VC4_OP_SILU`` (the same 2^f polynomial and 3x Newton reciprocal as
        sigmoid, with the final multiply folded into the epilogue), so the
        graph is rewritten here: the ``Sigmoid`` node is skipped and its
        consumer ``Mul`` becomes a single silu launch.

        The rewrite is only taken when the sigmoid output has exactly one
        consumer, i.e. when the fused kernel is guaranteed to compute the same
        value the graph would have (``sigmoid(x)*x``), and never for a graph
        output.
        """
        # keyed by node *index*: node names are optional in ONNX and empty
        # names must not make two nodes alias
        self._silu_skip = set()             # indices never dispatched
        self._silu_src = {}                 # Mul index -> x tensor name
        self._node_idx = None
        # A recorded rewrite wins over the implicit scan below: the optimiser
        # pass states which Mul is the SiLU and which Sigmoid it absorbed, and
        # that record survives VC4_NO_SILU_FUSE (which only disables the
        # executor's own habit, so the pass can be measured against it).
        try:
            from .optimizer.ir import Node as _IRNode
            producer = {}
            for i, n in enumerate(self.graph.node):
                for o in n.output:
                    if o:
                        producer[o] = i
            for i, n in enumerate(self.graph.node):
                rec = _IRNode(n, i).fused or {}
                f = (rec.get("rules") or {}).get("silu")
                if not f or not f.get("input"):
                    continue
                self._silu_src[i] = f["input"]
                for nm in f.get("replaces") or []:
                    j = producer.get(nm)
                    if j is not None:
                        self._silu_skip.add(j)
        except Exception:                   # pragma: no cover
            pass
        if os.environ.get("VC4_NO_SILU_FUSE"):
            return                          # A/B switch (no implicit rewrite)
        producer = {}
        for i, n in enumerate(self.graph.node):
            for o in n.output:
                if o:
                    producer[o] = (i, n)
        consumers = {}
        for n in self.graph.node:
            for inp in n.input:
                if inp:
                    consumers.setdefault(inp, []).append(n)
        gouts = set(o.name for o in self.graph.output)
        for idx, n in enumerate(self.graph.node):
            if n.op_type != "Mul" or len(n.input) != 2:
                continue
            hit = None
            for sig_name, x_name in ((n.input[0], n.input[1]),
                                     (n.input[1], n.input[0])):
                prod = producer.get(sig_name)
                if prod is None or prod[1].op_type != "Sigmoid":
                    continue
                pn = prod[1]
                if len(pn.input) != 1 or pn.input[0] != x_name:
                    continue
                if len(consumers.get(sig_name, [])) != 1 or sig_name in gouts:
                    continue
                hit = (prod[0], x_name)
            if hit:
                self._silu_skip.add(hit[0])
                self._silu_src[idx] = hit[1]

    # ---------------------------------------------------------------- buffers
    def _alloc(self, n):
        return self.pool.alloc(int(n) * 4 + 64)

    def _tmp(self, t):
        """Register a tensor as dead once this node finishes."""
        if t is not None:
            self._garbage.append(t)
        return t

    def _free(self, t):
        if isinstance(t, Tensor):
            self.pool.release(t.buf)
        elif isinstance(t, vc4infer.Buf):
            self.pool.release(t)

    def upload(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._charge_host("upload", arr.nbytes)
        buf = self._alloc(arr.size)
        self.eng.upload(arr.ravel(), buf)
        return Tensor(buf, arr.shape)

    def upload_nhwc(self, arr, shape, pitch=None):
        """arr is [n][H][W][C] contiguous -> Tensor with logical NCHW shape."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._charge_host("upload_nhwc", arr.nbytes)
        n, h, w, c = arr.shape
        buf = self.pool.alloc(int(arr.size) * 4)
        self.eng.upload(arr.ravel(), buf)
        return Tensor(buf, shape, "nhwc", pitch if pitch is not None else w)

    def upload_pitched(self, arr, shape, pitch):
        """arr is [n][H][pitch][C] -> Tensor whose logical width is shape[3]."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._charge_host("upload_pitched", arr.nbytes)
        buf = self.pool.alloc(int(arr.size) * 4)
        self.eng.upload(arr.ravel(), buf)
        return Tensor(buf, shape, "nhwc", pitch)

    def _compact(self, t):
        """NHWC tensor with no storage padding (pitch == W)."""
        if t.layout != "nhwc" or t.pitch == t.shape[3]:
            return t
        return self.upload_nhwc(self.download_nhwc(t), t.shape)

    def download(self, t):
        self._charge_host("download", t.size * 4)
        x = self.eng.download(t.buf, t.size * 4)
        return np.asarray(x, dtype=np.float32).reshape(t.shape).copy()

    def download_nhwc(self, t):
        """-> [n][H][W][C] numpy of the *logical* extent (pitch removed)."""
        self._charge_host("download_nhwc", t.size * 4)
        n, c, h, w = t.shape
        x = self.eng.download(t.buf, t.size * 4)
        a = np.asarray(x, dtype=np.float32).reshape(n, h, t.pitch, c)
        return np.ascontiguousarray(a[:, :, :w, :])

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

    def _hostconst(self, name):
        return self.host.get(name)

    # ------------------------------------------------------------- utilities
    # ------------------------------------------------- device layout copies
    def _x(self, src, dst, rows, vpr, srow, drow, svec, dvec, slane,
           nlane=16, nblk=1, sbstep=64, dbstep=64, units=None, depth=16):
        """One vc4_copy2d_x call.  src/dst are arena Bufs (views are fine)."""
        if units is None:
            units = 4 if vpr % 4 == 0 else (2 if vpr % 2 == 0 else 1)
        if depth < 16 and nlane == 16:
            # a partial store moves `depth` words, so only that many lanes may
            # be read (the rest are clamped to the vector base)
            nlane = depth
        self.eng.copy2d_x(src, dst, rows, vpr, srow, drow, svec, dvec, slane,
                          nlane, nblk, sbstep, dbstep, units, depth)

    def _buf_at(self, t, off):
        """t's arena buffer (a Tensor or a Buf), shifted by `off` bytes."""
        b = t.buf if isinstance(t, Tensor) else t
        return vc4infer.Buf(b.off + int(off), b.bytes - int(off))

    def _runs(self, src, dst, rows, n, srow, drow, svec=64, dvec=64,
              slane=4, units=4):
        """`rows` runs of `n` contiguous floats: whole vectors, then the
        ~4-vector tail that ends at the end of each run (identical overlap),
        then a k-word partial store when n is not a multiple of 16 floats."""
        nv = n // 16
        rem = n % 16
        if rem:
            off = (n - rem) * 4
            self._x(self._buf_at(src, off), self._buf_at(dst, off), rows, 1,
                    srow, drow, 64, 64, slane, 16, 1, 64, 64, 1, rem)
        if nv >= 4:
            full = (nv // 4) * 4
            self._x(src, dst, rows, full, srow, drow, svec, dvec, slane, 16,
                    1, 64, 64, 4, 16)
            if nv % 4:
                k = (nv - 4) * 16 * 4
                self._x(self._buf_at(src, k), self._buf_at(dst, k), rows, 4,
                        srow, drow, svec, dvec, slane, 16, 1, 64, 64, 4, 16)
        elif nv:
            self._x(src, dst, rows, nv, srow, drow, svec, dvec, slane, 16, 1,
                    64, 64, 1, 16)

    def _gather_flat(self, x, tab, shape):
        """out[flat i] = x[tab[i]]: one ``vc4_gather`` launch.

        ``tab`` is a flat *word* index per flat output element, i.e. integer
        index arithmetic on metadata (never on tensor data).  The engine's
        gather reads arena *byte* offsets, so the table becomes ``x``'s base
        offset plus 4*index and is uploaded as raw int32 -- a byte copy of
        metadata, the same class of host traffic as an index map in
        ``_dispatch_geom.word_gather``.
        """
        out = self._alloc_t(shape)
        n = int(tab.size)
        if n == 0:
            return out
        ent = np.asarray(tab, np.int64) * 4 + int(x.buf.off)
        tb = self.pool.alloc(n * 4)
        self._charge_host("upload", n * 4)
        self.eng.upload(ent.astype(np.int32).tobytes(), tb)
        self.eng.gather(tb, out.buf, n, 0.0)
        self.pool.release(tb)
        return out

    def _alloc_t(self, shape, layout="nchw", pitch=None, extra=0):
        """An activation buffer from the pool, released after this node
        unless it is one of the node's outputs."""
        n = int(np.prod(shape)) if layout != "nhwc" else \
            int(shape[0] * shape[2] * (pitch or shape[3]) * shape[1])
        return self._tmp(Tensor(self.pool.alloc((n + extra) * 4 + 64), shape,
                                layout, pitch))

    def nchw(self, t):
        """Host numpy array of a Tensor in ONNX (NCHW) order."""
        if t.layout == "nhwc":
            a = self.download_nhwc(t)                       # n,h,w,c
            return np.ascontiguousarray(a.transpose(0, 3, 1, 2))
        return self.download(t)

    def to_nhwc(self, t, cpad=None):
        """Tensor in NHWC layout, done on the QPUs.

        An NHWC tensor whose channel count already matches is returned as is
        (the old code downloaded it first and then threw the copy away, 6.5 MB
        per YOLOv8n run).  An NCHW source becomes NHWC with one copy2d_x call:
        the destination vector is one pixel's 16 channels and the 16 lanes are
        the channels, so lane stride = H*W*4 for the source plane and the
        lanes past Cin store zero (the conv's channel padding).
        """
        if len(t.shape) != 4:
            raise UnsupportedNode("to_nhwc: rank %d" % len(t.shape))
        n, c, h, w = t.shape
        cp = _align16(c) if cpad is None else int(cpad)
        if t.layout == "nhwc" and cp == t.shape[1] and w == t.shape[3]:
            return t
        if t.layout == "nhwc" and cp == c:
            return t
        out = self._alloc_t((n, cp, h, w), "nhwc", w)
        if t.layout == "nhwc":
            # same layout, different channel padding: not reachable from the
            # conv path (it always pads to 16 before the first call)
            raise UnsupportedNode("to_nhwc: NHWC re-pad C%d -> %d"
                                  % (c, cp))
        units = 4 if w % 4 == 0 else (2 if w % 2 == 0 else 1)
        # One call moves one 16-channel block of every pixel: the 16 lanes are
        # the channels (source lane stride H*W*4, NCHW planes), a row is a
        # pixel row of W pixels, and the vector stride inside a pixel is the
        # destination pixel pitch.  C > 16 needs one call per channel block:
        # copy2d_x's lane mask is shared by every block of a row, so a single
        # call cannot zero-pad a partial last block (nlane <= 16 always).
        for i in range(n):
            for k0 in range(0, c, 16):
                nlan = min(16, c - k0)
                self._x(self._buf_at(t, (i * c + k0) * h * w * 4),
                        self._buf_at(out, i * h * w * cp * 4 + k0 * 4),
                        h, w, w * 4, w * cp * 4, 4, cp * 4, h * w * 4,
                        nlane=nlan, units=units)
        return out

    def _prepad(self, x, pads, value=0.0):
        """pads = [top, left, bottom, right]; returns a new NCHW tensor."""
        a = self.download(x)
        p = np.pad(a, ((0, 0), (0, 0), (pads[0], pads[2]),
                       (pads[1], pads[3])), constant_values=value)
        return self.upload(p)

    # ------------------------------------------------------------ conv (NHWC)
    def _conv_nhwc(self, node, x, wa, ba, kh, kw, sh, pad, dil=1, groups=1,
                   act=0):
        """NHWC conv, native: batch, dilation and grouped channel blocks.

        x is [N][H][pitch][Cin] and wa is the ONNX [Cout][Cin/g][kh][kw].
        groups == 1 uses one im2col + GEMM; groups > 1 runs one im2col + GEMM
        per group but pads the input once and reuses the activation panel, and
        each group's GEMM writes straight into its channel range of the single
        output (the GEMM C stride is Cout).  No host interleave, no wasted
        MACs.
        """
        N = x.shape[0]
        Cin = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        Cout = wa.shape[0]
        if sh * Cin * 4 > 8191:
            raise UnsupportedNode("Conv %s: stride*Cin*4 = %d B > 8191 B"
                                  % (node.name, sh * Cin * 4))
        ho = (H + 2 * pad - dil * (kh - 1) - 1) // sh + 1
        wo = (W + 2 * pad - dil * (kw - 1) - 1) // sh + 1
        if ho <= 0 or wo <= 0:
            raise UnsupportedNode("Conv %s: empty %dx%d output" % (node.name,
                                                                    ho, wo))
        op = _align16(wo)

        if groups == 1:
            # Cout is zero-padded to 16 (the GEMM N): the DFL head's 1-channel
            # conv computes 16 and the caller slices back.
            Cout = _align16(Cout)
            if Cin % 16 or (Cin * kh * kw) % 16:
                raise UnsupportedNode("Conv %s: NHWC needs Cin,K %% 16 "
                                      "(%d,%d)" % (node.name, Cin,
                                                   Cin * kh * kw))
            out = self.pool.alloc(N * ho * op * Cout * 4 + 64)
            if wa.shape[1] != Cin:
                w2 = np.zeros((wa.shape[0], Cin) + wa.shape[2:], np.float32)
                w2[:, :wa.shape[1]] = wa
                wa = w2
            if wa.shape[0] != Cout:
                w3 = np.zeros((Cout,) + wa.shape[1:], np.float32)
                w3[:wa.shape[0]] = wa
                wa = w3
            if ba.size != Cout:
                b2 = np.zeros(Cout, np.float32)
                b2[:ba.size] = ba
                ba = b2
            key = ("panel", node.name, Cin, Cout, kh, kw, dil)
            wp = self._panels.get(key)
            if wp is None:
                wt = self.upload(np.ascontiguousarray(wa, np.float32))
                bb = self.upload(np.ascontiguousarray(ba, np.float32))
                wp = self.pool.alloc(self.eng.conv_nhwc_wpanel_bytes(
                    Cin, Cout, kh, kw, True))
                self.eng.conv_nhwc_pack_w(wt.buf, Cin, Cout, kh, kw, bb.buf,
                                          wp)
                self._panels[key] = wp
                self._free(wt)
                self._free(bb)
            h, w, o = self.eng.conv_nhwc_dil(
                x.buf, N, Cin, H, W, x.pitch, wp, Cout, kh, kw, sh, pad, dil,
                out, act=act)
            return Tensor(out, (N, Cout, h, w), "nhwc", pitch=o)

        # grouped conv: per-group panels, one shared padded input/panel
        if Cin % groups or Cout % groups:
            raise UnsupportedNode("Conv %s: groups %d does not divide %d/%d"
                                  % (node.name, groups, Cin, Cout))
        cg, og = Cin // groups, Cout // groups
        if cg % 16 or og % 16 or (cg * kh * kw) % 16:
            raise UnsupportedNode(
                "Conv %s: grouped NHWC needs per-group Cin,Cout,K %% 16 "
                "(%d,%d,%d)" % (node.name, cg, og, cg * kh * kw))
        if wa.shape[1] != cg:
            raise UnsupportedNode("Conv %s: weight has %d channels per group, "
                                  "expected %d" % (node.name, wa.shape[1], cg))
        out = self.pool.alloc(N * ho * op * Cout * 4 + 64)
        panels = []
        for g in range(groups):
            key = ("panelg", node.name, g, cg, og, kh, kw, dil)
            wp = self._panels.get(key)
            if wp is None:
                wg = np.ascontiguousarray(wa[g * og:(g + 1) * og], np.float32)
                bg = np.ascontiguousarray(ba[g * og:(g + 1) * og], np.float32)
                wt = self.upload(wg)
                bb = self.upload(bg)
                wp = self.pool.alloc(self.eng.conv_nhwc_wpanel_bytes(
                    cg, og, kh, kw, True))
                self.eng.conv_nhwc_pack_w(wt.buf, cg, og, kh, kw, bb.buf, wp)
                self._panels[key] = wp
                self._free(wt)
                self._free(bb)
            panels.append(wp)
        h, w, o = self.eng.conv_nhwc_group(
            x.buf, N, Cin, H, W, x.pitch, panels, Cout, groups, kh, kw, sh,
            pad, dil, out, act=act)
        return Tensor(out, (N, Cout, h, w), "nhwc", pitch=o)

    # the optimiser's activation name -> the kernel's vc4_act
    _ACT_OF = {"relu": vc4infer.ACT_RELU}

    def _act_of(self, node):
        """The activation the optimiser folded into this node, if any."""
        f = self._fused.get(self._node_idx, {}).get("activation")
        return self._ACT_OF.get((f or {}).get("act")) if f else None

    def _epilogue(self, node, t):
        """Apply an activation here because the kernel could not.

        The conv and GEMM kernels fold the activation into the store when the
        shape takes the path that has an epilogue (`_conv_nhwc`'s `act`), so
        this is the fallback: the legacy NCHW conv, or a graph whose activation
        the kernel refused.  It costs the elementwise launch the fusion was
        meant to remove, which is why the plan only credits the fold where the
        executor takes it (`plan.py`, `epilogue_ok`).
        """
        f = self._fused.get(self._node_idx, {}).get("activation")
        if not f or not isinstance(t, Tensor):
            return t
        act = f.get("act")
        out = self.pool.alloc(t.size * 4 + 64)
        if act == "relu":
            self.eng.elementwise(vc4infer.OP_RELU, t.buf, out, t.size)
        elif act == "relu6":
            self.eng.unary(vc4infer.UOP_CLIP, t.buf, out, t.size,
                           0.0, 6.0, 0.0)
        else:
            self.pool.release(out)
            return t
        self._tmp(t)
        return Tensor(out, t.shape, t.layout, t.pitch)

    def _conv(self, node):
        act = self._act_of(node)
        t, folded = self._conv_impl(node, act)
        return t if folded else self._epilogue(node, t)

    def _conv_impl(self, node, act=None):
        x = self._float(node.input[0])
        w = self._float(node.input[1])
        b = self._float(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if len(x.shape) != 4:
            raise UnsupportedNode("Conv %s: expected NCHW input" % node.name)
        N = int(x.shape[0])

        group = int(_attr(node, "group", 1))
        ks = _attr(node, "kernel_shape", list(w.shape[2:]))
        kh, kw = int(ks[0]), int(ks[1])
        strides = _attr(node, "strides", [1, 1])
        sh, sw = int(strides[0]), int(strides[1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        dil = _attr(node, "dilations", [1, 1])
        if int(dil[0]) != int(dil[1]):
            raise UnsupportedNode("Conv %s: non-square dilation" % node.name)
        dilv = int(dil[0])
        if sh != sw:
            raise UnsupportedNode("Conv %s: non-square stride" % node.name)

        Cin = x.shape[1]
        Cout = w.shape[0]
        wa = self._hostconst(node.input[1])
        wa = (np.asarray(wa, np.float32) if wa is not None
              else self.download(w))
        if b is not None:
            ba = self._hostconst(node.input[2]) if len(node.input) > 2 else None
            ba = (np.asarray(ba, np.float32) if ba is not None
                  else self.download(b))
        else:
            ba = np.zeros(Cout, np.float32)

        grouped = (group > 1 and group != Cin and Cin % group == 0
                   and Cout % group == 0)
        depthwise = (group == Cin)
        if depthwise and Cout != Cin:
            raise UnsupportedNode("Conv %s: depthwise Cout %d != Cin %d "
                                  "(channel multiplier is not implemented)"
                                  % (node.name, Cout, Cin))
        native_group = (grouped and (Cin // group) % 16 == 0
                        and (Cout // group) % 16 == 0)
        nhwc_ok = (kh in (1, 3, 5) and kw in (1, 3, 5) and sh in (1, 2)
                   and list(pads) == [pads[0]] * 4 and 1 <= dilv <= 8
                   and (group == 1 or native_group or depthwise))
        if nhwc_ok:
            xn = self.to_nhwc(x, _align16(Cin))
            if xn is not x:
                self._tmp(xn)
            if depthwise:
                # The dedicated depthwise kernel is not shipped (it wedges
                # the QPU); the verified block-diagonal dense expansion is
                # used instead.  It costs Cin x the real MACs.
                if kh != kw:
                    raise UnsupportedNode("Conv %s: non-square depthwise"
                                          % node.name)
                cp = xn.shape[1]
                dense = np.zeros((cp, cp, kh, kw), np.float32)
                for c in range(Cout):
                    dense[c, c] = wa[c, 0]
                wa = dense
                Cout = cp
            t = self._conv_nhwc(node, xn, wa, ba, kh, kw, sh, int(pads[0]),
                                dilv, 1 if depthwise else group,
                                act if act is not None else 0)
            if t.shape[1] != w.shape[0]:
                # Cout is padded to 16 for the GEMM; take the real channels
                # back on the QPUs (the DFL head's 1-channel conv)
                t = self._slice_ch(t, 0, int(w.shape[0]))
            return t, act is not None

        # ---------------- legacy NCHW path (non-square kernels, etc.) -------
        if N != 1:
            raise UnsupportedNode("Conv %s: batch %d needs the NHWC path"
                                  % (node.name, N))
        H, W = x.shape[2], x.shape[3]
        if dilv != 1:
            raise UnsupportedNode("Conv %s: NCHW path has no dilation"
                                  % node.name)
        if list(pads) == [pads[0]] * 4:
            p = pads[0]
            if sh == 1:
                hp, wp_ = H + 2 * p, _align16(W + 2 * p)
            elif sh == 2 and kh == 3:
                hp = (H + 1) // 2 + 2
                wp_ = _align16(((W + 1) // 2) + 2)
            else:
                hp = wp_ = 1
            if hp * wp_ * 4 > 8191:
                raise UnsupportedNode(
                    "Conv %s: padded plane %dx%d = %d B > 8191 B "
                    "(NCHW im2col VDR pitch limit)" % (node.name, hp, wp_,
                                                       hp * wp_ * 4))
        if group not in (1, Cin):
            raise UnsupportedNode("Conv %s: groups=%d (only 1, depthwise or "
                                  "a 16-aligned group block)" % (node.name,
                                                                 group))
        cpad = _align16(Cin)
        opad = _align16(Cout)
        if x.layout == "nhwc":
            x = self.upload(self.nchw(x))
        xp = self._pad_ch(x, cpad)
        if wa.shape[1] != cpad and group == 1:
            wa = np.pad(wa, ((0, 0), (0, cpad - wa.shape[1]), (0, 0), (0, 0)))
        if group == Cin:
            if sh != 1:
                raise UnsupportedNode("Conv %s: stride-2 depthwise" % node.name)
            dense = np.zeros((opad, cpad, kh, kw), np.float32)
            for c in range(Cout):
                dense[c, c] = wa[c, 0]
            wa = dense
        if wa.shape[0] != opad:
            wa = np.pad(wa, ((0, opad - Cout), (0, 0), (0, 0), (0, 0)))
        if ba.size != opad:
            ba = np.pad(ba, (0, opad - ba.size))
        wb = self.upload(np.ascontiguousarray(wa))
        bb = self.upload(ba)
        if list(pads) == [pads[0]] * 4:
            pad = int(pads[0])
        else:
            xp = self._prepad(xp, pads)
            pad = 0
        if sh == 1:
            t = self._conv_core(xp, wb, bb, kh, kw, 1, pad, node.name)
        elif sh == 2 and kh == 1 and kw == 1 and pad == 0:
            a = self.download(xp)
            ph = np.ascontiguousarray(a[:, :, 0::2, 0::2])
            t = self._conv_core(self.upload(ph), wb, bb, 1, 1, 1, 0, node.name)
        elif sh == 2 and kh == 3 and kw == 3 and pad == 1:
            t = self._conv_stride2_3x3(xp, wa, bb, cpad, Cout, opad, node.name)
        else:
            raise UnsupportedNode("Conv %s: stride %d k=%dx%d pads=%s"
                                  % (node.name, sh, kh, kw, pads))
        if t.shape[1] != Cout:
            a = self.download(t)[:, :Cout]
            t = self.upload(np.ascontiguousarray(a))
        return t, False

    def _pad_ch(self, x, cpad):
        a = self.download(x)
        if a.shape[1] == cpad:
            return self.upload(a)
        p = np.zeros((a.shape[0], cpad) + a.shape[2:], np.float32)
        p[:, :a.shape[1]] = a
        return self.upload(p)

    def _conv_core(self, x, w, b, kh, kw, stride, pad, name="Conv"):
        C = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        hp = H + 2 * pad
        wp = _align16(W + 2 * pad)
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

    def _conv_stride2_3x3(self, xp, wa, bb, cpad, Cout, opad, name="Conv"):
        """NCHW conv 3x3 stride 2 pad 1 == 4 phase images + one stride-1 conv."""
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

    # ------------------------------------------------------- elementwise ops
    def _same_layout(self, a, b):
        """b in a's layout *and storage pitch* (host byte gather if needed).

        Two NHWC tensors with the same logical shape can still have different
        storage pitches (a conv output is align16(wo) wide, a host gather is
        wo wide); a flat elementwise op over them would mix the wrong pixels,
        so such operands are compacted first.
        """
        if a.layout != b.layout or a.shape != b.shape:
            if a.layout == "nhwc":
                b = self.upload_nhwc(self.download_nhwc(b), b.shape)
            else:
                b = self.upload(self.nchw(b))
        if a.layout == "nhwc" and b.pitch != a.pitch:
            b = self._compact(b)
            self._tmp(b)
        return b

    def _expand_to(self, name, ref, layout="nchw"):
        """Broadcast a graph constant to `ref`'s logical shape.

        This is plan-time constant folding, not activation movement: the
        expanded operand depends only on the constant and the static shape, so
        it is built once and kept like a packed weight panel (a per-run
        expansion was 8.4 kB/run of host bytes on YOLOv8n's `Mul_2`).
        """
        c = self._hostconst(name)
        if c is None:
            raise UnsupportedNode("broadcast operand %s is not a constant"
                                  % name)
        key = ("expand", name, tuple(ref.shape), layout)
        t = self._panels.get(key)
        if t is not None:
            return t
        b = np.ascontiguousarray(
            np.broadcast_to(np.asarray(c, np.float32), ref.shape))
        if layout == "nhwc":
            t = self.upload_nhwc(
                np.ascontiguousarray(b.transpose(0, 2, 3, 1)), ref.shape)
        else:
            t = self.upload(b)
        self._panels[key] = t
        return t

    def _bin(self, a, b, op):
        b = self._same_layout(a, b)
        if a.layout == "nhwc" and b.pitch != a.pitch:
            a = self._compact(a)                      # b could not shrink
            self._tmp(a)
        out = self.pool.alloc(a.size * 4 + 64)
        self.eng.elementwise(op, a.buf, out, a.size, b=b.buf)
        return Tensor(out, a.shape, a.layout, a.pitch)

    def _binary(self, node, op):
        a = self._float(node.input[0])
        b = self._float(node.input[1])
        if a.shape != b.shape:
            which = 1 if self._hostconst(node.input[1]) is not None else 0
            if which == 1:
                b = self._expand_to(node.input[1], a, a.layout)
            else:
                a = self._expand_to(node.input[0], b, b.layout)
            # the expansion is a cached plan-time constant: it must not be
            # registered as a per-node temporary, or the pool would recycle
            # (and clobber) it for the next node and for the next run
        return self._bin(a, b, op)

    def _scaled(self, t, k):
        """t * k on the GPU; k is a graph constant (folded here)."""
        n = t.size
        out = self.pool.alloc(n * 4 + 64)
        c = self.pool.alloc(n * 4 + 64)
        self.eng.upload(np.full(n, np.float32(k)), c)
        self.eng.elementwise(vc4infer.OP_MUL, t.buf, out, n, b=c)
        self._free(c)
        return Tensor(out, t.shape, t.layout, t.pitch)

    def _addconst(self, t, arr):
        """t + broadcastable constant on the GPU (host expand of the const)."""
        if t.layout == "nhwc" and t.pitch != t.shape[3]:
            t = self._compact(t)
            self._tmp(t)
        b = np.ascontiguousarray(
            np.broadcast_to(np.asarray(arr, np.float32), t.shape))
        if t.layout == "nhwc":
            b = np.ascontiguousarray(b.transpose(0, 2, 3, 1))
            b = np.broadcast_to(b, (t.shape[0], t.shape[2], t.shape[3],
                                    t.shape[1]))
        cb = self.upload(np.ascontiguousarray(b))
        out = self.pool.alloc(t.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_ADD, t.buf, out, t.size, b=cb.buf)
        self._free(cb)
        return Tensor(out, t.shape, t.layout, t.pitch)

    def _relu(self, node):
        x = self._float(node.input[0])
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_RELU, x.buf, out, x.size)
        return Tensor(out, x.shape, x.layout, x.pitch)

    def _sigmoid(self, node):
        x = self._float(node.input[0])
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_SIGMOID, x.buf, out, x.size)
        return Tensor(out, x.shape, x.layout, x.pitch)

    def _silu(self, x):
        """x * sigmoid(x) in one QPU kernel (the fused SiLU rewrite)."""
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_SILU, x.buf, out, x.size)
        return Tensor(out, x.shape, x.layout, x.pitch)

    # ------------------------------------------------- generic QPU unary ops
    def _unary_uop(self, node, uop, p0=0.0, p1=0.0, p2=0.0, limit=None):
        """One QPU unary kernel over every stored element (layout-agnostic,
        exactly like Relu: the op touches all bytes it is handed, padding
        included, and keeps the operand's layout and pitch).

        ``limit`` is the op's documented absolute input range.  The kernels
        for Sin/Cos use a two-term Cody-Waite reduction whose accuracy is
        bounded (|x| <= 1e5, measured 4.8e-6); outside it the node raises
        instead of returning a number the kernel cannot stand behind.
        """
        x = self._float(node.input[0])
        if limit is not None:
            a = self.download_nhwc(x) if x.layout == "nhwc" else self.download(x)
            m = float(np.max(np.abs(a))) if a.size else 0.0
            if not np.isfinite(m) or m > limit:
                raise UnsupportedNode(
                    node.name, node.op_type,
                    "input range |x|max=%r exceeds the kernel's documented "
                    "range |x| <= %g" % (m, limit))
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.unary(uop, x.buf, out, x.size, p0, p1, p2)
        return Tensor(out, x.shape, x.layout, x.pitch)

    def _uop_on(self, x, uop, p0=0.0, p1=0.0, p2=0.0):
        """Apply one QPU unary kernel to an existing tensor."""
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.unary(uop, x.buf, out, x.size, p0, p1, p2)
        return Tensor(out, x.shape, x.layout, x.pitch)

    def _abs(self, node):
        return self._unary_uop(node, vc4infer.UOP_ABS)

    def _neg(self, node):
        return self._unary_uop(node, vc4infer.UOP_NEG)

    def _sign(self, node):
        return self._unary_uop(node, vc4infer.UOP_SIGN)

    def _leakyrelu(self, node):
        return self._unary_uop(node, vc4infer.UOP_LEAKYRELU,
                               float(_attr(node, "alpha", 0.01)))

    def _hardsigmoid(self, node):
        return self._unary_uop(node, vc4infer.UOP_HARDSIGMOID,
                               float(_attr(node, "alpha", 0.2)),
                               float(_attr(node, "beta", 0.5)))

    def _hardswish(self, node):
        # ONNX HardSwish (opset 14) has no attributes: x * clip(x/6 + 0.5, 0, 1)
        return self._unary_uop(node, vc4infer.UOP_HARDSWISH, 1.0 / 6.0, 0.5)

    def _clip(self, node):
        """Clip with attribute (opset < 11) or constant-tensor limits."""
        lo = _attr(node, "min", None)
        hi = _attr(node, "max", None)
        for idx, which in ((1, "min"), (2, "max")):
            if len(node.input) <= idx or not node.input[idx]:
                continue
            name = node.input[idx]
            c = self._hostconst(name)
            if c is None:
                raise UnsupportedNode(
                    node.name, "Clip",
                    "runtime tensor limit %s: the QPU kernel takes scalar "
                    "limits in uniform registers (vc4_select3 clip3 exists but "
                    "is not element-verified yet)" % name)
            arr = np.asarray(c)
            if arr.size != 1:
                raise UnsupportedNode(
                    node.name, "Clip",
                    "limit %s has %d elements; scalar limits only" %
                    (name, arr.size))
            if which == "min":
                lo = float(arr.reshape(-1)[0])
            else:
                hi = float(arr.reshape(-1)[0])
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if not (lo <= hi):
            raise UnsupportedNode(node.name, "Clip", "min %g > max %g" % (lo, hi))
        return self._unary_uop(node, vc4infer.UOP_CLIP, lo, hi)

    def _tanh(self, node):
        """Native QPU tanh: sign(x)*(1-t)/(1+t), t = exp(-2|x|).

        Element-verified at 1.2e-7 over the whole line (docs/ONNX.md sec. 12).
        """
        return self._unary_uop(node, vc4infer.UOP_TANH)

    # the rest of the ONNX unary family, one kernel each
    def _exp(self, node):
        return self._unary_uop(node, vc4infer.UOP_EXP)

    def _domain(self, node, x, lo=None, hi=None, desc=""):
        """Refuse inputs the kernel's documented domain excludes."""
        a = self.download_nhwc(x) if x.layout == "nhwc" else self.download(x)
        if a.size:
            vmin, vmax = float(np.min(a)), float(np.max(a))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or \
                    (lo is not None and vmin < lo) or \
                    (hi is not None and vmax > hi):
                raise UnsupportedNode(
                    node.name, node.op_type,
                    "%s: input range [%r, %r] is outside the kernel's "
                    "verified domain %s" % (node.op_type, vmin, vmax, desc))

    def _log(self, node):
        x = self._float(node.input[0])
        self._domain(node, x, lo=2.0 ** -126, desc="2^-126 <= x < inf")
        return self._unary_uop(node, vc4infer.UOP_LOG)

    def _sqrt(self, node):
        x = self._float(node.input[0])
        self._domain(node, x, lo=0.0, desc="0 <= x < inf")
        return self._unary_uop(node, vc4infer.UOP_SQRT)

    def _reciprocal(self, node):
        return self._unary_uop(node, vc4infer.UOP_RECIPROCAL)

    def _erf(self, node):
        return self._unary_uop(node, vc4infer.UOP_ERF)

    def _sin(self, node):
        return self._unary_uop(node, vc4infer.UOP_SIN, limit=1e5)

    def _cos(self, node):
        return self._unary_uop(node, vc4infer.UOP_COS, limit=1e5)

    def _floor(self, node):
        return self._unary_uop(node, vc4infer.UOP_FLOOR)

    def _ceil(self, node):
        return self._unary_uop(node, vc4infer.UOP_CEIL)

    def _round(self, node):
        # ONNX Round: nearest integer, ties to even
        return self._unary_uop(node, vc4infer.UOP_ROUND)

    def _softplus(self, node):
        return self._unary_uop(node, vc4infer.UOP_SOFTPLUS)

    def _softsign(self, node):
        return self._unary_uop(node, vc4infer.UOP_SOFTSIGN)

    def _elu(self, node):
        return self._unary_uop(node, vc4infer.UOP_ELU,
                               float(_attr(node, "alpha", 1.0)))

    def _selu(self, node):
        return self._unary_uop(node, vc4infer.UOP_SELU,
                               float(_attr(node, "alpha", 1.6732631921768188)),
                               float(_attr(node, "gamma", 1.0507010221481323)))

    def _gelu(self, node):
        """ONNX Gelu; the default/erf form is composed from verified kernels.

        erf saturates at 1-2^-24 with the truncating FPU, so the fused body
        needs a saturation select and does not fit the code slot at nv4; the
        composition (scale, native Erf, add 1, halve, multiply by x) is exact
        to the same tolerance and uses four verified launches.
        """
        approx = _attr(node, "approximate", "none") or "none"
        if approx == "tanh":
            return self._unary_uop(node, vc4infer.UOP_GELU)
        if approx != "none":
            raise UnsupportedNode(node.name, "Gelu",
                                  "approximate=%r (want none or tanh)" % approx)
        x = self._float(node.input[0])
        u = self._scaled(x, 0.7071067811865476)
        self._tmp(u)
        e = self._uop_on(u, vc4infer.UOP_ERF)
        self._tmp(e)
        s = self._addconst(e, 1.0)
        self._tmp(s)
        h = self._scaled(s, 0.5)
        self._tmp(h)
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_MUL, x.buf, out, x.size, b=h.buf)
        return Tensor(out, x.shape, x.layout, x.pitch)

    def _mish(self, node):
        """mish(x) = x*tanh(softplus(x)), composed from verified kernels."""
        x = self._float(node.input[0])
        sp = self._uop_on(x, vc4infer.UOP_SOFTPLUS)
        self._tmp(sp)
        th = self._uop_on(sp, vc4infer.UOP_TANH)
        self._tmp(th)
        out = self.pool.alloc(x.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_MUL, x.buf, out, x.size, b=th.buf)
        return Tensor(out, x.shape, x.layout, x.pitch)

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
        n, s = int(ks[0]), int(strides[0])
        if ks[0] != ks[1] or strides[0] != strides[1]:
            raise UnsupportedNode("MaxPool %s: non-square window" % node.name)
        if x.layout == "nhwc":
            if s == 1 and n in (3, 5) and list(pads) == [(n - 1) // 2] * 4:
                pad = (n - 1) // 2
                # ONNX MaxPool pads with -inf, the NHWC kernel with zero, so
                # the border is materialised on the QPUs: fill the whole
                # padded plane, then copy the interior back in
                if pad:
                    h2, w2 = H + 2 * pad, W + 2 * pad
                    op2 = _align16(w2)
                    y = self._alloc_t((N, C, h2, w2), "nhwc", op2)
                    nfl = N * h2 * op2 * C
                    self.eng.fill2d(y.buf, nfl // 16 // 4, 4, 256, 64,
                                    -np.inf)
                    vpr = W * C // 16
                    units = 4 if vpr % 4 == 0 else (2 if vpr % 2 == 0 else 1)
                    for i in range(N):
                        self._x(
                            self._buf_at(x, i * H * x.pitch * C * 4),
                            self._buf_at(y, (i * h2 * op2 + pad * op2 + pad)
                                         * C * 4),
                            H, vpr, x.pitch * C * 4, op2 * C * 4, 64, 64, 4,
                            16, 1, 64, 64, units, 16)
                    x = y
                    self._tmp(x)
                    H, W = h2, w2
                    pad = 0
                ho, wo = H - n + 1, W - n + 1
                op = _align16(wo)
                out = self.pool.alloc(N * ho * op * C * 4 + 64)
                h, w, o = self.eng.maxpool_nhwc(x.buf, N, C, H, W, x.pitch,
                                                n, pad, out)
                return Tensor(out, (N, C, h, w), "nhwc", pitch=o)
            x = self.upload(self.nchw(x))
            N, C, H, W = x.shape
        pad = int(pads[0])
        if list(pads) != [0, 0, 0, 0] and list(pads) != [pad] * 4:
            x = self._prepad(x, pads, value=-np.inf)
            pad = 0
        ho = (H + 2 * pad - n) // s + 1
        wo = (W + 2 * pad - n) // s + 1
        op = _align16(wo)
        hop = (ho + 3) & ~3
        if N == 1:
            out = self._alloc(C * hop * op)
            h, w = self.eng.maxpooln(x.buf, out, C, H, W, n, s, pad)
            t = Tensor(out, (1, C, hop, op))
            a = self.download(t)[:, :, :h, :w]
            return self.upload(np.ascontiguousarray(a))
        # vc4_maxpooln is a single-image op: one launch per image, still all
        # arithmetic on the QPU.
        out = self._alloc(N * C * hop * op)
        parts = []
        for k in range(N):
            sb = vc4infer.Buf(x.buf.off + k * C * H * W * 4, C * H * W * 4)
            ob = vc4infer.Buf(out.off + k * C * hop * op * 4, C * hop * op * 4)
            h, w = self.eng.maxpooln(sb, ob, C, H, W, n, s, pad)
            t = Tensor(ob, (1, C, hop, op))
            parts.append(self.download(t)[0, :, :h, :w])
        return self.upload(np.ascontiguousarray(np.stack(parts)))

    def _concat(self, node):
        axis = int(_attr(node, "axis", 0))
        ts = [self._float(n) for n in node.input if n]
        shape = ts[0].shape
        rank = len(shape)
        if axis < 0:
            axis += rank
        if rank == 4 and axis == 1:
            if all(t.layout == "nhwc" for t in ts):
                # pitched channel concat on the QPUs: one call per input, the
                # destination vector is one pixel's 16-channel block, `nblk`
                # walks the >16-channel inputs
                for t in ts:
                    if t.shape[0] != shape[0] or t.shape[2:] != shape[2:]:
                        raise UnsupportedNode("Concat %s: shape mismatch"
                                              % node.name)
                n, _, h, w = shape
                ct = sum(t.shape[1] for t in ts)
                out = self._alloc_t((n, ct, h, w), "nhwc", w)
                coff = 0
                for t in ts:
                    cs = t.shape[1]
                    if cs % 16 == 0:
                        units = 4 if w % 4 == 0 else \
                            (2 if w % 2 == 0 else 1)
                        nb, dp, nl = cs // 16, 16, 16
                    else:
                        units, nb, dp, nl = 1, 1, cs, cs
                    for i in range(n):
                        self._x(
                            self._buf_at(t, i * h * t.pitch * cs * 4),
                            self._buf_at(out, (i * h * w * ct + coff) * 4),
                            h, w, t.pitch * cs * 4, w * ct * 4, cs * 4,
                            ct * 4, 4, nl, nb, 64, 64, units, dp)
                    coff += cs
                return out
            a = np.concatenate([self.nchw(t) for t in ts], axis=1)
            return self.upload(np.ascontiguousarray(a))
        if rank == 3 and axis in (1, 2):
            # channel-major concat: `rows` runs of L floats each, placed at a
            # channel offset (axis 1) or a position offset inside every row
            # (axis 2), all on the QPUs
            n = shape[0]
            cl = sum(t.shape[axis] for t in ts)
            out_shape = list(shape)
            out_shape[axis] = cl
            out = self._alloc_t(out_shape)
            coff = loff = 0
            for t in ts:
                ci, li = t.shape[1], t.shape[2]
                if axis == 1:
                    base = coff * li * 4
                    rows, run = ci, li
                else:
                    base = loff * 4
                    rows, run = ci, li
                for i in range(n):
                    self._runs(self._buf_at(t, i * ci * li * 4),
                               self._buf_at(out, i * cl * (out_shape[2]) * 4
                                            + base),
                               rows, run, li * 4, out_shape[2] * 4, 64, 64,
                               4)
                if axis == 1:
                    coff += ci
                else:
                    loff += li
            return out
        raise UnsupportedNode("Concat %s: axis %d of rank %d"
                              % (node.name, axis, rank))

    def _split(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", 0))
        rank = len(x.shape)
        if axis < 0:
            axis += rank
        if rank != 4 or axis != 1:
            raise UnsupportedNode("Split %s: axis %d of rank %d"
                                  % (node.name, axis, rank))
        nout = len([o for o in node.output if o])
        sizes = None
        if len(node.input) > 1 and node.input[1]:
            v = self.host.get(node.input[1], self.values.get(node.input[1]))
            sizes = [int(s) for s in np.asarray(v).ravel()]
        if sizes is None:
            if x.shape[1] % nout:
                raise UnsupportedNode("Split %s: %d channels into %d"
                                      % (node.name, x.shape[1], nout))
            sizes = [x.shape[1] // nout] * nout
        if sum(sizes) != x.shape[1]:
            raise UnsupportedNode("Split %s: sizes %s vs C=%d"
                                  % (node.name, sizes, x.shape[1]))
        n, c, h, w = x.shape
        outs = []
        if x.layout == "nhwc":
            # channel-strided copy on the QPUs, keeping the source pitch
            off = 0
            for sz in sizes:
                out = self._alloc_t((n, sz, h, w), "nhwc", x.pitch)
                if sz % 16 == 0:
                    units = 4 if w % 4 == 0 else (2 if w % 2 == 0 else 1)
                    nb, dp, nl = sz // 16, 16, 16
                else:
                    units, nb, dp, nl = 1, 1, sz, sz
                for i in range(n):
                    self._x(self._buf_at(x, (i * h * x.pitch * c + off) * 4),
                            self._buf_at(out, i * h * x.pitch * sz * 4),
                            h, w, x.pitch * c * 4, x.pitch * sz * 4,
                            c * 4, sz * 4, 4, nl, nb, 64, 64, units, dp)
                outs.append(out)
                off += sz
        else:
            cl = int(np.prod(x.shape[2:]))
            off = 0
            for sz in sizes:
                out = self._alloc_t((n, sz) + tuple(x.shape[2:]))
                for i in range(n):
                    self._runs(self._buf_at(x, (i * c + off) * cl * 4),
                               self._buf_at(out, i * sz * cl * 4), sz, cl,
                               cl * 4, cl * 4)
                outs.append(out)
                off += sz
        return outs

    # Resize-19 attribute domain.  The coordinate mapping is one source
    # coordinate per output index (float64, computed the way the ONNX reference
    # evaluator computes it), and the nearest index or the bilinear tap pair
    # and weight both derive from that one table.
    _RESIZE_COORD_MODES = ("half_pixel", "half_pixel_symmetric",
                           "pytorch_half_pixel", "asymmetric", "align_corners",
                           "tf_crop_and_resize")
    _RESIZE_NEAREST_MODES = ("round_prefer_floor", "round_prefer_ceil",
                             "floor", "ceil")

    @staticmethod
    def _resize_src(n_in, n_out, scale, coord, roi=None):
        """Source coordinate of each output index along one axis (Resize-19).

        roi is (start, end) in input-pixel units, only used by
        tf_crop_and_resize.
        """
        s = np.arange(n_out, dtype=np.float64)
        if coord == "half_pixel":
            return (s + 0.5) / scale - 0.5
        if coord == "half_pixel_symmetric":
            # keeps the centre of the implicit ROI on the centre of the canvas
            adjustment = n_out / (scale * n_in)
            offset = (n_in / 2.0) * (1.0 - adjustment)
            return offset + (s + 0.5) / scale - 0.5
        if coord == "pytorch_half_pixel":
            if n_out > 1:
                return (s + 0.5) / scale - 0.5
            return np.zeros(n_out)
        if coord == "asymmetric":
            return s / scale
        if coord == "align_corners":
            if n_out > 1:
                return s * (n_in - 1) / (n_out - 1)
            return np.zeros(n_out)
        if coord == "tf_crop_and_resize":
            r0, r1 = float(roi[0]), float(roi[1])
            if n_out > 1:
                src = s * (r1 - r0) * (n_in - 1) / (n_out - 1)
            else:
                src = np.full(n_out, (r1 - r0) * (n_in - 1) / 2.0)
            return src + r0 * (n_in - 1)
        raise UnsupportedNode("Resize: coordinate_transformation_mode=%s"
                              % coord)

    @staticmethod
    def _resize_nearest(src, n_in, nearest_mode):
        """The one nearest source index per output position (Resize-19)."""
        if nearest_mode == "round_prefer_floor":
            idx = np.ceil(src - 0.5)        # exact .5 -> the lower index
        elif nearest_mode == "round_prefer_ceil":
            idx = np.floor(src + 0.5)       # exact .5 -> the higher index
        elif nearest_mode == "floor":
            idx = np.floor(src)
        elif nearest_mode == "ceil":
            idx = np.ceil(src)
        else:
            raise UnsupportedNode("Resize: nearest_mode=%s" % nearest_mode)
        return np.clip(idx, 0, n_in - 1).astype(np.int64)

    def _resize(self, node):
        x = self._float(node.input[0])
        if len(x.shape) != 4:
            raise UnsupportedNode("Resize %s: rank %d (4-D only)"
                                  % (node.name, len(x.shape)))
        mode = _attr(node, "mode", "nearest")
        coord = _attr(node, "coordinate_transformation_mode", "half_pixel")
        nmode = _attr(node, "nearest_mode", "round_prefer_floor")
        if mode not in ("nearest", "linear"):
            raise UnsupportedNode("Resize %s: mode=%s (nearest, linear)"
                                  % (node.name, mode))
        if coord not in self._RESIZE_COORD_MODES:
            raise UnsupportedNode("Resize %s: coord=%s" % (node.name, coord))
        if int(_attr(node, "antialias", 0)):
            raise UnsupportedNode("Resize %s: antialias=1" % node.name)
        if _attr(node, "axes", None):
            raise UnsupportedNode("Resize %s: axes=%s (spatial H/W only)"
                                  % (node.name, list(_attr(node, "axes"))))
        kap = _attr(node, "keep_aspect_ratio_policy", "stretch")
        if kap != "stretch":
            raise UnsupportedNode("Resize %s: keep_aspect_ratio_policy=%s"
                                  % (node.name, kap))
        # nearest_mode is only read by mode=nearest; linear ignores it.
        if mode == "nearest" and nmode not in self._RESIZE_NEAREST_MODES:
            raise UnsupportedNode("Resize %s: nearest_mode=%s"
                                  % (node.name, nmode))

        def operand(i):
            """A host-visible input (roi / scales / sizes)."""
            if len(node.input) <= i or not node.input[i]:
                return None
            nm = node.input[i]
            return self.host.get(nm, self.values.get(nm))

        roi = operand(1)
        scales = operand(2)
        sizes = operand(3)
        if roi is not None:
            roi = np.asarray(roi, dtype=np.float64).ravel()
        if scales is not None:
            scales = np.asarray(scales, dtype=np.float64).ravel()
        if sizes is not None:
            sizes = np.asarray(sizes, dtype=np.int64).ravel()

        nhwc = x.layout == "nhwc"
        n_in, c_in, h, w = (int(v) for v in x.shape)
        if sizes is not None:
            if sizes.size != 4:
                raise UnsupportedNode("Resize %s: sizes rank %d (4 only)"
                                      % (node.name, sizes.size))
            if int(sizes[0]) != n_in or int(sizes[1]) != c_in:
                raise UnsupportedNode("Resize %s: sizes scale N/C (%d,%d -> "
                                      "%d,%d); only H/W is resized"
                                      % (node.name, n_in, c_in,
                                         int(sizes[0]), int(sizes[1])))
            Ho, Wo = int(sizes[2]), int(sizes[3])
            scy, scx = Ho / float(h), Wo / float(w)
        elif scales is not None:
            if scales.size != 4:
                raise UnsupportedNode("Resize %s: scales rank %d (4 only)"
                                      % (node.name, scales.size))
            if float(scales[0]) != 1.0 or float(scales[1]) != 1.0:
                raise UnsupportedNode("Resize %s: scales %g/%g on N/C; only "
                                      "H/W is resized"
                                      % (node.name, float(scales[0]),
                                         float(scales[1])))
            scy, scx = float(scales[2]), float(scales[3])
            # ONNX: output_dim = floor(input_dim * scale)
            Ho = int(np.floor(h * scy))
            Wo = int(np.floor(w * scx))
        else:
            raise UnsupportedNode("Resize %s: no scales/sizes" % node.name)
        if Ho <= 0 or Wo <= 0:
            raise UnsupportedNode("Resize %s: empty output %dx%d"
                                  % (node.name, Ho, Wo))

        ry = rx = None
        if coord == "tf_crop_and_resize":
            if roi is None or roi.size != 8:
                raise UnsupportedNode("Resize %s: tf_crop_and_resize needs a "
                                      "roi input of 2*rank values" % node.name)
            ry = (roi[2], roi[6])
            rx = (roi[3], roi[7])
        sy = self._resize_src(h, Ho, scy, coord, ry)
        sx = self._resize_src(w, Wo, scx, coord, rx)
        if coord == "tf_crop_and_resize":
            if sy.min() < 0 or sy.max() > h - 1 or sx.min() < 0 or \
                    sx.max() > w - 1:
                raise UnsupportedNode("Resize %s: tf_crop_and_resize window "
                                      "leaves the input (extrapolation)"
                                      % node.name)

        if mode == "nearest":
            iy = self._resize_nearest(sy, h, nmode)
            ix = self._resize_nearest(sx, w, nmode)
            if nhwc and Ho == 2 * h and Wo == 2 * w and \
                    np.array_equal(iy, np.repeat(np.arange(h), 2)) and \
                    np.array_equal(ix, np.repeat(np.arange(w), 2)):
                # exact 2x pixel duplication, the shape both YOLOv8n resizes
                # have: the NHWC upsample kernel does the duplication on the
                # QPU.  The index test above is what proves the mode/coord
                # combination is that duplication (2x half_pixel
                # round_prefer_floor and 2x asymmetric both are).
                op = _align16(Wo)
                out = self._alloc_t((x.shape[0], x.shape[1], Ho, op), "nhwc",
                                    op)
                ho, wo, gp = self.eng.upsample2x_nhwc(
                    x.buf, x.shape[0], x.shape[1], h, w, x.pitch, out.buf)
                if (ho, wo, gp) != (Ho, Wo, op):
                    raise UnsupportedNode("Resize %s: up2 returned %dx%d/%d"
                                          % (node.name, ho, wo, gp))
                return Tensor(out.buf, (x.shape[0], x.shape[1], Ho, Wo),
                              "nhwc", op)
            if nhwc:
                # A nearest resize is a pure gather with no arithmetic: one
                # host pass over the NHWC image, not one QPU launch per output
                # pixel (the 2x fast path above keeps the hot case on the QPU).
                a = self.download_nhwc(x)
                y = np.ascontiguousarray(a[:, iy][:, :, ix])
                return self.upload_nhwc(y, (n_in, c_in, Ho, Wo))
            a = self.download(x)
            y = np.ascontiguousarray(a[:, :, iy][:, :, :, ix])
            return self.upload(y)

        # Bilinear.  The ARM only gathers the four neighbours and builds the
        # coordinate weights; the four multiplies and three adds that touch
        # the activations run on the QPUs (vc4_elementwise MUL/ADD).
        def tap1d(src, n_in):
            i0 = np.floor(src).astype(np.int64)
            return (np.clip(i0, 0, n_in - 1), np.clip(i0 + 1, 0, n_in - 1),
                    src - i0)

        iy0, iy1, wy = tap1d(sy, h)
        ix0, ix1, wx = tap1d(sx, w)
        a = self.download_nhwc(x) if nhwc else self.download(x)
        if nhwc:
            taps = [a[:, iy0][:, :, ix0], a[:, iy0][:, :, ix1],
                    a[:, iy1][:, :, ix0], a[:, iy1][:, :, ix1]]
            wy0, wy1 = 1.0 - wy, wy
            wx0, wx1 = 1.0 - wx, wx
            wts = [wy0[:, None] * wx0[None, :], wy0[:, None] * wx1[None, :],
                   wy1[:, None] * wx0[None, :], wy1[:, None] * wx1[None, :]]
            shp = (a.shape[0], x.shape[1], Ho, Wo)
            ashp = (a.shape[0], Ho, Wo, x.shape[1])
            wts = [np.broadcast_to(z[None, :, :, None], ashp) for z in wts]
        else:
            taps = [a[:, :, iy0][:, :, :, ix0], a[:, :, iy0][:, :, :, ix1],
                    a[:, :, iy1][:, :, :, ix0], a[:, :, iy1][:, :, :, ix1]]
            wy0, wy1 = 1.0 - wy, wy
            wx0, wx1 = 1.0 - wx, wx
            wts = [wy0[None, None, :, None] * wx0[None, None, None, :],
                   wy0[None, None, :, None] * wx1[None, None, None, :],
                   wy1[None, None, :, None] * wx0[None, None, None, :],
                   wy1[None, None, :, None] * wx1[None, None, None, :]]
            shp = (a.shape[0], x.shape[1], Ho, Wo)
            wts = [np.broadcast_to(z, shp) for z in wts]
        return self._lerp4(taps, wts, shp, nhwc)
    def _lerp4(self, taps, weights, shape, nhwc):
        """sum taps[i]*weights[i] with GPU MUL/ADD (no new QPU kernel)."""
        def up(arr):
            arr = np.ascontiguousarray(arr, np.float32)
            if nhwc:
                return self.upload_nhwc(arr, shape)
            return self.upload(arr)

        def ew(op, a, b):
            out = self.pool.alloc(a.size * 4 + 64)
            self.eng.elementwise(op, a.buf, out, a.size, b=b.buf)
            return Tensor(out, a.shape, a.layout, a.pitch)

        t = [up(z) for z in taps]
        w = [up(z) for z in weights]
        p0 = ew(vc4infer.OP_MUL, t[0], w[0])
        self._free(t[0].buf), self._free(w[0].buf)
        p1 = ew(vc4infer.OP_MUL, t[1], w[1])
        self._free(t[1].buf), self._free(w[1].buf)
        s0 = ew(vc4infer.OP_ADD, p0, p1)
        self._free(p0.buf), self._free(p1.buf)
        p2 = ew(vc4infer.OP_MUL, t[2], w[2])
        self._free(t[2].buf), self._free(w[2].buf)
        p3 = ew(vc4infer.OP_MUL, t[3], w[3])
        self._free(t[3].buf), self._free(w[3].buf)
        s1 = ew(vc4infer.OP_ADD, p2, p3)
        self._free(p2.buf), self._free(p3.buf)
        out = ew(vc4infer.OP_ADD, s0, s1)
        self._free(s0.buf), self._free(s1.buf)
        return out

    def _slice(self, node):
        """Slice, on the QPUs.

        The affine copy2d path covers the common case: one sliced axis whose
        trailing axes are all full, so every output row is a contiguous run of
        the source.  That is every Slice in YOLOv8n and costs one launch.

        Every other form -- several axes, an interior axis with sliced trailing
        axes, or any step other than 1 including a negative one -- becomes one
        ``vc4_gather`` over an index table: the table is built per output
        element by integer index arithmetic (numpy, metadata only), the QPUs
        read the table and move the words.  No tensor value is computed on the
        ARM either way.

        Endpoints and steps follow the ONNX definition (operator Slice-13,
        "Attributes/Inputs: starts, ends, axes, steps"): a negative start/end
        counts from the back, then the effective start/end are clamped to
        [0, dim] for a positive step and to [-1, dim-1] for a negative step,
        which is how INT_MAX / INT_MIN mean "to the end" / "before the start".
        A step of 0 is invalid.
        """
        x = self._float(node.input[0])

        def val(i, dflt=None):
            if len(node.input) <= i or not node.input[i]:
                return dflt
            v = self.host.get(node.input[i], self.values.get(node.input[i]))
            return np.asarray(v).ravel().tolist()

        starts = val(1)
        ends = val(2)
        axes = val(3, None)
        steps = val(4, None)
        if starts is None or ends is None:
            raise UnsupportedNode("Slice %s: no starts/ends" % node.name)
        if axes is None:
            axes = list(range(len(starts)))
        if steps is None:
            steps = [1] * len(starts)
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise UnsupportedNode("Slice %s: starts/ends/axes/steps lengths "
                                  "%d/%d/%d/%d" % (node.name, len(starts),
                                                    len(ends), len(axes),
                                                    len(steps)))
        if x.layout != "nchw":
            raise UnsupportedNode("Slice %s: needs the compact NCHW source"
                                  % node.name)
        shp = [int(d) for d in x.shape]
        rank = len(shp)
        norm = []
        used = set()
        for st, en, ax, sp in zip(starts, ends, axes, steps):
            ax = int(ax) % rank
            sp = int(sp)
            if sp == 0:
                raise UnsupportedNode("Slice %s: step 0 is invalid (axis %d)"
                                      % (node.name, ax))
            if ax in used:
                raise UnsupportedNode("Slice %s: axis %d named twice"
                                      % (node.name, ax))
            used.add(ax)
            d = shp[ax]
            st = int(st)
            en = int(en)
            if st < 0:
                st += d
            if en < 0:
                en += d
            if sp > 0:
                st = min(max(st, 0), d)
                en = min(max(en, 0), d)
            else:
                st = min(max(st, 0), d - 1)
                en = min(max(en, -1), d - 1)
            norm.append((ax, st, en, sp))
        cut = [(ax, st, en, sp) for ax, st, en, sp in norm
               if not (st == 0 and en == shp[ax] and sp == 1)]
        if not cut:
            return x
        if len(cut) == 1 and cut[0][3] == 1:
            # affine: one sliced axis, contiguous output rows
            ax, st, en, _ = cut[0]
            inner = int(np.prod(shp[ax + 1:]))
            out_shape = list(shp)
            out_shape[ax] = en - st
            out = self._alloc_t(out_shape)
            rows = int(np.prod(shp[:ax])) if ax else 1
            run = (en - st) * inner
            if run > 0:
                self._runs(self._buf_at(x, st * inner * 4), out.buf, rows,
                           run, shp[ax] * inner * 4, run * 4)
            return out
        # general: one index table, one gather launch
        rngs = [np.arange(d, dtype=np.int64) for d in shp]
        for ax, st, en, sp in norm:
            rngs[ax] = np.arange(st, en, sp, dtype=np.int64)
        tab, out_shape = _flat_index(shp, list(range(rank)), rngs)
        return self._gather_flat(x, tab, out_shape)

    def _softmax(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", -1))
        rank = len(x.shape)
        if axis < 0:
            axis += rank
        c = x.shape[axis]
        if c > 16:
            raise UnsupportedNode("Softmax %s: %d classes (max 16)"
                                  % (node.name, c))
        if x.layout == "nhwc" and axis == 1:
            # already [pixel][class] with pitch == W: the rows the fused
            # kernel wants are the pixels, and its output is the same layout
            if x.pitch != x.shape[3]:
                raise UnsupportedNode("Softmax %s: pitched NHWC %d != W %d"
                                      % (node.name, x.pitch, x.shape[3]))
            rows = x.shape[0] * x.shape[2] * x.shape[3]
            rp = (rows + 7) & ~7
            inb = self._alloc_t(x.shape, "nhwc", x.shape[3],
                                extra=(rp - rows) * c)
            out = self._alloc_t(x.shape, "nhwc", x.shape[3],
                                extra=(rp - rows) * c)
            self._x(x.buf, inb.buf, 1, rows,
                    0, 0, 64, 64, 4, 16, 1, 64, 64,
                    4 if rows % 4 == 0 else (2 if rows % 2 == 0 else 1), 16)
            self.eng.softmax_fwd(inb.buf, out.buf, rp, c)
            return out
        if x.layout == "nchw" and axis == rank - 1:
            # already compact [rows][classes]: the layout the kernel wants, for
            # any class count <= 16 (its scratch pads the row with -1e30).  The
            # row count must still be a multiple of 8, and the padded rows are
            # read from the arena, so the logits are copied into a padded
            # buffer first.
            rows = x.logical // c
            rp = (rows + 7) & ~7
            inb = self._alloc_t(x.shape, extra=(rp - rows) * c)
            self._runs(x.buf, inb.buf, 1, rows * c, 0, 0)
            out = self._alloc_t(x.shape, extra=(rp - rows) * c)
            self.eng.softmax_fwd(inb.buf, out.buf, rp, c)
            return out
        if c != 16:
            raise UnsupportedNode("Softmax %s: axis %d of %d classes needs the "
                                  "last axis or the NHWC layout (16 classes)"
                                  % (node.name, axis, c))
        # NCHW -> NHWC on the QPUs, one vector per pixel, then the kernel
        n, _, h, w = x.shape
        rows = n * h * w
        rp = (rows + 7) & ~7
        inb = self._alloc_t((n, c, h, w), "nhwc", w, extra=(rp - rows) * c)
        out = self._alloc_t((n, c, h, w), "nhwc", w, extra=(rp - rows) * c)
        for i in range(n):
            self._x(self._buf_at(x, i * c * h * w * 4),
                    self._buf_at(inb, i * h * w * c * 4), 1, h * w, 0, 0, 4,
                    64, h * w * 4, 16, 1, 64, 64,
                    4 if (h * w) % 4 == 0 else (2 if (h * w) % 2 == 0 else 1),
                    16)
        self.eng.softmax_fwd(inb.buf, out.buf, rp, c)
        return out

    def _transpose(self, node):
        """Arbitrary permutation, on the QPUs.

        The affine copy2d_x block loop covers the permutation a layout op
        needs: for perm [0,2,1,3] on [1][J][K][W] the vector is 16 consecutive
        W positions, the row is K (the outer swapped axis) and the block is J,
        with the two streams advancing by different block steps.

        Every other permutation, and any rank, is one ``vc4_gather`` over a
        permutation index table: destination element ``i`` names its source
        word, so an arbitrary perm (and the absent-attribute "reverse the
        axes" form) needs no new kernel and no host arithmetic on values.
        """
        x = self._float(node.input[0])
        perm = _attr(node, "perm", None)
        shp = [int(d) for d in x.shape]
        if perm is None:
            perm = list(range(len(shp)))[::-1]
        perm = [int(p) % len(shp) for p in perm]
        if sorted(perm) != list(range(len(shp))):
            raise UnsupportedNode("Transpose %s: perm %s" % (node.name, perm))
        if x.layout != "nchw":
            raise UnsupportedNode("Transpose %s: needs the compact NCHW "
                                  "source" % node.name)
        if len(shp) != 4 or perm != [0, 2, 1, 3]:
            rngs = [np.arange(shp[p], dtype=np.int64) for p in perm]
            tab, out_shape = _flat_index(shp, perm, rngs)
            return self._gather_flat(x, tab, out_shape)
        n, a, b, w = shp                 # source [n][a][b][w] -> [n][b][a][w]
        out = self._alloc_t((n, b, a, w))
        nv = w // 16
        rem = w % 16
        for i in range(n):
            sb = self._buf_at(x, i * a * b * w * 4)
            db = self._buf_at(out, i * a * b * w * 4)
            if nv:
                full = (nv // 4) * 4
                if full:
                    self._x(sb, db, b, full, w * 4, a * w * 4, 64, 64, 4, 16,
                            a, b * w * 4, w * 4, 4, 16)
                if nv % 4:
                    k = (nv - 4) * 16 * 4
                    self._x(self._buf_at(sb, k), self._buf_at(db, k), b, 4,
                            w * 4, a * w * 4, 64, 64, 4, 16, a, b * w * 4,
                            w * 4, 4, 16)
                elif full == 0:
                    self._x(sb, db, b, nv, w * 4, a * w * 4, 64, 64, 4, 16,
                            a, b * w * 4, w * 4, 1, 16)
            if rem:
                k = (w - rem) * 4
                self._x(self._buf_at(sb, k), self._buf_at(db, k), b, 1,
                        w * 4, a * w * 4, 64, 64, 4, 16, a, b * w * 4,
                        w * 4, 1, rem)
        return out

    def _matmul(self, node):
        return self._epilogue(node, self._matmul_impl(node))

    def _matmul_impl(self, node):
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
        return self._epilogue(node, self._gemm_impl(node))

    def _gemm_impl(self, node):
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
            key = ("gemmT", node.name)
            ba = self._panels.get(key)
            if ba is None:
                src = self._hostconst(node.input[1])
                src = (np.asarray(src, np.float32) if src is not None
                       else self.download(b))
                ba = np.ascontiguousarray(np.asarray(src).T)
                self._panels[key] = ba
        else:
            K2, N = b.shape
            src = self._hostconst(node.input[1])
            ba = (np.asarray(src, np.float32) if src is not None
                  else self.download(b))
        if K != K2:
            raise UnsupportedNode("Gemm %s: K mismatch" % node.name)
        keyb = ("gemmB", node.name)
        bb = self._panels.get(keyb)
        if bb is None:
            bb = self.upload(np.ascontiguousarray(ba))
            self._panels[keyb] = bb
        out = self._alloc(M * N)
        self.eng.matmul(a.buf, bb.buf, out, M, N, K)
        t = Tensor(out, (M, N))
        if beta == 1.0 and len(node.input) > 2 and node.input[2]:
            keyc = ("gemmC", node.name)
            cb = self._panels.get(keyc)
            if cb is None:
                c = self._float(node.input[2])
                ca = self._hostconst(node.input[2])
                ca = (np.asarray(ca, np.float32) if ca is not None
                      else self.download(c))
                if ca.size == N:
                    ca = np.broadcast_to(ca.reshape(1, N), (M, N))
                elif ca.shape != (M, N):
                    raise UnsupportedNode("Gemm %s: C shape %s"
                                          % (node.name, ca.shape))
                cb = self.upload(np.ascontiguousarray(ca))
                self._panels[keyc] = cb
            o2 = self._alloc(M * N)
            self.eng.elementwise(vc4infer.OP_ADD, t.buf, o2, M * N, b=cb.buf)
            t = Tensor(o2, (M, N))
        return t

    def _global_avg_pool(self, node):
        x = self._float(node.input[0])
        if len(x.shape) != 4:
            raise UnsupportedNode("GlobalAveragePool %s: expected NCHW"
                                  % node.name)
        N, C, H, W = x.shape
        if N != 1:
            raise UnsupportedNode("GlobalAveragePool %s: batch != 1" % node.name)
        if x.layout == "nhwc":
            x = self.upload(self.nchw(x))
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
        logical = list(x.shape)
        out = []
        for i, d in enumerate(target):
            out.append(logical[i] if d == 0 else int(d))
        if out.count(-1) > 1:
            raise UnsupportedNode("Reshape %s: two -1 dims" % node.name)
        if -1 in out:
            k = out.index(-1)
            known = int(np.prod([d for i, d in enumerate(out) if i != k]))
            if known == 0 or x.logical % known:
                raise UnsupportedNode("Reshape %s: %s -> %s"
                                      % (node.name, x.shape, out))
            out[k] = x.logical // known
        if int(np.prod(out)) != x.logical:
            raise UnsupportedNode("Reshape %s: %s -> %s"
                                  % (node.name, x.shape, out))
        if x.layout == "nhwc":
            return Tensor(self._to_nchw_buf(x).buf, out)
        return Tensor(x.buf, out)

    def _to_nchw_buf(self, x, shape=None):
        """NHWC tensor -> a compact NCHW buffer, on the QPUs.

        A destination vector is 16 consecutive NCHW words.  When the source
        is compact (pitch == W) those are 16 consecutive pixels of one
        channel (lane stride C*4); when it is pitched they are one source row
        of W words, so the lanes are the W columns (a W-word partial store).
        """
        n, c, h, w = x.shape
        tgt = shape or x.shape
        out = self._alloc_t(tgt)
        hw = h * w
        for i in range(n):
            sb = self._buf_at(x, i * h * x.pitch * c * 4)
            db = self._buf_at(out, i * hw * c * 4)
            if x.pitch == w:
                nv = hw // 16
                rem = hw % 16
                if nv >= 4:
                    self._x(sb, db, c, (nv // 4) * 4, 4, hw * 4, 16 * c * 4,
                            64, c * 4, 16, 1, 64, 64, 4, 16)
                    if nv % 4:
                        ks = (nv - 4) * 16 * c * 4
                        kd = (nv - 4) * 16 * 4
                        self._x(self._buf_at(sb, ks), self._buf_at(db, kd),
                                c, 4, 4, hw * 4, 16 * c * 4, 64, c * 4, 16,
                                1, 64, 64, 4, 16)
                elif nv:
                    self._x(sb, db, c, nv, 4, hw * 4, 16 * c * 4, 64, c * 4,
                            16, 1, 64, 64, 1, 16)
                if rem:
                    k = (hw - rem) * 4
                    self._x(self._buf_at(sb, k), self._buf_at(db, k), c, 1,
                            4, hw * 4, c * 4, 64, c * 4, 16, 1, 64, 64, 1,
                            rem)
            elif w <= 16:
                self._x(sb, db, c, h, 4, hw * 4, x.pitch * c * 4, w * 4,
                        c * 4, w, 1, 64, 64, 1, w)
            else:
                for w0 in range(0, w, 16):
                    k = min(16, w - w0)
                    self._x(self._buf_at(sb, w0 * c * 4),
                            self._buf_at(db, w0 * 4), c, h, 4, hw * 4,
                            x.pitch * c * 4, w * 4, c * 4, k, 1, 64, 64, 1,
                            k)
        return out

    def _slice_ch(self, x, c0, cn):
        """x's channels [c0, c0+cn) as a new NHWC tensor, on the QPUs."""
        n, c, h, w = x.shape
        out = self._alloc_t((n, cn, h, w), "nhwc", w)
        units = 4 if w % 4 == 0 else (2 if w % 2 == 0 else 1)
        if cn % 16 == 0:
            nb, dp, nl = cn // 16, 16, 16
        else:
            units, nb, dp, nl = 1, 1, cn, cn
        for i in range(n):
            self._x(self._buf_at(x, (i * h * x.pitch * c + c0) * 4),
                    out.buf if n == 1 else self._buf_at(out, i * h * w * cn
                                                       * 4),
                    h, w, x.pitch * c * 4, w * cn * 4, c * 4, cn * 4, 4, nl,
                    nb, 64, 64, units, dp)
        return out

    def _flatten(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", 1))
        if axis < 0:
            axis += len(x.shape)
        before = int(np.prod(x.shape[:axis])) if axis else 1
        after = int(np.prod(x.shape[axis:]))
        if x.layout == "nhwc":
            return Tensor(self._to_nchw_buf(x).buf, (before, after))
        return Tensor(x.buf, (before, after))

    def _squeeze(self, node):
        x = self._float(node.input[0])
        axes = _attr(node, "axes", None)
        if len(node.input) > 1 and node.input[1]:
            v = self.host.get(node.input[1], self.values.get(node.input[1]))
            axes = np.asarray(v).ravel().tolist()
        nb = self._to_nchw_buf(x) if x.layout == "nhwc" else None
        shape = list(x.shape)
        if axes is None:
            shape = [d for d in shape if d != 1]
        else:
            for ax in axes:
                ax = int(ax) % len(shape)
                if shape[ax] != 1:
                    raise UnsupportedNode("Squeeze %s: dim %d != 1"
                                          % (node.name, ax))
                shape[ax] = 0
            shape = [d for d in shape if d != 0]
        if nb is not None:
            return Tensor(nb.buf, shape)
        return Tensor(x.buf, shape)

    def _unsqueeze(self, node):
        x = self._float(node.input[0])
        v = self.host.get(node.input[1], self.values.get(node.input[1]))
        axes = sorted(int(a) for a in np.asarray(v).ravel())
        nb = self._to_nchw_buf(x) if x.layout == "nhwc" else None
        shape = list(x.shape)
        for ax in axes:
            shape.insert(ax, 1)
        if nb is not None:
            return Tensor(nb.buf, shape)
        return Tensor(x.buf, shape)

    def _cast(self, node):
        x = self.values[node.input[0]]
        to = int(_attr(node, "to", onnx.TensorProto.FLOAT))
        if isinstance(x, Tensor):
            if to != onnx.TensorProto.FLOAT:
                raise UnsupportedNode("Cast %s: fp32 -> %d needs a kernel"
                                      % (node.name, to))
            return x
        return np.asarray(x).astype(np.float32 if to == onnx.TensorProto.FLOAT
                                    else np.int64)

    def _identity(self, node):
        v = self.values[node.input[0]]
        return v

    def _const_node(self, node):
        v = _attr(node, "value")
        if v is None:
            raise UnsupportedNode("Constant %s: no value" % node.name)
        return self._constant(node.output[0], np.asarray(v))

    # --------------------------------------------------- host integer ops
    def _shape(self, node):
        x = self.values[node.input[0]]
        return np.asarray(x.shape, dtype=np.int64)

    def _gather(self, node):
        d = np.asarray(self.values[node.input[0]])
        idx = np.asarray(self.values[node.input[1]], dtype=np.int64)
        axis = int(_attr(node, "axis", 0))
        return np.take(d, idx, axis=axis).astype(np.int64)

    def _host_binop(self, node, op):
        a = self.values[node.input[0]]
        b = self.values[node.input[1]]
        if isinstance(a, Tensor) or isinstance(b, Tensor):
            return None
        return op(np.asarray(a), np.asarray(b)).astype(np.int64)

    def _add(self, node):
        f = self._fused.get(self._node_idx, {}).get("affine")
        if f:                                  # Add(Mul(x,a),b) -> AFFINE(x)
            # UOP_AFFINE is a*p0 + p1 with the same two roundings the two
            # kernels did (mul, then add), so the bytes agree.  Checked before
            # the host binop: the Mul this replaces is skipped, so its output
            # name has no value and must never be looked up.
            return self._uop_on(self._float(f["input"]), vc4infer.UOP_AFFINE,
                                float(f["alpha"]), float(f["beta"]))
        r = self._host_binop(node, np.add)
        if r is not None:
            return r
        return self._binary(node, vc4infer.OP_ADD)

    def _mul(self, node):
        src = self._silu_src.get(self._node_idx)
        if src is not None:                    # Mul(x, Sigmoid(x)) -> SiLU(x)
            return self._silu(self._float(src))
        f = self._fused.get(self._node_idx, {}).get("hardswish")
        if f:                                  # Mul(x, HardSigmoid(x))
            return self._uop_on(self._float(f["input"]),
                                vc4infer.UOP_HARDSWISH)
        r = self._host_binop(node, np.multiply)
        if r is not None:
            return r
        return self._binary(node, vc4infer.OP_MUL)

    def _div(self, node):
        r = self._host_binop(node, np.floor_divide)
        if r is not None:
            return r
        a = self._float(node.input[0])
        c = self._hostconst(node.input[1])
        if c is None or np.asarray(c).size != 1:
            raise UnsupportedNode("Div %s: divisor is not a scalar constant"
                                  % node.name)
        return self._scaled(a, 1.0 / float(np.asarray(c).ravel()[0]))

    def _sub(self, node):
        r = self._host_binop(node, np.subtract)
        if r is not None:
            return r
        a_t = isinstance(self.values[node.input[0]], Tensor)
        b_t = isinstance(self.values[node.input[1]], Tensor)
        if a_t and not b_t:
            c = self._hostconst(node.input[1])
            if c is None:
                raise UnsupportedNode("Sub %s: right operand is not constant"
                                      % node.name)
            a = self._float(node.input[0])
            return self._addconst(a, -np.asarray(c, np.float32))
        if not a_t and b_t:
            c = self._hostconst(node.input[0])
            if c is None:
                raise UnsupportedNode("Sub %s: left operand is not constant"
                                      % node.name)
            b = self._float(node.input[1])
            t = self._scaled(b, -1.0)          # -x on the QPUs
            return self._addconst(t, np.asarray(c, np.float32))
        # both runtime tensors: a + (-1)*b, both on the QPUs
        a = self._float(node.input[0])
        b = self._float(node.input[1])
        t = self._scaled(self._same_layout(a, b), -1.0)
        return self._bin(a, t, vc4infer.OP_ADD)

    # ---------------------------------------------------------------- execute
    def _dispatch(self, node):
        op = node.op_type
        d = {
            "Conv": self._conv,
            "Relu": self._relu,
            "Abs": self._abs,
            "Neg": self._neg,
            "Sign": self._sign,
            "Clip": self._clip,
            "LeakyRelu": self._leakyrelu,
            "HardSigmoid": self._hardsigmoid,
            "HardSwish": self._hardswish,
            "Sigmoid": self._sigmoid,
            "Tanh": self._tanh,
            "Exp": self._exp,
            "Log": self._log,
            "Sqrt": self._sqrt,
            "Reciprocal": self._reciprocal,
            "Erf": self._erf,
            "Sin": self._sin,
            "Cos": self._cos,
            "Floor": self._floor,
            "Ceil": self._ceil,
            "Round": self._round,
            "Softplus": self._softplus,
            "Softsign": self._softsign,
            "Elu": self._elu,
            "Selu": self._selu,
            "Gelu": self._gelu,
            "Mish": self._mish,
            "Silu": lambda node: self._silu(self._float(node.input[0])),
            "Add": self._add,
            "Mul": self._mul,
            "MaxPool": self._maxpool,
            "Concat": self._concat,
            "Split": self._split,
            "Slice": self._slice,
            "Div": self._div,
            "Sub": self._sub,
            "Resize": self._resize,
            "MatMul": self._matmul,
            "Gemm": self._gemm,
            "Softmax": self._softmax,
            "Transpose": self._transpose,
            "GlobalAveragePool": self._global_avg_pool,
            "Reshape": self._reshape,
            "Flatten": self._flatten,
            "Squeeze": self._squeeze,
            "Unsqueeze": self._unsqueeze,
            "Cast": self._cast,
            "Identity": self._identity,
            "Constant": self._const_node,
            "Shape": self._shape,
            "Gather": self._gather,
        }
        if not _UNARY_NV4_VERIFIED:
            for name in _UNARY_NV4_STAGED:
                d.pop(name, None)
        fn = d.get(op)
        # the family tables also own a few operators of the main table
        # (docs/OPS-NORM.md: Softmax > 16 classes)
        h = _extra_handlers().get(op)
        if h is not None and (fn is None or op in _EXTRA_OVERRIDES):
            fn = functools.partial(h, self)
        if fn is None:
            raise UnsupportedNode(node.name, op, "unsupported op %s" % op)
        try:
            return fn(node)
        except vc4infer.Unsupported as e:
            raise UnsupportedNode(node.name, op, str(e))
        except vc4infer.Error as e:
            raise UnsupportedNode(node.name, op, "engine error: %s" % e)
        except UnsupportedNode as e:
            # a deeper layer raised without the node context; add it
            raise UnsupportedNode(node.name if not e.node else e.node,
                                  op if not e.op else e.op, e.detail)

    # ---------------------------------------------------------------- execute
    def _drop_garbage(self):
        """Release per-node temporaries left by a failed VC4 dispatch."""
        live = set(v.buf.off for v in self.values.values()
                   if isinstance(v, Tensor))
        for t in self._garbage:
            if isinstance(t, Tensor) and t.buf.off not in live:
                self.pool.release(t.buf)
        self._garbage = []

    def _run_node_hybrid(self, idx, node, tier, reason):
        """Execute one node under the hybrid policy.

        Returns ``(out, backend, reason, assigned)``.  ``assigned`` is True when
        the host evaluator already stored the outputs in ``self.values``.
        """
        if tier == "vc4":
            self._promote(node)
            try:
                return self._dispatch(node), "vc4", reason, False
            except UnsupportedNode as e:
                if not self._fallback_allowed(tier, node, e):
                    if self.host_policy == "structural" and \
                            not str(e.detail).startswith("engine error"):
                        raise HostNotAllowed(
                            node.name or node.op_type, node.op_type,
                            "VC4 cannot express this shape/attribute and "
                            "host_policy='structural' forbids the numpy "
                            "fallback (%s)" % e.detail) from e
                    raise
                self._drop_garbage()
                self._host_node(node, idx)
                return None, "host-unwritten", \
                    "runtime fallback: %s" % e.detail, True

        if tier == "host-meta":
            # Metadata ops usually still have a cheap int64/byte host path in
            # the executor; use it.  If the engine cannot dispatch the shape,
            # the reference evaluator is allowed (still no activation math).
            try:
                return self._dispatch(node), "host-meta", reason, False
            except UnsupportedNode as e:
                if not self._fallback_allowed(tier, node, e):
                    raise
                self._drop_garbage()
                self._host_node(node, idx)
                return None, "host-meta", \
                    "%s (reference evaluator)" % reason, True

        if tier == "host-control":
            if self.host_policy == "device":
                raise NeedsDeviceDispatch(
                    node.name or node.op_type, node.op_type,
                    "If/Loop/Scan branch execution on the device")
            self._host_node(node, idx)
            return None, "host-control", reason, True

        if tier == "host-unwritten":
            if self.host_policy in ("device", "structural"):
                raise HostNotAllowed(node.name or node.op_type, node.op_type,
                                     reason, policy=self.host_policy)
            self._host_node(node, idx)
            return None, "host-unwritten", reason, True

        if tier == "host-meta" and self.host_policy == "device":
            raise HostNotAllowed(node.name or node.op_type, node.op_type,
                                 "metadata node not folded at plan time: %s"
                                 % reason, policy="device")

        raise UnsupportedNode(node.name, node.op_type, reason)

    def _plan(self):
        """last consumer index of every tensor name (graph outputs: inf)."""
        last = {}
        for i, node in enumerate(self.graph.node):
            for name in node.input:
                if name:
                    last[name] = i
            # subgraphs capture outer-scope values without listing them; keep
            # those live until the control-flow node that uses them
            for name in outer_refs(node):
                last[name] = i
        for o in self.graph.output:
            last[o.name] = len(self.graph.node)
        self._last_use = last

    def run(self, feeds, profile=False):
        self.check_policy()
        self.values = {k: v for k, v in self.values.items()
                       if k in self._consts}
        self._count_host = False
        self._node_ctx = None
        self.host_bytes = 0
        self._host_site_acc = {}
        self.host_sites = []
        self.boundary_uploads = 0
        self.boundary_downloads = 0
        for name, arr in feeds.items():
            self.values[name] = self.upload(arr)
        if not self._last_use:
            self._plan()
        self.profile = []
        self.backend_report = []
        skip = self._skip
        hybrid = self.mode == "hybrid"
        plan = self.backend_plan() if hybrid else None
        for idx, node in enumerate(self.graph.node):
            name = self._node_name(idx, node)
            if idx in skip or (plan is not None and
                              plan[idx]["backend"] == "folded"):
                if idx in skip:
                    self._apply_alias(idx)
                if hybrid:
                    backend = "vc4" if idx in skip else "folded"
                    reason = (self._skip_reason.get(idx)
                              if idx in skip else plan[idx]["reason"])
                    self.backend_report.append({
                        "index": idx, "name": name, "op": node.op_type,
                        "backend": backend, "reason": reason,
                        "wall_ms": 0.0, "gpu_ms": 0.0, "launches": 0})
                    if profile:
                        self.profile.append({
                            "index": idx, "name": name, "op": node.op_type,
                            "backend": backend, "reason": reason,
                            "wall_ms": 0.0, "gpu_ms": 0.0, "launches": 0,
                            "flops": 0, "macs": 0})
                continue
            self._node_idx = idx
            tier = plan[idx]["backend"] if hybrid else None
            reason = plan[idx]["reason"] if hybrid else ""
            t0 = time.perf_counter()
            s0 = self.eng.stats
            self._node_ctx = (idx, name, node.op_type)
            self._count_host = True
            try:
                if hybrid:
                    out, backend, reason, assigned = self._run_node_hybrid(
                        idx, node, tier, reason)
                else:
                    out = self._dispatch(node)
                    backend, assigned = None, False
            finally:
                self._count_host = False
            s1 = self.eng.stats
            dt = (time.perf_counter() - t0) * 1e3
            if profile:
                gpu = (s1.pack_ns - s0.pack_ns + s1.kernel_ns - s0.kernel_ns) / 1e6
                entry = {
                    "index": idx,
                    "name": name,
                    "op": node.op_type,
                    "wall_ms": dt,
                    "gpu_ms": gpu,
                    "launches": s1.launches - s0.launches,
                    "flops": s1.flops - s0.flops,
                    "macs": int(plan[idx].get("macs", 0)) if hybrid else 0,
                }
                if hybrid:
                    entry["backend"] = backend
                    entry["reason"] = reason
                self.profile.append(entry)
            if assigned:
                outs = []
            else:
                outs = out if isinstance(out, list) else [out]
                for i, oname in enumerate(node.output):
                    if oname:
                        self.values[oname] = (outs[i] if i < len(outs)
                                              else outs[-1])
            if hybrid:
                self.backend_report.append({
                    "index": idx, "name": name, "op": node.op_type,
                    "backend": backend, "reason": reason,
                    "macs": int(plan[idx].get("macs", 0)),
                    "wall_ms": dt,
                    "gpu_ms": (s1.pack_ns - s0.pack_ns +
                               s1.kernel_ns - s0.kernel_ns) / 1e6,
                    "launches": s1.launches - s0.launches})
            # recycle: this node's inputs that nothing else reads, and any
            # temporary the op created for itself
            # views (Reshape/Flatten/Squeeze) share their input buffer, so
            # liveness has to be tracked per buffer, not per Tensor object
            keep = set(t.buf.off for t in outs if isinstance(t, Tensor))
            for t in self._garbage:
                if isinstance(t, Tensor) and t.buf.off not in keep:
                    self.pool.release(t.buf)
            self._garbage = []
            for name in node.input:
                if not name or name in self._consts:
                    continue
                if self._last_use.get(name, idx) <= idx:
                    v = self.values.pop(name, None)
                    if isinstance(v, Tensor) and v.buf.off not in keep:
                        self.pool.release(v.buf)
        self.host_sites = sorted(self._host_site_acc.values(),
                                 key=lambda e: -e["bytes"])
        self.host_bytes = sum(e["bytes"] for e in self.host_sites)
        res = []
        for o in self.graph.output:
            v = self.values[o.name]
            res.append(self.nchw(v) if isinstance(v, Tensor)
                       else np.asarray(v))
        return res


def load_model(path):
    return onnx.load(path)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--input", help=".npy input (else random)")
    ap.add_argument("--out", help="write the first output to this .npy")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cache", default=default_cache_dir())
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
            print("%-28s %-14s %9s %9s %6s" %
                  ("node", "op", "wall ms", "gpu ms", "launch"))
            for p in r.profile:
                print("%-28s %-14s %9.3f %9.3f %6d" %
                      (p["name"][:28], p["op"], p["wall_ms"], p["gpu_ms"],
                       p["launches"]))
        if args.out:
            np.save(args.out, outs[0])
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
