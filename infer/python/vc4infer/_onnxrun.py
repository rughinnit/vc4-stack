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
    Resize        nearest or linear (linear: GPU Mul/Add), asymmetric/half_pixel
    MatMul, Gemm  2-D
    Flatten, Reshape, Squeeze, Unsqueeze, Cast, Identity, Constant
    Shape, Gather int64 host tensors (static shapes)

Batch N is supported for the NHWC conv path, pool, upsample, concat,
elementwise and softmax; the NCHW fallback, Gemm and GlobalAveragePool are
batch 1 and raise otherwise.
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
from ._hybrid import HostEvaluator, outer_refs             # noqa: E402

# Kept for the scripts that print the engine directory.
ENGINE = _first_dir(os.environ.get("VC4INFER_ENGINE"),
                    os.path.join(HERE, "..", "engine"),
                    "/usr/lib/vc4infer")


def _align16(v):
    return (int(v) + 15) & ~15


class Pool:
    """A bump/free allocator over one engine allocation.

    libvc4infer has no free(), so the runner carves every activation out of a
    single arena buffer and recycles it by ONNX liveness.  ``Buf`` objects are
    plain (off, bytes) pairs, so a sub-range of the pool is a perfectly good
    engine operand.
    """

    def __init__(self, eng, nbytes, reserve=0):
        self.eng = eng
        self.reserve = reserve
        while True:
            n = (nbytes + 63) & ~63
            if n < (1 << 20):
                raise UnsupportedNode("no room for a graph pool (%d B)"
                                      % nbytes)
            try:
                self.buf = eng.alloc(n)
                break
            except vc4infer.Error:
                nbytes //= 2
        self.base = self.buf.off
        self.size = n
        self.bump = 0
        self.free = []                  # [(off, size)], sorted, coalesced

    def alloc(self, nbytes):
        n = (nbytes + 63) & ~63
        for i, (off, sz) in enumerate(self.free):
            if sz >= n:
                if sz == n:
                    self.free.pop(i)
                else:
                    self.free[i] = (off + n, sz - n)
                return vc4infer.Buf(self.base + off, int(nbytes))
        off, self.bump = self.bump, self.bump + n
        if off + n > self.size:
            raise UnsupportedNode("graph pool exhausted: %d B needed, %d B "
                                  "used of %d B" % (n, off, self.size))
        return vc4infer.Buf(self.base + off, int(nbytes))

    def release(self, b):
        if b is None:
            return
        off = b.off - self.base
        if off < 0 or off >= self.size:
            return
        n = (int(b.bytes) + 63) & ~63
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

    @property
    def used_bytes(self):
        used = self.bump
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
    def __init__(self, model_path, eng, mode="strict", host_policy="device"):
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
        # The pool must leave room for the per-conv scratch (the padded image
        # plus the im2col panel): 6 MB at 160x160, 23 MB at 320x320, so a
        # bigger model wants a bigger reserve (VC4_POOL_RESERVE_MB).
        reserve = int(os.environ.get("VC4_POOL_RESERVE_MB", "24")) << 20
        need = max(eng.arena_bytes - reserve, 8 << 20)
        pool = getattr(eng, "_vc4_pool", None)
        if pool is None or pool.size < need:
            pool = Pool(eng, need)
            eng._vc4_pool = pool
        else:
            pool.bump = 0
            pool.free = []
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
        if os.environ.get("VC4_NO_SILU_FUSE"):
            return                          # A/B switch (no rewrite)
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
    def nchw(self, t):
        """Host numpy array of a Tensor in ONNX (NCHW) order."""
        if t.layout == "nhwc":
            a = self.download_nhwc(t)                       # n,h,w,c
            return np.ascontiguousarray(a.transpose(0, 3, 1, 2))
        return self.download(t)

    def to_nhwc(self, t, cpad=None):
        """Tensor in NHWC layout (host transpose if needed; C padded to 16)."""
        a = self.nchw(t)
        if len(a.shape) != 4:
            raise UnsupportedNode("to_nhwc: rank %d" % len(a.shape))
        c = a.shape[1]
        cp = _align16(c) if cpad is None else int(cpad)
        if cp != c:
            b = np.zeros((a.shape[0], cp) + a.shape[2:], np.float32)
            b[:, :c] = a
            a = b
        if t.layout == "nhwc" and c == cp and t.shape == a.shape:
            return t
        return self.upload_nhwc(np.ascontiguousarray(a.transpose(0, 2, 3, 1)),
                                (a.shape[0], cp, a.shape[2], a.shape[3]))

    def _prepad(self, x, pads, value=0.0):
        """pads = [top, left, bottom, right]; returns a new NCHW tensor."""
        a = self.download(x)
        p = np.pad(a, ((0, 0), (0, 0), (pads[0], pads[2]),
                       (pads[1], pads[3])), constant_values=value)
        return self.upload(p)

    # ------------------------------------------------------------ conv (NHWC)
    def _conv_nhwc(self, node, x, wa, ba, kh, kw, sh, pad, dil=1, groups=1):
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
                out)
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
            pad, dil, out)
        return Tensor(out, (N, Cout, h, w), "nhwc", pitch=o)

    def _conv(self, node):
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
                                dilv, 1 if depthwise else group)
            if t.shape[1] != w.shape[0]:
                a = self.download_nhwc(t)[:, :, :, :w.shape[0]]
                t = self.upload_nhwc(np.ascontiguousarray(a),
                                     (N, w.shape[0], t.shape[2], t.shape[3]))
            return t

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
        return t

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
        """Broadcast a graph constant to `ref`'s logical shape (host expand)."""
        c = self._hostconst(name)
        if c is None:
            raise UnsupportedNode("broadcast operand %s is not a constant"
                                  % name)
        b = np.ascontiguousarray(
            np.broadcast_to(np.asarray(c, np.float32), ref.shape))
        if layout == "nhwc":
            return self.upload_nhwc(np.ascontiguousarray(b.transpose(0, 2, 3, 1)),
                                    ref.shape)
        return self.upload(b)

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
                self._tmp(b)
            else:
                a = self._expand_to(node.input[0], b, b.layout)
                self._tmp(a)
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
    def _unary_uop(self, node, uop, p0=0.0, p1=0.0, p2=0.0):
        """One QPU unary kernel over every stored element (layout-agnostic,
        exactly like Relu: the op touches all bytes it is handed, padding
        included, and keeps the operand's layout and pitch)."""
        x = self._float(node.input[0])
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
        """tanh(x) = 2*sigmoid(2x) - 1, composed from verified QPU kernels.

        VC4_OP_SIGMOID is the fused element-wise kernel (degree-5 2^f
        polynomial + 3x Newton reciprocal, measured 7.15e-07) and the two
        scalings are the engine's correctly-rounded MUL plus a broadcast ADD,
        so the whole node is four GPU launches with no new kernel and no host
        arithmetic.  The direct tanh body in kops.py is not element-verified
        yet (docs/ONNX.md sec. 11), so this composition is the shipped path.
        """
        x = self._float(node.input[0])
        t = self._scaled(x, 2.0)
        self._tmp(t)
        y = self.pool.alloc(t.size * 4 + 64)
        self.eng.elementwise(vc4infer.OP_SIGMOID, t.buf, y, t.size)
        t2 = Tensor(y, t.shape, t.layout, t.pitch)
        self._tmp(t2)
        u = self._scaled(t2, 2.0)
        self._tmp(u)
        return self._addconst(u, -1.0)

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
                # the border is materialised on the host (a constant fill)
                if pad:
                    a = self.download_nhwc(x)
                    a = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                               constant_values=-np.inf)
                    x = self.upload_nhwc(np.ascontiguousarray(a),
                                         (N, C, H + 2 * pad, W + 2 * pad))
                    self._tmp(x)
                    H += 2 * pad
                    W += 2 * pad
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
                for t in ts:
                    if t.shape[0] != shape[0] or t.shape[2:] != shape[2:]:
                        raise UnsupportedNode("Concat %s: shape mismatch"
                                              % node.name)
                a = np.concatenate([self.download_nhwc(t) for t in ts], axis=3)
                return self.upload_nhwc(np.ascontiguousarray(a),
                                        (a.shape[0], a.shape[3], shape[2],
                                         shape[3]))
            a = np.concatenate([self.nchw(t) for t in ts], axis=1)
            return self.upload(np.ascontiguousarray(a))
        if rank == 3 and axis in (1, 2):
            a = np.concatenate([self.nchw(t) for t in ts], axis=axis)
            return self.upload(np.ascontiguousarray(a))
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
        outs = []
        if x.layout == "nhwc":
            n, c, h, w = x.shape
            full = self.eng.download(x.buf, x.size * 4).reshape(
                n, h, x.pitch, c)
            off = 0
            for s in sizes:
                part = np.ascontiguousarray(full[:, :, :, off:off + s])
                outs.append(self.upload_pitched(part, (n, s, h, w), x.pitch))
                off += s
        else:
            a = self.download(x)
            off = 0
            for s in sizes:
                outs.append(self.upload(
                    np.ascontiguousarray(a[:, off:off + s])))
                off += s
        return outs

    def _resize(self, node):
        x = self._float(node.input[0])
        mode = _attr(node, "mode", "nearest")
        coord = _attr(node, "coordinate_transformation_mode", "half_pixel")
        if mode not in ("nearest", "linear"):
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
        nhwc = x.layout == "nhwc"
        h, w = (x.shape[2], x.shape[3])
        if sizes is not None:
            Ho, Wo = int(sizes[2]), int(sizes[3])
        elif scales is not None:
            Ho = int(round(h * float(scales[2])))
            Wo = int(round(w * float(scales[3])))
        else:
            raise UnsupportedNode("Resize %s: no scales/sizes" % node.name)

        a = self.download_nhwc(x) if nhwc else self.download(x)

        def axis(n_in, n_out):
            s = np.arange(n_out, dtype=np.float64)
            if coord == "asymmetric":
                src = s * (n_in / n_out)
            else:
                src = (s + 0.5) * (n_in / n_out) - 0.5
            i0u = np.floor(src).astype(np.int64)
            if mode == "nearest":
                if coord == "asymmetric":
                    idx = np.floor(s * (n_in / n_out)).astype(np.int64)
                else:
                    idx = np.floor((s + 0.5) * (n_in / n_out) - 0.5).astype(np.int64)
                return np.clip(idx, 0, n_in - 1), None, None
            return (np.clip(i0u, 0, n_in - 1), np.clip(i0u + 1, 0, n_in - 1),
                    src - i0u)

        iy0, iy1, wy = axis(h, Ho)
        ix0, ix1, wx = axis(w, Wo)
        if mode == "nearest":
            if nhwc:
                y = np.ascontiguousarray(a[:, iy0][:, :, ix0])
                return self.upload_nhwc(y, (a.shape[0], x.shape[1], Ho, Wo))
            y = np.ascontiguousarray(a[:, :, iy0][:, :, :, ix0])
            return self.upload(y)

        # Bilinear.  The ARM only gathers the four neighbours and builds the
        # coordinate weights; the four multiplies and three adds that touch
        # the activations run on the QPUs (vc4_elementwise MUL/ADD).
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
        a = self.nchw(x)
        if axes is None:
            axes = list(range(len(starts)))
        if steps is None:
            steps = [1] * len(starts)
        sl = [slice(None)] * a.ndim
        for s, e, ax, st in zip(starts, ends, axes, steps):
            ax = int(ax) % a.ndim
            e = int(e)
            sl[ax] = slice(int(s), None if e >= 2 ** 31 else e, int(st))
        y = np.ascontiguousarray(a[tuple(sl)])
        return self.upload(y)

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
        a = self.nchw(x)
        ashape = a.shape
        perm = None
        if axis != rank - 1:
            perm = [i for i in range(rank) if i != axis] + [axis]
            a = np.ascontiguousarray(a.transpose(perm))
        rows = a.size // c
        rp = (rows + 7) & ~7
        flat = a.reshape(rows, c)
        if rp != rows:
            flat = np.concatenate([flat, np.zeros((rp - rows, c), np.float32)])
        zb = self.upload(np.ascontiguousarray(flat))
        out = self.pool.alloc(rp * c * 4 + 64)
        self.eng.softmax_fwd(zb.buf, out, rp, c)
        y = np.asarray(self.eng.download(out, rp * c * 4)[:rows * c],
                       np.float32).reshape(rows, c)
        y = y.reshape(a.shape)
        if perm is not None:
            y = np.ascontiguousarray(y.transpose(np.argsort(perm)))
        return self.upload(np.ascontiguousarray(y))

    def _transpose(self, node):
        x = self._float(node.input[0])
        perm = _attr(node, "perm", None)
        a = self.nchw(x)
        if perm is None:
            perm = list(range(a.ndim))[::-1]
        return self.upload(np.ascontiguousarray(a.transpose(perm)))

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
            return Tensor(self.upload(self.nchw(x)).buf, out)
        return Tensor(x.buf, out)

    def _flatten(self, node):
        x = self._float(node.input[0])
        axis = int(_attr(node, "axis", 1))
        if axis < 0:
            axis += len(x.shape)
        before = int(np.prod(x.shape[:axis])) if axis else 1
        after = int(np.prod(x.shape[axis:]))
        if x.layout == "nhwc":
            return Tensor(self.upload(self.nchw(x)).buf, (before, after))
        return Tensor(x.buf, (before, after))

    def _squeeze(self, node):
        x = self._float(node.input[0])
        axes = _attr(node, "axes", None)
        if len(node.input) > 1 and node.input[1]:
            v = self.host.get(node.input[1], self.values.get(node.input[1]))
            axes = np.asarray(v).ravel().tolist()
        a = self.nchw(x) if x.layout == "nhwc" else None
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
        if a is not None:
            return Tensor(self.upload(a).buf, shape)
        return Tensor(x.buf, shape)

    def _unsqueeze(self, node):
        x = self._float(node.input[0])
        v = self.host.get(node.input[1], self.values.get(node.input[1]))
        axes = sorted(int(a) for a in np.asarray(v).ravel())
        a = self.nchw(x) if x.layout == "nhwc" else None
        shape = list(x.shape)
        for ax in axes:
            shape.insert(ax, 1)
        if a is not None:
            return Tensor(self.upload(a).buf, shape)
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
        r = self._host_binop(node, np.add)
        if r is not None:
            return r
        return self._binary(node, vc4infer.OP_ADD)

    def _mul(self, node):
        src = self._silu_src.get(self._node_idx)
        if src is not None:                    # Mul(x, Sigmoid(x)) -> SiLU(x)
            return self._silu(self._float(src))
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
        fn = {
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
        }.get(op)
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
        skip = getattr(self, "_silu_skip", None) or set()
        hybrid = self.mode == "hybrid"
        plan = self.backend_plan() if hybrid else None
        for idx, node in enumerate(self.graph.node):
            name = self._node_name(idx, node)
            if idx in skip or (plan is not None and
                              plan[idx]["backend"] == "folded"):
                if hybrid:
                    backend = "vc4" if idx in skip else "folded"
                    reason = ("fused into the consumer Mul as one SiLU launch"
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
