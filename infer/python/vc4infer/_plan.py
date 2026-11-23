"""_plan.py -- freeze one frame's device calls and replay them in C.

The executor's per-node cost is Python: a dispatch lookup, parsed attributes,
Tensor and pool bookkeeping, the liveness pass.  All of it decides *what to
call*, and for a static graph the answer is the same on every frame.  A plan
records the answer once, as a straight line of device calls over arena
offsets, and ``vc4_plan_run`` walks it in C.  This module builds the record;
``src/vc4_plan.c`` runs it.

How it is built
    The first strict, non-profile frame that allocates no buffer outliving it
    runs exactly as before, with two additions: the engine is wrapped in a
    ``Trace`` that copies each call's decoded arguments into a list, and the
    pool labels every allocation so the offsets can be named.  A frame that
    does allocate such a buffer (a conv's packed weight panel, a broadcast
    expansion) is not recorded: the buffer took a range an earlier node had
    released, and replaying that node would overwrite it.  The next frame has
    it live from the start and is safe.  After the frame the trace is split
    per node.  A node becomes a C record when every call it made is
    one of the representable byte-moving or elementwise entry points and each
    of its outputs is an arena buffer; then its buffers are the ones the frame
    used, and replay needs no allocator.  A node with no calls whose outputs
    all alias existing buffers is an ALIAS record (a reshape, a flatten, an
    identity, a fused activation's relabel).

The boundary
    A node the plan cannot represent stays on the Python path, unchanged: the
    plan holds a ``("p", index)`` step before it, the replay driver hands that
    index to the ordinary ``Runner._dispatch``, and execution continues after
    it.  The rule is deliberately about *calls*, not about op types: a change
    that lowers an op through a new engine entry point is representable
    automatically if the entry point is on the list, and falls back if it is
    not.  Conv, GEMM, softmax, pooling and anything that moves bytes between
    the host and the arena are not on the list, so those nodes stay Python
    and the boundary is visible in ``Plan.py_nodes``.

    The plan is also not used for a profiling or hybrid run (the per-node
    stats and the backend report would stop being per node), for an engine
    artifact (its own pinned arena plan already owns the offsets) or when
    ``$VC4_NO_CPLAN=1``.

Limits, stated rather than hidden
    A frozen plan pins every buffer to the offset the recorded frame used, so
    it is valid for feeds of the recorded shape and element count; a different
    feed raises instead of running against the wrong bytes.  A handler that
    allocated a different number or size of buffers on a later frame is
    caught by the pinned pool (``ArtifactPlanError``); the runner then drops
    the plan, puts the ordinary allocator back and re-runs that frame on the
    Python path, so the failure costs the plan and not the answer.  Replay
    assumes nothing reads a host numpy value that a C step produced, which is
    why a node with a host output is never a C step in the first place.
"""
import ctypes
import os
import struct

import numpy as np

from . import _binding as vc4infer
from ._errors import UnsupportedNode

_ALIAS = vc4infer.PLAN_ALIAS
_EW = vc4infer.PLAN_EW
_UNARY = vc4infer.PLAN_UNARY
_COPY2D = vc4infer.PLAN_COPY2D
_COPY2D_N = vc4infer.PLAN_COPY2D_N
_COPY2D_X = vc4infer.PLAN_COPY2D_X
_FILL2D = vc4infer.PLAN_FILL2D
_GATHER = vc4infer.PLAN_GATHER
_BLIT = vc4infer.PLAN_BLIT
_REDUCE = vc4infer.PLAN_REDUCE
_CUMSUM = vc4infer.PLAN_CUMSUM


def _buf(b):
    """(offset, bytes) of an engine buffer, or None."""
    if b is None:
        return None
    return (int(b.off), int(b.bytes))


def _fbits(v):
    """A float's IEEE-754 bits as an int, so a parameter survives exactly."""
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _is_buffer(v):
    """True when the value is an arena tensor the C walker can address."""
    if v is None or not getattr(v, "has_buf", False):
        return False
    b = getattr(v, "buf", None)
    return b is not None and hasattr(b, "off") and hasattr(b, "bytes")


class Trace:
    """An Engine proxy that records the calls a frame makes.

    ``__getattr__`` forwards everything, so the handler sees the real engine.
    A callable that is not one of the representable entry points is recorded
    as ``("other", name)``: its node then cannot be a C step, which is what
    keeps a conv or a host transfer on the Python path.
    """

    def __init__(self, eng):
        object.__setattr__(self, "_eng", eng)
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "plan_static", False)

    def __getattr__(self, name):
        attr = getattr(self._eng, name)
        if callable(attr) and not name.startswith("_"):
            def unsupported(*a, **k):
                # `plan_static` marks a transfer the executor makes once and
                # keeps (the broadcast expansion of a graph constant), not
                # per-frame activation data: the plan omits it on replay.
                if name == "upload" and self.plan_static:
                    self.calls.append(("static", name))
                else:
                    self.calls.append(("other", name))
                return attr(*a, **k)
            return unsupported
        return attr

    # -- representable calls ------------------------------------------------
    def elementwise(self, op, a, out, n, b=None):
        self.calls.append(("elementwise", int(op), _buf(a), _buf(b),
                           _buf(out), int(n)))
        return self._eng.elementwise(op, a, out, n, b=b)

    def unary(self, op, a, out, n, p0=0.0, p1=0.0, p2=0.0):
        self.calls.append(("unary", int(op), _buf(a), _buf(out), int(n),
                           float(p0), float(p1), float(p2)))
        return self._eng.unary(op, a, out, n, p0, p1, p2)

    def copy2d(self, a, out, rows, vpr, srow, drow, svec, dvec, slane):
        self.calls.append(("copy2d", _buf(a), _buf(out),
                           (int(rows), int(vpr), int(srow), int(drow),
                            int(svec), int(dvec), int(slane))))
        return self._eng.copy2d(a, out, rows, vpr, srow, drow, svec, dvec,
                                slane)

    def copy2d_n(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane, units, depth):
        self.calls.append(("copy2d_n", _buf(a), _buf(out),
                           (int(rows), int(vpr), int(srow), int(drow),
                            int(svec), int(dvec), int(slane), int(nlane),
                            int(units), int(depth))))
        return self._eng.copy2d_n(a, out, rows, vpr, srow, drow, svec, dvec,
                                  slane, nlane, units, depth)

    def copy2d_x(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane=16, nblk=1, sbstep=64, dbstep=64, units=4, depth=16):
        self.calls.append(("copy2d_x", _buf(a), _buf(out),
                           (int(rows), int(vpr), int(srow), int(drow),
                            int(svec), int(dvec), int(slane), int(nlane),
                            int(nblk), int(sbstep), int(dbstep), int(units),
                            int(depth))))
        return self._eng.copy2d_x(a, out, rows, vpr, srow, drow, svec, dvec,
                                  slane, nlane, nblk, sbstep, dbstep, units,
                                  depth)

    def fill2d(self, out, rows, vpr, drow, dvec, value=0.0):
        self.calls.append(("fill2d", _buf(out),
                           (int(rows), int(vpr), int(drow), int(dvec)),
                           float(value)))
        return self._eng.fill2d(out, rows, vpr, drow, dvec, value)

    def gather(self, tab, out, n, fill=0.0):
        self.calls.append(("gather", _buf(tab), _buf(out), int(n),
                           float(fill)))
        return self._eng.gather(tab, out, n, fill)

    def blit(self, a, out, nbytes):
        self.calls.append(("blit", _buf(a), _buf(out), int(nbytes)))
        return self._eng.blit(a, out, nbytes)

    def reduce(self, a, out, outer, red, op):
        self.calls.append(("reduce", _buf(a), _buf(out),
                           (int(outer), int(red), int(op))))
        return self._eng.reduce(a, out, outer, red, op)

    def cumsum(self, a, out, rows, cols):
        self.calls.append(("cumsum", _buf(a), _buf(out),
                           (int(rows), int(cols))))
        return self._eng.cumsum(a, out, rows, cols)

    # -- control, not a device call ----------------------------------------
    # The executor brackets each node with a batch scope.  Without these two
    # methods the proxy records ("other", "batch_hold") as a device call and
    # every node becomes unrepresentable, so no plan is ever built.
    def batch_hold(self):
        fn = getattr(self._eng, "batch_hold", None)
        if fn is not None:
            fn()

    def batch_release(self):
        fn = getattr(self._eng, "batch_release", None)
        if fn is not None:
            fn()


def _convert(call):
    """A trace entry as (kind, a, b, out, n, p), or None if unrepresentable."""
    k = call[0]
    if k == "static":
        # a plan-time panel upload: its bytes are already in the arena and
        # the record only has to keep the buffer addressable
        return (_ALIAS, None, None, None, 0, [])
    if k == "elementwise":
        _, op, a, b, out, n = call
        return (_EW, a, b, out, n, [op, 1 if b else 0])
    if k == "unary":
        _, op, a, out, n, p0, p1, p2 = call
        return (_UNARY, a, None, out, n,
                [op, _fbits(p0), _fbits(p1), _fbits(p2)])
    if k == "copy2d":
        _, a, out, f = call
        return (_COPY2D, a, None, out, 0, list(f))
    if k == "copy2d_n":
        _, a, out, f = call
        return (_COPY2D_N, a, None, out, 0, list(f))
    if k == "copy2d_x":
        _, a, out, f = call
        return (_COPY2D_X, a, None, out, 0, list(f))
    if k == "fill2d":
        _, out, f, value = call
        return (_FILL2D, None, None, out, 0, list(f) + [_fbits(value)])
    if k == "gather":
        _, tab, out, n, fill = call
        return (_GATHER, tab, None, out, n, [_fbits(fill)])
    if k == "blit":
        _, a, out, nbytes = call
        return (_BLIT, a, None, out, nbytes, [])
    if k == "reduce":
        _, a, out, f = call
        return (_REDUCE, a, None, out, 0, list(f))
    if k == "cumsum":
        _, a, out, f = call
        return (_CUMSUM, a, None, out, 0, list(f))
    return None


def _record(conv):
    """One trace entry as a PlanOp."""
    kind, a, b, out, n, p = conv
    op = vc4infer.PlanOp()
    op.kind = kind
    if a is not None:
        op.a_off, op.a_bytes = a
    if b is not None:
        op.b_off, op.b_bytes = b
    if out is not None:
        op.c_off, op.c_bytes = out
    op.n = int(n)
    for i, v in enumerate(p):
        if i >= vc4infer.PLAN_ARGS:
            raise UnsupportedNode("plan: %s has more than %d arguments"
                                  % (kind, vc4infer.PLAN_ARGS))
        op.p[i] = int(v)
    return op


class Plan:
    """The frozen form of one graph: steps, output descriptors and the pins.

    ``steps`` is a list of ``("c", array)`` for a run of device calls,
    ``("p", index)`` for a node that must go through the Python executor and
    ``("s", index)`` for a node the optimiser already accounted for (a fused
    or skipped node; its output names alias the recorded buffers).
    """

    def __init__(self, steps, nodes, desc, feed_sig, c_ops, c_nodes,
                 py_nodes):
        self.steps = steps
        # nodes[i] is the node index, or the tuple of node indices, that
        # steps[i] covers.  A C node with no records (an alias) inside a run
        # of records stays in its tuple; a run that produced no records at
        # all makes no step, so nothing has to be replayed for it.
        self.nodes = nodes
        self.desc = desc
        self.feed_sig = feed_sig
        self.c_ops = c_ops
        self.c_nodes = c_nodes
        self.py_nodes = py_nodes
        self.pin = {}

    def tensor(self, name):
        """The Tensor an output name names, built from its recorded buffer."""
        from ._onnxrun import Tensor
        off, nbytes, shape, layout, pitch = self.desc[name]
        return Tensor(vc4infer.Buf(off, nbytes), shape, layout, pitch)

    def check_feeds(self, feeds):
        """Refuse a feed the recorded buffers were not sized for."""
        if set(feeds) != set(self.feed_sig):
            raise UnsupportedNode(
                "frozen plan: feeds %s, the plan was built for %s"
                % (sorted(feeds), sorted(self.feed_sig)))
        for name, arr in feeds.items():
            a = np.asarray(arr)
            sig = (tuple(int(d) for d in a.shape), int(a.size))
            want = self.feed_sig[name]
            if sig != want:
                raise UnsupportedNode(
                    "frozen plan: feed %r is %s (%d elements) and the plan "
                    "was built for %s (%d); a static plan pins the buffers, "
                    "so use a new Runner (or VC4_NO_CPLAN=1) for a different "
                    "shape" % (name, sig[0], sig[1], want[0], want[1]))

    def summary(self):
        return {"steps": len(self.steps), "c_ops": self.c_ops,
                "c_nodes": self.c_nodes, "py_nodes": len(self.py_nodes),
                "py": list(self.py_nodes)}


class Build:
    """Collects the per-node trace of one frame into a Plan."""

    def __init__(self, feeds):
        self.tracer = None
        self.steps = []
        self.nodes = []
        self.desc = {}
        self._pending = []
        self._group = []
        self.c_ops = 0
        self.c_nodes = 0
        self.py_nodes = []
        self.feed_sig = {name: (tuple(int(d) for d in np.asarray(arr).shape),
                                int(np.asarray(arr).size))
                         for name, arr in feeds.items()}
        self.reason = {}

    # -- collection ---------------------------------------------------------
    def note(self, idx, node, calls, values, skipped=False):
        """Record one node: a run of calls, or a Python step."""
        if skipped:
            self._flush()
            self.steps.append(("s", idx))
            self.nodes.append((idx,))
            self._capture(node, values)
            return
        convs = []
        for c in calls:
            v = _convert(c)
            if v is None:
                self.reason[idx] = "call %s is not representable" % (c[0],)
                convs = None
                break
            convs.append(v)
        if convs is not None:
            for name in node.output:
                if name and not _is_buffer(values.get(name)):
                    self.reason[idx] = "output %r is not an arena buffer" % name
                    convs = None
                    break
        if convs is None:
            self._flush()
            self.steps.append(("p", idx))
            self.nodes.append((idx,))
            self.py_nodes.append(idx)
            return
        self._group.append(idx)
        for v in convs:
            self._pending.append(_record(v))
        self.c_ops += len(convs)
        self.c_nodes += 1
        self._capture(node, values)

    def _capture(self, node, values):
        for name in node.output:
            if not name:
                continue
            v = values.get(name)
            if _is_buffer(v):
                b = v.buf
                self.desc[name] = (int(b.off), int(b.bytes),
                                   tuple(int(d) for d in v.shape),
                                   v.layout, int(v.pitch))
            else:
                self.desc.pop(name, None)

    def _flush(self):
        if self._pending:
            self.steps.append(("c", (vc4infer.PlanOp * len(self._pending))
                               (*self._pending)))
            self.nodes.append(tuple(self._group))
        self._pending = []
        self._group = []

    # -- finish -------------------------------------------------------------
    def finish(self, records):
        """Freeze the plan; ``records`` is the pool's allocation recording."""
        self._flush()
        if not any(kind == "c" for kind, _ in self.steps):
            return None
        pin = {e["key"]: (int(e["seg"]), int(e["offset"]), int(e["bytes"]))
               for e in records}
        plan = Plan(self.steps, self.nodes, self.desc, self.feed_sig,
                    self.c_ops, self.c_nodes, self.py_nodes)
        plan.pin = pin
        return plan


def supports(eng):
    """True when a Runner may build a plan on this engine."""
    if not getattr(eng, "plan_supported", False):
        return False
    if os.environ.get("VC4_NO_CPLAN", "") not in ("", "0"):
        return False
    fn = getattr(getattr(eng, "lib", None), "vc4_plan_op_size", None)
    if fn is None:
        return False
    fn.restype = ctypes.c_size_t
    return int(fn()) == ctypes.sizeof(vc4infer.PlanOp)
