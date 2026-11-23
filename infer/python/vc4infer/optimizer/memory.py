"""Plan-time liveness and static arena offsets for the vc4infer executor.

The executor allocates every activation from one graph pool and recycles the
buffers it no longer needs.  That works, but the *size* of the pool was an
estimate: ``_onnxrun.estimate_pool_bytes`` walked the graph with a last-use map
that also recorded a tensor's producer as a use, so every output stayed live
from the node that wrote it to the end of the graph.  The estimate was
therefore the sum of every tensor, not the peak, and a model could be refused
an arena it fits.

This module is the plan-time replacement.  It computes a real live range for
every buffer the executor will allocate - first definition to last *consumer* -
and assigns non-overlapping offsets with reuse:

    plan = plan_memory(model, shapes)
    plan.peak_bytes          # the pool the run needs, known before it starts
    plan.offsets["/x"]       # the static offset of a tensor
    print(plan.format())

Only the graph is read: no device, no kernel cache, no ctypes.  The result is
carried in the engine artifact (``plan.memory`` in the manifest) and sizes the
graph pool for a plain ONNX load.

What is planned
---------------
* float initialisers: permanent, uploaded once and never released;
* packed conv panels: permanent, built on first use and cached (the panel size
  is the kernel's own layout, ``align16(Cout) * (align16(align16(Cin)*kh*kw)
  + 16) * 4``);
* graph inputs: live from the start of the run to their last consumer;
* node outputs: live from the node that writes them to their last consumer,
  or to the end of the graph when they are graph outputs.

Views do not get a buffer: ``Reshape``/``Flatten``/``Squeeze``/``Unsqueeze``/
``Identity``/``Cast`` hand the input buffer on, so their output is an alias of
it and only the source is planned.  A node the optimiser fused is not
executed, so its output is not allocated either: a SiLU's ``Sigmoid`` output
is dropped, and a conv activation's output aliases the conv's buffer.

A node may allocate one scratch buffer while it runs (a layout copy, a GEMM
operand pack, a split or softmax transpose).  The plan cannot name those from
the graph, so it reserves one largest-tensor block, shared by every node,
after the permanent region and before the activation region.  One block is
enough because the executor releases every per-node temporary when the node
finishes; the block is the maximum, not the sum.  The build measures the run
against the plan and records the margin (``plan.margin_bytes``).

Offsets are assigned by a linear scan over the live ranges, sorted by first
definition: ranges whose last use is behind the scan are released and their
bytes join a free list, a range takes the first gap that fits, and the bump
pointer only moves when no gap does.  That is the same first-fit policy the
runtime free list uses, applied once, to a known set of ranges, so the result
is deterministic and the peak is a property of the graph rather than of the
order in which a particular run happened to free its buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Node as _IRNode

ALIGN = 64
MIB = 1 << 20

# Headroom between the modelled peak and the pool the runner asks for.  The
# planner names every graph tensor but not the executor's per-node temporary;
# one MiB covers the difference measured on YOLOv8n at 96/128/160 (0.3-0.6 MiB)
# with room for alignment, so the first segment does not have to grow mid-run.
HEADROOM = 1 << 20

# ops whose output is the input buffer with another shape (executor views)
VIEW_OPS = frozenset((
    "Identity", "Cast", "Reshape", "Flatten", "Squeeze", "Unsqueeze",
    "Dropout",
))

# float TensorProto type; the executor keeps these in the arena
FLOAT = 1

# Ops whose executor allocates a tensor-sized copy beside its input and output:
# a layout conversion, an operand pack, an index gather, a row copy.  The copy
# is part of the per-node scratch block; a plain elementwise op writes its
# output and needs nothing else.
COPY_OPS = frozenset((
    "Concat", "Split", "Slice", "Transpose", "Expand", "Tile", "Pad",
    "Gather", "GatherElements", "GatherND", "Resize", "Upsample", "Softmax",
    "LogSoftmax", "Pool", "MaxPool", "AveragePool", "GlobalAveragePool",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ArgMax", "ArgMin",
    "MatMul", "Gemm", "DepthToSpace", "SpaceToDepth", "ScatterND",
))


def align(n, a=ALIGN):
    return (int(n) + a - 1) // a * a


def align16(n):
    return (int(n) + 15) // 16 * 16


def tensor_bytes(shape, nhwc=False):
    """Bytes the executor's pool allocates for a tensor of this ONNX shape.

    A conv output is stored NHWC with the channel and width axes padded to 16
    (the QPU grid); every other tensor is contiguous fp32.  The +64 is the
    executor's per-buffer slack (``Runner._alloc``), aligned like the pool's
    own allocation.
    """
    if not shape or any(d is None or int(d) <= 0 for d in shape):
        return 0
    n = 1
    for d in shape:
        n *= int(d)
    if nhwc and len(shape) == 4:
        nb, c, h, w = (int(d) for d in shape)
        n = nb * h * align16(w) * align16(c)
    return align(n * 4 + 64)


def weight_bytes(dims):
    """Arena bytes for an initialiser's raw (NCHW) fp32 copy."""
    n = 1
    for d in dims:
        n *= int(d)
    return align(n * 4 + 64) if n > 0 else 0


def panel_bytes(shape, group=1):
    """Bytes of the k-major NHWC weight panel for a conv weight of `shape`."""
    if not shape or len(shape) != 4:
        return 0
    cout, cin, kh, kw = (int(d) for d in shape)
    if min(cout, cin, kh, kw) <= 0:
        return 0
    cin *= int(group)
    return align(align16(cout) * (align16(align16(cin) * kh * kw) + 16) * 4)


@dataclass
class TensorRange:
    """One planned buffer and the node interval it is live over."""

    name: str
    kind: str               # "const" | "panel" | "input" | "activation" | "temp"
    bytes: int
    first: int
    last: int
    offset: int = -1

    @property
    def permanent(self):
        return self.kind in ("const", "panel", "temp")

    @property
    def end(self):
        return self.offset + self.bytes

    def as_list(self):
        return [self.name, self.kind, int(self.bytes), int(self.first),
                int(self.last), int(self.offset)]


@dataclass
class MemoryPlan:
    """The result of one liveness pass: ranges, offsets and the peak."""

    ranges: list = field(default_factory=list)
    peak_bytes: int = 0
    permanent_bytes: int = 0
    temp_bytes: int = 0
    activation_bytes: int = 0
    reused_bytes: int = 0
    nodes: int = 0
    aliases: dict = field(default_factory=dict)
    dropped: tuple = ()

    @property
    def peak_mb(self):
        return self.peak_bytes / MIB

    @property
    def headroom_bytes(self):
        """Headroom for the executor's unnameable per-node temporary.

        One MiB is the cap; a small model gets one eighth of its peak, with a
        64 KiB floor, so a sub-MiB graph is not charged a MiB of slack.
        """
        return min(HEADROOM, max(64 << 10, self.peak_bytes // 8))

    @property
    def pool_bytes(self):
        """The first pool segment: the modelled peak plus the headroom."""
        return align(self.peak_bytes + self.headroom_bytes)

    @property
    def offsets(self):
        return {r.name: r.offset for r in self.ranges if r.offset >= 0}

    def offset(self, name):
        for r in self.ranges:
            if r.name == name:
                return r.offset
        return None

    def range(self, name):
        for r in self.ranges:
            if r.name == name:
                return r
        return None

    def as_dict(self, max_ranges=0):
        """Manifest form.  Ranges are sorted by offset, then name."""
        ranges = sorted((r for r in self.ranges if r.offset >= 0),
                        key=lambda r: (r.offset, r.name))
        if max_ranges and len(ranges) > max_ranges:
            ranges = ranges[:max_ranges]
        return {
            "planner": "liveness",
            "peak_bytes": int(self.peak_bytes),
            "permanent_bytes": int(self.permanent_bytes),
            "temp_bytes": int(self.temp_bytes),
            "activation_bytes": int(self.activation_bytes),
            "reused_bytes": int(self.reused_bytes),
            "headroom_bytes": int(self.headroom_bytes),
            "pool_bytes": int(self.pool_bytes),
            "nodes": int(self.nodes),
            "dropped": list(self.dropped),
            "ranges": [r.as_list() for r in ranges],
        }

    def format(self):
        lines = [
            "memory plan: peak %.2f MiB = permanent %.2f + temp %.2f"
            " + activation %.2f, reused %.2f over %d nodes"
            % (self.peak_mb, self.permanent_bytes / MIB,
               self.temp_bytes / MIB, self.activation_bytes / MIB,
               self.reused_bytes / MIB, self.nodes),
        ]
        for r in sorted(self.ranges, key=lambda r: (r.offset, r.name)):
            if r.kind == "activation" and r.bytes < (1 << 16):
                continue
            lines.append("  %10d  %8.1f KiB  %-10s %s [%d..%d]"
                         % (r.offset, r.bytes / 1024.0, r.kind, r.name,
                            r.first, r.last))
        return "\n".join(lines)


# ------------------------------------------------------------- graph reading
def _weight_shape(init):
    return [int(d) for d in init.dims] if init is not None else None


def _group(node):
    for a in node.attribute:
        if a.name == "group":
            return max(1, int(a.i))
    return 1


def _conv_kernel(node):
    for a in node.attribute:
        if a.name == "kernel_shape":
            return [int(v) for v in a.ints]
    return None


def _skipped_and_aliases(nodes, outputs):
    """Node indices the executor does not run, plus output aliases.

    Mirrors ``Runner._collect_fused`` and ``Runner._find_silu``: a provenance
    ``fused`` record marks a node absorbed by another, a ``silu`` record names
    the ``Sigmoid`` outputs it replaced, and a graph with no records still gets
    the executor's implicit ``Mul(x, Sigmoid(x))`` fold.
    """
    skip = set()
    dead = set()
    alias = {}
    by_name = {}
    for i, n in enumerate(nodes):
        by_name[n.name or ("%s_%s" % (n.op_type, i))] = i
    for i, n in enumerate(nodes):
        rec = _IRNode(n, i).fused or {}
        rules = rec.get("rules") or {}
        fused = rules.get("fused")
        if fused:
            skip.add(i)
            if fused.get("by") == "activation":
                j = by_name.get(fused.get("into"))
                if j is not None and n.output and nodes[j].output:
                    alias[n.output[0]] = nodes[j].output[0]
            else:
                dead.update(o for o in n.output if o)
        # A `silu` record sits on the Mul that is executed as one SiLU launch;
        # the Sigmoid it absorbed is skipped through that node's own `fused`
        # record, so the rule itself must not skip anything.
    # implicit SiLU fold: the executor does it with no optimiser record
    producers = {}
    consumers = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                producers[o] = (i, n)
        for x in n.input:
            if x:
                consumers[x] = consumers.get(x, 0) + 1
    for i, n in enumerate(nodes):
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        for cand, other in ((n.input[0], n.input[1]),
                            (n.input[1], n.input[0])):
            prod = producers.get(cand)
            if prod is None or prod[1].op_type != "Sigmoid":
                continue
            pn = prod[1]
            if len(pn.input) != 1 or pn.input[0] != other:
                continue
            if cand in outputs or consumers.get(cand, 0) != 1:
                continue
            skip.add(prod[0])
            dead.add(cand)
    return skip, dead, alias


def plan_memory(model, shapes=None, dropped=()):
    """Plan every arena buffer of `model` and assign static offsets.

    `shapes` is a name -> dims map (``coverage.refine_shapes``); when omitted
    it is computed from the model.  `dropped` names initialisers the artifact
    path does not upload because a packed panel replaces them; their bytes are
    left out of the permanent region.
    """
    if shapes is None:
        from .. import coverage
        shapes = coverage.refine_shapes(model)
    graph = model.graph
    nodes = list(graph.node)
    n_nodes = len(nodes)
    outputs = {o.name for o in graph.output}
    drop = set(dropped or ())
    skipped, dead, alias = _skipped_and_aliases(nodes, outputs)

    init_shape = {i.name: _weight_shape(i) for i in graph.initializer}
    init_names = set(init_shape)

    def canon(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    # Views first: their output is the input buffer, so every consumer of the
    # view is a use of the source tensor.  Topological order makes `canon`
    # resolve a chain of views in one pass.
    for i, n in enumerate(nodes):
        if i in skipped or n.op_type not in VIEW_OPS:
            continue
        if not n.input or not n.input[0]:
            continue
        src = canon(n.input[0])
        for o in n.output:
            if o and o not in dead:
                alias[o] = src

    # --- live intervals over canonical tensor names ----------------------
    first = {}
    last = {}
    for i, n in enumerate(nodes):
        if i in skipped:
            continue
        for x in n.input:
            if not x or x in dead:
                continue
            c = canon(x)
            if c not in first:
                first[c] = 0 if c in init_names else i
            last[c] = i
    for o in graph.output:
        c = canon(o.name)
        if c not in first:
            first[c] = 0
        last[c] = n_nodes

    # --- permanent: constants and packed conv panels ---------------------
    ranges = []
    for init in graph.initializer:
        if init.data_type != FLOAT or init.name in drop:
            continue
        b = weight_bytes(init.dims)
        if b:
            ranges.append(TensorRange("const:" + init.name, "const", b, 0,
                                      n_nodes))

    transient = 0
    copy = 0
    for i, n in enumerate(nodes):
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        ws = shapes.get(n.input[1]) or init_shape.get(n.input[1])
        if not ws or len(ws) != 4:
            continue
        ks = _conv_kernel(n)
        kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(ws[2]),
                                                      int(ws[3]))
        gp = _group(n)
        b = panel_bytes((int(ws[0]), int(ws[1]), kh, kw), gp)
        if b:
            ranges.append(TensorRange("panel:" + (n.name or str(i)), "panel",
                                      b, 0, n_nodes))
        transient = max(transient, align(int(ws[0]) * int(ws[1]) * gp * kh *
                                         kw * 4 + 64))
        # the NHWC conversion of the conv's input pads Cin to 16 and W to 16,
        # so it is much larger than the graph tensor (1.56 MiB for the 3-channel
        # 160x160 input) and is held beside the output while the conv runs
        xs = shapes.get(n.input[0]) or init_shape.get(n.input[0])
        if xs and len(xs) == 4:
            copy = max(copy, tensor_bytes(xs, nhwc=True))

    # The executor's own copy beside a node: a layout conversion, an operand
    # pack, a gather table's target, a row transpose.  One input-sized or
    # output-sized buffer is the model; a conv's raw weight upload and its
    # padded NHWC input are counted above.
    for i, n in enumerate(nodes):
        if i in skipped or n.op_type not in COPY_OPS:
            continue
        ins = max((tensor_bytes(shapes.get(x) or init_shape.get(x))
                   for x in n.input if x), default=0)
        outs = max((tensor_bytes(shapes.get(o))
                    for o in n.output if o), default=0)
        copy = max(copy, ins, outs)

    # --- activations ------------------------------------------------------
    acts = []
    seen = set()

    def add_act(name, kind, b, first_i, last_i):
        if not b or name in seen or name in init_names:
            return
        seen.add(name)
        acts.append(TensorRange(name, kind, b, first_i, last_i))

    for inp in graph.input:
        if inp.name in init_names:
            continue
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        c = canon(inp.name)
        add_act(c, "input", tensor_bytes(dims), 0, last.get(c, n_nodes))

    for i, n in enumerate(nodes):
        if i in skipped:
            continue
        for o in n.output:
            if not o or o in dead:
                continue
            c = canon(o)
            if c in dead:
                continue
            b = tensor_bytes(shapes.get(o), nhwc=(n.op_type == "Conv"))
            add_act(c, "activation", b, first.get(c, i),
                    last.get(c, n_nodes))

    # One shared block for the temporary a node allocates beside its input and
    # output.  It is the largest of the copy estimate and the raw weight
    # upload, not a sum: every node releases its temporary when it finishes.
    temp = align(max(transient, copy)) if (transient or copy) else 0
    if temp:
        ranges.append(TensorRange("scratch:per-node", "temp", temp, 0,
                                  n_nodes))

    # --- offsets: permanent first, then a linear scan over the ranges ----
    offset = 0
    for r in ranges:
        if r.permanent:
            r.offset = offset
            offset += r.bytes
    permanent = offset
    top = offset
    peak = offset
    active = []                     # [(last, offset, bytes)]
    free = []                       # [(start, end)]

    acts.sort(key=lambda r: (r.first, -r.bytes, r.name))
    for r in acts:
        live = []
        for a in active:
            if a[0] < r.first:
                free.append((a[1], a[1] + a[2]))
            else:
                live.append(a)
        active = live
        free.sort()
        merged = []
        for s, e in free:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        free = merged
        at = None
        for gi, (s, e) in enumerate(free):
            if e - s >= r.bytes:
                at = s
                if s + r.bytes < e:
                    free[gi] = (s + r.bytes, e)
                else:
                    free.pop(gi)
                break
        if at is None:
            at = top
            top += r.bytes
        r.offset = at
        active.append((r.last, at, r.bytes))
        if at + r.bytes > peak:
            peak = at + r.bytes

    total = sum(r.bytes for r in acts)
    act_peak = max((r.offset + r.bytes for r in acts), default=0) - permanent
    return MemoryPlan(
        ranges=ranges + acts,
        peak_bytes=int(peak),
        permanent_bytes=int(permanent - temp),
        temp_bytes=int(temp),
        activation_bytes=int(max(act_peak, 0)),
        reused_bytes=int(total - max(act_peak, 0)),
        nodes=n_nodes,
        aliases={k: v for k, v in alias.items() if k != v},
        dropped=tuple(sorted(drop)),
    )


def _numel(init):
    n = 1
    for d in init.dims:
        n *= int(d)
    return n


def legacy_pool_bytes(model, shapes=None):
    """The pre-liveness estimate, kept as the A/B control.

    This is the walk the pool was sized with before the live-range pass.  It
    records an output as used by the node that produces it, so every tensor
    stays live from its producer to the end of the graph and the result is the
    sum of every activation rather than the peak.  ``VC4_NO_MEM_PLAN=1``
    selects it, and ``docs/OPTIMIZER.md`` records the before/after table.
    """
    graph = model.graph
    if shapes is None:
        from .. import coverage
        shapes = coverage.refine_shapes(model)
    init = {i.name: [int(d) for d in i.dims] for i in graph.initializer}

    def shp(name):
        return shapes.get(name) or init.get(name)

    perma = 0
    for it in graph.initializer:
        if it.data_type != FLOAT:
            continue
        b = align(_numel(it) * 4 + 64)
        perma += b

    panels = 0
    transient = 0
    for n in graph.node:
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        ws = shp(n.input[1])
        if not ws or len(ws) != 4 or any(int(d) <= 0 for d in ws):
            continue
        ks = _conv_kernel(n)
        kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(ws[2]),
                                                      int(ws[3]))
        cout, cin = int(ws[0]), int(ws[1]) * _group(n)
        panels += panel_bytes((cout, int(ws[1]), kh, kw), _group(n))
        transient = max(transient, align(cout * cin * kh * kw * 4 + 64))

    last = {}
    for idx, n in enumerate(graph.node):
        for t in n.output:
            if t:
                last[t] = idx
    live = 0
    peak = perma
    sizes = {}
    for idx, n in enumerate(graph.node):
        for t in n.input:
            if t and last.get(t) == idx:
                live -= sizes.pop(t, 0)
        extra = 0
        for t in n.output:
            if not t or t in init:
                continue
            b = tensor_bytes(shapes.get(t), nhwc=(n.op_type == "Conv"))
            if not b:
                continue
            sizes[t] = b
            live += b
        if n.op_type == "Conv" and len(n.input) > 1:
            extra = transient
        peak = max(peak, perma + live + extra)
    return int(peak + panels)


def estimate_pool_bytes(model, shapes=None):
    """Compatibility wrapper: the plan's peak, for callers that only want it."""
    return plan_memory(model, shapes).peak_bytes
