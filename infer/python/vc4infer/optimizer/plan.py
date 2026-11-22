"""Plan-level launch accounting for the vc4infer optimiser.

The optimiser's gate is a launch count and a wall time; only the board can give
the second.  This module gives the first from the graph alone, so a rewrite can
be checked, bisected and reviewed before it ever runs on the QPUs:

    plan = plan_of(graph)
    plan.total          # modelled QPU launches
    plan.exact          # the classes the optimiser rewrites (validated)
    plan.copy           # layout copies (modelled, approximate)
    plan.by_op          # op -> launches, descending

The elementwise model is not a guess.  One launch covers ``EW_QPU`` groups of
``EW_GROUP`` floats (the engine's one-group-per-QPU rule in ``vc4_ew.c``), and a
tensor's stored extent is not its logical extent: a conv writes NHWC with a
pitch of ``align16(wo)``, so a [1,16,40,40] activation occupies
``40*48*16 = 30720`` floats, not 25600, and costs 10 launches rather than 9.
Against the recorded per-node profile in
``infer/tools/onnx/yolo160_bench.json`` this reproduces all 134 elementwise
nodes exactly (525 launches: 468 Mul, 38 Add, 4 Sub, 1 Div, 14 Sigmoid) and
all 64 conv rows (128 launches).  ``compare_profile`` re-checks that against a
fresh ``yolo_bench.py --json`` run, which is the command to run on silicon.

Everything else (layout copies, softmax's row copy, the pool border) is marked
approximate: it is included so the plan is not silently short, never used to
claim a gain.  ``Plan.unmodelled`` lists the node ops whose launches are not
modelled at all.
"""
from __future__ import annotations

from onnx import TensorProto

EW_GROUP = 256
EW_QPU = 12
ALIGN16 = 16


def align16(v):
    return (int(v) + ALIGN16 - 1) // ALIGN16 * ALIGN16


def stored_floats(shape, layout):
    """Floats a tensor actually occupies, pitched NHWC included."""
    if not shape or any(int(d) < 0 for d in shape):
        return 0
    if layout[0] == "nhwc" and len(shape) == 4:
        n, c, h, w = (int(d) for d in shape)
        return n * h * layout[1] * c
    n = 1
    for d in shape:
        n *= int(d)
    return n


def elementwise_launches(nstored):
    """ceil(ceil(n/256)/12): one 256-float group per QPU per launch."""
    if nstored <= 0:
        return 0
    groups = (int(nstored) + EW_GROUP - 1) // EW_GROUP
    return (groups + EW_QPU - 1) // EW_QPU


# ops whose output keeps the producer's layout and pitch
_PRESERVE = frozenset((
    "Relu", "Abs", "Neg", "Sign", "Clip", "LeakyRelu", "HardSigmoid",
    "HardSwish", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Reciprocal",
    "Erf", "Sin", "Cos", "Floor", "Ceil", "Round", "Softplus", "Softsign",
    "Elu", "Selu", "Gelu", "Mish", "Silu", "Add", "Sub", "Mul", "Div", "Max",
    "Min", "Pow", "Where", "PRelu", "Dropout", "Identity", "Hardmax",
))
_ELEMENTWISE = frozenset((
    "Relu", "Abs", "Neg", "Sign", "Clip", "LeakyRelu", "HardSigmoid",
    "HardSwish", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Reciprocal",
    "Erf", "Sin", "Cos", "Floor", "Ceil", "Round", "Softplus", "Softsign",
    "Elu", "Selu", "Gelu", "Mish", "Silu", "Add", "Mul", "Max", "Min", "Pow",
    "Where", "PRelu", "Sub", "Div",
))
_COPY = frozenset(("Concat", "Split", "Reshape", "Transpose", "Resize",
                   "Slice", "Gather", "Pad", "Expand", "Tile"))
_VIEW = frozenset(("Identity", "Flatten", "Squeeze", "Unsqueeze", "Dropout"))
_META = frozenset(("Constant", "Shape", "Size", "ConstantOfShape"))


def _int_attr(node, name, default):
    v = node.attr(name, default)
    return default if v is None else v


def implicit_pairs(graph):
    """Sigmoid nodes the executor already folds into their consumer Mul.

    ``Runner._find_silu`` has fused ``Mul(x, Sigmoid(x))`` since before the
    optimiser existed, so a plan of an unoptimised graph must not charge the
    Sigmoid a launch: it would double-count 467 of the 802 launches on
    YOLOv8n-160.  This mirrors that scan exactly (single consumer, not a graph
    output, the Sigmoid's input is the Mul's other operand) and is also what
    lets the plan show the ew pass's real contribution: it moves the rewrite
    from an implicit executor habit into a recorded graph rewrite.
    """
    gouts = graph.graph_outputs()
    producers = graph.producers()
    consumers = graph.consumers()
    skip = set()
    for node in graph.nodes:
        if node.op_type != "Mul" or len(node.proto.input) != 2:
            continue
        a, b = node.proto.input
        for cand, x in ((a, b), (b, a)):
            prod = producers.get(cand)
            if prod is None or prod.op_type != "Sigmoid":
                continue
            if len(prod.proto.input) != 1 or prod.proto.input[0] != x:
                continue
            if cand in gouts or len(consumers.get(cand, [])) != 1:
                continue
            skip.add(prod.name)
            break
    return skip


def conv_nhwc(node, xshape, wshape, group):
    """True when the executor takes the NHWC conv path (the pitch matters)."""
    if not xshape or not wshape or len(xshape) != 4 or len(wshape) != 4:
        return False
    if any(int(d) <= 0 for d in xshape[:2] + wshape[:2]):
        return False
    ks = node.attr("kernel_shape") or [wshape[2], wshape[3]]
    st = node.attr("strides") or [1, 1]
    pd = node.attr("pads") or [0, 0, 0, 0]
    dl = node.attr("dilations") or [1, 1]
    kh, kw = int(ks[0]), int(ks[1])
    if kh not in (1, 3, 5) or kw not in (1, 3, 5):
        return False
    if int(st[0]) != int(st[1]) or int(st[0]) not in (1, 2):
        return False
    if len(set(int(p) for p in pd)) != 1:
        return False
    if int(dl[0]) != int(dl[1]) or not 1 <= int(dl[0]) <= 8:
        return False
    cin = int(wshape[1]) * group
    cout = int(wshape[0])
    if group != 1:
        # depthwise and 16-aligned grouped blocks are the NHWC path
        depthwise = group == cin
        grouped = (cin % group == 0 and cout % group == 0 and
                   (cin // group) % 16 == 0 and (cout // group) % 16 == 0)
        if not (depthwise or grouped):
            return False
    if int(st[0]) * cin * 4 > 8191:
        return False
    return True


def layouts(graph):
    """name -> ("nhwc", pitch) | ("nchw", 0), following the executor's rules."""
    out = {}
    producers = graph.producers()
    for name in graph.initializers:
        out[name] = ("nchw", 0)
    for node in graph.nodes:
        op = node.op_type
        ins = [i for i in node.proto.input if i]
        src = out.get(ins[0], ("nchw", 0)) if ins else ("nchw", 0)
        loc = ("nchw", 0)
        if op == "Conv":
            x = graph.shape(ins[0]) if ins else None
            w = graph.shape(ins[1]) if len(ins) > 1 else None
            group = int(node.attr("group", 1))
            if conv_nhwc(node, x, w, group):
                ks = node.attr("kernel_shape") or [w[2], w[3]]
                st = node.attr("strides") or [1, 1]
                pd = node.attr("pads") or [0, 0, 0, 0]
                dl = node.attr("dilations") or [1, 1]
                wo = (int(x[3]) + int(pd[1]) + int(pd[3]) -
                      int(dl[1]) * (int(ks[1]) - 1) - 1) // int(st[1]) + 1
                loc = ("nhwc", align16(wo))
        elif op in _PRESERVE:
            loc = src
        elif op == "Split":
            loc = src
        elif op == "Concat":
            ls = [out.get(i, ("nchw", 0)) for i in ins]
            shp = graph.shape(node.proto.output[0]) if node.proto.output else None
            if ls and all(v[0] == "nhwc" for v in ls) and shp and len(shp) == 4:
                loc = ("nhwc", int(shp[3]))
        elif op in ("MaxPool", "Resize", "Softmax"):
            loc = src if src[0] == "nhwc" else ("nchw", 0)
        for name in node.proto.output:
            if name:
                out[name] = loc
    return out


def _softmax_launches(node, shape, layout):
    if not shape:
        return 0
    rank = len(shape)
    axis = int(node.attr("axis", -1))
    ax = axis + rank if axis < 0 else axis
    if layout[0] == "nhwc" and ax == 1 and rank == 4:
        rows = int(shape[0]) * int(shape[2]) * int(shape[3])
    else:
        c = int(shape[ax]) if 0 <= ax < rank else 1
        n = 1
        for d in shape:
            n *= int(d)
        rows = n // c if c else 0
    rp = (rows + 7) // 8 * 8
    # vc4_softmax_fwd: one launch per 12 groups of 8 rows, plus the host-side
    # row copy into the 16-column padded buffer (one copy2d call).
    return (rp // 8 + EW_QPU - 1) // EW_QPU + (1 if rows else 0)


def _act_launches(node, shape, lay):
    """An epilogue activation is still an elementwise launch on this engine."""
    if not node.rule("activation"):
        return 0
    return elementwise_launches(stored_floats(shape, lay))


def node_launches(graph, node, loc):
    """Modelled launches for one node.  Returns (exact, copy)."""
    op = node.op_type
    outs = [o for o in node.proto.output if o]
    out = outs[0] if outs else None
    shape = graph.shape(out) if out else None
    lay = loc.get(out, ("nchw", 0))
    ins = [i for i in node.proto.input if i]

    # Integer and boolean tensors are shape plumbing under the device policy:
    # the executor resolves them at plan time (coverage.fold_metadata), so they
    # cost no launch.  A numeric integer kernel (ArgMax's int64 output, a Cast
    # to int) is outside this model; the corpus is fp32 detector graphs and the
    # plan is only used to claim launch deltas for the passes below.
    if graph.dtype(out) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        return 0, 0

    if op in _META or op in _VIEW:
        return 0, 0
    if op == "Conv":
        w = graph.shape(ins[1]) if len(ins) > 1 else None
        group = int(node.attr("group", 1))
        cin = int(w[1]) * group if w else 1
        if group == cin and group > 1:
            n = 1                           # depthwise: one kernel
        elif group > 1:
            n = 1 + group                   # one GEMM per group
        else:
            n = 2                           # im2col + GEMM
        return n + _act_launches(node, shape, lay), 0
    if op in ("MatMul",):
        return 2 + _act_launches(node, shape, lay), 0
    if op == "Gemm":
        n = 3 if len(ins) > 2 and node.proto.input[2] else 2
        return n + _act_launches(node, shape, lay), 0
    if op in ("GlobalAveragePool",):
        return 2, 0
    if op == "Softmax":
        return _softmax_launches(node, shape, lay), 0
    if op == "MaxPool":
        ks = node.attr("kernel_shape")
        pads = node.attr("pads") or [0, 0, 0, 0]
        st = node.attr("strides") or [1, 1]
        if (ks and int(st[0]) == 1 and pad_needs_border(ks, pads) and
                lay[0] == "nhwc"):
            return 1, 2                     # fill2d + interior copy + kernel
        return 1, 0
    if op == "Resize":
        return 0, 1
    if op in ("Concat", "Split"):
        n = len(ins) if op == "Concat" else len(outs)
        if lay[0] != "nhwc" and op == "Concat":
            return 0, 0                     # the host concatenate path
        batch = int(shape[0]) if shape and len(shape) == 4 else 1
        return 0, max(1, n) * max(1, batch)
    if op in ("Reshape", "Transpose", "Pad"):
        return 0, 1
    if op in ("Slice", "Gather", "Expand", "Tile"):
        return 0, 1
    if op in ("Sub", "Div"):
        # Sub: the executor's Mul(-1)+Add unless one operand is a constant
        # (then one addconst).  Div by a constant: one Mul by the reciprocal.
        if graph.dtype(out) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            return 0, 0         # integer plumbing: host-meta, no launch
        n = elementwise_launches(stored_floats(shape, lay))
        left = node.proto.input[0] if len(ins) > 0 else ""
        right = node.proto.input[1] if len(ins) > 1 else ""
        if op == "Div":
            return (n if graph.is_const(right) else 0), 0
        if graph.is_const(right) and not graph.is_const(left):
            return n, 0
        return 2 * n, 0
    if op in _ELEMENTWISE:
        if graph.dtype(out) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            return 0, 0         # integer/boolean plumbing: host-meta, no launch
        return elementwise_launches(stored_floats(shape, lay)), 0
    return 0, 0


def pad_needs_border(ks, pads):
    if not ks or len(ks) < 2 or ks[0] != ks[1]:
        return False
    n = int(ks[0])
    return n in (3, 5) and list(int(p) for p in pads) == [(n - 1) // 2] * 4


class Plan:
    """Launch and node accounting for one graph.

    ``implicit`` (default) counts the Sigmoid half of a SiLU pair as fused,
    because ``Runner._find_silu`` has always done that rewrite.  Pass
    ``implicit=False`` to count what the graph would cost if that special case
    were removed; the difference is what the ew pass takes ownership of.
    """

    def __init__(self, graph, implicit=True):
        self.graph = graph
        self.nodes = len(graph.nodes)
        self.implicit_pairs = implicit_pairs(graph) if implicit else set()
        implicit_nodes = self.implicit_pairs
        self.fused_away = {n.name for n in graph.nodes if n.rule("fused")}
        self.dispatches = sum(
            1 for n in graph.nodes
            if n.name not in self.fused_away and n.name not in implicit_nodes)
        self.histogram = graph.histogram()
        loc = layouts(graph)
        self.by_op = {}
        self.per_node = {}
        self.exact = 0
        self.copy = 0
        self.unmodelled = set()
        for node in graph.nodes:
            if node.rule("fused") or node.name in implicit_nodes:
                # a fused-away node launches nothing; the surviving node of
                # the pair carries the launch (and shows up as rule=silu)
                self.per_node[node.name] = 0
                continue
            exact, copy = node_launches(graph, node, loc)
            self.per_node[node.name] = exact + copy
            if exact or copy:
                self.by_op[node.op_type] = \
                    self.by_op.get(node.op_type, 0) + exact + copy
            self.exact += exact
            self.copy += copy
            if not exact and not copy and node.op_type not in (
                    _META | _VIEW) and node.op_type not in _COPY:
                if graph.shape(node.proto.output[0] if node.proto.output else ""):
                    self.unmodelled.add(node.op_type)
        self.by_op = dict(sorted(self.by_op.items(),
                                 key=lambda kv: (-kv[1], kv[0])))

    @property
    def total(self):
        return self.exact + self.copy

    def summary(self):
        return {"nodes": self.nodes, "dispatches": self.dispatches,
                "launches": self.total, "exact": self.exact,
                "copy": self.copy, "ops": self.histogram}

    def diff(self, other):
        """other - self, per op and in total."""
        ops = {}
        for op in set(self.by_op) | set(other.by_op):
            d = other.by_op.get(op, 0) - self.by_op.get(op, 0)
            if d:
                ops[op] = d
        return {"nodes": other.nodes - self.nodes,
                "dispatches": other.dispatches - self.dispatches,
                "launches": other.total - self.total,
                "exact": other.exact - self.exact,
                "copy": other.copy - self.copy,
                "ops": dict(sorted(ops.items(), key=lambda kv: kv[1]))}


def plan_of(graph, implicit=True):
    return Plan(graph, implicit=implicit)


# Ops whose launch count the model reproduces from the graph exactly.  These
# are the classes every optimiser pass in this module rewrites.
EXACT_OPS = frozenset(_ELEMENTWISE | {"Conv", "MatMul", "Gemm"})


def compare_profile(plan, profile):
    """Compare the model with a runner profile (``yolo_bench.py --json``).

    Returns ``(checked, exact_bad, copy_bad)``.  ``exact_bad`` must be empty:
    it lists a node whose launch count in the modelled class disagrees with a
    recorded run.  ``copy_bad`` is expected when the profile predates the QPU
    layout copies (the recorded 678-launch profile does: 40 such rows, all
    label/concat/resize/reshape/softmax-copy nodes, none of them a class a pass
    rewrites).  Only recorded nodes are compared, so a partial profile cannot
    hide an over-count.
    """
    want = {p["name"]: p for p in profile}
    checked = 0
    exact_bad, copy_bad = [], []
    for name, get in plan.per_node.items():
        rec = want.get(name)
        if rec is None:
            continue
        checked += 1
        op = rec.get("op")
        if int(rec.get("launches", 0)) != get:
            row = (name, op, get, int(rec.get("launches", 0)))
            (exact_bad if op in EXACT_OPS else copy_bad).append(row)
    return checked, exact_bad, copy_bad


def format_plan(plan, label=""):
    head = "%-14s %5s %9s" % ("op", "nodes", "launches")
    lines = []
    if label:
        lines.append(label)
    lines.append("nodes %d  dispatches %d  launches %d "
                 "(exact %d + copy ~%d)"
                 % (plan.nodes, plan.dispatches, plan.total, plan.exact,
                    plan.copy))
    lines.append(head)
    ops = [n.op_type for n in plan.graph.nodes]
    for op, launches in plan.by_op.items():
        lines.append("%-14s %5d %9d" % (op, ops.count(op), launches))
    if plan.unmodelled:
        lines.append("ops with unmodelled nodes: %s"
                     % ", ".join(sorted(plan.unmodelled)))
    return "\n".join(lines)
