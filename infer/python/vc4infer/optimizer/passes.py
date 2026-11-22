"""The optimiser pass pipeline.

Every pass is a graph-to-graph rewrite with a switch.  The default for every
pass is **off**: a pass ships only after its operator sweep and end-to-end
parity have been measured on the board (docs/OPTIMIZER.md, "Correctness
gates").  Enable them by name:

    VC4_OPT=canonicalise           one pass
    VC4_OPT=all                    every pass in pipeline order
    VC4_OPT=all,-epilogue          all but one (a bisection)
    VC4_OPT=canonicalise,ew        explicit list

Pass names are the :attr:`Pass.key` values.  ``optimise(model, passes=...)``
takes the same grammar and overrides the environment, which is what the tests
and ``infer/tools/plan_diff.py`` use.

What a fused node records
-------------------------
Some rewrites replace several nodes with one *dispatch* while leaving the ONNX
graph valid and semantically exact for any other reader (a fused activation, a
fused SiLU pair).  Those mark the surviving node and its partner instead of
deleting a node whose output is still referenced:

    Mul     rule=silu       replaces=[/model.0/act/Sigmoid]  input=<x>
    Sigmoid rule=fused      into=<Mul name>                  (skipped at run)

Rewrites that remove nodes outright (constant folding, identity elimination,
bias folding, dead-node removal) simply delete them; the surviving node carries
the provenance.  ``Graph.summary()["fused"]`` counts marked dispatches, and the
plan printer reports node count, dispatch count and the op histogram.

Pass 1: canonicalise
    constant-fold      fold a node whose inputs are all plan-time constants
    dead-node          drop nodes whose outputs nothing reads
    identity           remove no-op Reshape/Cast/Transpose/Identity/...
    div-const          Div(x, k) -> Mul(x, 1/k), the executor's own lowering
    sub-const          Sub(x, k) -> Add(x, -k), likewise
Pass 2: ew
    silu               Mul(x, Sigmoid(x))            -> one SiLU launch
    hardswish          Mul(x, HardSigmoid(x))        -> one HardSwish launch
    affine             Add(Mul(x, a), b)             -> one AFFINE launch
Pass 3: epilogue
    conv-bias          Conv + Add(per-channel constant) -> Conv bias
    conv-activation    Conv/MatMul + Relu / Clip[0,6] / silu -> the GEMM's
                       store
"""
from __future__ import annotations

import math
import os

import numpy as np
import onnx
from onnx import TensorProto

from .ir import Graph

# Ops whose value this module can compute on the host from constants alone.
# Deliberately excludes Div, Exp, Sigmoid and the transcendentals: the device
# evaluates those with its own kernels (a Newton reciprocal, a 2^f polynomial)
# and a numpy fold would move the last bits for no gain.  Add/Sub/Mul/Neg/Abs
# are exact IEEE fp32 in both places.
FOLD_OPS = frozenset((
    "Shape", "Size", "Constant", "Identity", "Cast", "Reshape", "Flatten",
    "Squeeze", "Unsqueeze", "Transpose", "Concat", "Gather", "Slice",
    "Add", "Sub", "Mul", "Neg", "Abs",
))
MAX_FOLD_BYTES = 32 << 20


# ----------------------------------------------------------------- records
class RuleStat:
    """What one rule did, for the plan printer and the pass log."""

    __slots__ = ("rule", "matched", "removed", "added", "notes")

    def __init__(self, rule):
        self.rule = rule
        self.matched = 0
        self.removed = []
        self.added = []
        self.notes = []

    def __repr__(self):
        return "<%s matched=%d removed=%d added=%d>" % (
            self.rule, self.matched, len(self.removed), len(self.added))

    def as_dict(self):
        return {"rule": self.rule, "matched": self.matched,
                "removed": list(self.removed), "added": list(self.added),
                "notes": list(self.notes)}


class PassResult:
    """A pass's before/after plans and its per-rule records."""

    __slots__ = ("key", "title", "before", "after", "rules", "error")

    def __init__(self, key, title, before, after, rules, error=None):
        self.key = key
        self.title = title
        self.before = before
        self.after = after
        self.rules = rules
        self.error = error

    @property
    def changed(self):
        return self.before != self.after


class OptimiseResult:
    """The optimised model plus the plan of every pass that ran."""

    __slots__ = ("model", "graph", "results", "enabled")

    def __init__(self, model, graph, results, enabled):
        self.model = model
        self.graph = graph
        self.results = results
        self.enabled = enabled

    @property
    def changed(self):
        return any(r.changed for r in self.results)


# -------------------------------------------------------------------- pass
class Pass:
    key = ""
    title = ""
    default = False

    def run(self, graph, ctx=None):
        raise NotImplementedError


class _Ctx:
    """Small shared helpers for the rules."""

    def __init__(self, graph):
        self.graph = graph
        self.kept_outputs = graph.graph_outputs()

    def consumers(self):
        return self.graph.consumers()

    def producers(self):
        return self.graph.producers()

    def single_consumer(self, name):
        c = self.graph.consumers().get(name, [])
        return c[0] if len(c) == 1 else None

    def detach(self, node):
        """Remove a node and rewire its consumers to its first input."""
        src = node.proto.input[0]
        dst = node.proto.output[0]
        if dst in self.kept_outputs and src in self.kept_outputs:
            return False                 # two outputs cannot share one name
        self.graph.remove(node)
        if dst in self.kept_outputs:
            for vi in self.graph.outputs:
                if vi.name == dst:
                    vi.name = src
            for other in self.graph.nodes:
                for i, nm in enumerate(other.proto.input):
                    if nm == dst:
                        other.proto.input[i] = src
            return True
        self.graph.rename_uses(dst, src)
        return True


# --------------------------------------------------------------- canonical
class Canonicalise(Pass):
    key = "canonicalise"
    title = "constant folding, dead nodes, identity elimination"

    def run(self, graph, ctx=None):
        out = []
        out.append(_fold_constants(graph))
        out.append(_dead_nodes(graph))
        out.append(_identities(graph))
        out.append(_div_const(graph))
        out.append(_sub_const(graph))
        out.append(_dead_nodes(graph))
        graph.drop_unused_initializers()
        graph.reindex()
        graph.refresh()
        return out


def _fold_constants(graph):
    st = RuleStat("constant-fold")
    before = len(graph.nodes)
    folded = {}
    budget = MAX_FOLD_BYTES
    keep = []
    for node in graph.nodes:
        op = node.op_type
        outs = [o for o in node.proto.output if o]
        if (op not in FOLD_OPS or not outs or len(outs) != 1 or
                any(a.type in (onnx.AttributeProto.GRAPH,
                               onnx.AttributeProto.GRAPHS)
                    for a in node.proto.attribute) or
                outs[0] in graph.graph_outputs() or op == "Constant"):
            keep.append(node)
            continue
        ins = [i for i in node.proto.input if i]
        ok = bool(ins)
        for name in ins:
            if name not in folded and name not in graph.initializers and \
                    not graph.is_const(name):
                ok = False
                break
        if not ok:
            keep.append(node)
            continue
        arr = graph.eval_const(node, {**graph.initializers, **folded})
        if arr is None or arr.dtype == object:
            keep.append(node)
            continue
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.nbytes > budget:
            keep.append(node)
            continue
        budget -= arr.nbytes
        name = outs[0]
        folded[name] = arr
        graph.initializers[name] = arr
        st.matched += 1
        st.removed.append(node.name)
    graph.nodes = keep
    st.notes.append("%d -> %d nodes" % (before, len(graph.nodes)))
    graph.reindex()
    graph.refresh()
    return st


def _dead_nodes(graph):
    st = RuleStat("dead-node")
    gouts = graph.graph_outputs()
    changed = True
    while changed:
        changed = False
        used = set(gouts)
        for node in graph.nodes:
            used.update(i for i in node.proto.input if i)
        keep = []
        for node in graph.nodes:
            outs = [o for o in node.proto.output if o]
            if outs and not any(o in used for o in outs):
                st.removed.append(node.name)
                st.matched += 1
                changed = True
                continue
            keep.append(node)
        graph.nodes = keep
    if st.matched:
        graph.reindex()
        graph.refresh()
    return st


def _identities(graph):
    st = RuleStat("identity")
    for node in list(graph.nodes):
        op = node.op_type
        outs = [o for o in node.proto.output if o]
        ins = [i for i in node.proto.input if i]
        if len(outs) != 1 or not ins:
            continue
        keep = None
        if op == "Identity":
            keep = True
        elif op == "Dropout":
            tm = node.attr("training_mode", 0)
            if not tm:
                keep = True
        elif op == "Cast":
            to = int(node.attr("to", TensorProto.FLOAT))
            keep = graph.dtype(ins[0]) == to
        elif op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
            a, b = graph.shape(ins[0]), graph.shape(outs[0])
            if a and b and a == b:
                keep = True
        elif op == "Transpose":
            perm = node.attr("perm")
            rank = len(graph.shape(ins[0]) or [])
            keep = (perm is None and rank <= 1) or (perm == list(range(rank)))
        elif op == "Pad":
            pads = node.attr("pads")
            keep = bool(pads) and all(int(v) == 0 for v in pads)
        elif op == "Concat" and len(ins) == 1:
            keep = True
        if not keep:
            continue
        before = len(graph.nodes)
        _Ctx(graph).detach(node)
        if len(graph.nodes) != before:
            st.matched += 1
            st.removed.append(node.name)
    if st.matched:
        graph.reindex()
        graph.refresh()
    return st


def _new_scalar(graph, base, value, dtype=np.float32):
    name = base
    i = 0
    while name in graph.value_names():
        i += 1
        name = "%s_%d" % (base, i)
    arr = np.asarray(value, dtype).reshape(())
    graph.initializers[name] = arr
    return name


def _const_scalar(graph, name):
    v = graph.const(name)
    if v is None:
        return None
    a = np.asarray(v)
    if a.size != 1:
        return None
    return np.float32(a.ravel()[0])


def _div_const(graph):
    """Div(x, k) -> Mul(x, 1/k); the executor already dispatches it that way."""
    st = RuleStat("div-const")
    for node in list(graph.nodes):
        if node.op_type != "Div" or len(node.proto.input) != 2:
            continue
        x, k = node.proto.input
        c = _const_scalar(graph, k)
        if c is None or graph.is_const(x):
            continue
        if graph.dtype(x) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            continue        # integer Div is a floor/truncate, not a multiply
        val = float(c)
        if val == 0.0 or not math.isfinite(val):
            continue
        recip = _new_scalar(graph, (node.raw_name or "div") + "/recip",
                            1.0 / val)
        node.proto.op_type = "Mul"
        node.proto.input[1] = recip
        st.matched += 1
        st.notes.append("%s: 1/%g" % (node.name, val))
    if st.matched:
        graph.refresh()
    return st


def _sub_const(graph):
    """Sub(x, k) -> Add(x, -k): the executor's _addconst path, one launch."""
    st = RuleStat("sub-const")
    for node in list(graph.nodes):
        if node.op_type != "Sub" or len(node.proto.input) != 2:
            continue
        x, k = node.proto.input
        c = graph.const(k)
        if c is None or graph.is_const(x):
            continue
        neg = np.negative(np.asarray(c, np.float32))
        name = (node.raw_name or "sub") + "/neg"
        i = 0
        while name in graph.value_names():
            i += 1
            name = "%s/neg_%d" % (node.raw_name or "sub", i)
        graph.initializers[name] = np.ascontiguousarray(neg)
        node.proto.op_type = "Add"
        node.proto.input[1] = name
        st.matched += 1
        st.notes.append(node.name)
    if st.matched:
        graph.refresh()
    return st


# -------------------------------------------------------------- elementwise
class ElementwiseFusion(Pass):
    key = "ew"
    title = "elementwise chains into one launch"

    def run(self, graph, ctx=None):
        out = [_fuse_silu(graph), _fuse_hardswish(graph), _fuse_affine(graph)]
        # anything that became dead after a pair fusion
        out.append(_dead_nodes(graph))
        graph.reindex()
        graph.refresh()
        return out


def _mark_skip(graph, node, target, rule):
    node.mark("fused", into=target, rule=rule)


def _fuse_silu(graph):
    st = RuleStat("silu")
    gouts = graph.graph_outputs()
    producers = graph.producers()
    consumers = graph.consumers()
    for node in list(graph.nodes):
        if node.op_type != "Mul" or len(node.proto.input) != 2 or node.fused:
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
            node.mark("silu", replaces=[prod.name], input=x)
            prod.mark("fused", into=node.name, by="silu")
            st.matched += 1
            st.notes.append("%s <- %s" % (node.name, prod.name))
            break
    return st


def _fuse_hardswish(graph):
    st = RuleStat("hardswish")
    gouts = graph.graph_outputs()
    producers = graph.producers()
    consumers = graph.consumers()
    for node in list(graph.nodes):
        if node.op_type != "Mul" or len(node.proto.input) != 2 or node.fused:
            continue
        a, b = node.proto.input
        for cand, x in ((a, b), (b, a)):
            prod = producers.get(cand)
            if prod is None or prod.op_type != "HardSigmoid":
                continue
            if len(prod.proto.input) != 1 or prod.proto.input[0] != x:
                continue
            if cand in gouts or len(consumers.get(cand, [])) != 1:
                continue
            alpha = float(prod.attr("alpha", 0.2))
            beta = float(prod.attr("beta", 0.5))
            # AttributeProto stores float32, so compare with a tolerance: the
            # UOP_HARDSWISH body is the 0.2/0.5 form and nothing else.
            if abs(alpha - 0.2) > 1e-6 or abs(beta - 0.5) > 1e-6:
                continue
            node.mark("hardswish", replaces=[prod.name], input=x)
            prod.mark("fused", into=node.name, by="hardswish")
            st.matched += 1
            st.notes.append("%s <- %s" % (node.name, prod.name))
            break
    return st


def _fuse_affine(graph):
    """Add(Mul(x, a), b) -> one AFFINE launch when a and b are scalars.

    Only this order: (x*a) + b maps to out = x*a + b exactly.  The reversed
    order (x + b)*a would need b*a folded on the host first, which changes one
    rounding, so it is left to the executor's two launches until there is a
    measured reason to take it.
    """
    st = RuleStat("affine")
    gouts = graph.graph_outputs()
    producers = graph.producers()
    consumers = graph.consumers()
    for node in list(graph.nodes):
        if node.op_type != "Add" or len(node.proto.input) != 2 or node.fused:
            continue
        a, b = node.proto.input
        for mul_name, bias_name in ((a, b), (b, a)):
            prod = producers.get(mul_name)
            if prod is None or prod.op_type != "Mul" or prod.fused:
                continue
            if len(prod.proto.input) != 2:
                continue
            alpha = beta = None
            x = None
            c1, c2 = prod.proto.input
            s1, s2 = _const_scalar(graph, c1), _const_scalar(graph, c2)
            bias = _const_scalar(graph, bias_name)
            if bias is None:
                continue
            if s1 is not None:
                alpha, x = s1, c2
            elif s2 is not None:
                alpha, x = s2, c1
            if alpha is None or graph.is_const(x):
                continue
            if mul_name in gouts or len(consumers.get(mul_name, [])) != 1:
                continue
            if not graph.shape(x) or graph.shape(x) != graph.shape(node.outputs[0]):
                continue
            beta = bias
            node.mark("affine", replaces=[prod.name], input=x,
                      alpha=float(alpha), beta=float(beta))
            prod.mark("fused", into=node.name, by="affine")
            st.matched += 1
            st.notes.append("%s <- %s (a=%g b=%g)"
                            % (node.name, prod.name, alpha, beta))
            break
    return st


# ---------------------------------------------------------------- epilogue
class Epilogue(Pass):
    key = "epilogue"
    title = "bias and activation folded into the producing conv"

    def run(self, graph, ctx=None):
        out = [_fold_conv_bias(graph), _fuse_conv_activation(graph)]
        out.append(_dead_nodes(graph))
        graph.reindex()
        graph.refresh()
        return out


def _conv_bias_ok(graph, node):
    """The executor's NHWC conv takes a [Cout] bias; check the geometry."""
    x = graph.shape(node.proto.input[0])
    w = graph.shape(node.proto.input[1])
    if not x or not w or len(x) != 4 or len(w) != 4:
        return False
    if any(d <= 0 for d in x[:2] + w[:2]):
        return False
    ks = node.attr("kernel_shape") or [w[2], w[3]]
    st = node.attr("strides") or [1, 1]
    pd = node.attr("pads") or [0, 0, 0, 0]
    dl = node.attr("dilations") or [1, 1]
    group = int(node.attr("group", 1))
    cin, cout = int(w[1]) * group, int(w[0])
    if ks[0] != ks[1] or ks[0] not in (1, 3, 5):
        return False
    if st[0] != st[1] or st[0] not in (1, 2):
        return False
    if len(set(int(p) for p in pd)) != 1:
        return False
    if dl[0] != dl[1] or not 1 <= int(dl[0]) <= 8:
        return False
    if group != 1:
        return False
    if cin % 16 or cout % 16:
        return False
    if st[0] * cin * 4 > 8191:
        return False
    return True


def _per_channel(value, cout):
    a = np.asarray(value, np.float32)
    if a.size == 1:
        return np.full(cout, float(a.ravel()[0]), np.float32)
    if a.size != cout:
        return None
    return np.ascontiguousarray(a.ravel().astype(np.float32))


def _fold_conv_bias(graph):
    """Conv [+ bias] followed by Add(per-channel constant) -> Conv bias."""
    st = RuleStat("conv-bias")
    gouts = graph.graph_outputs()
    producers = graph.producers()
    consumers = graph.consumers()
    for node in list(graph.nodes):
        if node.op_type != "Conv" or not node.proto.output:
            continue
        out = node.proto.output[0]
        if out in gouts or len(consumers.get(out, [])) != 1:
            continue
        add = consumers[out][0]
        if add.op_type != "Add" or len(add.proto.input) != 2:
            continue
        if add.proto.output[0] in gouts and \
                len(consumers.get(add.proto.output[0], [])) != 0:
            pass
        a, b = add.proto.input
        const = b if graph.is_const(b) else a if graph.is_const(a) else None
        if const is None:
            continue
        shp = graph.shape(out)
        if not shp or len(shp) != 4:
            continue
        cout = int(shp[1])
        cval = graph.const(const)
        if cval is None:
            continue
        bias = _per_channel(cval, cout)
        if bias is None:
            continue
        if not _conv_bias_ok(graph, node):
            st.notes.append("%s: NHWC geometry refused" % node.name)
            continue
        if len(node.proto.input) > 2 and node.proto.input[2]:
            old = graph.const(node.proto.input[2])
            if old is None:
                continue
            old = np.asarray(old, np.float32).ravel()
            if old.size != cout:
                continue
            bias = (old + bias).astype(np.float32)
        name = (node.raw_name or "conv") + "/bias"
        i = 0
        while name in graph.value_names():
            i += 1
            name = "%s/bias_%d" % (node.raw_name or "conv", i)
        graph.initializers[name] = bias
        old_bias = node.proto.input[2] if len(node.proto.input) > 2 else ""
        node.set_inputs([node.proto.input[0], node.proto.input[1], name])
        node.mark("conv-bias", replaces=[add.name, old_bias] if old_bias
                  else [add.name], bias=name)
        graph.remove(add)
        graph.rename_uses(add.proto.output[0], out)
        st.matched += 1
        st.removed.append(add.name)
        st.notes.append("%s: %d-channel bias folded" % (node.name, cout))
    if st.matched:
        graph.reindex()
        graph.refresh()
    return st


def _clip_rule(graph, act):
    """`relu6` for Clip(0, 6), else None."""
    lo = act.attr("min", None)
    hi = act.attr("max", None)
    # opset 11+ carries them as optional inputs instead of attributes
    if lo is None and len(act.proto.input) > 1 and act.proto.input[1]:
        c = graph.const(act.proto.input[1])
        if c is not None and np.asarray(c).size == 1:
            lo = float(np.asarray(c).ravel()[0])
    if hi is None and len(act.proto.input) > 2 and act.proto.input[2]:
        c = graph.const(act.proto.input[2])
        if c is not None and np.asarray(c).size == 1:
            hi = float(np.asarray(c).ravel()[0])
    return "relu6" if lo in (0.0, 0) and hi in (6.0, 6) else None


def _fuse_conv_activation(graph):
    """Conv/MatMul -> Relu | Clip(0,6) | SiLU becomes one launch.

    The GEMM applies the activation to the value it stores, so the second
    pass over the output disappears.  Only the pairs the kernel can take are
    marked (`plan.epilogue_ok` is the same predicate the launch model uses):
    the NHWC conv folds all three, a MatMul relu and clip(0,6) (the asm10
    tile has one free B register, not twelve), a Gemm none.  A pair outside
    that set is left alone rather than marked for a launch that would not
    happen.

    SiLU arrives here already collapsed: the `ew` pass has replaced
    `Mul(x, Sigmoid(x))` with one node whose `silu` record names x, so the
    pattern is Conv -> Mul-with-a-silu-record.  The Sigmoid it absorbed is
    still a graph node marked `fused` (it is what keeps the graph readable),
    which is why the consumer count filters fused nodes out.
    """
    from .plan import epilogue_ok
    st = RuleStat("conv-activation")
    gouts = graph.graph_outputs()
    consumers = graph.consumers()
    for node in list(graph.nodes):
        if node.op_type not in ("Conv", "Gemm", "MatMul") or not node.proto.output:
            continue
        out = node.proto.output[0]
        if out in gouts:
            continue
        # a consumer is gone only if it carries the `fused` rule: `Node.fused`
        # is the whole provenance record, and a SiLU Mul has one
        live = [c for c in consumers.get(out, []) if not c.rule("fused")]
        if len(live) != 1:
            continue
        act = live[0]
        rule = None
        if act.op_type == "Relu":
            rule = "relu"
        elif act.op_type == "Clip":
            rule = _clip_rule(graph, act)
        elif act.op_type == "Mul":
            silu = act.rule("silu")
            if silu and silu.get("input") == out:
                rule = "silu"
        # `act.fused` is truthy for any node with a provenance record, and a
        # SiLU Mul has one (`silu`), so test the fused *rule*: only a node
        # already absorbed by another node must be skipped here.
        if rule is None or act.rule("fused"):
            continue
        if not epilogue_ok(graph, node, rule):
            st.notes.append("%s <- %s (%s): the kernel cannot fold it here"
                            % (node.name, act.name, rule))
            continue
        node.mark("activation", replaces=[act.name], act=rule)
        act.mark("fused", into=node.name, by="activation")
        st.matched += 1
        st.notes.append("%s <- %s (%s)" % (node.name, act.name, rule))
    return st


# --------------------------------------------------------------- pipeline
PIPELINE = (Canonicalise(), ElementwiseFusion(), Epilogue())
PASSES = {p.key: p for p in PIPELINE}


def resolve_passes(spec=None):
    """Turn a pass spec into the ordered list to run.

    ``None`` reads ``$VC4_OPT`` (absent or empty = no passes, the default).
    ``"all"`` enables every pass; a ``-name`` entry disables one after it.
    Unknown names raise ``ValueError`` so a typo cannot silently plan nothing.
    """
    if spec is None:
        spec = os.environ.get("VC4_OPT", "")
    if isinstance(spec, (list, tuple)):
        items = [str(s) for s in spec]
    else:
        items = [s.strip() for s in str(spec).replace(";", ",").split(",")
                 if s.strip()]
    enabled = set()
    for item in items:
        name = item[1:] if item.startswith("-") else item
        if name in ("", "none"):
            continue
        if name == "all":
            if item.startswith("-"):
                enabled.clear()
            else:
                enabled.update(PASSES)
            continue
        if name not in PASSES:
            raise ValueError("unknown optimiser pass %r (have: %s)"
                             % (name, ", ".join(PASSES)))
        if item.startswith("-"):
            enabled.discard(name)
        else:
            enabled.add(name)
    return [p for p in PIPELINE if p.key in enabled]


def optimise(model, passes=None, on_pass=None):
    """Run the enabled passes over ``model`` and return an ``OptimiseResult``.

    ``on_pass(PassResult, graph)`` is called after each pass, which is how the
    plan printer shows before/after for every rewrite; ``graph`` is the live
    IR, valid only for the duration of the call (copy it to keep it).  With no
    pass enabled the input model is returned unchanged (no copy, no inference).
    """
    enabled = resolve_passes(passes)
    if not enabled:
        graph = Graph.from_model(model, copy_model=False)
        return OptimiseResult(model, graph, [], [])
    graph = Graph.from_model(model)
    results = []
    for p in enabled:
        before = graph.summary()
        err = None
        try:
            rules = p.run(graph)
        except Exception as e:                       # noqa: BLE001
            err = e
            rules = []
        after = graph.summary()
        res = PassResult(p.key, p.title, before, after, rules, err)
        results.append(res)
        if on_pass is not None:
            on_pass(res, graph)
        if err is not None:
            raise err
    return OptimiseResult(graph.to_model(), graph, results, enabled)
