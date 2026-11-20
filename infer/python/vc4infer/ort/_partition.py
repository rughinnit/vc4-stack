"""Turn one ONNX graph into VC4-runnable and ORT-runnable sub-models.

This is the partitioning half of ``vc4infer.ort``.  It answers one question
per node -- "can libvc4infer execute exactly this node with this shape?" --
using the same static analysis that backs ``vc4infer nodes`` (``coverage.py``),
and it never guesses:

    tier "vc4"        a VC4 kernel exists and the static constraints hold
    tier "host-meta"  Constant/Shape and integer shape arithmetic; the strict
                      executor resolves these at plan time or with host integer
                      bookkeeping, so they may be folded into a VC4 component
    anything else     left to ORT, and reported

A component is claimed only when every node in it is claimable *and* every
tensor that crosses its boundary is fp32 with a fully static shape.  A dynamic
dimension or an integer boundary tensor downgrades the whole component: the
engine has no runtime fallback, and this provider will not invent one.

Components are maximal connected sets under the tensor graph.  Each becomes a
self-contained ONNX ModelProto: nodes in the original order, the initialisers
those nodes reference, and one graph input/output per boundary tensor.  The
VC4 sub-model is written to the cache once (initialiser layout conversion
happens there, not per run); the ORT sub-model is kept in memory.
"""
import hashlib
import os
import tempfile

import onnx
from onnx import ModelProto, TensorProto

from .. import coverage

# Bump when the sub-model shape changes: invalidates the on-disk cache.
FORMAT = "vc4infer-ort-1"

CLAIM_TIERS = ("vc4", "host-meta")


class Partition:
    """One sub-graph and the backend that executes it."""

    __slots__ = ("index", "backend", "indices", "inputs", "outputs", "macs",
                 "path", "model_bytes", "deps")

    def __init__(self, index, backend, indices, inputs, outputs, macs):
        self.index = index
        self.backend = backend
        self.indices = list(indices)
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.macs = int(macs)
        self.path = None          # VC4 sub-model file
        self.model_bytes = None   # ORT sub-model bytes
        self.deps = set()

    @property
    def first_index(self):
        return min(self.indices) if self.indices else -1

    def __repr__(self):
        return "Partition(%d, %s, %d nodes)" % (
            self.index, self.backend, len(self.indices))


def _shape_infer(model):
    try:
        return onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True)
    except Exception:
        return model


def _value_map(model):
    """name -> ValueInfoProto from shape inference (best effort)."""
    inf = _shape_infer(model)
    out = {}
    for vi in list(inf.graph.input) + list(inf.graph.value_info) + \
            list(inf.graph.output):
        out[vi.name] = vi
    return out


def _dtype_map(model, values):
    """name -> TensorProto elem_type, including initialisers."""
    out = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + \
            list(model.graph.output):
        out[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        out[init.name] = init.data_type
    for name, vi in values.items():
        out.setdefault(name, vi.type.tensor_type.elem_type)
    return out


def _fixed_dims(vi):
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return False
    return all(d.HasField("dim_value") and d.dim_value > 0 for d in tt.shape.dim)


def _boundaries(indices, graph, produced_by, consumed_by, inits, gouts):
    """Tensors entering and leaving a node set, in first-use order."""
    inside = set(indices)
    ins, outs = [], []
    seen_in, seen_out = set(), set()
    for idx in indices:
        node = graph.node[idx]
        for t in node.input:
            if not t or t in inits or t in seen_in:
                continue
            p = produced_by.get(t)
            if p is None or p not in inside:
                ins.append(t)
                seen_in.add(t)
        for t in node.output:
            if not t or t in seen_out:
                continue
            users = consumed_by.get(t, ())
            if t in gouts or any(u not in inside for u in users):
                outs.append(t)
                seen_out.add(t)
    return ins, outs


def _components(nodes, adj):
    """Maximal connected components of ``nodes`` under ``adj``."""
    left = set(nodes)
    out = []
    while left:
        seed = min(left)
        stack = [seed]
        left.discard(seed)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb in left:
                    left.discard(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    out.sort(key=lambda c: c[0])
    return out


def _adjacency(graph):
    produced_by, consumed_by = {}, {}
    for idx, node in enumerate(graph.node):
        for t in node.output:
            if t:
                produced_by[t] = idx
        for t in node.input:
            if t:
                consumed_by.setdefault(t, []).append(idx)
    adj = {}
    for t, prod in produced_by.items():
        for user in consumed_by.get(t, ()):
            if user != prod:
                adj.setdefault(prod, set()).add(user)
                adj.setdefault(user, set()).add(prod)
    return produced_by, consumed_by, adj


def _has_subgraph(node):
    return any(a.type in (onnx.AttributeProto.GRAPH,
                          onnx.AttributeProto.GRAPHS) for a in node.attribute)


def _vi_for(name, values, shapes):
    vi = values.get(name)
    if vi is None or not vi.type.tensor_type.HasField("elem_type"):
        vi = onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        dims = shapes.get(name)
        if dims is not None:
            vi = onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                                    list(dims))
    return vi


def _submodel(model, indices, inputs, outputs, values, shapes):
    g = onnx.GraphProto()
    g.name = "vc4_partition"
    for idx in indices:
        g.node.add().CopyFrom(model.graph.node[idx])
    used = set()
    for idx in indices:
        for t in model.graph.node[idx].input:
            used.add(t)
    for init in model.graph.initializer:
        if init.name in used:
            g.initializer.add().CopyFrom(init)
    for name in inputs:
        g.input.add().CopyFrom(_vi_for(name, values, shapes))
    for name in outputs:
        g.output.add().CopyFrom(_vi_for(name, values, shapes))
    m = ModelProto()
    m.ir_version = model.ir_version
    m.producer_name = "vc4infer.ort"
    m.producer_version = "partition"
    m.opset_import.extend(model.opset_import)
    if len(model.functions):
        m.functions.extend(model.functions)
    m.graph.CopyFrom(g)
    return m


def _cache_root():
    base = os.environ.get("VC4INFER_ORT_CACHE")
    if not base:
        base = os.path.join(
            os.environ.get("XDG_CACHE_HOME") or
            os.path.join(os.path.expanduser("~"), ".cache"),
            "vc4infer", "ort")
    return base


def _write_cached(model_bytes, part, proto):
    """Write one VC4 sub-model once, atomically; return its path."""
    digest = hashlib.sha256(model_bytes).hexdigest()[:16]
    d = os.path.join(_cache_root(), digest, FORMAT)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "p%d.onnx" % part.index)
    if not os.path.isfile(path):
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        os.close(fd)
        onnx.save(proto, tmp)
        os.replace(tmp, path)
    return path


def build(model, model_bytes):
    """Partition ``model``; return ``(partitions, report)``.

    ``partitions`` is dependency-ordered.  ``report`` is the coverage dict from
    ``coverage.analyse_model`` so the caller can build the honesty table
    without analysing the model twice.
    """
    rep = coverage.analyse_model(model)
    graph = model.graph
    produced_by, consumed_by, adj = _adjacency(graph)
    inits = set(i.name for i in graph.initializer)
    gouts = set(o.name for o in graph.output)
    values = _value_map(model)
    shapes = rep.get("shapes") or {}
    dtypes = _dtype_map(model, values)

    claimable = set()
    for e in rep["nodes"]:
        if e["tier"] in CLAIM_TIERS and e["status"] != "unsupported":
            claimable.add(e["index"])

    # Control flow is never split: a subgraph attribute can capture tensors
    # from the outer graph, and reconstructing that capture safely is ORT's
    # job, not this partitioner's.
    if any(_has_subgraph(n) for n in graph.node):
        claimable = set()

    # Refuse to claim a component unless its whole boundary is fp32/static.
    def valid(indices):
        if not any(rep["nodes"][i]["tier"] == "vc4" for i in indices):
            return False          # only folded metadata: let ORT have it
        ins, outs = _boundaries(indices, graph, produced_by, consumed_by,
                                inits, gouts)
        for name in ins + outs:
            if dtypes.get(name) != TensorProto.FLOAT:
                return False
            vi = values.get(name)
            if vi is None or not _fixed_dims(vi):
                return False
        return True

    while True:
        comps = _components(claimable, adj)
        bad = [c for c in comps if not valid(c)]
        if not bad:
            break
        for c in bad:
            claimable.difference_update(c)
        if not claimable:
            break

    vc4_nodes = set(claimable)
    ort_nodes = set(range(len(graph.node))) - vc4_nodes
    groups = [(c, "vc4") for c in _components(vc4_nodes, adj)]
    groups += [(c, "ort-cpu") for c in _components(ort_nodes, adj)]

    parts = []
    for i, (indices, backend) in enumerate(groups):
        ins, outs = _boundaries(indices, graph, produced_by, consumed_by,
                                inits, gouts)
        macs = sum(int(rep["nodes"][j].get("macs") or 0) for j in indices)
        p = Partition(i, backend, indices, ins, outs, macs)
        if backend == "vc4":
            proto = _submodel(model, indices, ins, outs, values, shapes)
            p.path = _write_cached(model_bytes, p, proto)
        else:
            proto = _submodel(model, indices, ins, outs, values, shapes)
            p.model_bytes = proto.SerializeToString()
        parts.append(p)

    # Dependency edges and a deterministic topological order.
    producer = {}
    for p in parts:
        for t in p.outputs:
            producer[t] = p
    for p in parts:
        for t in p.inputs:
            q = producer.get(t)
            if q is not None and q is not p:
                p.deps.add(q.index)

    order, done = [], set()
    remaining = sorted(parts, key=lambda p: p.first_index)
    while remaining:
        ready = [p for p in remaining if p.deps <= done]
        if not ready:                      # a cycle cannot happen in an ONNX DAG
            ready = remaining[:1]
        p = ready[0]
        order.append(p)
        done.add(p.index)
        remaining.remove(p)

    return order, rep
