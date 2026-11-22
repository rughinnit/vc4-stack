"""A small graph IR for the vc4infer optimiser.

The executor is a node-by-node interpreter over ``onnx.GraphProto``: a node's
inputs and outputs are tensor names, its attributes are ``AttributeProto``, and
the only plan-time state it keeps is a shape map, an initialiser table and the
SiLU rewrite.  An optimiser needs to rewrite that graph, so this module wraps
the same information in objects a pass can edit:

    Node   one ONNX node: op, name, inputs, outputs, attributes, provenance
    Graph  an ordered node list plus initialisers, graph inputs/outputs and
           static shapes/dtypes

Fidelity matters more than convenience.  ``Node.proto`` is the original
``NodeProto``, so a pass that does not touch a node leaves its attributes,
domain and subgraphs byte-identical; a rewritten node is a copy with the
fields the pass changed.  ``Graph.to_model`` therefore emits a model that
still passes ``onnx.checker`` for every graph the executor accepted.

Provenance
----------
A fused node has to say what it replaced: the executor is debugged from the
plan, and a regression is bisected from the plan diff.  ONNX has no general
"replaces" attribute and ``onnx.checker`` rejects unknown attributes on
standard ops, so provenance is written on one line of ``NodeProto.doc_string``:

    vc4-opt: {"rule": "silu", "replaces": ["/model.0/act/Sigmoid", ...],
              "input": "/model.0/conv/Conv_output_0"}

``Node.fused`` parses it back into a dict, and ``Node.mark`` writes it while
keeping any documentation the model already carried.  The record is data, not
prose: the plan printer, the coverage report and the executor all read it,
and a node with no marker behaves exactly as it did before the optimiser ran.

Static shapes
-------------
``Graph.shape(name)`` is not ONNX shape inference: for a fixed-shape model ONNX
leaves the head's dynamic ``Reshape``/``Slice`` chain as ``-1`` because it does
not follow an initialiser through ``Gather`` and ``Slice``.  This module
propagates shapes itself, with the constants resolved, and falls back to ONNX
inference for any name it cannot compute.  The launch model needs the stored
extent of every elementwise tensor, so an unresolved dim there would silently
undercount launches.
"""
from __future__ import annotations

import copy
import json
from itertools import zip_longest

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

# Marker prefix and the node-name fallback the rest of the executor uses.
OPT_PREFIX = "vc4-opt: "
_UNARY = frozenset((
    "Abs", "Neg", "Sign", "Not", "Reciprocal", "Sqrt", "Exp", "Log", "Round",
    "Floor", "Ceil", "Relu", "LeakyRelu", "Elu", "Selu", "Softplus",
    "Softsign", "HardSigmoid", "HardSwish", "Sigmoid", "Tanh", "Erf", "Sin",
    "Cos", "Asin", "Acos", "Atan", "Asinh", "Acosh", "Atanh", "Sinh", "Cosh",
    "Tan", "Gelu", "Mish", "Silu", "IsNaN", "IsInf", "Identity", "Dropout",
    "Cast", "Clip", "Shrink", "ThresholdedRelu", "Celu",
))
_BINARY = frozenset((
    "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mod", "And", "Or",
    "Xor", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "PRelu",
))
_COMPARE = frozenset((
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "IsNaN",
    "IsInf", "And", "Or", "Xor", "Not", "Where",
))
_INT_OPS = frozenset((
    "Shape", "Size", "ArgMax", "ArgMin", "NonZero", "TopK", "Gather",
    "GatherElements", "GatherND", "ScatterElements", "ScatterND",
))


# --------------------------------------------------------------- attributes
def _attr_to_py(attr):
    """AttributeProto -> a plain Python value (GRADIENT of subgraphs kept raw)."""
    t = attr.type
    if t == AttributeProto.INT:
        return int(attr.i)
    if t == AttributeProto.FLOAT:
        return float(attr.f)
    if t == AttributeProto.STRING:
        return attr.s.decode("utf-8", "replace")
    if t == AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if t == AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if t == AttributeProto.STRINGS:
        return [v.decode("utf-8", "replace") for v in attr.strings]
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if t == AttributeProto.TENSORS:
        return [numpy_helper.to_array(v) for v in attr.tensors]
    return attr  # GRAPH/GRAPHS/SPARSE_*: never rewritten by a pass here


def _py_to_attr(name, value):
    if isinstance(value, bool):
        return helper.make_attribute(name, int(value))
    if isinstance(value, (int, np.integer)):
        return helper.make_attribute(name, int(value))
    if isinstance(value, (float, np.floating)):
        return helper.make_attribute(name, float(value))
    if isinstance(value, str):
        return helper.make_attribute(name, value)
    if isinstance(value, np.ndarray):
        return helper.make_attribute(name, numpy_helper.from_array(value))
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str):
            return helper.make_attribute(name, [str(v) for v in value])
        if value and isinstance(value[0], (float, np.floating)):
            return helper.make_attribute(name, [float(v) for v in value])
        return helper.make_attribute(name, [int(v) for v in value])
    raise TypeError("attribute %s: cannot encode %r" % (name, type(value)))


# --------------------------------------------------------------------- node
class Node:
    """One ONNX node, editable in place, with an optional provenance record."""

    __slots__ = ("proto", "index")

    def __init__(self, proto, index=None):
        self.proto = proto
        self.index = index

    # -- identity ----------------------------------------------------------
    @property
    def op_type(self):
        return self.proto.op_type

    @property
    def name(self):
        return self.proto.name or "%s_%s" % (self.proto.op_type, self.index)

    @property
    def raw_name(self):
        return self.proto.name

    @property
    def inputs(self):
        return list(self.proto.input)

    @property
    def outputs(self):
        return list(self.proto.output)

    def set_inputs(self, names):
        del self.proto.input[:]
        self.proto.input.extend([n or "" for n in names])

    def set_outputs(self, names):
        del self.proto.output[:]
        self.proto.output.extend([n or "" for n in names])

    def __repr__(self):
        return "<Node %s %s>" % (self.op_type, self.name)

    def clone(self):
        return Node(copy.deepcopy(self.proto), self.index)

    # -- attributes --------------------------------------------------------
    def attr(self, name, default=None):
        for a in self.proto.attribute:
            if a.name == name:
                return _attr_to_py(a)
        return default

    def attrs(self):
        return {a.name: _attr_to_py(a) for a in self.proto.attribute}

    def set_attr(self, name, value):
        attr = _py_to_attr(name, value)
        for i, a in enumerate(self.proto.attribute):
            if a.name == name:
                self.proto.attribute[i].CopyFrom(attr)
                return
        self.proto.attribute.append(attr)

    def del_attr(self, name):
        keep = [a for a in self.proto.attribute if a.name != name]
        if len(keep) != len(self.proto.attribute):
            del self.proto.attribute[:]
            self.proto.attribute.extend(keep)
            return True
        return False

    # -- provenance --------------------------------------------------------
    @property
    def fused(self):
        """The provenance record a pass left, or None.

        A node can carry more than one rewrite (a conv may have its bias
        folded *and* an activation attached), so the record is a map from rule
        name to that rule's fields plus a ``replaces`` list.
        """
        rec = None
        for line in (self.proto.doc_string or "").splitlines():
            if not line.startswith(OPT_PREFIX):
                continue
            try:
                got = json.loads(line[len(OPT_PREFIX):])
            except ValueError:
                continue
            if "rules" in got:
                rec = rec or {"rules": {}, "replaces": []}
                rec["rules"].update(got.get("rules") or {})
                rec["replaces"].extend(got.get("replaces") or [])
        return rec

    def rule(self, name):
        rec = self.fused
        return (rec or {}).get("rules", {}).get(name)

    def rules(self):
        rec = self.fused
        return sorted((rec or {}).get("rules", {}))

    def mark(self, rule, replaces=(), **fields):
        """Record "this node stands for `replaces` after `rule`"."""
        rec = self.fused or {"rules": {}, "replaces": []}
        entry = dict(fields)
        entry["replaces"] = [str(r) for r in replaces if r]
        rec["rules"][rule] = entry
        for r in replaces:
            if r and str(r) not in rec["replaces"]:
                rec["replaces"].append(str(r))
        line = OPT_PREFIX + json.dumps(rec, sort_keys=True)
        lines = [ln for ln in (self.proto.doc_string or "").splitlines()
                 if not ln.startswith(OPT_PREFIX)]
        lines.append(line)
        self.proto.doc_string = "\n".join(lines)

    def unmark(self):
        lines = [ln for ln in (self.proto.doc_string or "").splitlines()
                 if not ln.startswith(OPT_PREFIX)]
        self.proto.doc_string = "\n".join(lines)


# -------------------------------------------------------------------- graph
class Graph:
    """An ordered node list with static shapes, types and initialisers."""

    def __init__(self, nodes, initializers, inputs, outputs, meta=None):
        self.nodes = list(nodes)
        self.initializers = dict(initializers)      # name -> np.ndarray
        self.inputs = list(inputs)                  # ValueInfoProto
        self.outputs = list(outputs)                # ValueInfoProto
        self.meta = dict(meta or {})
        self._shapes = {}
        self._dtypes = {}
        self._const = {}
        self._infer()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_model(cls, model, copy_model=True):
        model = copy.deepcopy(model) if copy_model else model
        inits = {}
        for init in model.graph.initializer:
            try:
                inits[init.name] = numpy_helper.to_array(init)
            except Exception:                       # external data, etc.
                inits[init.name] = None
        nodes = [Node(n, i) for i, n in enumerate(model.graph.node)]
        meta = {
            "graph_name": model.graph.name,
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "model_version": model.model_version,
            "doc_string": model.doc_string,
            "opset": [(o.domain, int(o.version)) for o in model.opset_import],
            "metadata": {p.key: p.value for p in model.metadata_props},
            "functions": list(model.functions) if len(model.functions) else [],
        }
        return cls(nodes, inits, model.graph.input, model.graph.output, meta)

    def to_model(self):
        graph = onnx.GraphProto()
        graph.name = self.meta.get("graph_name") or "vc4opt"
        for vi in self.inputs:
            graph.input.append(copy.deepcopy(vi))
        for vi in self.outputs:
            graph.output.append(copy.deepcopy(vi))
        for name, arr in self.initializers.items():
            if arr is None:
                continue
            graph.initializer.append(numpy_helper.from_array(arr, name))
        for n in self.nodes:
            graph.node.append(copy.deepcopy(n.proto))
        model = onnx.ModelProto()
        model.ir_version = self.meta.get("ir_version") or \
            onnx.IR_VERSION
        if self.meta.get("producer_name"):
            model.producer_name = self.meta["producer_name"]
        if self.meta.get("producer_version"):
            model.producer_version = self.meta["producer_version"]
        model.graph.CopyFrom(graph)
        for fn in (self.meta.get("functions") or []):
            try:
                model.functions.append(copy.deepcopy(fn))
            except Exception:
                pass
        for domain, version in self.meta.get("opset") or [("", 17)]:
            model.opset_import.add(domain=domain, version=version)
        for k, v in (self.meta.get("metadata") or {}).items():
            p = model.metadata_props.add()
            p.key, p.value = k, v
        return model

    # -- queries -----------------------------------------------------------
    def is_const(self, name):
        return name in self.initializers or name in self._const

    def const(self, name):
        """Plan-time value of a constant tensor, or None."""
        if name in self.initializers:
            return self.initializers[name]
        return self._const.get(name)

    def shape(self, name):
        return self._shapes.get(name)

    def dtype(self, name):
        return self._dtypes.get(name)

    def producer(self, name):
        for n in self.nodes:
            if name in n.proto.output:
                return n
        return None

    def producers(self):
        out = {}
        for n in self.nodes:
            for o in n.proto.output:
                if o:
                    out[o] = n
        return out

    def consumers(self):
        out = {}
        for n in self.nodes:
            for i in n.proto.input:
                if i:
                    out.setdefault(i, []).append(n)
        return out

    def histogram(self):
        """op -> node count, deterministic order."""
        h = {}
        for n in self.nodes:
            h[n.op_type] = h.get(n.op_type, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self):
        """One-line plan description used by the printer and its diffs."""
        return {"nodes": len(self.nodes), "ops": self.histogram(),
                "fused": sum(1 for n in self.nodes if n.fused)}

    def value_names(self):
        names = set(self.initializers)
        for vi in self.inputs:
            names.add(vi.name)
        for n in self.nodes:
            names.update(i for i in n.proto.input if i)
            names.update(o for o in n.proto.output if o)
        return names

    def graph_outputs(self):
        return {vi.name for vi in self.outputs}

    # -- edits -------------------------------------------------------------
    def remove(self, node):
        self.nodes.remove(node)

    def rename_uses(self, old, new):
        """Point every consumer of `old` at `new`; returns the rewires made."""
        n = 0
        for node in self.nodes:
            hit = False
            for i, name in enumerate(node.proto.input):
                if name == old:
                    node.proto.input[i] = new
                    hit = True
            if hit:
                n += 1
        for vi in self.outputs:
            if vi.name == old:
                vi.name = new
        return n

    def drop_unused_initializers(self):
        used = set(self.graph_outputs())
        for n in self.nodes:
            used.update(i for i in n.proto.input if i)
        keep = {k: v for k, v in self.initializers.items() if k in used}
        dropped = [k for k in self.initializers if k not in keep]
        self.initializers = keep
        return dropped

    def reindex(self):
        for i, n in enumerate(self.nodes):
            n.index = i

    # -- shape/type propagation -------------------------------------------
    def refresh(self):
        """Recompute shapes/types/constants after a pass mutated the graph."""
        self._infer()

    def eval_const(self, node, lookup=None):
        """Value of a node from plan-time constants, or None."""
        vals = {}
        for i in node.proto.input:
            if not i:
                continue
            v = None if lookup is None else lookup.get(i)
            if v is None:
                v = self.initializers.get(i)
            if v is None:
                v = self._const.get(i)
            if v is None:
                return None
            vals[i] = np.asarray(v)
        if not vals and node.op_type not in ("Shape", "Size"):
            return None
        return self._fold(node, vals)

    def _infer(self):
        """Fill _shapes/_dtypes/_const; ONNX inference is the fallback."""
        shapes, dtypes = {}, {}
        for vi in self.inputs:
            tt = vi.type.tensor_type
            if tt.elem_type:
                dtypes[vi.name] = int(tt.elem_type)
            if tt.HasField("shape"):
                shapes[vi.name] = [int(d.dim_value) if d.HasField("dim_value")
                                   else -1 for d in tt.shape.dim]
        for name, arr in self.initializers.items():
            if arr is None:
                continue
            shapes[name] = list(arr.shape)
            dtypes[name] = numpy_helper.from_array(arr).data_type

        const = {}
        for n in self.nodes:
            op = n.op_type
            ins = [i for i in n.proto.input]
            outs = [o for o in n.proto.output if o]
            if op == "Constant":
                arr = self._const_value(n)
                if arr is not None and outs:
                    const[outs[0]] = arr
                    shapes[outs[0]] = list(arr.shape)
                    dtypes[outs[0]] = _np_dtype(arr)
                continue
            if op in ("Shape", "Size") and outs and ins:
                src = shapes.get(ins[0])
                if src and all(int(d) >= 0 for d in src):
                    if op == "Shape":
                        arr = np.asarray(src, np.int64)
                        start = int(n.attr("start", 0))
                        end = n.attr("end", None)
                        rank = len(src)
                        start = start + rank if start < 0 else start
                        end = rank if end is None else int(end)
                        end = end + rank if end < 0 else end
                        arr = arr[max(0, start):min(rank, end)]
                    else:
                        arr = np.asarray(int(np.prod(src)), np.int64)
                    const[outs[0]] = arr
                    shapes[outs[0]] = list(arr.shape)
                    dtypes[outs[0]] = _np_dtype(arr)
                    continue
            got = self._propagate(n, shapes, dtypes, const)
            for name, sh in zip(outs, got or []):
                if sh is None:
                    continue
                if isinstance(sh, tuple):
                    sh, dt = sh
                    if dt is not None:
                        dtypes[name] = dt
                if sh is None:
                    continue
                shapes[name] = list(sh)
            # a fully constant node is a plan-time value
            if op != "Constant" and ins and all(
                    i in const or i in self.initializers for i in ins if i):
                arr = self._fold(n, const)
                if arr is not None:
                    for name in outs:
                        const[name] = arr
                        shapes[name] = list(arr.shape)
                        dtypes[name] = _np_dtype(arr)
        self._shapes, self._dtypes, self._const = shapes, dtypes, const

        # ONNX inference fills gaps this propagation does not model (rare ops,
        # unknown ranks).  It is never allowed to override a resolved dim.
        try:
            m = onnx.shape_inference.infer_shapes(self.to_model(),
                                                  strict_mode=False,
                                                  data_prop=False)
            for vi in (list(m.graph.value_info) + list(m.graph.output) +
                       list(m.graph.input)):
                tt = vi.type.tensor_type
                if tt.elem_type and vi.name not in dtypes:
                    dtypes[vi.name] = int(tt.elem_type)
                if not tt.HasField("shape"):
                    continue
                ref = shapes.get(vi.name)
                if ref and all(d >= 0 for d in ref):
                    continue
                shapes[vi.name] = [int(d.dim_value) if d.HasField("dim_value")
                                   else -1 for d in tt.shape.dim]
        except Exception:                            # noqa: BLE001
            pass

    def _const_value(self, node):
        for a in node.proto.attribute:
            if a.name == "value" and a.t is not None:
                return numpy_helper.to_array(a.t)
            if a.name == "value_float":
                return np.asarray(a.f, np.float32)
            if a.name == "value_int":
                return np.asarray(a.i, np.int64)
            if a.name == "value_floats":
                return np.asarray(list(a.floats), np.float32)
            if a.name == "value_ints":
                return np.asarray(list(a.ints), np.int64)
        return None

    def _propagate(self, node, shapes, dtypes, const):
        """Output shapes for one node, or None when this module has no rule."""
        op = node.op_type
        ins = node.proto.input
        outs = [o for o in node.proto.output if o]

        def sh(i):
            return shapes.get(ins[i]) if i < len(ins) and ins[i] else None

        if op in _UNARY:
            if op == "Cast":
                to = int(node.attr("to", TensorProto.FLOAT))
                return [(sh(0), to)]
            if op in ("IsNaN", "IsInf", "Not"):
                return [(sh(0), TensorProto.BOOL)]
            return [sh(0)]
        if op in _BINARY:
            dt = (TensorProto.BOOL if op in _COMPARE else
                  dtypes.get(ins[0], TensorProto.FLOAT))
            try:
                return [(_broadcast(sh(0), sh(1)), dt)]
            except ValueError:
                return None       # not a valid broadcast: leave it to ONNX
        if op == "Where":
            return [(_broadcast(_broadcast(sh(0), sh(1)), sh(2)),
                     dtypes.get(ins[1], TensorProto.FLOAT))]
        if op == "Conv":
            x, w = sh(0), shapes.get(ins[1]) if len(ins) > 1 else None
            if not x or not w or len(x) != 4 or len(w) != 4:
                return None
            ks = node.attr("kernel_shape") or [w[2], w[3]]
            st = node.attr("strides") or [1, 1]
            pd = node.attr("pads") or [0, 0, 0, 0]
            dl = node.attr("dilations") or [1, 1]
            ho = (x[2] + pd[0] + pd[2] - dl[0] * (ks[0] - 1) - 1) // st[0] + 1
            wo = (x[3] + pd[1] + pd[3] - dl[1] * (ks[1] - 1) - 1) // st[1] + 1
            return [[x[0], w[0], ho, wo]]
        if op in ("MatMul", "Gemm"):
            a, b = sh(0), sh(1)
            if not a or not b:
                return None
            ta = int(node.attr("transA", 0)) if op == "Gemm" else 0
            tb = int(node.attr("transB", 0)) if op == "Gemm" else 0
            m = a[-1] if ta else (a[-2] if len(a) > 1 else 1)
            ka = a[-2] if ta else a[-1]
            n = b[-2] if tb else b[-1]
            kb = b[-1] if tb else (b[-2] if len(b) > 1 else 1)
            if ka >= 0 and kb >= 0 and ka != kb:
                return None
            if len(a) == 1:
                return [[n]]
            return [list(a[:-2]) + [m, n]]
        if op == "Concat":
            axis = int(node.attr("axis", 0))
            dims = [shapes.get(i) for i in ins if i]
            if not dims or any(d is None for d in dims):
                return None
            rank = len(dims[0])
            ax = axis + rank if axis < 0 else axis
            out = list(dims[0])
            out[ax] = sum(int(d[ax]) for d in dims)
            return [out]
        if op == "Split":
            x = sh(0)
            if not x:
                return None
            axis = int(node.attr("axis", 0))
            ax = axis + len(x) if axis < 0 else axis
            n = len(outs)
            sizes = None
            if len(ins) > 1 and ins[1]:
                v = const.get(ins[1])
                if v is None and ins[1] in self.initializers:
                    v = self.initializers[ins[1]]
                if v is not None:
                    sizes = [int(s) for s in np.asarray(v).ravel()]
            if sizes is None:
                sp = node.attr("split")
                if sp:
                    sizes = [int(s) for s in sp]
                elif int(x[ax]) > 0 and int(x[ax]) % n == 0:
                    sizes = [int(x[ax]) // n] * n
            if sizes is None:
                return None
            return [[d if i != ax else s for i, d in enumerate(x)]
                    for s in sizes]
        if op == "Reshape":
            x = sh(0)
            tgt = const.get(ins[1]) if len(ins) > 1 else None
            if tgt is None and len(ins) > 1:
                tgt = self.initializers.get(ins[1])
            if x is None or tgt is None:
                return None
            out = [int(v) for v in np.asarray(tgt).ravel()]
            for i, d in enumerate(out):
                if d == 0:
                    out[i] = x[i]
            if out.count(-1) == 1:
                known = int(np.prod([d for d in out if d != -1]))
                tot = int(np.prod([d for d in x if d >= 0]))
                out[out.index(-1)] = tot // known if known else -1
            return [out]
        if op == "Flatten":
            x = sh(0)
            if not x:
                return None
            axis = int(node.attr("axis", 1))
            ax = axis + len(x) if axis < 0 else axis
            a = int(np.prod(x[:ax]))
            b = int(np.prod(x[ax:]))
            return [[a, b]]
        if op in ("Squeeze", "Unsqueeze"):
            x = sh(0)
            if not x:
                return None
            axes = node.attr("axes")
            if axes is None and len(ins) > 1 and ins[1]:
                v = const.get(ins[1], self.initializers.get(ins[1]))
                axes = [int(a) for a in np.asarray(v).ravel()] if v is not None \
                    else None
            if axes is None:
                if op == "Squeeze":
                    return [[d for d in x if d != 1]]
                return None
            axes = [a + len(x) if a < 0 else a for a in axes]
            if op == "Squeeze":
                return [[d for i, d in enumerate(x) if i not in axes]]
            out = list(x)
            for a in sorted(axes):
                out.insert(a, 1)
            return [out]
        if op == "Transpose":
            x = sh(0)
            if not x:
                return None
            perm = node.attr("perm") or list(range(len(x)))[::-1]
            return [[x[p] for p in perm]]
        if op == "Slice":
            x = sh(0)
            if not x:
                return None
            axes = const.get(ins[3], self.initializers.get(ins[3])) \
                if len(ins) > 3 else None
            starts = const.get(ins[1], self.initializers.get(ins[1])) \
                if len(ins) > 1 else None
            ends = const.get(ins[2], self.initializers.get(ins[2])) \
                if len(ins) > 2 else None
            steps = const.get(ins[4], self.initializers.get(ins[4])) \
                if len(ins) > 4 else None
            if starts is None or ends is None:
                return None
            starts = [int(v) for v in np.asarray(starts).ravel()]
            ends = [int(v) for v in np.asarray(ends).ravel()]
            axes = ([int(v) for v in np.asarray(axes).ravel()]
                    if axes is not None else list(range(len(starts))))
            steps = ([int(v) for v in np.asarray(steps).ravel()]
                     if steps is not None else [1] * len(starts))
            out = list(x)
            for s, e, a, st in zip(starts, ends, axes, steps):
                d = int(x[a])
                if d < 0:
                    out[a] = -1
                    continue
                s = max(0, s + d if s < 0 else s)
                e = min(d, e + d if e < 0 else e)
                out[a] = max(0, (e - s + (st - 1)) // st) if st > 0 else -1
            return [out]
        if op == "Gather":
            x, idx = sh(0), sh(1)
            if not x or not idx:
                return None
            axis = int(node.attr("axis", 0))
            ax = axis + len(x) if axis < 0 else axis
            if not 0 <= ax < len(x):
                return None
            return [x[:ax] + idx + x[ax + 1:]]
        if op == "MaxPool":
            x = sh(0)
            if not x:
                return None
            ks = node.attr("kernel_shape")
            st = node.attr("strides") or [1, 1]
            pd = node.attr("pads") or [0, 0, 0, 0]
            dl = node.attr("dilations") or [1, 1]
            if not ks:
                return None
            ceil_mode = int(node.attr("ceil_mode", 0))
            ho = _pool_out(x[2], ks[0], st[0], pd[0] + pd[2], dl[0], ceil_mode)
            wo = _pool_out(x[3], ks[1], st[1], pd[1] + pd[3], dl[1], ceil_mode)
            return [[x[0], x[1], ho, wo]]
        if op == "GlobalAveragePool" or op == "GlobalMaxPool":
            x = sh(0)
            return [[x[0], x[1], 1, 1]] if x and len(x) == 4 else None
        if op in ("Resize",):
            x = sh(0)
            if not x or len(ins) < 3:
                return None
            sizes = const.get(ins[3], self.initializers.get(ins[3])) \
                if len(ins) > 3 else None
            if sizes is not None:
                return [[int(v) for v in np.asarray(sizes).ravel()]]
            scales = const.get(ins[2], self.initializers.get(ins[2]))
            if scales is None:
                return None
            sc = [float(v) for v in np.asarray(scales).ravel()]
            return [[int(np.floor(d * s)) for d, s in zip(x, sc)]]
        if op in ("Softmax", "LogSoftmax", "Hardmax"):
            return [sh(0)]
        if op in ("Shape", "Size"):
            if op == "Shape":
                return [[len(sh(0) or [])]]
            return [[]]
        if op in ("ArgMax", "ArgMin"):
            x = sh(0)
            if not x:
                return None
            axis = int(node.attr("axis", 0))
            keep = int(node.attr("keepdims", 1))
            ax = axis + len(x) if axis < 0 else axis
            out = list(x)
            if keep:
                out[ax] = 1
            else:
                out.pop(ax)
            return [(out, TensorProto.INT64)]
        if op in _INT_OPS and op not in ("TopK",):
            return [sh(0)]
        return None

    def _fold(self, node, const):
        """Fold a node whose every input is a plan-time constant (numpy)."""
        op = node.op_type
        vals = []
        for i in node.proto.input:
            if not i:
                continue
            v = const.get(i)
            if v is None:
                v = self.initializers.get(i)
            if v is None:
                return None
            vals.append(np.asarray(v))
        try:
            if op in ("Shape", "Size"):
                a = vals[0]
                return (np.asarray(a.shape, np.int64) if op == "Shape" else
                        np.asarray(a.size, np.int64))
            if op == "Identity":
                return vals[0]
            if op == "Cast":
                return _cast(vals[0], int(node.attr("to", TensorProto.FLOAT)))
            if op == "ConstantOfShape":
                # A constant fill is a tensor, not arithmetic: the ONNX test
                # corpus stores every weight and bias this way, and a detector
                # head may carry its anchor table the same way.  ONNX's default
                # for an absent ``value`` attribute is float32 zero.
                dims = [int(v) for v in vals[0].ravel()]
                if not dims or any(d < 0 for d in dims):
                    return None
                val = node.attr("value")
                if val is None:
                    return np.zeros(dims, np.float32)
                a = np.asarray(val)
                if a.size != 1:
                    return None
                return np.full(dims, a.ravel()[0], dtype=a.dtype)
            if op == "Reshape":
                return vals[0].reshape([int(v) for v in vals[1].ravel()])
            if op == "Flatten":
                axis = int(node.attr("axis", 1))
                return vals[0].reshape(int(np.prod(vals[0].shape[:axis])), -1)
            if op in ("Squeeze", "Unsqueeze"):
                axes = node.attr("axes")
                if axes is None and len(vals) > 1:
                    axes = [int(v) for v in vals[1].ravel()]
                if axes is None:
                    return np.squeeze(vals[0])
                axes = tuple(a + vals[0].ndim if a < 0 else a for a in axes)
                return (np.squeeze(vals[0], axis=axes) if op == "Squeeze"
                        else np.expand_dims(vals[0], axis=axes))
            if op == "Transpose":
                perm = node.attr("perm") or list(range(vals[0].ndim))[::-1]
                return np.transpose(vals[0], perm)
            if op == "Concat":
                return np.concatenate(vals, axis=int(node.attr("axis", 0)))
            if op == "Gather":
                return np.take(vals[0], vals[1].astype(np.int64),
                               axis=int(node.attr("axis", 0)))
            if op == "Slice":
                return _np_slice(vals)
            if op in ("Add", "Sub", "Mul", "Min", "Max"):
                f = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
                     "Min": np.minimum, "Max": np.maximum}[op]
                return f(vals[0], vals[1])
            if op == "Div":
                # Integer shape arithmetic only: ONNX Div on integers truncates
                # toward zero.  Float Div is deliberately not folded (the
                # device's reciprocal is not numpy's division).
                if not np.issubdtype(vals[0].dtype, np.integer) or \
                        not np.issubdtype(vals[1].dtype, np.integer):
                    return None
                a = vals[0].astype(np.int64)
                b = vals[1].astype(np.int64)
                if np.any(b == 0):
                    return None
                q = np.abs(a) // np.abs(b)
                return np.where((a < 0) != (b < 0), -q, q).astype(np.int64)
            if op == "Neg":
                return np.negative(vals[0])
            if op == "Abs":
                return np.abs(vals[0])
        except Exception:                            # noqa: BLE001
            return None
        return None


def _pool_out(n, k, s, pad, dil, ceil_mode):
    if n < 0:
        return -1
    v = (n + pad - dil * (k - 1) - 1) / s + 1
    return int(np.ceil(v) if ceil_mode else np.floor(v))


def _np_dtype(arr):
    return numpy_helper.from_array(arr).data_type


def _cast(arr, to):
    table = {
        TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
        TensorProto.BOOL: np.bool_,
    }
    return arr.astype(table.get(to, np.float32))


def _np_slice(vals):
    a, starts, ends = vals[0], vals[1].astype(np.int64), vals[2].astype(np.int64)
    axes = (vals[3].astype(np.int64) if len(vals) > 3 else
            np.arange(starts.size))
    steps = (vals[4].astype(np.int64) if len(vals) > 4 else
             np.ones(starts.size, np.int64))
    sl = [slice(None)] * a.ndim
    for s, e, ax, st in zip(starts, ends, axes, steps):
        sl[int(ax)] = slice(int(s), int(e), int(st))
    return a[tuple(sl)]


def _broadcast(a, b):
    if a is None or b is None:
        return None
    out = []
    for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif x < 0 or y < 0:
            out.append(-1)
        else:
            raise ValueError("shapes %s and %s do not broadcast" % (a, b))
    return list(reversed(out))
