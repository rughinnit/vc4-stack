"""Host execution for hybrid mode: one node at a time through the ONNX
reference evaluator (numpy).

The unit of work is a *single* NodeProto from the original graph.  Its
attributes (including subgraphs: ``If``/``Loop``/``Scan`` bodies), its node
name and its domain are used unmodified; the opset imports are copied from the
original model.  The node's inputs are fed as graph inputs with the types and
shapes of the actual numpy arrays, so the reference evaluator computes exactly
the value that node contributes at that point in the graph.

Subgraphs may capture values from the enclosing scope without listing them as
inputs (ONNX lexical scoping).  :meth:`HostEvaluator.outer_refs` walks the
subgraph attributes and reports those captures, so the caller can supply them
from the executor's tensor table.  Initialisers declared *inside* a subgraph
are left to the evaluator.

Nothing here is used by strict mode.
"""
import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper
from onnx.reference import ReferenceEvaluator

_NP2ONNX = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int16"): TensorProto.INT16,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("bool"): TensorProto.BOOL,
}


def _elem_type(arr):
    return _NP2ONNX.get(np.dtype(arr.dtype), TensorProto.UNDEFINED)


def _graph_attrs(node):
    """Yield every GraphProto attribute of a node (recursively)."""
    for a in node.attribute:
        if a.type == AttributeProto.GRAPH and a.HasField("g"):
            yield a.g
        elif a.type == AttributeProto.GRAPHS:
            for g in a.graphs:
                yield g


def _walk_subgraph(graph, defined, refs):
    inner = set(defined)
    inner.update(i.name for i in graph.input)
    inner.update(i.name for i in graph.initializer)
    for n in graph.node:
        for name in n.input:
            if name and name not in inner:
                refs.add(name)
        inner.update(o for o in n.output if o)
        for sub in _graph_attrs(n):
            _walk_subgraph(sub, inner, refs)
    for out in graph.output:
        if out.name and out.name not in inner:
            refs.add(out.name)


def outer_refs(node):
    """Names a node's subgraphs capture from the enclosing graph, in order."""
    refs = []
    seen = set()
    for g in _graph_attrs(node):
        found = set()
        _walk_subgraph(g, set(), found)
        for name in found:
            if name not in seen:
                seen.add(name)
                refs.append(name)
    return refs


class HostEvaluator:
    """Run individual ONNX nodes with ``onnx.reference.ReferenceEvaluator``."""

    def __init__(self, model):
        self.model = model
        self.opsets = list(model.opset_import)
        self._evaluators = {}
        self.elem_types = {}
        try:
            inferred = onnx.shape_inference.infer_shapes(
                model, strict_mode=False, data_prop=False)
            for vi in list(inferred.graph.value_info) + \
                    list(inferred.graph.output):
                tt = vi.type.tensor_type
                if tt.elem_type:
                    self.elem_types[vi.name] = int(tt.elem_type)
        except Exception:
            pass
        for vi in list(model.graph.output) + list(model.graph.input):
            tt = vi.type.tensor_type
            if tt.elem_type:
                self.elem_types.setdefault(vi.name, int(tt.elem_type))

    # ------------------------------------------------------- lexical capture
    def outer_refs(self, node):
        """Names a subgraph captures from the enclosing graph, in order."""
        return outer_refs(node)

    # -------------------------------------------------------------- evaluate
    def _evaluator(self, node, idx, feeds):
        ev = self._evaluators.get(idx)
        if ev is not None:
            return ev
        ins = [helper.make_tensor_value_info(
            name, _elem_type(arr), list(np.asarray(arr).shape))
            for name, arr in feeds.items()]
        outs = []
        for name in node.output:
            if not name:
                continue
            elem = self.elem_types.get(name, TensorProto.UNDEFINED)
            outs.append(helper.make_tensor_value_info(name, elem, None))
        graph = helper.make_graph([node], "hybrid_%d_%s" % (idx, node.op_type),
                                  ins, outs)
        model = helper.make_model(graph, opset_imports=self.opsets,
                                  producer_name="vc4infer-hybrid")
        # Keep the original IR level: it is at least as new as the opsets the
        # node was written for, and some reference ops consult it.
        if self.model.ir_version:
            model.ir_version = self.model.ir_version
        ev = ReferenceEvaluator(model)
        self._evaluators[idx] = ev
        return ev

    def run(self, node, feeds, idx=0):
        """Evaluate one node; return values aligned with non-empty outputs."""
        ev = self._evaluator(node, idx, feeds)
        outs = ev.run(None, feeds)
        return [np.asarray(o) for o in outs]
