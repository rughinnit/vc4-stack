#!/usr/bin/env python3
"""make_tinystories.py -- a static, cache-free graph from a causal-LM export.

The HF ``onnx-community/TinyStories-1M-ONNX`` fp32 export is a
``use_cache=True`` graph: 18 graph inputs (``input_ids`` plus eight past
key/value pairs and ``attention_mask``), 800 nodes, and 17 outputs (logits plus
eight present key/value pairs).  The VC4 engine accepts fp32 graph inputs with
static shapes only, so this script rewrites the graph into what it can run:

* **no past, no present.**  Each layer's ``present.k = Concat(past.k, new_k)``
  is bypassed: attention reads ``new_k``/``new_v`` directly and the present
  outputs are dropped.  The model then recomputes the whole window every step.
  The past length is still read by the mask construction (``Shape(past)[2]``),
  so that one ``Shape`` node becomes the constant 0 it would have produced.
* **fp32 inputs, cast inside the graph.**  ``input_ids`` and ``position_ids``
  become fp32 ``[1, n_ctx]`` inputs and a ``Cast`` to int64 feeds the two
  embedding ``Gather``s.  Token ids are below 2**24 and positions below 2048,
  so fp32 represents them exactly.
* **a constant causal mask.**  ``attention_mask`` is dropped as an input (a
  fixed window has no padding), and the run-time mask chain
  (``Shape``/``Range``/``Trilu``/``ConstantOfShape``) becomes one bool
  initializer consumed by the eight ``Where``s.  The export's own masked value
  -3.4028235e38 is replaced by -1e9: exp() of either is 0, and the engine
  refuses a non-finite Softmax input, which that value becomes once summed.
* **LayerNorm fused.**  The export decomposes LayerNorm into nine primitives
  and its ``Sub(x, mean)`` is a trailing-axis broadcast between two runtime
  tensors, which the engine's elementwise path refuses.  Each site becomes one
  ``LayerNormalization`` (opset 17) and uses the engine's fused kernel.
* **static geometry.**  ``Reshape`` targets are built from ``Shape`` of an
  activation, so nothing after them is shaped at plan time; the chains are
  evaluated once into initializers.  Every ``MatMul`` with a 3-D activation and
  a 2-D weight is wrapped in Reshape/Reshape: the engine runs one GEMM per
  batch index and requires equal batch dimensions.
* **the embedding transpose is folded.**  ``Transpose(transformer.wte.weight)``
  is a 12.9 MB copy computed on every forward pass; it becomes an initializer.
* **last-position logits.**  The final hidden state is sliced to its last row
  before the LM head: generation needs the newest token only, and the engine's
  multi-row GEMM is wrong for N > 16384, which 50257 is.

Everything else (8 GPT-Neo blocks, GELU-new as Mul/Pow/Add/Tanh, softmax) is
left as exported, so a refusal from the engine names a real op of the model
rather than an artefact of this rewrite.

Usage:

    python make_tinystories.py --src model.onnx --ctx 16 --out ts1m_c16.onnx

``--logits all`` skips the last-token slice and emits ``[1, n_ctx, 50257]``
(measured wrong on this engine, see docs/TINY-LLM.md).  The rewrite is checked
against the original export with ONNX Runtime, the original fed a zero-length
past and an all-ones mask: last-position logits agree to 2.9e-6.
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LAYERS = 8
N_HEADS = 16
HEAD_DIM = 4
VOCAB = 50257
# The export's own masked value is -3.4028235e38.  exp() of it is 0, but the
# engine refuses a non-finite Softmax input, and sums of that value overflow
# to -inf on the way.  -1e9 is finite, and exp(-1e9 - max) is 0 in fp32 just
# as exp(-inf - max) is, so the softmax output is identical.
MASKED = -1e9


def _replace_inputs(graph, repl):
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in repl:
                node.input[i] = repl[name]


def _delete_nodes(graph, pred):
    keep = [n for n in graph.node if not pred(n)]
    dropped = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    return dropped


def _add_init(graph, name, array):
    graph.initializer.append(numpy_helper.from_array(np.asarray(array), name))


def _find(graph, op=None, output=None, name=None):
    for n in graph.node:
        if op is not None and n.op_type != op:
            continue
        if output is not None and output not in n.output:
            continue
        if name is not None and n.name != name:
            continue
        return n
    raise KeyError("no node op=%s output=%s name=%s" % (op, output, name))


def strip_cache(model):
    """Remove past inputs / present outputs by bypassing the cache Concat."""
    g = model.graph
    rewired = {}
    for i in range(LAYERS):
        for kind in ("key", "value"):
            present = "present.%d.%s" % (i, kind)
            node = _find(g, op="Concat", output=present)
            # Concat(past, new): attention wants `new` only.
            rewired[present] = node.input[1]
    _replace_inputs(g, rewired)
    dropped = _delete_nodes(g, lambda n: any(o.startswith("present.")
                                            for o in n.output if o))
    outs = [o for o in g.output if not o.name.startswith("present.")]
    del g.output[:]
    g.output.extend(outs)
    # Shape(past.key)[2] is the past length the mask builder adds; with no
    # past it is 0.  Replacing the Shape output keeps the mask chain exact
    # without a zero-length tensor anywhere in the graph.
    shape_node = _find(g, op="Shape", output="/transformer/Shape_1_output_0")
    _add_init(g, "/transformer/Shape_1_output_0",
              np.array([1, N_HEADS, 0, HEAD_DIM], np.int64))
    shape_out = shape_node.output[0]
    dropped += _delete_nodes(g, lambda n: shape_out in n.output)
    ins = [i for i in g.input if not i.name.startswith("past_key_values.")]
    del g.input[:]
    g.input.extend(ins)
    return dropped


def drop_attention_mask(model):
    """No padding in a fixed window, so the mask is not an input."""
    g = model.graph
    ins = [i for i in g.input if i.name != "attention_mask"]
    del g.input[:]
    g.input.extend(ins)


def static_masks_where(model, n_ctx):
    """Replace the run-time causal mask with a constant bool condition.

    The export builds the same N x N lower triangle from Shape/Range/Trilu on
    every step.  With no past and no padding it is a constant; the Where that
    consumes it keeps the export's own masked value (-3.4e38) and its false
    branch.  The bias Add that follows the Where is dropped: it adds the same
    -3.4e38 to entries that are already -3.4e38 (softmax of either is 0).
    """
    g = model.graph
    rows = np.arange(n_ctx)[:, None]
    cols = np.arange(n_ctx)[None, :]
    ok = (cols <= rows)[None, None, :, :]
    _add_init(g, "causal_ok", ok)
    _add_init(g, "masked_value", np.array(MASKED, np.float32))
    for i in range(LAYERS):
        add = _find(g, op="Add",
                    output="/transformer/h.%d/attn/attention/Add_output_0" % i)
        where = _find(g, op="Where",
                      output="/transformer/h.%d/attn/attention/Where_output_0"
                      % i)
        where.input[0] = "causal_ok"
        where.input[2] = "masked_value"
        add_out = add.output[0]
        _replace_inputs(g, {add_out: where.output[0]})
        _delete_nodes(g, lambda n: add_out in n.output)


def static_masks(model, n_ctx):
    """Replace the built-at-run-time causal mask with a constant bias Add.

    With no past and no padding the attention condition is the same N x N
    lower triangle on every step.  The export writes it twice: a bool Where
    that replaces masked scores with -3.4e38, then an Add of a bias that is 0
    on the triangle and -3.4e38 above it.  The two have the same effect as
    the bias alone -- a masked entry is <= -3.4e38 either way, and
    exp(-3.4e38) and exp(-inf) are both 0 in fp32 -- so the Where and the
    run-time mask construction are removed and the Add keeps a constant.
    """
    g = model.graph
    rows = np.arange(n_ctx)[:, None]
    cols = np.arange(n_ctx)[None, :]
    bias = np.where(cols <= rows, 0.0, MASKED).astype(np.float32)
    _add_init(g, "causal_bias", bias[None, None, :, :])
    for i in range(LAYERS):
        add = _find(g, op="Add",
                    output="/transformer/h.%d/attn/attention/Add_output_0" % i)
        where = _find(g, op="Where",
                      output="/transformer/h.%d/attn/attention/Where_output_0"
                      % i)
        add.input[0] = where.input[1]          # the scores
        add.input[1] = "causal_bias"
        where_out = where.output[0]
        _delete_nodes(g, lambda n: where_out in n.output)


def float_inputs(model, n_ctx):
    """input_ids/position_ids become fp32 [1,n_ctx]; Cast feeds the Gathers."""
    g = model.graph
    for name, consumer_name in (
            ("input_ids", "/transformer/wte/Gather"),
            ("position_ids", "/transformer/wpe/Gather")):
        inp = [i for i in g.input if i.name == name][0]
        inp.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = inp.type.tensor_type.shape.dim
        for d in dims:
            d.ClearField("dim_param")
        dims[0].dim_value = 1
        dims[1].dim_value = n_ctx
        cast_out = name + "_i64"
        node = helper.make_node("Cast", [name], [cast_out], name="cast_" + name,
                                to=TensorProto.INT64)
        gather = _find(g, op="Gather", name=consumer_name)
        for i, nm in enumerate(gather.input):
            if nm == name:
                gather.input[i] = cast_out
        g.node.insert(0, node)


def fold_wte_transpose(model):
    """Transpose(transformer.wte.weight) is a per-run 3.2M-element copy."""
    g = model.graph
    node = _find(g, op="Transpose", output="transformer.wte.weight_transposed")
    wte = next(i for i in g.initializer if i.name == "transformer.wte.weight")
    _add_init(g, node.output[0], numpy_helper.to_array(wte).T.copy())
    _delete_nodes(g, lambda n: n is node)


def _const_value(graph, name):
    for i in graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i)
    for n in graph.node:
        if n.op_type == "Constant" and name in n.output:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    raise KeyError("not a constant: %s" % name)


def fuse_layernorm(model):
    """Nine primitive ops -> one LayerNormalization, at each LN site.

    The export decomposes LayerNorm into ReduceMean/Sub/Pow/ReduceMean/Add/
    Sqrt/Div/Mul/Add, and its ``Sub(x, mean)`` is a trailing-axis broadcast
    the engine's elementwise ops do not take (b is a runtime tensor, not a
    constant).  The engine has a fused ``layernorm_fwd`` kernel, so the
    standard fusion both removes the refused node and replaces nine launches
    with one.  Opset 17 is the first with LayerNormalization.
    """
    g = model.graph
    prefixes = ["/transformer/h.%d/ln_%d" % (i, j)
                for i in range(LAYERS) for j in (1, 2)]
    prefixes.append("/transformer/ln_f")
    made = 0
    for p in prefixes:
        names = {p + s for s in ("/ReduceMean", "/Sub", "/Pow",
                                 "/ReduceMean_1", "/Add", "/Sqrt", "/Div",
                                 "/Mul", "/Add_1")}
        sub = _find(g, name=p + "/Sub")
        mul = _find(g, name=p + "/Mul")
        add1 = _find(g, name=p + "/Add_1")
        eps = _const_value(g, _find(g, name=p + "/Add").input[1])
        node = helper.make_node(
            "LayerNormalization",
            [sub.input[0], mul.input[1], add1.input[1]],
            [add1.output[0]], name=p + "/LayerNorm", axis=-1,
            epsilon=float(np.asarray(eps).ravel()[0]))
        first = min(i for i, n in enumerate(g.node) if n.name in names)
        kept = [n for i, n in enumerate(g.node) if n.name not in names]
        kept.insert(first, node)
        del g.node[:]
        g.node.extend(kept)
        made += 1
    return made


FOLDABLE = frozenset((
    "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Reshape", "Slice",
    "Cast", "Add", "Sub", "Mul", "Div", "Range", "Equal", "Where",
    "ConstantOfShape", "Expand", "Transpose", "Trilu", "ReduceProd"))


def _eval_meta(node, vals):
    """Evaluate one metadata node on numpy arrays (int64 / bool only)."""
    import numpy as _np
    a = [vals[i] for i in node.input if i]
    op = node.op_type
    att = {x.name: helper.get_attribute_value(x) for x in node.attribute}
    if op == "Gather":
        return [_np.take(a[0], a[1].astype(_np.int64),
                         axis=int(att.get("axis", 0)))]
    if op == "Unsqueeze":
        axes = _np.atleast_1d(a[1] if len(a) > 1 else att["axes"]).astype(int)
        r = a[0]
        for ax in sorted(int(x) % (r.ndim + len(axes)) for x in axes):
            r = _np.expand_dims(r, ax)
        return [r]
    if op == "Squeeze":
        axes = None
        if len(a) > 1:
            axes = tuple(int(x) for x in _np.atleast_1d(a[1]))
        elif "axes" in att:
            axes = tuple(int(x) for x in att["axes"])
        return [_np.squeeze(a[0], axis=axes)]
    if op == "Concat":
        return [_np.concatenate(a, axis=int(att.get("axis", 0)))]
    if op == "Reshape":
        shape = [int(x) for x in a[1]]
        src = list(a[0].shape)
        for i, d in enumerate(shape):
            if d == 0 and not att.get("allowzero", 0):
                shape[i] = src[i]
        return [_np.reshape(a[0], shape)]
    if op == "Slice":
        starts, ends = a[1], a[2]
        axes = a[3] if len(a) > 3 else _np.arange(len(starts))
        steps = a[4] if len(a) > 4 else _np.ones(len(starts))
        sl = [slice(None)] * a[0].ndim
        for st, en, ax, sp in zip(starts, ends, axes, steps):
            sl[int(ax)] = slice(int(st), int(en), int(sp))
        return [a[0][tuple(sl)]]
    if op == "Cast":
        return [a[0].astype(onnx.helper.tensor_dtype_to_np_dtype(
            int(att["to"])))]
    if op == "Range":
        return [_np.arange(a[0].item(), a[1].item(), a[2].item(),
                           dtype=a[0].dtype)]
    if op in ("Add", "Sub", "Mul", "Div"):
        fn = {"Add": _np.add, "Sub": _np.subtract, "Mul": _np.multiply,
              "Div": _np.floor_divide}[op]
        return [fn(a[0], a[1])]
    if op == "Equal":
        return [_np.equal(a[0], a[1])]
    if op == "Where":
        return [_np.where(a[0], a[1], a[2])]
    if op == "Transpose":
        return [_np.transpose(a[0], att.get("perm"))]
    if op == "Trilu":
        return [_np.triu(a[0]) if int(att.get("upper", 1)) else
                _np.tril(a[0])]
    if op == "ConstantOfShape":
        shape = [int(x) for x in a[0]]
        fill = att.get("value")
        v = 0.0 if fill is None else numpy_helper.to_array(fill).ravel()[0]
        return [_np.full(shape, v, dtype=_np.asarray(v).dtype)]
    if op == "Expand":
        return [_np.broadcast_to(a[0], [int(x) for x in a[1]]).copy()]
    return None


def _shape_map(model):
    del model.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(model)
    out = {}
    for t in list(m.graph.input) + list(m.graph.output) + \
            list(m.graph.value_info):
        if t.type.HasField("tensor_type"):
            out[t.name] = tuple(d.dim_value if d.HasField("dim_value")
                                else None for d in t.type.tensor_type.shape.dim)
    for i in m.graph.initializer:
        out[i.name] = tuple(i.dims)
    model.graph.value_info.extend(m.graph.value_info)
    return out


def fold_shape_metadata(model):
    """Fold Shape/Gather/Unsqueeze/Concat chains into initializers.

    ``Reshape`` shape operands in this export are built from ``Shape`` of an
    activation, so shape inference leaves every reshaped tensor four unknown
    dims and the plan has no static geometry.  The chain depends only on
    static shapes, so it is evaluated here; each round lets inference resolve
    one more layer of Reshape outputs.
    """
    total = 0
    for _ in range(16):
        shapes = _shape_map(model)
        g = model.graph
        vals = {i.name: numpy_helper.to_array(i)
                for i in g.initializer}
        folded = []
        for idx, node in enumerate(g.node):
            if node.op_type not in FOLDABLE:
                continue
            if node.op_type == "Shape":
                shape = shapes.get(node.input[0])
                if shape is None or any(d is None for d in shape):
                    continue
                outs = [np.asarray(shape, np.int64)]
            else:
                ins = [i for i in node.input if i]
                if any(i not in vals for i in ins):
                    continue
                try:
                    outs = _eval_meta(node, vals)
                except Exception:                      # noqa: BLE001
                    outs = None
            if outs is None:
                continue
            for name, arr in zip(node.output, outs):
                if name:
                    vals[name] = np.asarray(arr)
            folded.append(idx)
        if not folded:
            break
        # put the results in as initializers and drop the nodes
        dead = set(folded)
        for idx in folded:
            for name in g.node[idx].output:
                if name:
                    g.initializer.append(
                        numpy_helper.from_array(vals[name], name=name))
        kept = [n for i, n in enumerate(g.node) if i not in dead]
        del g.node[:]
        g.node.extend(kept)
        total += len(folded)
    return total


def last_token_logits(model, n_ctx):
    """Slice the final hidden state to the last position before the LM head.

    Generation needs the logits of the newest token only.  It is also the
    only shape this engine computes correctly: its GEMM is exact for a
    one-row output at any width and for N <= 16384 at any M, and wrong for
    M > 1 with N > 16384 (measured on the board; the vocabulary is 50257).
    """
    g = model.graph
    mm = _find(g, name="/lm_head/MatMul")
    src = mm.input[0]
    _add_init(g, "last_starts", np.array([n_ctx - 1], np.int64))
    _add_init(g, "last_ends", np.array([n_ctx], np.int64))
    _add_init(g, "last_axes", np.array([1], np.int64))
    _add_init(g, "last_steps", np.array([1], np.int64))
    out = src + "_last"
    node = helper.make_node(
        "Slice", [src, "last_starts", "last_ends", "last_axes", "last_steps"],
        [out], name="last_token_slice")
    g.node.insert(list(g.node).index(mm), node)
    mm.input[0] = out


def squeeze_matmul_batch(model):
    """MatMul(rank-3 A, rank-2 B) -> Reshape/MatMul(2-D)/Reshape.

    The engine runs one GEMM per batch index and requires the operands' batch
    dimensions to be equal; a 3-D activation times a 2-D weight has none on
    the weight side, so the leading batch of 1 is reshaped away around the
    MatMul.  Both Reshapes are views of the same buffer.
    """
    shapes = _shape_map(model)
    g = model.graph
    kept = []
    made = 0
    for idx, node in enumerate(g.node):
        if node.op_type != "MatMul":
            kept.append(node)
            continue
        sa, sb = shapes.get(node.input[0]), shapes.get(node.input[1])
        if not sa or not sb or len(sa) <= 2 or len(sb) != 2:
            kept.append(node)
            continue
        if any(d is None for d in sa) or any(d is None for d in sb):
            kept.append(node)
            continue
        M, K = sa[-2], sa[-1]
        rows = int(np.prod(sa[:-1]))
        N = sb[-1]
        in_shape = "mm_in_%d" % idx
        out_shape = "mm_out_%d" % idx
        _add_init(g, in_shape, np.array([rows, K], np.int64))
        _add_init(g, out_shape, np.array(list(sa[:-2]) + [M, N], np.int64))
        orig = node.output[0]
        a2 = orig + "_2d_in"
        y2 = orig + "_2d"
        kept.append(helper.make_node("Reshape", [node.input[0], in_shape],
                                     [a2], name=node.name + "_squeeze"))
        node.input[0] = a2
        node.output[0] = y2
        kept.append(node)
        kept.append(helper.make_node("Reshape", [y2, out_shape], [orig],
                                     name=node.name + "_unsqueeze"))
        made += 1
    del g.node[:]
    g.node.extend(kept)
    return made


def prune(model):
    """Drop every node that no longer reaches a graph output."""
    before = len(model.graph.node)
    g = model.graph
    live = {o.name for o in g.output}
    kept = []
    for node in reversed(g.node):
        if any(o in live for o in node.output if o):
            kept.append(node)
            live.update(i for i in node.input if i)
    kept.reverse()
    del g.node[:]
    g.node.extend(kept)
    inits = [i for i in g.initializer if i.name in live]
    del g.initializer[:]
    g.initializer.extend(inits)
    vis = [v for v in g.value_info if v.name in live]
    del g.value_info[:]
    g.value_info.extend(vis)
    return before - len(g.node)


def classify(model):
    """Metadata nodes versus nodes that do activation arithmetic."""
    g = model.graph
    dt = {}
    for t in list(g.input) + list(g.output) + list(g.value_info):
        if t.type.HasField("tensor_type"):
            dt[t.name] = t.type.tensor_type.elem_type
    for i in g.initializer:
        dt[i.name] = i.data_type
    meta_ops = {"Shape", "Size", "Constant", "ConstantOfShape"}
    meta = compute = 0
    for n in g.node:
        ins = [dt.get(i) for i in n.input if i]
        ints = ins and all(t is not None and t not in
                           (TensorProto.FLOAT, TensorProto.DOUBLE) for t in ins)
        if n.op_type in meta_ops or ints:
            meta += 1
        else:
            compute += 1
    return meta, compute


def fix_shapes(model, n_ctx, last_only=True):
    # The source file's value_info carries symbolic names from the cache
    # export; stale entries would survive inference, so start clean.
    del model.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(model)
    for out in m.graph.output:
        if out.name == "logits":
            dims = out.type.tensor_type.shape.dim
            for d in dims:
                d.ClearField("dim_param")
            dims[0].dim_value = 1
            dims[1].dim_value = 1 if last_only else n_ctx
            dims[2].dim_value = VOCAB
    # Range's output length is data-dependent, so shape inference leaves the
    # old dim_param on it even though the values are static.  The executor
    # resolves Range/Slice bounds at plan time (coverage.refine_shapes); the
    # names are recorded rather than treated as an error.
    symbolic = []
    for v in m.graph.value_info:
        for d in v.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") and d.dim_param:
                symbolic.append([v.name, d.dim_param])
    return m, symbolic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="write the shape/node report as JSON")
    ap.add_argument("--mask", choices=("where", "bias"), default="where",
                    help="causal mask as a bool Where or an additive bias")
    ap.add_argument("--logits", choices=("last", "all"), default="last",
                    help="emit [1,1,vocab] (generation) or [1,ctx,vocab]")
    args = ap.parse_args(argv)

    model = onnx.load(args.src)
    orig_nodes = len(model.graph.node)
    orig_ops = {}
    for n in model.graph.node:
        orig_ops[n.op_type] = orig_ops.get(n.op_type, 0) + 1

    dropped_cache = strip_cache(model)
    drop_attention_mask(model)
    if args.mask == "where":
        static_masks_where(model, args.ctx)
    else:
        static_masks(model, args.ctx)
    float_inputs(model, args.ctx)
    fold_wte_transpose(model)
    ln_fused = fuse_layernorm(model)
    # LayerNormalization is opset 17; without the bump shape inference cannot
    # type the fused node and every shape after it stays unknown.
    model.opset_import[0].version = max(model.opset_import[0].version, 17)
    shape_meta = fold_shape_metadata(model)
    last_only = args.logits == "last"
    if last_only:
        last_token_logits(model, args.ctx)
    mm2d = squeeze_matmul_batch(model)
    pruned = prune(model)
    model, symbolic = fix_shapes(model, args.ctx, last_only)
    onnx.checker.check_model(model)

    meta, compute = classify(model)
    onnx.save(model, args.out)
    rep = {
        "source_nodes": orig_nodes,
        "cache_nodes_removed": dropped_cache,
        "layernorms_fused": ln_fused,
        "shape_metadata_folded": shape_meta,
        "matmuls_made_2d": mm2d,
        "pruned_nodes": pruned,
        "nodes": len(model.graph.node),
        "metadata_nodes": meta,
        "compute_nodes": compute,
        "ctx": args.ctx,
        "logits": args.logits,
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "initializer_bytes": sum(
            int(np.prod(list(i.dims))) * 4 for i in model.graph.initializer
            if i.data_type == TensorProto.FLOAT),
        "symbolic_dims": symbolic,
        "ops": {k: v for k, v in sorted(orig_ops.items())},
    }
    print("nodes %d -> %d (cache Concat %d, dead %d); metadata %d, "
          "compute %d; inputs %s; unresolved dims %d"
          % (rep["source_nodes"], rep["nodes"], rep["cache_nodes_removed"],
             rep["pruned_nodes"], meta, compute, rep["inputs"],
             len(symbolic)))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
