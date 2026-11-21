"""Pattern table for the VC4 Relax BYOC backend.

One composite is claimed: ``matmul(x, w) + bias`` where ``bias`` is a
constant.  The engine has exactly the two ops that composite needs
(``vc4_matmul`` and ``vc4_elementwise`` with ``VC4_OP_ADD``), so a match is
also a lowering contract: the codegen in :mod:`tvm_byoc.vc4.codegen` can always
emit a module call for it.  Anything that does not match stays on the host
graph, which is the BYOC fallback and keeps the engine invariant intact.

The partition is deliberately narrow:

* float32 only; the QPU FPU has no other dtype;
* rank-2 matmul with static M, K, N; the engine resolves every shape at plan
  time and the descriptor carries them;
* the add operand is a constant of shape (), (N,), (1, N) or (M, N), with at
  most :data:`MAX_EMBED_ELEMS` elements.  The constant is folded into the JSON
  descriptor, so it is an operand of the compiled module rather than a runtime
  input.
"""

from __future__ import annotations

import tvm
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix, register_patterns
from tvm.relax.dpl.pattern import is_op, wildcard

BACKEND = "vc4"
PATTERN_NAME = "vc4.matmul_bias_add"
MAX_EMBED_ELEMS = 4096


def _op_name(expr):
    op = getattr(expr, "op", None)
    return getattr(op, "name", None) if op is not None else None


def const_value(expr):
    """Return the ``tvm.runtime.Tensor`` behind a Relax constant, or None."""
    if isinstance(expr, tvm.ir.expr.GenericConst):
        return expr.value
    return None


def static_shape(shape):
    """Return the dimensions as Python ints, or None when any is symbolic."""
    out = []
    for dim in shape:
        try:
            out.append(int(dim))
        except (TypeError, ValueError):
            return None
    return out


def _bias_mode(bias_shape, m, n):
    """Map the constant's shape to the descriptor's broadcast mode."""
    if bias_shape == []:
        return 1
    if bias_shape == [n]:
        return 2
    if bias_shape == [1, n]:
        return 2
    if bias_shape == [m, n]:
        return 3
    return None


def match_operands(x_expr, w_expr, bias_const):
    """Validate the operand shapes; returns ``(m, n, k)`` or None.

    A Call's own ``ty`` is not resolved outside a built function, so the shapes
    are read from the operand expressions, whose Var and Constant types are
    always populated.
    """
    if x_expr is None or w_expr is None or bias_const is None:
        return None
    xs = static_shape(getattr(getattr(x_expr, "ty", None), "shape", ()))
    ws = static_shape(getattr(getattr(w_expr, "ty", None), "shape", ()))
    if xs is None or ws is None or len(xs) != 2 or len(ws) != 2:
        return None
    if str(getattr(x_expr.ty, "dtype", "")) != "float32":
        return None
    if str(getattr(w_expr.ty, "dtype", "")) != "float32":
        return None
    m, k = xs
    k2, n = ws
    if k != k2 or m <= 0 or n <= 0 or k <= 0:
        return None
    bias_shape = static_shape(bias_const.shape)
    if bias_shape is None or _bias_mode(bias_shape, m, n) is None:
        return None
    if bias_const.numel() > MAX_EMBED_ELEMS:
        return None
    return m, n, k


def _bound_value(values, expr):
    """Follow a variable to its binding; return the call itself if already one."""
    if expr is None:
        return None
    if _op_name(expr) is not None:
        return expr
    for var, value in values.items():
        if var is expr or (hasattr(expr, "same_as") and expr.same_as(var)):
            return value
    return None


def match_bindings(bindings):
    """Find ``add(matmul(x, w), const)`` in a dataflow block.

    Returns ``(add_binding, matmul_call, bias_const, m, n, k)`` or None.
    """
    values = {b.var: b.value for b in bindings}
    for binding in bindings:
        add = binding.value
        if _op_name(add) != "relax.add" or len(add.args) != 2:
            continue
        for ci, mi in ((1, 0), (0, 1)):
            bias_const = const_value(add.args[ci])
            if bias_const is None:
                continue
            matmul = _bound_value(values, add.args[mi])
            if matmul is None or _op_name(matmul) != "relax.matmul" or len(matmul.args) != 2:
                continue
            shape = match_operands(matmul.args[0], matmul.args[1], bias_const)
            if shape is None:
                continue
            return binding, matmul, bias_const, *shape
    return None


def _check(ctx) -> bool:
    """Pattern check: the annotated operands must be lowerable by the codegen."""
    ann = ctx.annotated_expr
    get = ann.get if hasattr(ann, "get") else ann.__getitem__
    try:
        bias = get("bias")
    except KeyError:
        return False
    return match_operands(get("x"), get("w"), const_value(bias)) is not None


def matmul_bias_pattern():
    """``matmul(x, w) + const`` marked for the VC4 codegen."""
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    matmul = is_op("relax.matmul")(x, w)
    root = is_op("relax.add")(matmul, bias)
    return (
        PATTERN_NAME,
        root,
        {"x": x, "w": w, "bias": bias, "matmul": matmul, "root": root},
        _check,
    )


def register_vc4_patterns():
    """Register the VC4 pattern table once and return the entries."""
    existing = list(get_patterns_with_prefix(BACKEND))
    if existing:
        return existing
    register_patterns([matmul_bias_pattern()])
    return list(get_patterns_with_prefix(BACKEND))
