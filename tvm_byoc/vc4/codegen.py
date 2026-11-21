"""Code generation for the VC4 Relax BYOC backend.

Two entry points matter:

``relax.ext.vc4``
    The callback that ``relax.transform.RunCodegen`` invokes with the grouped
    functions, the target options and the constant-name map.  It lowers each
    composite function to a JSON descriptor and constructs a runtime module
    through ``runtime.VC4JSONRuntimeCreate`` (see ``runtime_vc4.cc``).

``build_for_vc4``
    Partition, run the codegen and return the offloaded IRModule, ready for
    ``tvm.compile``.

The descriptor is the contract between the two halves of the backend and is
readable on its own: it names the symbol, the input shapes, the output shape
and, per fused node, M/N/K plus the folded bias.  ``VC4_TVM_DUMP_DIR`` writes
it to disk so the text can be inspected without building anything.
"""

from __future__ import annotations

import json
import os
import sys

import tvm
from tvm import relax

from . import runtime as _runtime
from .patterns import (
    MAX_EMBED_ELEMS,
    PATTERN_NAME,
    _bias_mode,
    match_bindings,
    register_vc4_patterns,
)

DESCRIPTOR_VERSION = 1
MODES = ("record", "host", "vc4")
NODE_OP = "vc4.fused_matmul_add"


def _debug(msg):
    if os.environ.get("VC4_TVM_DEBUG"):
        print(f"vc4 codegen: {msg}", file=sys.stderr)


def _function_symbol(func):
    attrs = func.attrs
    sym = getattr(attrs, "global_symbol", None) if attrs is not None else None
    return str(sym) if sym is not None else ""


def lower_function(func, options=None):
    """Lower one grouped Relax function to the VC4 descriptor (a dict)."""
    options = dict(options or {})
    mode = str(options.get("mode", "record"))
    if mode not in MODES:
        raise ValueError(f"vc4 codegen: mode must be one of {MODES}, got {mode!r}")

    # MergeCompositeFunctions wraps the grouped subgraph in a function annotated
    # with Codegen, whose body binds the real composite as a local function and
    # calls it with the outer parameters.  Descend into that local function; if
    # the codegen is handed the composite directly, use it as-is.
    blocks = func.body.blocks
    if len(blocks) != 1:
        raise ValueError("vc4 codegen: composite function must have one dataflow block")
    scope = func
    for binding in blocks[0].bindings:
        if isinstance(binding.value, relax.Function):
            scope = binding.value
            break
    matched = match_bindings(scope.body.blocks[0].bindings)
    if matched is None:
        raise ValueError(
            "vc4 codegen: composite does not match matmul + constant bias; "
            "the pattern check accepted something the codegen cannot lower"
        )
    add_binding, matmul, bias_const, m, n, k = matched

    bias_shape = [int(d) for d in bias_const.shape]
    bias_mode = _bias_mode(bias_shape, m, n)
    if bias_const.numel() > MAX_EMBED_ELEMS:
        raise ValueError(
            f"vc4 codegen: bias has {bias_const.numel()} elements; the descriptor "
            f"embeds at most {MAX_EMBED_ELEMS}. Bind the bias as a parameter "
            "until the constant-name runtime path exists."
        )
    bias_data = [float(v) for v in bias_const.numpy().reshape(-1)]

    # The module's call arguments follow the outer function's parameters, in
    # order; each must be one of the two matmul operands.  Shapes come from the
    # composite's parameters, which carry the resolved tensor types.
    if len(func.params) != len(scope.params):
        raise ValueError(
            "vc4 codegen: the Codegen wrapper and the composite disagree on "
            f"their parameters ({len(func.params)} vs {len(scope.params)})"
        )
    roles = []
    for sparam in scope.params:
        if matmul.args[0].same_as(sparam):
            roles.append(0)
        elif matmul.args[1].same_as(sparam):
            roles.append(1)
        else:
            raise ValueError(
                f"vc4 codegen: composite parameter '{sparam.name}' is not a matmul "
                "operand; the composite must consist of matmul(x, w) + bias"
            )
    inputs = []
    for index, param in enumerate(func.params):
        inputs.append(
            {
                "name": str(param.name),
                "role": "x" if roles[index] == 0 else "w",
                "shape": [int(d) for d in scope.params[index].ty.shape],
                "dtype": "float32",
            }
        )

    symbol = _function_symbol(func)
    if not symbol:
        raise ValueError("vc4 codegen: composite function has no global symbol")

    descriptor = {
        "version": DESCRIPTOR_VERSION,
        "backend": "vc4",
        "symbol": symbol,
        "mode": mode,
        "inputs": inputs,
        "outputs": [{"name": "out", "shape": [m, n], "dtype": "float32"}],
        "nodes": [
            {
                "op": NODE_OP,
                "name": f"{symbol}.matmul_bias_add",
                "inputs": [i["name"] for i in inputs],
                "attrs": {
                    "M": m,
                    "N": n,
                    "K": k,
                    "bias_mode": bias_mode,
                    "bias_shape": bias_shape,
                    "bias_data": bias_data,
                    "pattern": PATTERN_NAME,
                    "dtype": "float32",
                },
            }
        ],
    }
    _debug(f"{symbol}: {m}x{n}x{k} bias_mode={bias_mode} mode={mode}")
    return descriptor


def _dump(descriptor):
    out_dir = os.environ.get("VC4_TVM_DUMP_DIR")
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{descriptor['symbol']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(descriptor, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def codegen_vc4(funcs, options=None, constant_names=None):
    """The ``relax.ext.vc4`` codegen: one runtime module per grouped function."""
    options = dict(options or {})
    modules = []
    for func in funcs:
        descriptor = lower_function(func, options)
        _dump(descriptor)
        modules.append(
            _runtime.create_module(
                descriptor["symbol"],
                json.dumps(descriptor, sort_keys=True),
                list((constant_names or {}).values()),
            )
        )
    return modules


def register_codegen():
    """Register ``relax.ext.vc4`` (idempotent)."""
    if tvm.get_global_func("relax.ext.vc4", allow_missing=True) is None:
        import tvm_ffi

        tvm_ffi.register_global_func("relax.ext.vc4", codegen_vc4)


def partition_for_vc4(mod):
    """Group every VC4 composite into a function annotated for the codegen."""
    register_vc4_patterns()
    from tvm.relax.backend.pattern_registry import get_patterns_with_prefix

    patterns = get_patterns_with_prefix("vc4")
    if not patterns:
        raise RuntimeError("vc4 backend: the pattern table is empty")
    # annotate_codegen=True puts the backend name on each grouped function and
    # leaves one function per composite.  MergeCompositeFunctions would instead
    # wrap a whole chain of composites in a single Codegen function, which needs
    # a module that models the dataflow between them; the engine path here is
    # one fused op per module call.
    return relax.transform.FuseOpsByPattern(
        patterns, bind_constants=True, annotate_codegen=True
    )(mod)


def build_for_vc4(mod, mode="record", options=None):
    """Partition and codegen; returns an IRModule ready for ``tvm.compile``."""
    target_options = dict(options or {})
    target_options["mode"] = mode
    register_codegen()
    partitioned = partition_for_vc4(mod)
    return relax.transform.RunCodegen({"vc4": target_options})(partitioned)
