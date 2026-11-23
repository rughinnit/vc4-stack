"""Graph optimiser for the vc4infer executor.

Phases A to C of ``docs/OPTIMIZER.md``:

    ir         the graph representation (nodes, attributes, shapes, provenance)
    passes     the switchable pass pipeline (canonicalise, ew, epilogue,
               layout)
    plan       launch accounting from the plan, before and after a pass
    memory     per-tensor live ranges and static arena offsets with reuse

The pipeline is off by default; an operator sweep and end-to-end parity are
needed on the board before a pass is enabled, and the record of that gate lives
in ``docs/OPTIMIZER.md``.  Nothing here imports the ctypes binding or opens the
device, so the whole optimiser is testable on a machine with no VideoCore IV:

    from vc4infer.optimizer import optimise, plan_of, format_plan, plan_memory
    res = optimise(model, passes="all", on_pass=printer)
    plan_of(res.graph).total
    plan_memory(res.model, ...).peak_bytes
"""
from .ir import Graph, Node, OPT_PREFIX
from .passes import (PASSES, PIPELINE, OptimiseResult, Pass, PassResult,
                     RuleStat, fold_plan_constants, optimise,
                     resolve_passes)
from .plan import (EW_GROUP, EW_QPU, Plan, compare_profile,
                   elementwise_launches, epilogue_ok, format_plan, plan_of,
                   stored_floats)
from .memory import (MemoryPlan, TensorRange, legacy_pool_bytes, panel_bytes,
                     plan_memory, tensor_bytes)

__all__ = [
    "Graph", "Node", "OPT_PREFIX",
    "PASSES", "PIPELINE", "Pass", "PassResult", "RuleStat", "OptimiseResult",
    "optimise", "resolve_passes", "fold_plan_constants",
    "Plan", "plan_of", "format_plan", "compare_profile", "EW_GROUP", "EW_QPU",
    "elementwise_launches", "stored_floats", "epilogue_ok",
    "MemoryPlan", "TensorRange", "plan_memory", "legacy_pool_bytes",
    "tensor_bytes", "panel_bytes",
]
