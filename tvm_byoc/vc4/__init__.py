"""VC4 backend for Apache TVM Relax (BYOC).

The 0.27 wheel has no ``tvm/relay`` package, so ``relay.ext`` does not exist
and the Relay BYOC path is unavailable.  This package implements the Relax
path instead:

* :mod:`tvm_byoc.vc4.patterns` - the pattern table (``matmul + constant bias``)
  consumed by ``relax.transform.FuseOpsByPattern``;
* :mod:`tvm_byoc.vc4.codegen` - ``relax.ext.vc4``, which turns a grouped
  function into a JSON descriptor and a module call;
* :mod:`tvm_byoc.vc4.runtime` - the loader for ``libvc4_tvm.so``, the JSON
  runtime that executes the calls (``record``, ``host`` or ``vc4`` mode).

Typical use::

    from tvm_byoc.vc4 import build_for_vc4
    options, offloaded = build_for_vc4(mod, mode="vc4")
    ex = tvm.compile(offloaded, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](x, w)
"""

from . import codegen, patterns, runtime

__all__ = ["build_for_vc4", "codegen", "partition_for_vc4", "patterns", "runtime"]

build_for_vc4 = codegen.build_for_vc4
partition_for_vc4 = codegen.partition_for_vc4
