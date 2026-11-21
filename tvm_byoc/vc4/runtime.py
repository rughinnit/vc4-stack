"""Loader for ``libvc4_tvm.so``, the VC4 BYOC runtime module.

The library registers two globals when it is dlopened::

    runtime.VC4JSONRuntimeCreate(symbol, descriptor_json, const_names) -> Module
    ffi.Module.load_from_bytes.vc4_json

so loading it with ``ctypes`` is enough to make the codegen construct modules.
``VC4_TVM_LIB`` overrides the path; the default is ``../build/libvc4_tvm.so``
next to this package, which is where ``vc4/build.sh`` puts it.
"""

from __future__ import annotations

import ctypes
import os

import tvm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.environ.get(
    "VC4_TVM_LIB",
    os.path.join(os.path.dirname(_HERE), "build", "libvc4_tvm.so"),
)

_loaded_path = None


def load(path=None):
    """dlopen the runtime module and return its path (idempotent)."""
    global _loaded_path
    if _loaded_path is not None:
        return _loaded_path
    path = os.path.abspath(path or DEFAULT_LIB)
    if not os.path.exists(path):
        raise RuntimeError(
            f"vc4 backend: {path} is missing. Build it with\n"
            f"  TVM_PYTHON={os.environ.get('PYTHON', 'python3')} "
            "tvm_byoc/vc4/build.sh\n"
            "or point VC4_TVM_LIB at an existing libvc4_tvm.so."
        )
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    if tvm.get_global_func("runtime.VC4JSONRuntimeCreate", allow_missing=True) is None:
        raise RuntimeError(
            f"vc4 backend: {path} was loaded but runtime.VC4JSONRuntimeCreate "
            "is not registered; the library and the installed TVM runtime disagree"
        )
    _loaded_path = path
    return path


def create_module(symbol, descriptor_json, const_names=()):
    """Construct the runtime module for one lowered subgraph."""
    load()
    create = tvm.get_global_func("runtime.VC4JSONRuntimeCreate")
    return create(symbol, descriptor_json, list(const_names))
