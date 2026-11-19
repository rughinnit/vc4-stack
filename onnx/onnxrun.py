#!/usr/bin/env python3
"""onnxrun.py -- compatibility wrapper for the ONNX executor.

The executor now lives in the ``vc4infer`` package
(``vc4infer/_onnxrun.py``) so the CLI, the Python API and the scripts in this
directory share one implementation:

    python onnx/onnxrun.py model.onnx --input in.npy --profile
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vc4infer._onnxrun import (                                 # noqa: E402,F401
    ENGINE, Pool, Runner, Tensor, UnsupportedNode, _align16, _attr,
    _first_dir, default_cache_dir, main, vc4infer,
)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
