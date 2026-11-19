"""Minimal torchvision stand-in for ONNX export on this board.

The venv's torchvision 0.29.2 wheel registers no operators against the
custom torch 2.15.0+cpu build ("operator torchvision::nms does not exist"),
so importing the real package fails.  Ultralytics only imports torchvision
inside classification-calibration and NMS-export functions, neither of which
runs for a plain detection ONNX export.  This shim provides the names those
modules touch and fails loudly if they are actually used.
"""
__version__ = "0.29.2+shim"
from . import transforms, ops  # noqa: F401
