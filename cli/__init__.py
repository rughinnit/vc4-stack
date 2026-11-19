"""vc4infer - run fixed-shape ONNX graphs on the VideoCore IV QPUs.

The package is deliberately small: :mod:`vc4infer_cli.cli` holds the command
line, :mod:`vc4infer_cli.decode` the YOLO head decode + NMS, and
:mod:`vc4infer_cli.coverage` the static node coverage report.  Everything that
computes belongs to the engine (``libvc4infer``) through ``vc4onnxrun``.
"""

__version__ = "1.0.0"

PRODUCT = "vc4infer"
