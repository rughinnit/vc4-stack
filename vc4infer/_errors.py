"""The public exception hierarchy.

Every error the package raises on purpose is one of these:

    UnsupportedNode   the model needs an operator/attribute the engine cannot
                      dispatch (strict: there is no CPU fallback)
    DeviceError       the requested device is not available, or vc4d/the engine
                      could not be opened
    ShapeError        a feed does not match the graph's input (name, shape or
                      dtype); subclasses ValueError, so plain ``except
                      ValueError`` still catches it

Engine-level failures from ``libvc4infer`` keep their own ``vc4infer.Error``
(they are not usage errors and carry the library's message).
"""


class Vc4InferError(Exception):
    """Base class for the package's own errors."""


class DeviceError(Vc4InferError, RuntimeError):
    """No such device, or the device/engine could not be opened."""


class ShapeError(Vc4InferError, ValueError):
    """A feed's name, shape or dtype does not match the graph input."""


class UnsupportedNode(Vc4InferError):
    """A graph node the engine cannot execute.

    Attributes
    ----------
    node : str
        The ONNX node name ("" when the runner raised without one).
    op : str
        The ONNX op type.
    detail : str
        The engine's reason (shape, attribute, missing kernel, ...).
    """

    def __init__(self, node="", op="", detail=""):
        # ``UnsupportedNode("Conv x: too big")`` from the executor's deeper
        # layers carries no node/op split; keep the message intact.
        if node and not op and not detail:
            detail, node = str(node), ""
        self.node = str(node or "")
        self.op = str(op or "")
        self.detail = str(detail or "")
        super().__init__(str(self))

    def __str__(self):
        if self.node and self.op:
            head = "node %r (%s)" % (self.node, self.op)
        elif self.node:
            head = "node %r" % self.node
        elif self.op:
            head = self.op
        else:
            head = "unsupported graph"
        return "%s: %s" % (head, self.detail) if self.detail else head
