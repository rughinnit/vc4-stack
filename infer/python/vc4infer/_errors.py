"""The public exception hierarchy.

Every error the package raises on purpose is one of these:

    UnsupportedNode   the model needs an operator/attribute the engine cannot
                      dispatch (strict: there is no CPU fallback)
    HostNotAllowed    hybrid mode with host_policy="structural": the node is a
                      numeric operator that has no VC4 kernel yet (host debt)
                      and the policy forbids running it on the host
    HostNodeError     hybrid mode: the host reference evaluator itself failed
                      on a node it was asked to run
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


class HostNotAllowed(Vc4InferError, RuntimeError):
    """A node the host policy refuses to move to the host.

    This is the visible form of host *debt*: the operator has no VC4 kernel
    yet, so ``host_policy="device"`` (the default) and ``"structural"`` stop
    instead of quietly computing it in numpy.  ``host_policy="any"`` is the
    explicit compatibility escape.

    Attributes
    ----------
    node : str
        The ONNX node name.
    op : str
        The ONNX op type.
    reason : str
        Why the node would have needed the host.
    """

    def __init__(self, node="", op="", reason="", policy="device"):
        self.node = str(node or "")
        self.op = str(op or "")
        self.reason = str(reason or "")
        self.policy = str(policy or "device")
        super().__init__(str(self))

    def __str__(self):
        head = ("node %r (%s)" % (self.node, self.op)
                if self.node and self.op else self.op or "node")
        return ("%s may not run on the host under host_policy=%r: %s\n"
                "(host_policy='any' runs it in the numpy reference "
                "evaluator; the operator still needs a VC4 kernel)"
                % (head, self.policy,
                   self.reason or "no VC4 implementation"))


class NeedsDeviceDispatch(Vc4InferError, RuntimeError):
    """A control-flow node that the device policy cannot execute.

    ``If``/``Loop``/``Scan`` need branch/iteration decisions *on the device*:
    a persistent QPU kernel pulling work items from a device-side queue.  The
    host-scheduled executor cannot run them, so the device policy names the
    missing feature instead of pretending the node is merely "slow".

    Attributes
    ----------
    node, op : str
        The ONNX node and operator.
    feature : str
        The runtime feature that would make the node device-executable.
    """

    def __init__(self, node="", op="", feature=""):
        self.node = str(node or "")
        self.op = str(op or "")
        self.feature = str(feature or "")
        super().__init__(str(self))

    def __str__(self):
        head = ("node %r (%s)" % (self.node, self.op)
                if self.node and self.op else self.op or "node")
        return ("%s needs device-side dispatch: %s\n"
                "(persistent QPU kernel + device-side work queue; the current "
                "executor is host-scheduled, so control flow cannot be "
                "emulated without the host making the branch decision)"
                % (head, self.feature or "If/Loop/Scan"))


class HostNodeError(Vc4InferError, RuntimeError):
    """The host reference evaluator failed on a node in hybrid mode."""

    def __init__(self, node="", op="", detail=""):
        self.node = str(node or "")
        self.op = str(op or "")
        self.detail = str(detail or "")
        super().__init__(str(self))

    def __str__(self):
        head = ("node %r (%s)" % (self.node, self.op)
                if self.node and self.op else self.op or "node")
        return "%s: host reference evaluator failed: %s" % (head, self.detail)
