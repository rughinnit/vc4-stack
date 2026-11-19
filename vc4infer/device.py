"""The one device vc4infer targets: the VideoCore IV QPU, reached via vc4d.

``Device`` is deliberately thin: it describes what the accelerator can do so a
caller can branch without importing the engine.  Memory and queue depth are
properties of the *backend* (vc4d hands each client one dma-buf arena and
serialises launches), not of a particular model.
"""
import os

from ._errors import DeviceError

_OPS = None


def ops():
    """ONNX operators the engine dispatches today (from the coverage table)."""
    global _OPS
    if _OPS is None:
        from . import coverage
        _OPS = tuple(sorted(coverage.OPS))
    return _OPS

_ARENA_DEFAULT_MB = 64


class Device:
    """A VideoCore IV device.

    Parameters
    ----------
    name : str
        Only ``"vc4"`` exists.  Anything else raises :class:`DeviceError`.

    Attributes
    ----------
    name : str
        ``"vc4"``.
    memory : dict
        The memory model the engine runs in: one shared dma-buf arena served by
        vc4d, plus the size the library will ask for.
    queue_depth : int
        How many launches can be in flight.  1: vc4d keeps exactly one launch
        in flight per client and serialises between clients.
    """

    def __init__(self, name="vc4"):
        if name != "vc4":
            raise DeviceError(
                "device %r is not available; vc4infer only runs on the "
                "VideoCore IV QPU (device='vc4'). Available: %r"
                % (name, devices()))
        self.name = "vc4"

    # ------------------------------------------------------------ properties
    @property
    def memory(self):
        mb = int(os.environ.get("VC4INFER_ARENA_MB", _ARENA_DEFAULT_MB))
        return {
            "kind": "dma-buf arena",
            "arena_mb": mb,
            "shared": True,
            "backend": "vc4d",
            "socket": "/run/vc4d.sock",
            "note": "one client arena at a time; vc4d caps the size",
        }

    @property
    def queue_depth(self):
        return 1

    @property
    def ops(self):
        return ops()

    def supports(self, op):
        """True when the engine has a QPU implementation for ONNX op ``op``."""
        return op in ops()

    # --------------------------------------------------------------- helpers
    def describe(self):
        m = self.memory
        return ("Device(vc4, arena=%d MiB via %s, queue_depth=%d, ops=%d)"
                % (m["arena_mb"], m["backend"], self.queue_depth, len(ops())))

    def __repr__(self):
        return "Device(%r)" % self.name

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


def devices():
    """The devices this build can run on: ``["vc4"]``."""
    return ["vc4"]
