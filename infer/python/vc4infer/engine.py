"""Engine: one ONNX graph on the VideoCore IV QPUs.

    import vc4infer
    eng = vc4infer.Engine("yolov8n.onnx")          # device=Device("vc4") by default
    eng.inputs, eng.outputs                        # TensorSpec name/shape/dtype
    out = eng.run({"images": x})                   # {"output0": np.ndarray}
    eng.profile(reps=3)                            # per-node + total ms

The engine is device-agnostic in shape: it takes and returns numpy, resolves
every tensor from the ONNX graph, and has no notion of YOLO.  ``run`` never
post-processes.  Strictness is preserved from the executor: an operator or
attribute without a QPU implementation raises :class:`UnsupportedNode` naming
the node; there is no CPU arithmetic fallback.
"""
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import _runtime, coverage
from ._errors import DeviceError, ShapeError, UnsupportedNode
from .device import Device


@dataclass(frozen=True)
class TensorSpec:
    """A graph tensor as the engine sees it."""

    name: str
    shape: tuple
    dtype: str = "float32"

    def __repr__(self):
        return "TensorSpec(%r, %s, %s)" % (
            self.name, "x".join(str(d) for d in self.shape), self.dtype)


class NodeInfo(NamedTuple):
    """One ONNX node's static verdict.

    ``supported`` is True/False, or None when the constraint depends on a shape
    that static inference could not resolve ("unverified").  ``constraint`` is
    the implementation note or the reason it would raise.
    """

    name: str
    op: str
    supported: bool
    constraint: str

    @property
    def status(self):
        if self.supported is None:
            return "unverified"
        return "supported" if self.supported else "unsupported"


@dataclass
class Plan:
    """A validated static execution plan (see :meth:`Engine.plan`)."""

    model: str
    device: str
    inputs: tuple
    outputs: tuple
    nodes: list
    dispatches: int
    unverified: list = field(default_factory=list)

    def to_dict(self):
        return {
            "model": self.model,
            "device": self.device,
            "inputs": [(i.name, list(i.shape), i.dtype) for i in self.inputs],
            "outputs": [(o.name, list(o.shape), o.dtype) for o in self.outputs],
            "nodes": len(self.nodes),
            "dispatches": self.dispatches,
            "unverified": [n.name for n in self.unverified],
        }


def _as_device(device):
    if device is None:
        return Device("vc4")
    if isinstance(device, Device):
        return device
    return Device(device)


class Engine:
    """A loaded ONNX graph plus its engine context.

    Parameters
    ----------
    model_path : str
        Fixed-shape, batch-1, fp32 ONNX file.
    device : Device or str
        ``Device("vc4")`` or ``"vc4"``.  This is the only device.
    arena_mb : int, optional
        Arena size to request from vc4d (default: the daemon's, 64 MiB).
    cache_dir : str, optional
        Kernel blob cache (``~/.cache/vc4infer/kcache`` by default).

    The engine owns one arena.  Only one engine should be open at a time; a
    second one cannot map a second 64 MiB arena and raises :class:`DeviceError`.
    """

    def __init__(self, model_path, device=None, *, arena_mb=None,
                 cache_dir=None):
        self.device = _as_device(device)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._eng = _runtime.open_engine(cache_dir=cache_dir,
                                             arena_mb=arena_mb)
        except _runtime.Error as e:
            raise DeviceError(str(e)) from e
        try:
            self._runner = _runtime.Runner(self.model_path, self._eng)
            if not self._runner.float_inputs:
                raise UnsupportedNode("", "Input",
                                      "graph has no fp32 input; vc4infer runs "
                                      "fp32 graphs only")
        except BaseException:
            self._eng.close()
            raise
        self.inputs = tuple(
            TensorSpec(name, tuple(dims))
            for name, dims in self._runner.float_inputs)
        self.outputs = tuple(
            TensorSpec(o.name, tuple(
                d.dim_value if d.HasField("dim_value") else -1
                for d in o.type.tensor_type.shape.dim))
            for o in self._runner.graph.output)
        self.last_profile = []
        self.stats = None
        self._plan_cache = None

    # ------------------------------------------------------------------- run
    def _check_feeds(self, feeds):
        if isinstance(feeds, np.ndarray):
            if len(self.inputs) != 1:
                raise ShapeError(
                    "model has %d inputs (%s); pass a dict"
                    % (len(self.inputs), ", ".join(i.name for i in self.inputs)))
            feeds = {self.inputs[0].name: feeds}
        if not isinstance(feeds, dict):
            raise ShapeError("feeds must be a dict of {input_name: ndarray} "
                             "(or one ndarray for a single-input model)")
        known = {i.name for i in self.inputs}
        extra = set(feeds) - known
        if extra:
            raise ShapeError("unknown input(s) %s; the graph expects %s"
                             % (sorted(extra), sorted(known)))
        missing = known - set(feeds)
        if missing:
            raise ShapeError("missing input(s) %s; the graph expects %s"
                             % (sorted(missing), sorted(known)))
        out = {}
        for spec in self.inputs:
            a = feeds[spec.name]
            if not isinstance(a, np.ndarray):
                a = np.asarray(a)
            if a.dtype.kind not in "fiu":
                raise ShapeError("input %r has dtype %s; expected float32"
                                 % (spec.name, a.dtype))
            if tuple(a.shape) != tuple(spec.shape):
                raise ShapeError(
                    "input %r has shape %s, the graph expects %s"
                    % (spec.name, tuple(a.shape),
                       "x".join(str(d) for d in spec.shape)))
            out[spec.name] = np.ascontiguousarray(a, dtype=np.float32)
        return out

    def run(self, feeds, profile=False):
        """Run the graph; return ``{output_name: np.ndarray}``.

        ``feeds`` is ``{input_name: ndarray}`` (or one ndarray for a
        single-input model).  Outputs are raw graph outputs in ONNX order; no
        decoding, thresholding or NMS happens here.  ``profile=True`` also
        stores per-node timings in :attr:`last_profile`.
        """
        x = self._check_feeds(feeds)
        with self._lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            s0 = self._eng.stats
            outs = self._runner.run(x, profile=profile)
            s1 = self._eng.stats
        if profile:
            self.last_profile = list(self._runner.profile)
        self.stats = {
            "launches": int(s1.launches - s0.launches),
            "gpu_ms": (s1.pack_ns - s0.pack_ns +
                       s1.kernel_ns - s0.kernel_ns) / 1e6,
        }
        return {name: np.asarray(arr)
                for name, arr in zip([o.name for o in self.outputs], outs)}

    # --------------------------------------------------------------- reports
    def _coverage(self):
        if self._plan_cache is None:
            self._plan_cache = coverage.analyse(self.model_path)
        return self._plan_cache

    def nodes(self):
        """Every node as :class:`NodeInfo` (name, op, supported, constraint)."""
        status = {"supported": True, "unsupported": False, "unknown": None}
        return [NodeInfo(n["name"], n["op"], status[n["status"]],
                         n["reason"]) for n in self._coverage()["nodes"]]

    def unsupported(self):
        """Nodes the static check rejects."""
        return [n for n in self.nodes() if n.supported is False]

    def plan(self):
        """Validate the graph and return a :class:`Plan`.

        Strict: raises :class:`UnsupportedNode` for the first node the engine
        cannot dispatch, so a caller can fail before touching the GPU.  The
        check is static (graph, attributes, initialiser shapes, shape
        inference); :meth:`run` is authoritative for a concrete input.
        """
        rep = self._coverage()
        nodes = self.nodes()
        bad = [n for n in nodes if n.supported is False]
        if bad:
            raise UnsupportedNode(bad[0].name, bad[0].op, bad[0].constraint)
        return Plan(model=self.model_path, device=self.device.name,
                    inputs=self.inputs, outputs=self.outputs, nodes=nodes,
                    dispatches=rep["dispatches"],
                    unverified=[n for n in nodes if n.supported is None])

    def memory(self):
        """Arena, graph pool and peak usage in bytes and MiB."""
        arena = int(self._eng.arena_bytes)
        pool = int(self._runner.pool.size)
        peak = int(self._runner.pool.used_bytes)
        return {
            "arena_bytes": arena, "arena_mb": arena / 2 ** 20,
            "pool_bytes": pool, "pool_mb": pool / 2 ** 20,
            "peak_bytes": peak, "peak_mb": peak / 2 ** 20,
            "device": self.device.name,
        }

    def _random_feeds(self):
        rng = np.random.default_rng(0)
        return {i.name: rng.standard_normal(
            [d if d > 0 else 1 for d in i.shape]).astype(np.float32)
            for i in self.inputs}

    def profile(self, reps=1, feeds=None):
        """Time the graph and every node.

        Returns total ms (mean/median/min), FPS, launches, GPU vs wall, and the
        per-node breakdown (mean over ``reps``).  The warm-up run that performs
        the kernel JIT is reported separately and not counted.
        """
        x = self._check_feeds(feeds) if feeds is not None else \
            self._random_feeds()
        t0 = time.perf_counter()
        self.run(x, profile=True)                    # warm-up (JIT)
        warm = (time.perf_counter() - t0) * 1e3
        times, acc = [], {}
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            self.run(x, profile=True)
            times.append((time.perf_counter() - t0) * 1e3)
            for p in self.last_profile:
                e = acc.setdefault(p["name"], {"name": p["name"],
                                               "op": p["op"], "wall_ms": 0.0,
                                               "gpu_ms": 0.0, "launches": 0})
                e["wall_ms"] += p["wall_ms"] / reps
                e["gpu_ms"] += p["gpu_ms"] / reps
                e["launches"] = p["launches"]
        mean = statistics.mean(times)
        return {
            "model": self.model_path,
            "device": self.device.name,
            "input_shape": [list(i.shape) for i in self.inputs],
            "reps": int(reps),
            "warmup_ms": warm,
            "runs_ms": times,
            "mean_ms": mean,
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "fps_mean": 1000.0 / mean,
            "fps_best": 1000.0 / min(times),
            "launches": int(sum(p["launches"] for p in self.last_profile)),
            "gpu_ms": float(sum(p["gpu_ms"] for p in self.last_profile)),
            "wall_sum_ms": float(sum(p["wall_ms"] for p in self.last_profile)),
            "nodes": [acc[k] for k in acc],
            **self.memory(),
        }

    # ------------------------------------------------------------- lifecycle
    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self._runner = None
        if getattr(self, "_eng", None) is not None:
            self._eng.close()
            self._eng = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return ("<vc4infer.Engine %s device=%s inputs=%s outputs=%s>"
                % (os.path.basename(self.model_path), self.device.name,
                   [i.name for i in self.inputs],
                   [o.name for o in self.outputs]))
