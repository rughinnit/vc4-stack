"""Engine: one ONNX graph on the VideoCore IV QPUs.

    import vc4infer
    eng = vc4infer.Engine("yolov8n.onnx")          # device=Device("vc4") by default
    eng.inputs, eng.outputs                        # TensorSpec name/shape/dtype
    out = eng.run({"images": x})                   # {"output0": np.ndarray}
    eng.profile(reps=3)                            # per-node + total ms

The engine is device-agnostic in shape: it takes and returns numpy, resolves
every tensor from the ONNX graph, and has no notion of YOLO.  ``run`` never
post-processes.

Two modes, selected explicitly:

``mode="strict"`` (default)
    Strictness is preserved from the executor: an operator or attribute
    without a QPU implementation raises :class:`UnsupportedNode` naming the
    node.  There is no CPU arithmetic fallback.

``mode="hybrid"``
    The graph is partitioned by tier.  ``host_policy="device"`` (the default)
    lets the host do only plan-time constant/shape folding and orchestration;
    every runtime node must be VC4, a numeric operator without a kernel raises
    :class:`HostNotAllowed`, and If/Loop/Scan raise
    :class:`NeedsDeviceDispatch`.  ``"structural"`` also allows runtime
    metadata/control on the host; ``"any"`` additionally runs unwritten numeric
    operators in the numpy reference evaluator.  Hybrid timings are never
    VC4-only timings.
"""
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import _runtime, coverage
from ._errors import (DeviceError, HostNotAllowed, ShapeError,
                      UnsupportedNode)
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


class NodeBackend(NamedTuple):
    """Where one node runs (or would run), and why.

    ``backend`` is one of ``vc4``, ``host-meta``, ``host-control``,
    ``host-unwritten``, ``unsupported-graph``.  Only ``host-unwritten`` is
    numeric host debt.
    """

    name: str
    op: str
    backend: str
    reason: str

    @property
    def on_host(self):
        return self.backend.startswith("host")


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
    mode: str = "strict"
    host_policy: str = "structural"
    backends: list = field(default_factory=list)
    host_debt: list = field(default_factory=list)

    def to_dict(self):
        tiers = {}
        for b in self.backends:
            tiers[b.backend] = tiers.get(b.backend, 0) + 1
        return {
            "model": self.model,
            "device": self.device,
            "mode": self.mode,
            "host_policy": self.host_policy,
            "inputs": [(i.name, list(i.shape), i.dtype) for i in self.inputs],
            "outputs": [(o.name, list(o.shape), o.dtype) for o in self.outputs],
            "nodes": len(self.nodes),
            "dispatches": self.dispatches,
            "unverified": [n.name for n in self.unverified],
            "tiers": tiers,
            "host_debt": [b.name for b in self.host_debt],
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
    mode : {"strict", "hybrid"}
        ``strict`` (default) raises on the first node without a VC4 kernel.
        ``hybrid`` may move allowed node tiers to the host.
    host_policy : {"device", "structural", "any"}
        Hybrid only.  ``device`` (default) permits only plan-time metadata
        folding and orchestration; runtime host compute raises
        :class:`HostNotAllowed` (numeric debt) or :class:`NeedsDeviceDispatch`
        (If/Loop/Scan).  ``structural`` permits runtime host-meta/host-control
        and refuses ``host-unwritten``.  ``any`` runs ``host-unwritten`` in the
        numpy reference evaluator.
    arena_mb : int, optional
        Arena size to request from vc4d (default: the daemon's, 64 MiB).
    cache_dir : str, optional
        Kernel blob cache (``~/.cache/vc4infer/kcache`` by default).

    The engine owns one arena.  Only one engine should be open at a time; a
    second one cannot map a second 64 MiB arena and raises :class:`DeviceError`.
    """

    def __init__(self, model_path, device=None, *, mode="strict",
                 host_policy="device", arena_mb=None, cache_dir=None):
        if mode not in ("strict", "hybrid"):
            raise ValueError("mode must be 'strict' or 'hybrid'")
        if host_policy not in ("device", "structural", "any"):
            raise ValueError("host_policy must be 'device', 'structural' "
                             "or 'any'")
        self.mode = mode
        self.host_policy = host_policy
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
            self._runner = _runtime.Runner(self.model_path, self._eng,
                                           mode=mode,
                                           host_policy=host_policy)
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
        self.last_boundary = {}
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
        self.last_boundary = {
            "tensors": int(self._runner.boundary_uploads +
                           self._runner.boundary_downloads),
            "host_to_vc4": int(self._runner.boundary_uploads),
            "vc4_to_host": int(self._runner.boundary_downloads),
        }
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
            self._plan_cache = self._runner.coverage_report()
        return self._plan_cache

    def nodes(self):
        """Every node as :class:`NodeInfo` (name, op, supported, constraint)."""
        status = {"supported": True, "unsupported": False, "unknown": None}
        return [NodeInfo(n["name"], n["op"], status[n["status"]],
                         n["reason"]) for n in self._coverage()["nodes"]]

    def node_backends(self):
        """Where each node runs, as ``[(name, op, backend, reason)]``.

        After a hybrid run this is the *actual* verdict of that run (including
        runtime fallbacks); before it, the static partition.  Backends are
        ``vc4``, ``host-meta``, ``host-control``, ``host-unwritten`` and
        ``unsupported-graph``.  Only ``host-unwritten`` is numeric host debt.
        """
        if self.mode == "hybrid":
            self._runner.prepare()
        rep = self._runner.backend_report or self._runner.backend_plan()
        return [NodeBackend(n["name"], n["op"], n["backend"], n["reason"])
                for n in rep]

    def host_debt(self):
        """Only the ``host-unwritten`` nodes: numeric ops with no VC4 kernel.

        In ``mode="strict"`` this is the to-do list the engine refuses to
        paper over; in ``mode="hybrid"`` + ``host_policy="device"`` it is why
        :meth:`plan`/:meth:`run` raise.  ``host_policy="any"`` runs them in the
        reference evaluator, but they remain debt.
        """
        return [b for b in self.node_backends()
                if b.backend == "host-unwritten"]

    def host_compute_nodes(self):
        """Nodes that performed *runtime* host compute (never under device).

        Folded metadata nodes are not executed at runtime and are excluded.
        In ``host_policy="device"`` this list must be empty after a run; the
        plan asserts it.
        """
        if self.mode == "hybrid":
            self._runner.prepare()
        rep = self._runner.backend_report
        if rep:
            return [NodeBackend(n["name"], n["op"], n["backend"],
                                n["reason"]) for n in rep
                    if n["backend"] in ("host-meta", "host-control",
                                        "host-unwritten")]
        return [NodeBackend(n["name"], n["op"], n["backend"], n["reason"])
                for n in self._runner.host_compute_plan()]

    def unsupported(self):
        """Nodes the static check rejects."""
        return [n for n in self.nodes() if n.supported is False]

    def plan(self):
        """Validate the graph and return a :class:`Plan`.

        Strict: raises :class:`UnsupportedNode` for the first node the engine
        cannot dispatch, so a caller can fail before touching the GPU.  Hybrid
        + structural: raises :class:`HostNotAllowed` for the first
        ``host-unwritten`` node.  ``unsupported-graph`` raises in every mode.
        """
        rep = self._coverage()
        nodes = self.nodes()
        self._runner.check_policy()
        if self.mode == "strict":
            bad = [n for n in nodes if n.supported is False]
            if bad:
                raise UnsupportedNode(bad[0].name, bad[0].op,
                                      bad[0].constraint)
        backs = self.node_backends()
        if self.mode == "hybrid" and self.host_policy == "device":
            hc = [b for b in backs
                  if b.backend in ("host-meta", "host-control",
                                   "host-unwritten")]
            if hc:
                raise HostNotAllowed(hc[0].name, hc[0].op,
                                     "device policy asserts zero runtime host "
                                     "compute (%s)" % hc[0].reason,
                                     policy="device")
        return Plan(model=self.model_path, device=self.device.name,
                    inputs=self.inputs, outputs=self.outputs, nodes=nodes,
                    dispatches=rep["dispatches"],
                    unverified=[n for n in nodes if n.supported is None],
                    mode=self.mode, host_policy=self.host_policy,
                    backends=backs,
                    host_debt=[b for b in backs
                               if b.backend == "host-unwritten"])

    def memory(self):
        """Arena, graph pool and peak usage in bytes and MiB."""
        arena = int(self._eng.arena_bytes)
        pool = int(self._runner.pool.size)
        peak = int(self._runner.pool.high)          # high-water this run
        plan = getattr(self._runner, "pool_plan", {}) or {}
        return {
            "arena_bytes": arena, "arena_mb": arena / 2 ** 20,
            "pool_bytes": pool, "pool_mb": pool / 2 ** 20,
            "pool_segments": int(getattr(self._runner.pool, "segments", 1)),
            "peak_bytes": peak, "peak_mb": peak / 2 ** 20,
            "estimate_bytes": int(plan.get("pool_bytes", 0)),
            "scratch_bytes": int(plan.get("scratch_bytes", 0)),
            "device": self.device.name,
        }

    def _random_feeds(self):
        rng = np.random.default_rng(0)
        return {i.name: rng.standard_normal(
            [d if d > 0 else 1 for d in i.shape]).astype(np.float32)
            for i in self.inputs}

    def utilization(self, reps=1, feeds=None):
        """The utilisation accounting for one cell of the invariant.

        Returns the fields of :meth:`profile` that matter for the device
        policy plus the measured MAC share, host byte movement, QPU duty and
        launch-path share.  Under ``host_policy="device"`` the invariant is
        ``host_compute_nodes == 0`` and ``bytes_copied_by_host == 0``;
        ``device_ok`` reports both explicitly.
        """
        p = self.profile(reps=reps, feeds=feeds)
        p["device_ok"] = (p["host_compute_nodes"] == 0 and
                          p["bytes_copied_by_host"] == 0)
        return p

    def profile(self, reps=1, feeds=None):
        """Time the graph and every node.

        Returns total ms (mean/median/min), FPS, launches, GPU vs wall, and the
        per-node breakdown (mean over ``reps``).  The warm-up run that performs
        the kernel JIT is reported separately and not counted.

        In hybrid mode every node also carries ``backend`` and ``reason``, and
        the summary splits node counts and wall time by tier (``vc4``,
        ``host-meta``, ``host-control``, ``host-unwritten``).  A hybrid timing
        is never a VC4-only figure; the summary says so explicitly.
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
                key = p.get("index", p["name"])
                e = acc.setdefault(key, {"index": p.get("index"),
                                         "name": p["name"], "op": p["op"],
                                         "backend": p.get("backend"),
                                         "reason": p.get("reason", ""),
                                         "wall_ms": 0.0, "gpu_ms": 0.0,
                                         "launches": 0})
                if p.get("backend"):
                    e["backend"] = p["backend"]
                    e["reason"] = p.get("reason", "")
                e["wall_ms"] += p["wall_ms"] / reps
                e["gpu_ms"] += p["gpu_ms"] / reps
                e["launches"] = p["launches"]
        mean = statistics.mean(times)
        nodes = [acc[k] for k in acc]
        for e in nodes:
            e["qpu_duty"] = (100.0 * e["gpu_ms"] / e["wall_ms"]
                             if e["wall_ms"] > 0 else 0.0)
        tiers = {}
        for e in nodes:
            t = e["backend"] or "vc4"
            d = tiers.setdefault(t, {"nodes": 0, "wall_ms": 0.0,
                                     "gpu_ms": 0.0})
            d["nodes"] += 1
            d["wall_ms"] += e["wall_ms"]
            d["gpu_ms"] += e["gpu_ms"]
        # Node counts are over the whole graph (including fused Sigmoid nodes),
        # not only the dispatched subset, so the percentages have a fixed
        # denominator regardless of the SiLU rewrite.
        backs = self._runner.backend_report or self._runner.backend_plan()
        all_tiers = {t: 0 for t in ("vc4", "folded", "host-meta",
                                    "host-control", "host-unwritten",
                                    "unsupported-graph")}
        for b in backs:
            all_tiers[b["backend"]] = all_tiers.get(b["backend"], 0) + 1
        compute = all_tiers["vc4"] + all_tiers["host-control"] + \
            all_tiers["host-unwritten"] + all_tiers["unsupported-graph"]
        wall_total = sum(d["wall_ms"] for d in tiers.values())
        host_wall = sum(d["wall_ms"] for t, d in tiers.items()
                        if t not in ("vc4", "folded"))
        hybrid = self.mode == "hybrid"
        # MACs: total from the static plan, VC4 share from executed nodes.
        plan_by_idx = {b["index"]: b for b in self._runner.backend_plan()}
        macs_total = sum(int(b.get("macs") or 0) for b in
                         self._runner.backend_plan())
        macs_vc4 = 0
        for e in nodes:
            b = plan_by_idx.get(e.get("index"))
            if b is not None and (e["backend"] or "vc4") == "vc4":
                macs_vc4 += int(b.get("macs") or 0)
        gpu_ms = float(sum(p["gpu_ms"] for p in self.last_profile))
        launches = int(sum(p["launches"] for p in self.last_profile))
        dispatched = sum(1 for e in nodes if e["wall_ms"] > 0)
        host_nodes = [NodeBackend(n["name"], n["op"], n["backend"],
                                  n["reason"]) for n in
                      self._runner.host_compute_nodes()]
        host_bytes = int(self._runner.host_bytes)
        note = ""
        if hybrid:
            note = ("hybrid (%s): %d/%d compute nodes on VC4; %.1f%% of "
                    "node wall time went to the host"
                    % (self.host_policy, all_tiers["vc4"], compute,
                       100.0 * host_wall / wall_total if wall_total else 0.0))
        if self.host_policy == "device" and hybrid:
            note += ("; host_compute_nodes=%d, bytes_copied_by_host=%d"
                     % (len(host_nodes), host_bytes))
        return {
            "model": self.model_path,
            "device": self.device.name,
            "mode": self.mode,
            "host_policy": self.host_policy,
            "input_shape": [list(i.shape) for i in self.inputs],
            "reps": int(reps),
            "warmup_ms": warm,
            "runs_ms": times,
            "mean_ms": mean,
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "fps_mean": 1000.0 / mean,
            "fps_best": 1000.0 / min(times),
            "launches": launches,
            "gpu_ms": gpu_ms,
            "wall_sum_ms": float(sum(p["wall_ms"] for p in self.last_profile)),
            "nodes": nodes,
            # tier split -----------------------------------------------------
            "nodes_total": len(backs),
            "nodes_vc4": all_tiers["vc4"],
            "nodes_folded": all_tiers["folded"],
            "nodes_host_meta": all_tiers["host-meta"],
            "nodes_host_control": all_tiers["host-control"],
            "nodes_host_unwritten": all_tiers["host-unwritten"],
            "nodes_unsupported_graph": all_tiers["unsupported-graph"],
            "pct_nodes_vc4": (100.0 * all_tiers["vc4"] / len(backs)
                              if backs else 0.0),
            "pct_compute_vc4": (100.0 * all_tiers["vc4"] / compute
                                if compute else 0.0),
            # utilisation ----------------------------------------------------
            "host_compute_nodes": len(host_nodes),
            "host_compute_node_names": [b.name for b in host_nodes],
            "arithmetic_on_vc4": (100.0 * macs_vc4 / macs_total
                                  if macs_total else 100.0),
            "macs_total": int(macs_total),
            "macs_vc4": int(macs_vc4),
            "bytes_copied_by_host": host_bytes,
            "host_copy_sites": self._runner.host_copy_report(),
            "qpu_duty": (100.0 * gpu_ms / mean if mean else 0.0),
            "qpu_busy_ms": gpu_ms,
            "qpu_idle_ms": max(0.0, mean - gpu_ms),
            "launches_per_node": (float(launches) / dispatched
                                  if dispatched else 0.0),
            "launch_path_ms": max(0.0, mean - gpu_ms),
            "launch_path_pct": (100.0 * max(0.0, mean - gpu_ms) / mean
                                if mean else 0.0),
            "time_ms": {t: d["wall_ms"] for t, d in tiers.items()},
            "pct_time_host": (100.0 * host_wall / wall_total
                              if wall_total else 0.0),
            "boundary": dict(self.last_boundary),
            "hybrid_note": note,
            "summary": ("%d/%d compute nodes on VC4, %.1f%% of time on host"
                        % (all_tiers["vc4"], compute,
                           100.0 * host_wall / wall_total
                           if wall_total else 0.0)),
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
