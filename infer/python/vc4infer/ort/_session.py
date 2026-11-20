"""``vc4infer.ort`` -- an ONNX Runtime execution provider for the VC4 QPUs.

Why this is a graph-partitioning provider and not an in-process EP plugin:
the onnxruntime 1.32.0 wheel installs **no C/C++ headers at all** (the
``capi`` directory holds shared objects only), and the out-of-tree EP ABI
lives in ``onnxruntime_ep_c_api.h``, which is not shipped and is not
obtainable from this board (GitHub is unreachable).  The runtime advertises
``RegisterExecutionProviderLibrary``/``CreateEpFactories`` and the Python
binding exposes ``register_execution_provider_library``, so a plugin *would*
be possible with a version-locked header -- but not with what is installed.
So the provider is built on the public Python API instead, which the
installed package does support: the graph is cut into VC4 and ORT
partitions, and the VC4 partitions are executed by libvc4infer.

Usage is ORT-shaped::

    import vc4infer.ort
    sess = vc4infer.ort.InferenceSession("yolov8n_160.onnx", providers=["vc4"])
    out = sess.run(None, {"images": x})
    sess.report()            # vc4 | vc4-folded | ort-cpu, %nodes, %MACs, ms

One session per process: the engine maps the daemon's single 64 MiB arena, so
a second live VC4 session raises ``DeviceError``.  Use one session and call
``run`` repeatedly (the engine holds the weights); close it before opening
another.
"""
import os
import threading
import time
import weakref

import numpy as np
import onnx

from .. import UnsupportedNode, __version__
from .._errors import DeviceError
from .. import _runtime
from . import _partition, _report

PROVIDER = "vc4"
PROVIDER_ALIASES = ("vc4", "Vc4ExecutionProvider", "VC4ExecutionProvider")
_LIVE = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


class _NodeArg:
    """The subset of ``onnxruntime.NodeArg`` users actually touch."""

    def __init__(self, name, shape, elem_type, dtype):
        self.name = name
        self.shape = list(shape)
        self.type = dtype
        self._elem_type = elem_type

    def __repr__(self):
        return "NodeArg(name=%r, shape=%r, type=%r)" % (
            self.name, self.shape, self.type)


def _elem_name(t):
    try:
        return onnx.TensorProto.DataType.Name(t)
    except Exception:
        return "unknown"


def _normalise(providers):
    """Return (names, vc4_options) from ORT-style provider entries."""
    if providers is None:
        return [PROVIDER], {}
    names, vc4_opts = [], {}
    for p in providers:
        if isinstance(p, (tuple, list)):
            name, opts = p[0], dict(p[1] or {})
        else:
            name, opts = p, {}
        if name in PROVIDER_ALIASES:
            names.append(PROVIDER)
            vc4_opts.update(opts)
        else:
            names.append(name)
    return names, vc4_opts


def get_available_providers():
    """ORT-style list.  ``vc4`` first, then what onnxruntime itself has."""
    out = []
    try:
        import vc4infer
        if "vc4" in vc4infer.devices():
            out.append(PROVIDER)
    except Exception:
        pass
    try:
        import onnxruntime as ort
        out.extend(ort.get_available_providers())
    except Exception:
        pass
    return out


def is_available():
    """True when vc4d is reachable and the engine can be opened."""
    try:
        import vc4infer
        return "vc4" in vc4infer.devices() and _runtime.daemon_reachable()
    except Exception:
        return False


class InferenceSession:
    """An ORT-compatible session with ``vc4`` in the provider list.

    ``providers=["vc4"]`` runs every node libvc4infer can execute exactly, and
    leaves the rest to ORT's CPU EP (that is ORT's own contract, and
    :meth:`report` names each such node).  ``fallback="error"`` turns the
    fallback into an exception for callers who want a hard guarantee.
    """

    def __init__(self, path_or_bytes, sess_options=None, providers=None,
                 provider_options=None, **kwargs):
        self.version = __version__
        self._lock = threading.RLock()
        self._closed = False
        self._eng = None
        self._runner = None
        self._runner_part = None
        self._last_vc4_profile = {}
        self.last_timing = None
        self.profile = bool(kwargs.pop("profile", False))
        self._fallback = "cpu"
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError("unexpected keyword argument(s): %s" % unknown)

        names, vc4_opts = _normalise(providers)
        if provider_options:
            for o in provider_options:
                vc4_opts.update(o or {})
        self._fallback = str(vc4_opts.get("fallback", "cpu"))
        if self._fallback not in ("cpu", "error"):
            raise ValueError("fallback must be 'cpu' or 'error'")
        self._arena_mb = vc4_opts.get("arena_mb")
        self._cache_dir = vc4_opts.get("cache_dir")
        self._startup_s = float(vc4_opts.get(
            "startup_timeout_s",
            os.environ.get("VC4INFER_ORT_STARTUP_S", 3)))
        self._min_arena = int(float(vc4_opts.get(
            "min_arena_mb",
            os.environ.get("VC4INFER_ORT_MIN_ARENA_MB", 0))) * (1 << 20))
        self.providers = [PROVIDER] + [n for n in names if n != PROVIDER] \
            if PROVIDER in names else list(names)

        # An ORT binary is required either way: for the CPU side, and because
        # this shim deliberately does not reimplement ORT's session plumbing.
        import onnxruntime as ort
        self._ort = ort

        self._model_bytes, self._path = self._load_bytes(path_or_bytes)
        self.model_name = os.path.basename(self._path) if self._path else \
            "<bytes>"
        self.model = onnx.load_from_string(self._model_bytes)
        from .. import coverage
        self.coverage = coverage.analyse_model(self.model)
        self._dtypes = _partition._dtype_map(
            self.model, _partition._value_map(self.model))

        self._inputs = self._node_args(self.model.graph.input, True)
        self._outputs = self._node_args(self.model.graph.output, False)
        self._output_names = [o.name for o in self._outputs]
        self._input_names = [i.name for i in self._inputs]

        self.partitions = []
        self._order = []
        self._ort_sessions = {}
        self._claimed = False
        if PROVIDER in self.providers:
            self._claim_process()
            try:
                if not _runtime.daemon_reachable():
                    raise DeviceError(
                        "vc4 provider requested but vc4d is not running "
                        "(no %s); `sudo systemctl start vc4d`, or use "
                        "providers=['CPUExecutionProvider']"
                        % _runtime.SOCKET)
                self.partitions, _ = _partition.build(self.model,
                                                      self._model_bytes)
                self._order = list(self.partitions)
                self._check_fallback()
                self._open_ort_partitions(sess_options, vc4_opts)
            except BaseException:
                _LIVE.discard(self)
                self._claimed = False
                raise
        else:
            self._whole_ort(sess_options, vc4_opts)
            self._order = list(self.partitions)

    # ------------------------------------------------------------- loading
    def _load_bytes(self, path_or_bytes):
        if isinstance(path_or_bytes, onnx.ModelProto):
            b = path_or_bytes.SerializeToString()
            return b, None
        if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
            return bytes(path_or_bytes), None
        path = os.path.abspath(os.fspath(path_or_bytes))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return f.read(), path

    def _node_args(self, vis, is_input):
        out = []
        for vi in vis:
            if is_input and vi.name in set(i.name for i in
                                           self.model.graph.initializer):
                continue
            tt = vi.type.tensor_type
            shape = [d.dim_value if d.HasField("dim_value") else -1
                     for d in tt.shape.dim]
            out.append(_NodeArg(vi.name, shape, tt.elem_type,
                                _elem_name(tt.elem_type)))
        return out

    def _check_fallback(self):
        has_ort = any(p.backend == "ort-cpu" for p in self.partitions)
        if has_ort and self._fallback == "error":
            debt = []
            for p in self.partitions:
                if p.backend == "ort-cpu":
                    debt += [self.coverage["nodes"][i] for i in p.indices]
            first = debt[0]
            raise UnsupportedNode(
                first["name"], first["op"],
                "fallback='error': %d node(s) cannot run on VC4: %s"
                % (len(debt), ", ".join(e["name"] for e in debt[:6])))

    def _whole_ort(self, sess_options, vc4_opts):
        opts = sess_options or self._ort.SessionOptions()
        self._ort_sessions = {-1: self._ort.InferenceSession(
            self._model_bytes, sess_options=opts,
            providers=["CPUExecutionProvider"])}
        self.partitions = [_partition.Partition(
            -1, "ort-cpu", list(range(len(self.model.graph.node))),
            list(self._input_names), list(self._output_names),
            sum(int(n.get("macs") or 0) for n in self.coverage["nodes"]))]
        self.partitions[0].model_bytes = self._model_bytes

    def _open_ort_partitions(self, sess_options, vc4_opts):
        if any(p.backend == "ort-cpu" for p in self.partitions):
            opts = sess_options or self._ort.SessionOptions()
            for p in self.partitions:
                if p.backend == "ort-cpu":
                    self._ort_sessions[p.index] = self._ort.InferenceSession(
                        p.model_bytes, sess_options=opts,
                        providers=["CPUExecutionProvider"])

    # ------------------------------------------------------------- device
    def _claim_process(self):
        with _LIVE_LOCK:
            for other in list(_LIVE):
                if other is not self and not getattr(other, "_closed", True):
                    raise DeviceError(
                        "another vc4infer.ort session is live in this process; "
                        "vc4d serves one 64 MiB arena, so use one session per "
                        "process and call run() repeatedly")
            _LIVE.add(self)
            self._claimed = True

    def _ensure_engine(self):
        if self._eng is not None:
            return self._eng
        if not self._claimed:
            self._claim_process()
        # vc4d shares one arena between clients.  If another process already
        # holds most of it, vc4_init succeeds with a smaller arena and a large
        # model fails half-way through weight upload (or, worse, two clients
        # share the QPUs).  With min_arena_mb set, wait for a quiet GPU.
        deadline = time.monotonic() + max(0.0, self._startup_s)
        try:
            while True:
                eng = _runtime.open_engine(
                    cache_dir=self._cache_dir, arena_mb=self._arena_mb,
                    retry_s=min(3.0, max(0.0, deadline - time.monotonic())))
                if not self._min_arena or \
                        eng.arena_bytes >= self._min_arena or \
                        time.monotonic() >= deadline:
                    self._eng = eng
                    break
                eng.close()
                time.sleep(0.5)
        except Exception:
            _LIVE.discard(self)
            raise
        return self._eng

    def _runner_for(self, part):
        if self._runner is not None and self._runner_part == part.index:
            return self._runner
        eng = self._ensure_engine()
        # One Runner per engine at a time: constructing a Runner resets the
        # shared pool, so the previous partition's weights are gone once the
        # next partition starts.  That is why a multi-partition model pays a
        # weight upload per partition per run, while the common single
        # partition case keeps its weights for the session's lifetime.
        self._runner = _runtime.Runner(part.path, eng, mode="strict")
        self._runner_part = part.index
        return self._runner

    # ---------------------------------------------------------------- run
    def _feeds(self, part, values):
        out = {}
        for t in part.inputs:
            if t not in values:
                raise ValueError(
                    "missing input %r for partition %d (graph inputs: %s)"
                    % (t, part.index, ", ".join(sorted(values)) or "none"))
            a = values[t]
            if self._dtypes.get(t) == onnx.TensorProto.FLOAT:
                a = np.ascontiguousarray(a, dtype=np.float32)
            else:
                a = np.ascontiguousarray(a)
            out[t] = a
        return out

    def run(self, output_names, input_feed, run_options=None):
        """ORT-compatible run.  ``output_names=None`` returns every output."""
        if self._closed:
            raise RuntimeError("session is closed")
        if not isinstance(input_feed, dict):
            raise TypeError("input_feed must be a dict of {name: ndarray}")
        wanted = list(output_names) if output_names is not None \
            else list(self._output_names)
        unknown = [n for n in wanted if n not in self._output_names]
        if unknown:
            raise ValueError("unknown output name(s): %s" % ", ".join(unknown))

        values = {}
        for k, v in input_feed.items():
            if self._dtypes.get(k) == onnx.TensorProto.FLOAT:
                values[k] = np.ascontiguousarray(v, dtype=np.float32)
            else:
                values[k] = np.ascontiguousarray(v)

        timing = {"vc4_ms": 0.0, "ort_ms": 0.0, "transfer_ms": 0.0,
                  "vc4_runs": 0, "ort_runs": 0}
        profile = {}
        with self._lock:
            for part in self._order:
                t0 = time.perf_counter()
                feeds = self._feeds(part, values)
                t1 = time.perf_counter()
                timing["transfer_ms"] += (t1 - t0) * 1e3
                if part.backend == "vc4":
                    rn = self._runner_for(part)
                    outs = rn.run(feeds, profile=self.profile)
                    if self.profile:
                        for e in rn.profile:
                            if 0 <= e["index"] < len(part.indices):
                                profile[part.indices[e["index"]]] = e
                    timing["vc4_ms"] += (time.perf_counter() - t1) * 1e3
                    timing["vc4_runs"] += 1
                else:
                    sess = self._ort_sessions[part.index]
                    res = sess.run(None, feeds)
                    outs = res
                    timing["ort_ms"] += (time.perf_counter() - t1) * 1e3
                    timing["ort_runs"] += 1
                for name, arr in zip(part.outputs, outs):
                    values[name] = arr
        self._last_vc4_profile = profile
        self.last_timing = timing
        return [values[n] for n in wanted]

    # ------------------------------------------------------------- reports
    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def get_providers(self):
        return list(self.providers)

    def node_backends(self):
        """Per-node attribution rows (see :mod:`vc4infer.ort._report`)."""
        return _report.rows(self)

    def report(self, per_node=False, out=None):
        """Print the ``vc4 | ort-cpu`` table; returns the summary dict."""
        return _report.render(self, per_node=per_node, out=out)

    # ----------------------------------------------------------- lifecycle
    def close(self):
        with self._lock:
            self._closed = True
            self._runner = None
            self._runner_part = None
            if self._eng is not None:
                try:
                    self._eng.close()
                finally:
                    self._eng = None
            self._ort_sessions = {}
            _LIVE.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def available_providers():
    """Alias kept for symmetry with the C engines' ``available_devices``."""
    return get_available_providers()
