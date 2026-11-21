#!/usr/bin/env python3
"""Hybrid-mode acceptance tests (GPU, one engine at a time).

    cd vc4-stack
    VC4INFER_DAEMON=1 PYTHONPATH=infer/python \\
        python3 infer/tests/test_hybrid.py

Managed mode only: every engine here is a vc4d client, so an engine close
returns its region to the daemon instead of releasing a dma-buf of its own.
Direct mode is refused because it allocates a fresh CMA arena per engine and
performs no validation; on a board whose 256 MiB CMA pool is already shared by
two daemons and by page cache, that path is what produces the kernel's
`__cma_alloc: ... ret: -16` lines, and it is not the configuration this test
describes.

Covers the three host tiers and the policy:

(a) hybrid_meta   Shape/Gather/Slice/Mul/Concat -> computed Resize sizes.
                  Structural hybrid runs it; those nodes are host-meta and the
                  output matches a full ReferenceEvaluator run.
(b) hybrid_erf    Erf used to be numeric host debt.  It is a QPU kernel now
                  (docs/COMPLETENESS.md), so strict runs it, host_debt() is
                  empty, and hybrid "any" produces the same output with no
                  host-unwritten node.
    hybrid_mixed  Split/Shape/Gather + Einsum/TopK/NonZero/LSTM, all four of
                  which were host debt until the QPU implementations landed.
                  Strict, device, structural and any agree with the reference;
                  there is no host debt and no boundary copy left.
(c) hybrid_if     Slice+Cast predicate into If with a captured x: host-control
                  under structural, both branches verified.
(d) YOLOv8n       structural hybrid must have zero host debt and identical
                  output/launches to strict (100% of compute nodes on VC4).
(e) lifecycle     N engines opened and closed in one process: the daemon's
                  arena goes back to 0 B, and CmaFree returns to its baseline
                  within CMA_TOL_KB.

The models are built by ``infer/tools/onnx/make_hybrid_models.py``; their
references come from a full ``onnx.reference.ReferenceEvaluator`` run, i.e. an
implementation independent of both the VC4 executor and this test.
"""
import ctypes
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(INFER, "python"))
sys.path.insert(0, os.path.join(INFER, "tools", "onnx"))

import vc4infer                                             # noqa: E402
import vc4infer._binding as _binding                        # noqa: E402
import vc4infer._runtime as _runtime                        # noqa: E402
import make_hybrid_models as gen                            # noqa: E402

ATOL = 1e-4
RTOL = 1e-4
TOL = 5e-4                       # YOLO reference tolerance (docs)
MODELS = os.environ.get("HYBRID_MODELS", "/home/raspi/work/onnx/models")
YOLO = os.environ.get("YOLO_MODEL",
                      "/home/raspi/work/onnx/models/yolov8n_160.onnx")

# How far CmaFree may sit below its baseline after a loop of engine
# open/close cycles.  Page cache is allowed to occupy the reusable CMA pool
# (measured: 31 MiB over one short Python run), so this is a leak bound, not a
# memory-accounting identity: a per-engine arena would be 24-96 MiB.
CMA_TOL_KB = 8 * 1024

FAILED = []
CMA_LOG = []


def check(cond, what):
    if cond:
        print("  ok   %s" % what)
    else:
        print("  FAIL %s" % what)
        FAILED.append(what)


def compare(got, ref, atol=ATOL, rtol=RTOL):
    """fp32 tolerance used throughout the engine: atol + rtol*max(|ref|,1)."""
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    if g.shape != r.shape:
        return {"shape_mismatch": (g.shape, r.shape)}
    if r.dtype.kind in "iu" or np.asarray(ref).dtype.kind in "iu":
        bad = int((g != r).sum())
        return {"max_abs": float(np.abs(g - r).max()) if g.size else 0.0,
                "max_rel": 0.0, "over": bad, "elements": int(g.size),
                "atol": 0.0, "rtol": 0.0}
    d = np.abs(g - r)
    den = np.maximum(np.abs(r), 1.0)
    tol = atol + rtol * den
    return {"max_abs": float(d.max()) if d.size else 0.0,
            "max_rel": float((d / np.maximum(np.abs(r), 1e-30)).max())
            if d.size else 0.0,
            "max_rel_scaled": float((d / den).max()) if d.size else 0.0,
            "over": int((d > tol).sum()), "elements": int(d.size),
            "atol": atol, "rtol": rtol}


def load_case(stem, case="default"):
    suffix = "" if case == "default" else "." + case
    ip = os.path.join(MODELS, stem + suffix + ".in.npy")
    rp = os.path.join(MODELS, stem + suffix + ".ref.npz")
    x = np.load(ip)
    z = np.load(rp)
    return x, [z["out%d" % i] for i in range(len(z.files))]


def run_model(path, x, **kw):
    with vc4infer.Engine(path, **kw) as eng:
        outs = eng.run({"x": x}, profile=True)
        names = [o.name for o in eng.outputs]
        return ([np.asarray(outs[n]) for n in names],
                eng.node_backends(), eng.last_profile, dict(eng.last_boundary),
                len(eng.last_profile))


def report(label, results):
    for i, c in enumerate(results):
        if "shape_mismatch" in c:
            print("    out%d SHAPE %s" % (i, c["shape_mismatch"]))
        else:
            print("    out%-2d max abs %.3e  max rel %.3e  over %d/%d"
                  % (i, c["max_abs"], c["max_rel"], c["over"], c["elements"]))
    return all(c.get("over", 1) == 0 and "shape_mismatch" not in c
               for c in results)


# ------------------------------------------------------------------- CMA probe
def cma_free_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("CmaFree:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


class cma_probe:
    """Sample CmaFree before, during and after a sub-test.

    The pool is declared ``reusable``, so ordinary page cache can occupy it
    between allocations and CmaFree drops for reasons unrelated to the GPU
    (measured 31 MiB over one Python run).  The leak signal is therefore the
    *minimum*, and the assertion is that it comes back afterwards.
    """

    def __init__(self, label, interval=0.2):
        self.label = label
        self.interval = interval
        self.before = cma_free_kb()
        self.low = self.before
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            v = cma_free_kb()
            if v >= 0 and (self.low < 0 or v < self.low):
                self.low = v
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        after = cma_free_kb()
        used = self.before - self.low if self.low >= 0 else -1
        CMA_LOG.append((self.label, self.before, self.low, after))
        print("    cma[%s]: before %d kB, low %d kB (used %d kB), after %d kB"
              % (self.label, self.before, self.low, used, after))
        return False


# --------------------------------------------------------------- daemon stats
class _DaemonStats(ctypes.Structure):
    """struct vc4d_stats_rsp from vc4d/proto/vc4d_proto.h (96 bytes)."""
    _fields_ = [("launches", ctypes.c_uint64),
                ("launches_failed", ctypes.c_uint64),
                ("bytes_uploaded", ctypes.c_uint64),
                ("reg_writes", ctypes.c_uint64),
                ("rejects", ctypes.c_uint64),
                ("arena_bytes", ctypes.c_uint64),
                ("arena_used", ctypes.c_uint64),
                ("gpu_ns", ctypes.c_uint64),
                ("clients", ctypes.c_uint32),
                ("max_clients", ctypes.c_uint32),
                ("programs", ctypes.c_uint32),
                ("dead", ctypes.c_uint32),
                ("evictions", ctypes.c_uint64),
                ("arena_want", ctypes.c_uint32),
                ("arena_degraded", ctypes.c_uint32)]


_DAEMON_LIB = None


def daemon_stats(socket="/run/vc4d.sock"):
    """Daemon arena counters over a short-lived control connection.

    The engine's own ``stats`` are per-context (launches, cycles); only vc4d
    knows how much of its single arena is handed out, which is the number a
    leaked client region would move.  Returns None when the symbols or the
    socket are absent, so the CMA check still runs.
    """
    global _DAEMON_LIB
    assert ctypes.sizeof(_DaemonStats) == 96, "vc4d_stats_rsp layout changed"
    try:
        if _DAEMON_LIB is None:
            L = ctypes.CDLL(_binding._LIB)
            L.vc4c_connect.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_char_p]
            L.vc4c_get_stats.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(_DaemonStats)]
            L.vc4c_close.argtypes = [ctypes.c_void_p]
            _DAEMON_LIB = L
        c = ctypes.c_void_p()
        if _DAEMON_LIB.vc4c_connect(ctypes.byref(c), socket.encode()):
            return None
        d = _DaemonStats()
        rc = _DAEMON_LIB.vc4c_get_stats(c, ctypes.byref(d))
        _DAEMON_LIB.vc4c_close(c)
        if rc:
            return None
        return dict(arena_bytes=int(d.arena_bytes),
                    arena_used=int(d.arena_used),
                    clients=int(d.clients),
                    want=int(d.arena_want),
                    degraded=int(d.arena_degraded))
    except (OSError, AttributeError):
        return None


def tiers_of(backs):
    """name -> backend for the node_backends() rows."""
    return {b.name: b.backend for b in backs}


# ------------------------------------------------------------------- (a) meta
def test_meta():
    print("\n(a) Shape/Gather/Slice/Mul/Concat -> Resize sizes (host-meta)")
    models = gen.ensure(MODELS)
    p = models["hybrid_meta"]["model"]
    x, ref = load_case("hybrid_meta")
    with cma_probe("meta/structural"):
        with vc4infer.Engine(p, mode="hybrid",
                            host_policy="device") as eng:
            plan = eng.plan()
            check(not eng.host_debt(), "no host debt")
            folded = {b.name for b in plan.backends if b.backend == "folded"}
            check({"shape", "slice_nc", "gather_hw", "mul_hw", "concat"} <= folded,
                  "Shape/Slice/Gather/Mul/Concat folded at plan time")
            check(any(b.name == "resize" and b.backend == "vc4"
                      for b in plan.backends), "Resize stays on VC4")
            check(eng.host_compute_nodes() == [], "host_compute_nodes == []")
            outs = eng.run({"x": x}, profile=True)
            got = [np.asarray(outs[o.name]) for o in eng.outputs]
            prof = eng.last_profile
            check(eng.host_compute_nodes() == [], "no runtime host compute")
            check(all(p.get("backend") for p in prof),
                  "every profile row has a backend")
    res = [compare(g, r) for g, r in zip(got, ref)]
    check(report("meta", res), "structural output matches ReferenceEvaluator")


# ------------------------------------------------- (b) former numeric host debt
FORMER_DEBT = ("einsum", "topk", "nonzero", "lstm")


def test_numeric_ops():
    print("\n(b) former numeric host debt runs on the QPU: Erf, Einsum, "
          "TopK, NonZero, LSTM")
    models = gen.ensure(MODELS)

    # ---- Erf: strict / device / any must all compute it on the QPU
    x, ref = load_case("hybrid_erf")
    ep = models["hybrid_erf"]["model"]
    with cma_probe("erf/strict"):
        with vc4infer.Engine(ep) as eng:
            check(not eng.host_debt(), "erf strict: no host debt")
            outs = eng.run({"x": x}, profile=True)
            strict_erf = [np.asarray(outs[o.name]) for o in eng.outputs]
            tiers = tiers_of(eng.node_backends())
            bnd = dict(eng.last_boundary)
            check(tiers.get("erf") == "vc4", "erf strict: Erf on VC4")
            check("host-unwritten" not in tiers.values(),
                  "erf strict: no host-unwritten node")
            check(bnd["tensors"] == 0, "erf strict: no boundary copies")
    res = [compare(g, r) for g, r in zip(strict_erf, ref)]
    check(report("erf/strict", res),
          "erf strict matches ReferenceEvaluator")

    with cma_probe("erf/device"):
        with vc4infer.Engine(ep, mode="hybrid",
                            host_policy="device") as eng:
            check(not eng.host_debt(), "erf device: no host debt")
            plan = eng.plan()
            check(any(b.name == "erf" and b.backend == "vc4"
                      for b in plan.backends),
                  "erf device plan() keeps Erf on VC4")
            outs = eng.run({"x": x}, profile=True)
            got = [np.asarray(outs[o.name]) for o in eng.outputs]
    res = [compare(g, r) for g, r in zip(got, ref)]
    check(report("erf/device", res),
          "erf device matches ReferenceEvaluator")

    with cma_probe("erf/any"):
        with vc4infer.Engine(ep, mode="hybrid",
                            host_policy="any") as eng:
            check(not eng.host_debt(), "erf any: no host debt")
            outs = eng.run({"x": x}, profile=True)
            got = [np.asarray(outs[o.name]) for o in eng.outputs]
            tiers = tiers_of(eng.node_backends())
            check("host-unwritten" not in tiers.values(),
                  "erf any: no host-unwritten node")
            check(dict(eng.last_boundary)["tensors"] == 0,
                  "erf any: no boundary copies")
    check(all(np.array_equal(a, b) for a, b in zip(got, strict_erf)),
          "erf any: bit-identical to strict")

    # ---- mixed: Split/Shape/Gather + the four former debt operators
    print("\n(b2) hybrid_mixed: Split + Einsum/TopK/NonZero/LSTM on the QPU")
    x, ref = load_case("hybrid_mixed")
    mp = models["hybrid_mixed"]["model"]
    with cma_probe("mixed/strict"):
        with vc4infer.Engine(mp) as eng:
            check(not eng.host_debt(), "mixed strict: no host debt")
            outs = eng.run({"x": x}, profile=True)
            strict_mixed = [np.asarray(outs[o.name]) for o in eng.outputs]
            tiers = tiers_of(eng.node_backends())
            launches_strict = sum(p.get("launches", 0)
                                  for p in eng.last_profile)
            bnd = dict(eng.last_boundary)
            for name in FORMER_DEBT:
                check(tiers.get(name) == "vc4",
                      "mixed strict: %s on VC4" % name)
            check("host-unwritten" not in tiers.values(),
                  "mixed strict: no host-unwritten node")
            check(bnd["tensors"] == 0,
                  "mixed strict: no boundary copies")
            check(launches_strict > 0,
                  "mixed strict: %d QPU launches" % launches_strict)
    res = [compare(g, r) for g, r in zip(strict_mixed, ref)]
    check(report("mixed/strict", res),
          "mixed strict matches ReferenceEvaluator")

    for policy in ("device", "structural", "any"):
        with cma_probe("mixed/" + policy):
            with vc4infer.Engine(mp, mode="hybrid",
                                host_policy=policy) as eng:
                check(not eng.host_debt(), "%s: no host debt" % policy)
                outs = eng.run({"x": x}, profile=True)
                got = [np.asarray(outs[o.name]) for o in eng.outputs]
                tiers = tiers_of(eng.node_backends())
                bnd = dict(eng.last_boundary)
                check("host-unwritten" not in tiers.values(),
                      "%s: no host-unwritten node" % policy)
                check(bnd["tensors"] == 0,
                      "%s: no boundary copies" % policy)
        res = [compare(g, r) for g, r in zip(got, ref)]
        check(report("mixed/" + policy, res),
              "%s matches ReferenceEvaluator" % policy)
        check(all(np.array_equal(a, b) for a, b in zip(got, strict_mixed)),
              "%s: bit-identical to strict" % policy)


# ------------------------------------------------------------- (c) host-control
def test_if():
    print("\n(c) If with a data-dependent branch (host-control)")
    models = gen.ensure(MODELS)
    p = models["hybrid_if"]["model"]
    with cma_probe("if/device"):
        with vc4infer.Engine(p, mode="hybrid",
                            host_policy="device") as eng:
            try:
                eng.plan()
                check(False, "device plan() raises NeedsDeviceDispatch for If")
            except vc4infer.NeedsDeviceDispatch as e:
                check(e.node == "if" and e.op == "If",
                      "device plan() names if/If as NeedsDeviceDispatch")
    with cma_probe("if/structural"):
        with vc4infer.Engine(p, mode="hybrid",
                            host_policy="structural") as eng:
            plan = eng.plan()
            ctl = [b for b in plan.backends if b.backend == "host-control"]
            check([b.name for b in ctl] == ["if"],
                  "If classified host-control")
            check(not eng.host_debt(), "no host debt")
            for case in ("true", "false"):
                x, ref = load_case("hybrid_if", case)
                outs = eng.run({"x": x}, profile=True)
                got = [np.asarray(outs[o.name]) for o in eng.outputs]
                res = [compare(g, r) for g, r in zip(got, ref)]
                check(report("if/" + case, res),
                      "If(%s) matches ReferenceEvaluator" % case)
                check(eng.last_boundary["vc4_to_host"] >= 1,
                      "If(%s): captured x crossed the boundary" % case)
    with cma_probe("if/strict"):
        with vc4infer.Engine(p) as eng:
            try:
                eng.plan()
                check(False, "strict plan() refuses If")
            except vc4infer.UnsupportedNode as e:
                check(e.node == "if" and e.op == "If",
                      "strict plan() raises UnsupportedNode naming if/If")
            x, _ = load_case("hybrid_if", "true")
            try:
                eng.run({"x": x})
                check(False, "strict run() refuses If")
            except vc4infer.UnsupportedNode as e:
                # plan() refuses If up front; run() walks the graph in order
                # and stops at the first node it cannot place, the host-meta
                # Squeeze.  Slice and Cast are QPU kernels now.
                check(e.node == "squeeze" and e.op == "Squeeze",
                      "strict run() stops at Squeeze (node=%s op=%s)"
                      % (e.node, e.op))


# ---------------------------------------------------------------- (d) YOLOv8n
def test_yolo():
    print("\n(d) YOLOv8n: device hybrid must be 100% compute-on-VC4")
    if not os.path.isfile(YOLO):
        print("  skip (no %s)" % YOLO)
        return
    x = np.random.default_rng(0).standard_normal((1, 3, 160, 160))
    x = x.astype(np.float32)
    with cma_probe("yolo/strict"):
        with vc4infer.Engine(YOLO) as eng:
            t0 = time.perf_counter()
            strict = eng.run({"images": x}, profile=True)
            strict_wall = (time.perf_counter() - t0) * 1e3
            strict_name = eng.outputs[0].name
            strict_launch = sum(p["launches"] for p in eng.last_profile)
            strict_gpu = eng.stats["gpu_ms"]
    with cma_probe("yolo/structural"):
        with vc4infer.Engine(YOLO, mode="hybrid",
                            host_policy="device") as eng:
            check(not eng.host_debt(), "no host debt")
            backs = eng.node_backends()
            tiers = {}
            for b in backs:
                tiers[b.backend] = tiers.get(b.backend, 0) + 1
            check(tiers.get("host-control", 0) == 0 and
                  tiers.get("host-unwritten", 0) == 0 and
                  tiers.get("host-meta", 0) == 0,
                  "zero runtime host nodes")
            compute = tiers.get("vc4", 0)
            check(compute == sum(v for k, v in tiers.items()
                                 if k != "folded"),
                  "100%% of compute nodes on VC4 (%d)" % compute)
            check(eng.host_compute_nodes() == [], "host_compute_nodes == []")
            t0 = time.perf_counter()
            yh = eng.run({"images": x}, profile=True)
            hybrid_wall = (time.perf_counter() - t0) * 1e3
            hybrid_launch = sum(p["launches"] for p in eng.last_profile)
            hybrid_gpu = eng.stats["gpu_ms"]
            util = eng.utilization(reps=1, feeds={"images": x})
            prof = eng.last_profile
    d = np.abs(np.asarray(strict[strict_name], np.float64) -
               np.asarray(yh[strict_name], np.float64))
    check(d.size > 0 and float(d.max()) == 0.0,
          "hybrid output identical to strict (max abs %.3e)" % float(d.max()))
    check(hybrid_launch == strict_launch,
          "same launch count (%d == %d)" % (hybrid_launch, strict_launch))
    print("    strict: %.1f ms wall / %.1f ms GPU / %d launches"
          % (strict_wall, strict_gpu, strict_launch))
    print("    hybrid: %.1f ms wall / %.1f ms GPU / %d launches"
          % (hybrid_wall, hybrid_gpu, hybrid_launch))
    check(all(p.get("backend") in ("vc4", "folded") for p in prof),
          "profile rows are vc4/folded only")
    check(util["host_compute_nodes"] == 0, "util host_compute_nodes == 0")
    check(util["arithmetic_on_vc4"] == 100.0,
          "arithmetic_on_vc4 == 100%")
    print("    util: bytes_copied_by_host=%d  qpu_duty=%.1f%%  "
          "launches=%d  launch_path=%.1f%%"
          % (util["bytes_copied_by_host"], util["qpu_duty"],
             util["launches"], util["launch_path_pct"]))
    if util["bytes_copied_by_host"]:
        top = sorted(util["host_copy_sites"], key=lambda s: -s["bytes"])[:8]
        print("    OUTSTANDING host copies:")
        for s in top:
            print("      %-34s %-10s %-14s %8d B x%d"
                  % (s["name"][:34], s["op"], s["site"], s["bytes"],
                     s["count"]))
    # Reference at 5e-4, when the fixed input/reference files are present.
    ref = os.environ.get("YOLO_REF", "/tmp/yolo160_ref.npy")
    inp = os.environ.get("YOLO_IN", "/tmp/yolo160_in.npy")
    if os.path.isfile(ref) and os.path.isfile(inp):
        xr = np.load(inp).astype(np.float32)
        yr = np.load(ref).astype(np.float32)
        with vc4infer.Engine(YOLO, mode="hybrid",
                            host_policy="structural") as eng:
            out = eng.run({eng.inputs[0].name: xr})[eng.outputs[0].name]
        res = compare(out, yr, atol=TOL, rtol=TOL)
        print("    ref@5e-4: max abs %.4g  max rel %.3e  over %d/%d"
              % (res["max_abs"], res["max_rel_scaled"], res["over"],
                 res["elements"]))
        check(res["over"] == 0, "YOLO reference 0/%d over 5e-4"
              % res["elements"])
    else:
        print("  skip reference (no %s)" % ref)


# ------------------------------------------------------------- (e) lifecycle
def test_engine_lifecycle():
    print("\n(e) engine open/close: daemon arena and CMA return to baseline")
    models = gen.ensure(MODELS)
    p = models["hybrid_meta"]["model"]
    saved = os.environ.get("VC4INFER_ARENA_MB")
    base = cma_free_kb()
    d0 = daemon_stats()
    check(base > 0, "CmaFree is readable (%d kB)" % base)
    if d0 is None:
        print("  skip daemon arena checks (no vc4c symbols/socket)")
    else:
        check(d0["arena_used"] == 0,
              "daemon arena is empty before the loop (%d B)"
              % d0["arena_used"])
        check(d0["degraded"] == 0,
              "daemon holds its full %d MiB arena" % (d0["want"] >> 20))

    # Three engines open at once: the tiers the old test named.  24 MiB each
    # fits three times into the daemon's 96 MiB arena; 64 MiB (the engine
    # default) does not fit twice.
    concurrent = []
    try:
        with cma_probe("3 concurrent engines"):
            for kw in ({}, {"mode": "hybrid", "host_policy": "structural"},
                       {"mode": "hybrid", "host_policy": "any"}):
                concurrent.append(vc4infer.Engine(p, arena_mb=24, **kw))
            check(len(concurrent) == 3, "three engines open in one process")
            d = daemon_stats()
            if d:
                check(0 < d["arena_used"] <= d["arena_bytes"],
                      "daemon handed out 3 regions (%d B of %d B)"
                      % (d["arena_used"], d["arena_bytes"]))
                check(d["clients"] >= 3,
                      "daemon sees %d client connections" % d["clients"])
    finally:
        for e in concurrent:
            e.close()
        if saved is None:
            os.environ.pop("VC4INFER_ARENA_MB", None)
        else:
            os.environ["VC4INFER_ARENA_MB"] = saved
    d = daemon_stats()
    if d:
        check(d["arena_used"] == 0,
              "all three regions released after close (%d B)" % d["arena_used"])

    # Sequential churn: the pattern that would accumulate a per-engine arena.
    n = int(os.environ.get("ENGINE_CHURN", "16"))
    bad = 0
    worst_used = 0
    try:
        with cma_probe("%d engine open/close" % n):
            for _ in range(n):
                try:
                    e = vc4infer.Engine(p, arena_mb=24)
                    e.close()
                except vc4infer.DeviceError:
                    bad += 1
                d = daemon_stats()
                if d:
                    worst_used = max(worst_used, d["arena_used"])
    finally:
        if saved is None:
            os.environ.pop("VC4INFER_ARENA_MB", None)
        else:
            os.environ["VC4INFER_ARENA_MB"] = saved
    check(bad == 0, "%d/%d engines opened and closed" % (n - bad, n))
    if d0 is not None:
        check(worst_used == 0,
              "daemon arena_used stayed 0 B across the loop (worst %d B)"
              % worst_used)
    end = cma_free_kb()
    check(end >= base - CMA_TOL_KB,
          "CmaFree returned to baseline: %d kB -> %d kB (tolerance %d kB)"
          % (base, end, CMA_TOL_KB))


def main():
    print("hybrid acceptance tests: models=%s" % MODELS)
    backend = _runtime.backend()
    if backend != "managed":
        print("REFUSING direct backend: this test is about vc4d client "
              "arenas.\nRun with VC4INFER_DAEMON=1 (managed).")
        return 2
    print("backend: %s" % _runtime.backend_summary())
    test_meta()
    test_numeric_ops()
    test_if()
    test_yolo()
    test_engine_lifecycle()
    print()
    if CMA_LOG:
        peak = max((b - lo, label) for label, b, lo, a in CMA_LOG if lo >= 0)
        print("CMA: %d sample(s), largest drop %d kB during %s"
              % (len(CMA_LOG), peak[0], peak[1]))
    if FAILED:
        print("HYBRID TESTS FAILED: %d check(s)" % len(FAILED))
        for f in FAILED:
            print("  - %s" % f)
        return 1
    print("HYBRID TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
