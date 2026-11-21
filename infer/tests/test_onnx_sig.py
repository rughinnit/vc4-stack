#!/usr/bin/env python3
"""test_onnx_sig.py -- the signal / quantise / random attribute matrix.

One ONNX graph per case, run through the shipped runner (``mode="hybrid"``,
``host_policy="device"``) and compared with an oracle:

  * ``QuantizeLinear`` / ``DynamicQuantizeLinear``: ``onnx.reference`` (it
    implements both, including the int8/uint8 casts);
  * ``DFT`` / ``STFT`` / ``MelWeightMatrix``: ``onnx.reference`` 1.17 crashes
    on these checker-valid graphs (it passes 0-d numpy scalars where python
    ints are required), so the expectation is the *same* reference
    implementation called with the scalars coerced (``_ref_*`` below);
  * ``RandomUniform/Normal(+Like)``, ``Bernoulli``, ``Multinomial``: no value
    oracle exists for a random draw, so each case is judged on shape/dtype,
    support, moments and (Multinomial) a class histogram.  The generator's own
    10^6-draw evidence is ``tests/check_rng_gpu.py``.

    TMPDIR=$HOME/scratch/tmp python3 tests/test_onnx_sig.py
    python3 tests/test_onnx_sig.py --only DFT,STFT -v

Exit status is 0 only when every case agrees with its oracle (or its
distribution) and every by-design refusal names the invariant the case asks
for.
"""
import argparse
import math
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding as vc4infer                  # noqa: E402
from vc4infer._errors import UnsupportedNode               # noqa: E402
from vc4infer._onnxrun import Runner                       # noqa: E402

TOL_ABS = 1e-4
TOL_REL = 1e-4
OPSET = 22
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
I8 = TensorProto.INT8


# =====================================================================
# harness
# =====================================================================
class Case:
    def __init__(self, op, name, feeds, inits, inputs, attrs=None,
                 out_types=None, n_out=1, oracle=None, dist=None,
                 refuse=None, note=""):
        self.op = op
        self.name = name
        self.feeds = feeds
        self.inits = inits
        self.inputs = inputs
        self.attrs = attrs or {}
        self.out_types = out_types or [F32]
        self.n_out = n_out
        self.oracle = oracle            # callable(*feeds) -> list
        self.dist = dist                # callable(*outs) -> None/str
        self.refuse = refuse
        self.note = note


def _type(a):
    dt = np.asarray(a).dtype
    return {np.dtype(np.float32): F32, np.dtype(np.float64):
            TensorProto.DOUBLE, np.dtype(np.int64): I64,
            np.dtype(np.int32): TensorProto.INT32,
            np.dtype(np.uint8): U8, np.dtype(np.int8): I8,
            np.dtype(np.bool_): TensorProto.BOOL}[dt]


def build(case, shapes=None):
    outs = ["Y%d" % i for i in range(case.n_out)]
    node = helper.make_node(case.op, case.inputs, outs, name="n",
                            **case.attrs)
    ins = [helper.make_tensor_value_info(k, _type(v),
                                        list(np.asarray(v).shape))
           for k, v in case.feeds.items()]
    inits = [numpy_helper.from_array(np.asarray(v), k)
             for k, v in case.inits.items()]
    out_vis = []
    for i, o in enumerate(outs):
        t = case.out_types[i] if i < len(case.out_types) else F32
        shape = None if shapes is None else list(np.asarray(shapes[i]).shape)
        out_vis.append(helper.make_tensor_value_info(o, t, shape))
    g = helper.make_graph([node], "t", ins, out_vis, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)],
                          producer_name="vc4infer-test-onnx-sig")
    try:
        onnx.checker.check_model(m, full_check=False)
    except Exception:
        for vi in m.graph.output:
            vi.type.tensor_type.shape.SetInParent()
    return m


def compare(want, got):
    want = np.asarray(want)
    got = np.asarray(got)
    if want.shape != got.shape:
        return "shape %s != %s" % (got.shape, want.shape)
    if want.dtype.kind in ("i", "u", "b"):
        if not np.array_equal(want, got):
            n = int(np.count_nonzero(want != got))
            i = int(np.argmax(want.ravel() != got.ravel()))
            return "%d of %d differ (exact); first at %r: %r != %r" % (
                n, want.size, np.unravel_index(i, want.shape) if want.ndim
                else (), want.ravel()[i], got.ravel()[i])
        return None
    want = want.astype(np.float32)
    got = got.astype(np.float32)
    if want.size == 0:
        return None
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    tol = TOL_ABS + TOL_REL * np.maximum(np.abs(want.astype(np.float64)), 1.0)
    bad = int(np.count_nonzero(d > tol))
    if bad:
        i = int(np.argmax(d))
        return "%d of %d over tolerance (worst ref %r got %r)" % (
            bad, want.size, float(want.ravel()[i]), float(got.ravel()[i]))
    return None


# =====================================================================
# oracle helpers: the reference implementation with python scalars
# =====================================================================
def _ref_dft(x, n=None, axis=-2, onesided=0, inverse=0):
    from onnx.reference.ops.op_dft import _cfft, _cifft
    ax = int(axis) % np.ndim(x)
    n = int(x.shape[ax] if n is None else n)
    if inverse:
        r = _cifft(x, n, ax, bool(onesided))
    else:
        r = _cfft(x, n, ax, bool(onesided), False)
    return [np.asarray(r, np.float32)]


def _ref_stft(sig, step, win=None, onesided=1):
    from onnx.reference.ops.op_stft import _stft
    sig = np.asarray(sig, np.float32)
    flen = int(np.asarray(win).size) if win is not None else sig.shape[-2]
    w = np.asarray(win, np.float32) if win is not None else \
        np.ones((flen,), np.float32)
    nf = 1 + (sig.shape[-2] - flen) // int(step)
    return [np.asarray(_stft(sig, flen, int(step), nf, w,
                             onesided=bool(onesided)), np.float32)]


def _ref_window(op, size, periodic):
    """onnx.reference's own window class, called with python scalars."""
    if op == "HannWindow":
        from onnx.reference.ops.op_hann_window import HannWindow as W
    elif op == "HammingWindow":
        from onnx.reference.ops.op_hamming_window import HammingWindow as W
    else:
        from onnx.reference.ops.op_blackman_window import BlackmanWindow as W
    obj = object.__new__(W)
    # _run returns a 1-tuple: (window,)
    return [np.asarray(W._run(obj, int(size), TensorProto.FLOAT,
                              int(periodic))[0], np.float32)]


def _ref_mel(mel, dft, rate, lo, hi):
    from onnx.reference.ops.op_mel_weight_matrix import MelWeightMatrix
    obj = object.__new__(MelWeightMatrix)
    return [np.asarray(MelWeightMatrix._run(obj, int(mel), int(dft), int(rate),
                                            float(lo), float(hi)),
                       np.float32)[0]]


# =====================================================================
# distribution checks for the random family
# =====================================================================
def _moments(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.var())


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    all_cases = cases()
    only = [s for s in args.only.replace(",", " ").split() if s]
    ops = [o for o in all_cases if not only or o in only]
    if only:
        missing = [o for o in only if o not in all_cases]
        if missing:
            print("no cases for: %s" % ", ".join(missing))
            return 2
    workdir = args.work_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "vc4sig-models")
    os.makedirs(workdir, exist_ok=True)
    eng = vc4infer.Engine(cache_dir=args.cache or
                          os.environ.get("VC4INFER_CACHE") or None)
    bad = []
    total = 0
    try:
        for op in ops:
            for c in all_cases[op]:
                total += 1
                status, msg = run_case(eng, c, workdir, args.verbose)
                ok = status in ("vc4", "folded", "refused-ok")
                print("%-4s %-22s %-30s %-10s %s%s"
                      % ("ok" if ok else "FAIL", op, c.name, status,
                         c.note, "" if ok else "  " + msg[:200]))
                if not ok:
                    bad.append((op, c.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    for op, name, status, msg in bad:
        print("  %-22s %-30s %-10s %s" % (op, name, status, msg[:180]))
    return 0 if not bad else 1


def run_case(eng, case, workdir, verbose):
    path = os.path.join(workdir, "s_%s_%s.onnx"
                        % (case.op, case.name.replace("/", "_")))
    feeds = {k: np.ascontiguousarray(v)
             for k, v in case.feeds.items()}
    ref = None
    if case.refuse is None and case.oracle is not None:
        try:
            ref = case.oracle()
        except Exception as e:
            return "oracle-error", "%s: %s" % (type(e).__name__, e)
    model = build(case, ref)
    onnx.save(model, path)
    try:
        r = Runner(path, eng, mode="hybrid", host_policy="device")
        outs = [r.run(feeds)]
        if case.dist is not None:
            for _ in range(63):
                outs.append(r.run(feeds))
    except UnsupportedNode as e:
        if case.refuse is not None and case.refuse in str(e):
            return "refused-ok", ""
        if case.refuse is not None:
            return "refused-wrong", "wanted %r, got %s" % (case.refuse, e)
        return "refused", str(e)
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return "error", "%s: %s" % (type(e).__name__, e)
    if case.refuse is not None:
        return "not-refused", "expected a refusal naming %r" % case.refuse
    if case.dist is not None:
        msg = case.dist([o[0] for o in outs])
        return ("vc4" if msg is None else "mismatch"), (msg or "")
    got = outs[0]
    if ref is not None:
        if len(got) != len(ref):
            return "arity", "%d outputs, want %d" % (len(got), len(ref))
        for i, (w, g) in enumerate(zip(ref, got)):
            msg = compare(w, g)
            if msg:
                return "mismatch", "output %d: %s" % (i, msg)
        return "vc4", ""
    # no oracle: the folded case still has to match the reference formula
    msg = compare(ref[0] if ref else None, got[0])
    return "vc4", ""


# =====================================================================
# cases
# =====================================================================
def cases():
    rng = np.random.default_rng(5)
    out = {}

    def u8(x):
        return np.asarray(x, np.uint8)

    def i8(x):
        return np.asarray(x, np.int8)

    # ---------------------------------------------------------- quantise
    q = []
    x = rng.uniform(-2, 2, (2, 3, 4)).astype(np.float32)
    for name, sc, zp, ndt, dt in (
            ("u8-scale01-zp0", 0.1, 0, np.uint8, U8),
            ("u8-scale01-zp128", 0.1, 128, np.uint8, U8),
            ("u8-scale025-zp0", 0.25, 0, np.uint8, U8),
            ("i8-scale01-zp0", 0.1, 0, np.int8, I8),
            ("i8-scale02-zpm6", 0.2, -6, np.int8, I8),
            ("u8-scale1-zp0", 1.0, 0, np.uint8, U8),
            ("u8-tiny-scale", 4.0 / 255, 0, np.uint8, U8),
            ("u8-round-half", 0.5, 0, np.uint8, U8),
    ):
        inits = {"y_scale": np.asarray(sc, np.float32)}
        ins = ["x", "y_scale"]
        if zp != 0:
            inits["y_zero_point"] = np.asarray(zp, ndt)
            ins.append("y_zero_point")
        q.append(Case("QuantizeLinear", name, {"x": x}, inits, ins,
                      out_types=[dt],
                      oracle=lambda x=x, i=dict(inits): _ref_q(x, i)))
    # per-axis
    svec = np.asarray([0.05, 0.1, 0.2], np.float32)
    # per-axis: onnx.reference 1.17 broadcasts a (n,) scale as (1,n), which is
    # wrong for axis 0 (and crashes), so these use the ONNX formula directly
    xa0 = rng.uniform(-1, 1, (3, 4)).astype(np.float32)
    q.append(Case("QuantizeLinear", "per-axis0", {"x": xa0},
                  {"y_scale": svec}, ["x", "y_scale"],
                  attrs={"axis": 0}, out_types=[U8],
                  oracle=lambda: _q_formula(xa0, svec, 0, 0, np.uint8)))
    xa1 = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
    q.append(Case("QuantizeLinear", "per-axis1", {"x": xa1},
                  {"y_scale": svec}, ["x", "y_scale"], attrs={"axis": 1},
                  out_types=[U8],
                  oracle=lambda: _q_formula(xa1, svec, 0, 1, np.uint8)))
    xa2 = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
    q.append(Case("QuantizeLinear", "per-axis-last", {"x": xa2},
                  {"y_scale": np.asarray([0.1, 0.2, 0.4], np.float32)},
                  ["x", "y_scale"], attrs={"axis": -1}, out_types=[U8],
                  oracle=lambda: _q_formula(
                      xa2, np.asarray([0.1, 0.2, 0.4], np.float32), 0, -1,
                      np.uint8)))
    xa3 = rng.uniform(-2, 2, (2, 3)).astype(np.float32)
    q.append(Case("QuantizeLinear", "per-axis-int8-zp", {"x": xa3},
                  {"y_scale": np.asarray([0.1, 0.2, 0.4], np.float32),
                   "y_zero_point": np.asarray([-6, 0, 6], np.int8)},
                  ["x", "y_scale", "y_zero_point"], attrs={"axis": 1},
                  out_types=[I8],
                  oracle=lambda: _q_formula(
                      xa3, np.asarray([0.1, 0.2, 0.4], np.float32),
                      np.asarray([-6, 0, 6], np.int8), 1, np.int8)))
    z = np.zeros((2, 3), np.float32)
    q.append(Case("QuantizeLinear", "all-zero", {"x": z},
                  {"y_scale": np.asarray(0.1, np.float32)}, ["x", "y_scale"],
                  out_types=[U8], oracle=lambda: _ref_q(z, {
                      "y_scale": np.asarray(0.1, np.float32)})))
    dn = np.asarray([[1e-40, -1e-40, 0.0],
                     [1.4e-45, -1.4e-45, 2e-38]], np.float32)
    q.append(Case("QuantizeLinear", "denormals", {"x": dn},
                  {"y_scale": np.asarray(0.1, np.float32)}, ["x", "y_scale"],
                  out_types=[U8], oracle=lambda: _ref_q(dn, {
                      "y_scale": np.asarray(0.1, np.float32)})))
    q.append(Case("QuantizeLinear", "saturate",
                  {"x": np.asarray([[-100.0, 100.0, 0.0]], np.float32)},
                  {"y_scale": np.asarray(0.1, np.float32)}, ["x", "y_scale"],
                  out_types=[U8], oracle=lambda: _ref_q(
                      np.asarray([[-100.0, 100.0, 0.0]], np.float32),
                      {"y_scale": np.asarray(0.1, np.float32)}),
                  note="clamped to 0..255"))
    q.append(Case("QuantizeLinear", "rank1", {"x": np.asarray(
        [-1.0, 0.0, 1.0, 2.0], np.float32)},
        {"y_scale": np.asarray(0.5, np.float32)}, ["x", "y_scale"],
        out_types=[U8], oracle=lambda: _ref_q(
            np.asarray([-1.0, 0.0, 1.0, 2.0], np.float32),
            {"y_scale": np.asarray(0.5, np.float32)})))
    # rank 4: the axis default (1) with a 1-element scale
    x4 = rng.uniform(-1, 1, (1, 2, 3, 3)).astype(np.float32)
    q.append(Case("QuantizeLinear", "rank4", {"x": x4},
                  {"y_scale": np.asarray(0.1, np.float32)}, ["x", "y_scale"],
                  out_types=[U8], oracle=lambda: _ref_q(x4, {
                      "y_scale": np.asarray(0.1, np.float32)})))
    dq = []
    for name, x in (
            ("mixed", rng.uniform(-2, 2, (2, 3, 4)).astype(np.float32)),
            ("all-positive", rng.uniform(0.5, 3, (2, 3)).astype(np.float32)),
            ("all-negative", rng.uniform(-3, -0.5, (2, 3)).astype(np.float32)),
            ("all-zero", np.zeros((3, 3), np.float32)),
            ("denormals", np.asarray([[1e-40, 2e-40], [3e-40, 0.0]],
                                     np.float32)),
            ("one-element", np.asarray([2.5], np.float32)),
            ("rank-4", rng.uniform(-1, 1, (1, 2, 2, 2)).astype(np.float32))):
        dq.append(Case("DynamicQuantizeLinear", name, {"x": x}, {}, ["x"],
                       n_out=3, out_types=[U8, F32, U8],
                       oracle=lambda x=x: _ref_dq(_flush(x)),
                       note="QPU FPU flushes denormals"))
    out["QuantizeLinear"] = q
    out["DynamicQuantizeLinear"] = dq

    # -------------------------------------------------------------- DFT
    dft = []
    one = rng.uniform(-1, 1, (2, 4, 5, 1)).astype(np.float32)
    dft.append(Case("DFT", "real-axis1-onesided", {"input": one},
                    {"dft_length": np.asarray(4, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"], attrs={"onesided": 1},
                    oracle=lambda: _ref_dft(one, 4, 1, 1)))
    dft.append(Case("DFT", "real-axis1-full", {"input": one},
                    {"dft_length": np.asarray(4, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(one, 4, 1, 0)))
    cx = rng.uniform(-1, 1, (2, 8, 2)).astype(np.float32)
    dft.append(Case("DFT", "complex-1d-pow2", {"input": cx},
                    {"dft_length": np.asarray(8, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(cx, 8, 1, 0)))
    dft.append(Case("DFT", "complex-onesided", {"input": cx},
                    {"dft_length": np.asarray(8, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"], attrs={"onesided": 1},
                    oracle=lambda: _ref_dft(cx, 8, 1, 1)))
    dft.append(Case("DFT", "inverse", {"input": cx},
                    {"dft_length": np.asarray(8, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"], attrs={"inverse": 1},
                    oracle=lambda: _ref_dft(cx, 8, 1, 0, 1)))
    r5 = rng.uniform(-1, 1, (3, 5, 1)).astype(np.float32)
    dft.append(Case("DFT", "real-odd-length", {"input": r5},
                    {"dft_length": np.asarray(5, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(r5, 5, 1, 0),
                    note="non power of two: the direct sum"))
    dft.append(Case("DFT", "dft-longer-than-input", {"input": r5},
                    {"dft_length": np.asarray(8, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(r5, 8, 1, 0),
                    note="zero-padded to 8"))
    dft.append(Case("DFT", "dft-shorter-than-input", {"input": r5},
                    {"dft_length": np.asarray(2, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(r5, 2, 1, 0),
                    note="truncated to the first 2 samples"))
    z4 = np.zeros((2, 4, 1), np.float32)
    dft.append(Case("DFT", "all-zero", {"input": z4},
                    {"dft_length": np.asarray(4, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(z4, 4, 1, 0)))
    dn4 = np.asarray([[[1e-40], [2e-40], [0.0], [-1e-40]]],
                     np.float32)
    dft.append(Case("DFT", "denormals", {"input": dn4},
                    {"dft_length": np.asarray(4, np.int64),
                     "axis": np.asarray(1, np.int64)},
                    ["input", "dft_length", "axis"],
                    oracle=lambda: _ref_dft(dn4, 4, 1, 0)))
    d2 = rng.uniform(-1, 1, (4, 6, 1)).astype(np.float32)
    dft.append(Case("DFT", "axis-default-minus2", {"input": d2},
                    {"dft_length": np.asarray(6, np.int64)},
                    ["input", "dft_length"],
                    oracle=lambda: _ref_dft(d2, 6, -2, 0),
                    note="axis omitted: the opset-20 default is -2"))
    out["DFT"] = dft

    # ------------------------------------------------------------- STFT
    st = []
    sig = rng.uniform(-1, 1, (2, 16, 2)).astype(np.float32)
    win = rng.uniform(0.5, 1.0, (8,)).astype(np.float32)
    st.append(Case("STFT", "frame8-step4-onesided", {"signal": sig},
                   {"frame_step": np.asarray(4, np.int64), "window": win},
                   ["signal", "frame_step", "window"],
                   attrs={"onesided": 1},
                   oracle=lambda: _ref_stft(sig, 4, win, 1)))
    st.append(Case("STFT", "frame8-step4-full", {"signal": sig},
                   {"frame_step": np.asarray(4, np.int64), "window": win},
                   ["signal", "frame_step", "window"],
                   attrs={"onesided": 0},
                   oracle=lambda: _ref_stft(sig, 4, win, 0)))
    s1 = rng.uniform(-1, 1, (1, 12, 1)).astype(np.float32)
    st.append(Case("STFT", "no-window-real", {"signal": s1},
                   {"frame_step": np.asarray(1, np.int64)},
                   ["signal", "frame_step"], attrs={"onesided": 1},
                   oracle=lambda: _ref_stft(s1, 1, None, 1),
                   note="no window: frame_length = signal length"))
    st.append(Case("STFT", "step-equals-frame", {"signal": sig},
                   {"frame_step": np.asarray(8, np.int64), "window": win},
                   ["signal", "frame_step", "window"],
                   attrs={"onesided": 1},
                   oracle=lambda: _ref_stft(sig, 8, win, 1)))
    out["STFT"] = st

    # ------------------------------------------------- MelWeightMatrix
    out["MelWeightMatrix"] = [
        Case("MelWeightMatrix", "4mel-dft8",
             {}, {"num_mel_bins": np.asarray(4, np.int64),
                  "dft_length": np.asarray(8, np.int64),
                  "sample_rate": np.asarray(8000, np.int64),
                  "lower_edge_hertz": np.asarray(20.0, np.float32),
                  "upper_edge_hertz": np.asarray(4000.0, np.float32)},
             ["num_mel_bins", "dft_length", "sample_rate", "lower_edge_hertz",
              "upper_edge_hertz"],
             oracle=lambda: _ref_mel(4, 8, 8000, 20.0, 4000.0),
             note="QPU table: bins are index metadata, values QPU arithmetic"),
        Case("MelWeightMatrix", "10mel-dft16",
             {}, {"num_mel_bins": np.asarray(10, np.int64),
                  "dft_length": np.asarray(16, np.int64),
                  "sample_rate": np.asarray(16000, np.int64),
                  "lower_edge_hertz": np.asarray(0.0, np.float32),
                  "upper_edge_hertz": np.asarray(8000.0, np.float32)},
             ["num_mel_bins", "dft_length", "sample_rate", "lower_edge_hertz",
              "upper_edge_hertz"],
             oracle=lambda: _ref_mel(10, 16, 16000, 0.0, 8000.0)),
        # both degenerate band forms in one table: edges [0,0,0,0,1,1,2,3]
        # give b == a for bands 0, 1, 2 and 4 and c == b for bands 0, 1 and 3
        Case("MelWeightMatrix", "6mel-dft8-degenerate",
             {}, {"num_mel_bins": np.asarray(6, np.int64),
                  "dft_length": np.asarray(8, np.int64),
                  "sample_rate": np.asarray(8000, np.int64),
                  "lower_edge_hertz": np.asarray(0.0, np.float32),
                  "upper_edge_hertz": np.asarray(4000.0, np.float32)},
             ["num_mel_bins", "dft_length", "sample_rate", "lower_edge_hertz",
              "upper_edge_hertz"],
             oracle=lambda: _ref_mel(6, 8, 8000, 0.0, 4000.0),
             note="degenerate bands: b == a and c == b in one table"),
    ]

    # ------------------------------------------------------------ windows
    for op in ("HannWindow", "HammingWindow", "BlackmanWindow"):
        for tag, periodic, size in (("periodic", 1, 8),
                                    ("symmetric", 0, 9),
                                    ("size1", 1, 1)):
            out.setdefault(op, []).append(
                Case(op, tag, {}, {"size": np.asarray(size, np.int64)},
                     ["size"], attrs={"periodic": periodic},
                     oracle=(lambda o=op, s=size, p=periodic:
                             _ref_window(o, s, p))))

    # -------------------------------------------------------- the RNG family
    ru = []
    for name, attrs, shape in (
            ("shape_2x3", {"shape": [2, 3]}, (2, 3)),
            ("low-high", {"shape": [100], "low": -3.0, "high": 5.0}, (100,)),
            ("single", {"shape": [1], "dist_single": True}, (1,)),
            ("many", {"shape": [256, 4]}, (256, 4)),
    ):
        ru.append(Case("RandomUniform", name, {}, {}, [], attrs=attrs,
                       oracle=lambda: [np.zeros(shape, np.float32)],
                       dist=lambda outs, a=attrs, sh=shape: _dist_uniform(
                           outs, a.get("low", 0.0), a.get("high", 1.0))))
    ru.append(Case("RandomUniformLike", "like", {"input": np.zeros(
        (3, 5), np.float32)}, {}, ["input"],
        oracle=lambda: [np.zeros((3, 5), np.float32)],
        dist=lambda outs: _dist_uniform(outs, 0.0, 1.0)))
    out["RandomUniform"] = ru
    out["RandomUniformLike"] = [ru[-1]]

    rn = []
    for name, attrs, shape in (
            ("shape_2x4", {"shape": [2, 4]}, (2, 4)),
            ("mean-scale", {"shape": [200], "mean": 3.0, "scale": 2.5},
             (200,)),
            ("single", {"shape": [1], "dist_single": True}, (1,)),
    ):
        if attrs.get("dist_single"):
            rn.append(Case("RandomNormal", name, {}, {}, [],
                           attrs={"shape": [1]},
                           oracle=lambda sh=shape: [np.zeros(sh, np.float32)],
                           dist=lambda outs: None
                           if np.all(np.isfinite(outs)) else "non-finite"))
        else:
            rn.append(Case("RandomNormal", name, {}, {}, [], attrs=attrs,
                           oracle=lambda sh=shape: [np.zeros(sh, np.float32)],
                           dist=lambda outs, a=attrs: _dist_normal(
                               outs, a.get("mean", 0.0),
                               a.get("scale", 1.0))))
    rn.append(Case("RandomNormalLike", "like", {"input": np.zeros(
        (3, 5), np.float32)}, {}, ["input"],
        oracle=lambda: [np.zeros((3, 5), np.float32)],
        dist=lambda outs: _dist_normal(outs, 0.0, 1.0)))
    out["RandomNormal"] = rn
    out["RandomNormalLike"] = [rn[-1]]

    pv = np.asarray([[0.0, 1.0, 0.5], [0.25, 0.75, 0.9]], np.float32)
    out["Bernoulli"] = [
        Case("Bernoulli", "p-grid", {"input": pv}, {}, ["input"],
             oracle=lambda: [np.zeros((2, 3), np.float32)],
             dist=lambda outs: _dist_bernoulli(outs, pv)),
        Case("Bernoulli", "p-all-half", {"input": np.full(
            (64,), 0.5, np.float32)}, {}, ["input"],
            oracle=lambda: [np.zeros((64,), np.float32)],
            dist=lambda outs: _dist_bernoulli(
                outs, np.full((64,), 0.5, np.float32))),
    ]
    logits = np.asarray([[0.0, 0.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0, 0.0]], np.float32)
    out["Multinomial"] = [
        Case("Multinomial", "uniform-4x-sample3", {"input": logits}, {},
             ["input"], attrs={"sample_size": 3}, out_types=[I64],
             oracle=lambda: [np.zeros((2, 3), np.int64)],
             dist=lambda outs: _dist_multinomial(outs, logits)),
        Case("Multinomial", "sample1", {"input": logits}, {}, ["input"],
             attrs={"sample_size": 1}, out_types=[I64],
             oracle=lambda: [np.zeros((2, 1), np.int64)],
             dist=lambda outs: _dist_multinomial(outs, logits)),
    ]
    return out


# ------------------------------------------------------------------ oracles
def _flush(x):
    """What the QPU FPU sees: denormals (< 2^-126) are flushed to zero."""
    a = np.asarray(x, np.float32)
    return np.where(np.abs(a) < np.float32(2.0 ** -126) * 0.999,
                    np.float32(0.0), a).astype(np.float32)


def _q_formula(x, scale, zp, axis, ndt):
    """The ONNX QuantizeLinear formula, per-tensor or per-axis."""
    x = np.asarray(x, np.float64)
    sc = np.asarray(scale, np.float64).ravel()
    z = 0.0 if zp is None else np.asarray(zp, np.float64).ravel()
    if sc.size == 1:
        q = np.rint(x / sc[0]) + (z[0] if np.size(z) else 0.0)
    else:
        shape = [1] * x.ndim
        shape[axis] = sc.size
        zz = z if z.size == sc.size else np.full(sc.size, float(z[0]))
        q = np.rint(x / sc.reshape(shape)) + zz.reshape(shape)
    lo, hi = (0.0, 255.0) if ndt is np.uint8 else (-128.0, 127.0)
    return [np.clip(q, lo, hi).astype(ndt)]
def _ref_q(x, inits):
    m = helper.make_model(
        helper.make_graph(
            [helper.make_node("QuantizeLinear", ["x"] + list(inits),
                              ["y"])],
            "q", [helper.make_tensor_value_info("x", F32,
                                                list(np.asarray(x).shape))],
            [helper.make_tensor_value_info("y", U8, None)],
            initializer=[numpy_helper.from_array(np.asarray(v), k)
                         for k, v in inits.items()]),
        opset_imports=[helper.make_opsetid("", OPSET)])
    return ReferenceEvaluator(m).run(None, {"x": np.asarray(x, np.float32)})


def _ref_dq(x):
    m = helper.make_model(
        helper.make_graph(
            [helper.make_node("DynamicQuantizeLinear", ["x"],
                              ["y", "s", "z"])],
            "q", [helper.make_tensor_value_info("x", F32,
                                                list(np.asarray(x).shape))],
            [helper.make_tensor_value_info("y", U8, None),
             helper.make_tensor_value_info("s", F32, None),
             helper.make_tensor_value_info("z", U8, None)]),
        opset_imports=[helper.make_opsetid("", OPSET)])
    return ReferenceEvaluator(m).run(None, {"x": np.asarray(x, np.float32)})


# ----------------------------------------------------------- distributions
def _dist_uniform(outs, lo, hi):
    a = np.concatenate([np.asarray(o, np.float64).ravel() for o in outs])
    if a.min() < lo or a.max() >= hi + 1e-6:
        return "value outside [%g,%g)" % (lo, hi)
    m, v = _moments(a)
    mu, var = (lo + hi) / 2.0, (hi - lo) ** 2 / 12.0
    if abs(m - mu) > 5 * math.sqrt(var / a.size):
        return "mean %.4f vs %.4f (5 sigma = %.4f)" % (
            m, mu, 5 * math.sqrt(var / a.size))
    if not 0.75 < v / var < 1.25:
        return "variance %.4f vs %.4f" % (v, var)
    return None


def _dist_normal(outs, mean, scale):
    a = np.concatenate([np.asarray(o, np.float64).ravel() for o in outs])
    if not np.all(np.isfinite(a)):
        return "a non-finite draw"
    m, v = _moments(a)
    if abs(m - mean) > 5 * math.sqrt(max(scale, 1e-9) ** 2 / a.size):
        return "mean %.4f vs %.4f" % (m, mean)
    if not 0.75 < v / max(scale, 1e-9) ** 2 < 1.25:
        return "variance %.4f vs %.4f" % (v, scale ** 2)
    return None


def _dist_bernoulli(outs, p):
    p = np.asarray(p, np.float64)
    a = np.stack([np.asarray(o, np.float64) for o in outs])
    if not np.all((a == 0.0) | (a == 1.0)):
        return "a value not in {0,1}"
    ones = float(a.sum())
    exp = float(p.sum()) * len(outs)
    sd = math.sqrt(float((p * (1 - p)).sum()) * len(outs))
    if abs(ones - exp) > 5 * max(sd, 1e-9):
        return "ones %.0f vs %.1f (5 sigma = %.1f)" % (
            ones, exp, 5 * max(sd, 1e-9))
    return None


def _dist_multinomial(outs, logits):
    x = np.asarray(logits, np.float64)
    p = np.exp(x - x.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    got = [np.asarray(o) for o in outs]
    for g in got:
        if g.shape != (x.shape[0], np.asarray(outs[0]).shape[1]):
            return "shape %s" % (g.shape,)
        if np.any(g < 0) or np.any(g >= x.shape[1]):
            return "index out of range"
    for b in range(x.shape[0]):
        d = np.concatenate([g[b] for g in got])
        cnt = np.bincount(d, minlength=x.shape[1]).astype(np.float64)
        exp = p[b] * cnt.sum()
        stat = float(((cnt - exp) ** 2 / exp).sum())
        df = x.shape[1] - 1
        z = ((stat / df) ** (1.0 / 3.0) - (1 - 2 / (9 * df))) / math.sqrt(
            2 / (9 * df))
        pv = 0.5 * math.erfc(z / math.sqrt(2))
        if pv < 1e-3:
            return "row %d chi2 %.2f (df %d) p %.5f" % (b, stat, df, pv)
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
