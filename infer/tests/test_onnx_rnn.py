#!/usr/bin/env python3
"""test_onnx_rnn.py -- the recurrent family: RNN, GRU, LSTM.

Every case is one ONNX graph with one node, synthesised in-process, run through
the shipped runner (``mode="hybrid"``, ``host_policy="device"``) and compared
with the ONNX *specification* formula evaluated in numpy.

Why the expectation is not ``onnx.reference``: its ``RNN``/``GRU``/``LSTM``
implementations (1.17) ignore ``direction``, ``clip``, ``activations``,
``activation_alpha/beta``, ``sequence_lens`` and ``linear_before_reset`` in
GRU, and ``_step`` raises for ``num_directions != 1`` -- so a bidirectional or
clipped model has no oracle there at all.  ``op_matrix.py`` still compares the
default-attribute form against ``onnx.reference`` (that is the one form the
reference does implement); this file is the attribute matrix.

    TMPDIR=$HOME/scratch/tmp python3 tests/test_onnx_rnn.py
    python3 tests/test_onnx_rnn.py --only LSTM -v

Exit status is 0 only when every case ran on the VC4 kernels inside the
matrix tolerance, or was refused with the exact invariant the case names.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

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


def rnd(rng, shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


# =====================================================================
# the specification formula (float64, cast at the end)
# =====================================================================
def _act(name, alpha=None, beta=None):
    n = name.lower()
    if n == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if n == "tanh":
        return np.tanh
    if n == "relu":
        return lambda x: np.maximum(x, 0.0)
    if n == "affine":
        return lambda x: alpha * x + beta
    if n == "hardsigmoid":
        a = 0.2 if alpha is None else alpha
        b = 0.5 if beta is None else beta
        return lambda x: np.clip(a * x + b, 0.0, 1.0)
    if n == "leakyrelu":
        a = 0.01 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * x)
    if n == "elu":
        a = 1.0 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * (np.exp(x) - 1.0))
    if n == "softsign":
        return lambda x: x / (1.0 + np.abs(x))
    if n == "softplus":
        return lambda x: np.log1p(np.exp(x))
    raise ValueError(name)


GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}
DEFAULT_ACT = {"RNN": ["Tanh"], "GRU": ["Sigmoid", "Tanh"],
               "LSTM": ["Sigmoid", "Tanh", "Tanh"]}


def ref(op, X, W, R, B=None, h0=None, c0=None, P=None, attrs=None):
    """The ONNX spec, in float64.  Returns [Y, Y_h] or [Y, Y_h, Y_c]."""
    attrs = attrs or {}
    G = GATES[op]
    nd = W.shape[0]
    layout = int(attrs.get("layout", 0))
    direction = attrs.get("direction", "forward")
    clip = attrs.get("clip", None)
    lbr = int(attrs.get("linear_before_reset", 0))
    H = R.shape[2]
    X = np.asarray(X, np.float64)
    if layout == 1:
        X = np.swapaxes(X, 0, 1)
    T, Bn, I = X.shape
    acts = attrs.get("activations") or DEFAULT_ACT[op] * nd
    alphas = attrs.get("activation_alpha") or []
    betas = attrs.get("activation_beta") or []
    if isinstance(alphas, (int, float)):
        alphas = [alphas]
    if isinstance(betas, (int, float)):
        betas = [betas]
    K = len(DEFAULT_ACT[op])
    fns = []
    for d in range(nd):
        row = []
        for k in range(K):
            a = alphas[d * K + k] if len(alphas) == K * nd else None
            b = betas[d * K + k] if len(betas) == K * nd else None
            row.append(_act(acts[d * K + k], a, b))
        fns.append(row)

    Y = np.zeros((T, nd, Bn, H))
    Yh = np.zeros((nd, Bn, H))
    Yc = np.zeros((nd, Bn, H))
    for d in range(nd):
        Wd = np.asarray(W[d], np.float64)
        Rd = np.asarray(R[d], np.float64)
        if B is None:
            bias = np.zeros(2 * G * H)
        else:
            bias = np.asarray(B[d], np.float64)
        wb, rb = bias[:G * H], bias[G * H:]
        h = np.zeros((Bn, H)) if h0 is None else \
            np.asarray(h0[d] if layout == 0 else h0[:, d], np.float64)
        c = np.zeros((Bn, H)) if c0 is None else \
            np.asarray(c0[d] if layout == 0 else c0[:, d], np.float64)
        order = list(range(T))
        if direction == "reverse" or d == 1:
            order = order[::-1]
        for t in order:
            xt = X[t]
            if op == "RNN":
                pre = xt @ Wd.T + h @ Rd.T + (wb + rb)
                if clip is not None:
                    pre = np.clip(pre, -clip, clip)
                h = fns[d][0](pre)
            elif op == "GRU":
                Wzr, Wh = Wd[:2 * H], Wd[2 * H:]
                Rzr, Rh = Rd[:2 * H], Rd[2 * H:]
                pre = xt @ Wzr.T + h @ Rzr.T + (wb[:2 * H] + rb[:2 * H])
                if clip is not None:
                    pre = np.clip(pre, -clip, clip)
                z = fns[d][0](pre[:, :H])
                r = fns[d][0](pre[:, H:])
                if lbr:
                    pre_h = xt @ Wh.T + wb[2 * H:] + r * (
                        h @ Rh.T + rb[2 * H:])
                else:
                    pre_h = xt @ Wh.T + (r * h) @ Rh.T + (
                        wb[2 * H:] + rb[2 * H:])
                if clip is not None:
                    pre_h = np.clip(pre_h, -clip, clip)
                hh = fns[d][1](pre_h)
                h = (1.0 - z) * hh + z * h
            else:
                # gate order i, o, f, c
                pre = xt @ Wd.T + h @ Rd.T + (wb + rb)
                pi = po = pf = 0.0
                if P is not None:
                    Pd = np.asarray(P[d], np.float64)
                    pi, po, pf = Pd[:H], Pd[H:2 * H], Pd[2 * H:]
                gi, go, gf, gc = np.split(pre, 4, axis=-1)
                qi = gi + pi * c
                qf = gf + pf * c
                if clip is not None:
                    qi = np.clip(qi, -clip, clip)
                    qf = np.clip(qf, -clip, clip)
                    gc = np.clip(gc, -clip, clip)
                i = fns[d][0](qi)
                f = fns[d][0](qf)
                cc = fns[d][1](gc)
                c = f * c + i * cc
                qo = go + po * c
                if clip is not None:
                    qo = np.clip(qo, -clip, clip)
                o = fns[d][0](qo)
                h = o * fns[d][2](c)
            Y[t, d] = h
        Yh[d] = h
        Yc[d] = c
    out = [Y, Yh] + ([Yc] if op == "LSTM" else [])
    if layout == 1:
        # [b][t][d][h] and [b][d][h]
        out = [np.transpose(Y, (2, 0, 1, 3)),
               np.transpose(Yh, (1, 0, 2))] + \
              ([np.transpose(Yc, (1, 0, 2))] if op == "LSTM" else [])
    return [np.ascontiguousarray(np.asarray(a, np.float32)) for a in out]


# =====================================================================
# harness
# =====================================================================
class Case:
    def __init__(self, op, name, feeds, inits, inputs, attrs=None, n_out=1,
                 ref_override=None, refuse=None, note=""):
        self.op = op
        self.name = name
        self.feeds = feeds
        self.inits = inits
        self.inputs = inputs
        self.attrs = attrs or {}
        self.n_out = n_out
        self.ref_override = ref_override
        self.refuse = refuse
        self.note = note


def build(case, out_shapes=None):
    outs = ["Y%d" % i for i in range(case.n_out)]
    node = helper.make_node(case.op, case.inputs, outs, name="n",
                            **case.attrs)
    ins = [helper.make_tensor_value_info(k, F32, list(np.asarray(v).shape))
           for k, v in case.feeds.items()]
    inits = [numpy_helper.from_array(np.asarray(v), k)
             for k, v in case.inits.items()]
    out_vis = []
    for i, o in enumerate(outs):
        shape = None if out_shapes is None else \
            list(np.asarray(out_shapes[i]).shape)
        out_vis.append(helper.make_tensor_value_info(o, F32, shape))
    graph = helper.make_graph([node], "t", ins, out_vis, initializer=inits)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="vc4infer-test-onnx-rnn")
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception:
        for vi in model.graph.output:
            vi.type.tensor_type.shape.SetInParent()
    return model


def compare(want, got):
    want = np.asarray(want, np.float32)
    got = np.asarray(got, np.float32)
    if want.shape != got.shape:
        return "shape %s != %s" % (got.shape, want.shape)
    if want.size == 0:
        return None
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    tol = TOL_ABS + TOL_REL * np.maximum(np.abs(want.astype(np.float64)), 1.0)
    bad = int(np.count_nonzero(d > tol))
    if bad:
        i = int(np.argmax(d))
        return "%d of %d over tolerance (worst ref %r got %r)" % (
            bad, want.size, float(want.reshape(-1)[i]),
            float(got.reshape(-1)[i]))
    return None


def run_case(eng, case, workdir, verbose):
    feeds = {k: np.ascontiguousarray(v, np.float32)
             for k, v in case.feeds.items()}
    path = os.path.join(workdir, "r_%s_%s.onnx" %
                        (case.op, case.name.replace("/", "_")))
    ref = None
    if case.refuse is None:
        try:
            ref = case.ref_override()
        except Exception as e:
            return "oracle-error", "%s: %s" % (type(e).__name__, e)
    model = build(case, None if ref is None else ref)
    onnx.save(model, path)
    try:
        r = Runner(path, eng, mode="hybrid", host_policy="device")
        got = r.run(feeds)
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
    if len(got) != len(ref):
        return "arity", "%d outputs, want %d" % (len(got), len(ref))
    for i, (w, g) in enumerate(zip(ref, got)):
        msg = compare(w, g)
        if msg:
            return "mismatch", "output %d (%s): %s" % (
                i, ("Y", "Y_h", "Y_c")[i], msg)
    return "vc4", ""


# =====================================================================
# cases
# =====================================================================
def _mk(op, name, T, Bn, I, H, nd=1, attrs=None, rand=None, zero_x=False,
        h0=None, c0=None, P=None, with_b=True, refuse=None, note=""):
    import zlib
    rng = np.random.default_rng(zlib.crc32(
        ("%s|%s|%d|%d|%d|%d" % (op, name, T, Bn, I, H)).encode()) % 2 ** 31)
    G = GATES[op]
    layout = int((attrs or {}).get("layout", 0))
    xs = (T, Bn, I) if layout == 0 else (Bn, T, I)
    X = np.zeros(xs, np.float32) if zero_x else rnd(rng, xs, -1, 1)
    W = rnd(rng, (nd, G * H, I), -0.7, 0.7)
    R = rnd(rng, (nd, G * H, H), -0.7, 0.7)
    inits = {"W": W, "R": R}
    # the ONNX input order is X, W, R, B, sequence_lens, initial_h,
    # initial_c, P -- an absent optional input is an empty name
    slots = ["X", "W", "R", "", "", "", "", ""]
    if with_b:
        inits["B"] = rnd(rng, (nd, 2 * G * H), -0.3, 0.3)
        slots[3] = "B"
    if h0 is not None:
        inits["initial_h"] = h0
        slots[5] = "initial_h"
    if op == "LSTM" and c0 is not None:
        inits["initial_c"] = c0
        slots[6] = "initial_c"
    if op == "LSTM" and P is not None:
        inits["P"] = P
        slots[7] = "P"
    while slots and slots[-1] == "":
        slots.pop()
    inputs = slots
    n_out = 3 if op == "LSTM" else 2
    at = dict(attrs or {})
    at["hidden_size"] = H

    def oracle():
        return ref(op, X, W, R, inits.get("B"), inits.get("initial_h"),
                   inits.get("initial_c"), inits.get("P"), at)

    return Case(op, name, {"X": X}, inits, inputs, attrs=at, n_out=n_out,
                ref_override=oracle, refuse=refuse, note=note)


def _states(op, nd, Bn, H, layout, scale=0.5):
    rng = np.random.default_rng(7)
    a = rnd(rng, (nd, Bn, H), -scale, scale)
    if layout == 1:
        a = np.ascontiguousarray(np.transpose(a, (1, 0, 2)))
    return a


def cases():
    out = {}
    rng = np.random.default_rng(11)

    rnn, gru, lstm = [], [], []

    # ---- RNN -------------------------------------------------------
    for layout in (0, 1):
        tag = "-layout%d" % layout
        rnn.append(_mk("RNN", "t4-b2-i3-h2" + tag, 4, 2, 3, 2,
                       attrs={"layout": layout}))
        rnn.append(_mk("RNN", "t1-single-step" + tag, 1, 2, 3, 2,
                       attrs={"layout": layout}))
        rnn.append(_mk("RNN", "hidden1" + tag, 3, 2, 2, 1,
                       attrs={"layout": layout}))
        rnn.append(_mk("RNN", "hidden16-aligned" + tag, 2, 8, 16, 16,
                       attrs={"layout": layout},
                       note="batch 8 and hidden 16: the GEMM needs no pad"))
        rnn.append(_mk("RNN", "zero-input" + tag, 3, 2, 3, 2, zero_x=True,
                       attrs={"layout": layout}))
    rnn.append(_mk("RNN", "init-h", 3, 2, 3, 2,
                   h0=_states("RNN", 1, 2, 2, 0)))
    rnn.append(_mk("RNN", "no-bias", 3, 2, 3, 2, with_b=False))
    rnn.append(_mk("RNN", "relu", 3, 2, 3, 2, attrs={"activations": ["Relu"]}))
    rnn.append(_mk("RNN", "affine-alpha-beta", 3, 2, 3, 2,
                   attrs={"activations": ["Affine"], "activation_alpha": [0.5],
                          "activation_beta": [-0.25]}))
    rnn.append(_mk("RNN", "hardsigmoid-defaults", 3, 2, 3, 2,
                   attrs={"activations": ["HardSigmoid"]}))
    rnn.append(_mk("RNN", "leakyrelu-defaults", 3, 2, 3, 2,
                   attrs={"activations": ["LeakyRelu"]}))
    rnn.append(_mk("RNN", "clip", 3, 2, 3, 2,
                   attrs={"clip": 0.4}))
    rnn.append(_mk("RNN", "reverse", 4, 2, 3, 2,
                   attrs={"direction": "reverse"}))
    rnn.append(_mk("RNN", "bidirectional", 3, 2, 3, 2, nd=2,
                   attrs={"direction": "bidirectional"}))
    rnn.append(_mk("RNN", "seqlen-refused", 3, 2, 3, 2,
                   attrs={"hidden_size": 2},
                   refuse="sequence_lens"))
    rnn[-1].inputs = ["X", "W", "R", "B", "sequence_lens"]
    rnn[-1].inits["sequence_lens"] = np.asarray([2, 3], np.int64)
    rnn.append(_mk("RNN", "scaledtanh-refused", 3, 2, 3, 2,
                   attrs={"activations": ["ScaledTanh"],
                          "activation_alpha": [1.0],
                          "activation_beta": [1.0]},
                   refuse="activation"))

    # ---- GRU -------------------------------------------------------
    for lbr in (0, 1):
        t = "-lbr%d" % lbr
        for layout in (0, 1):
            tag = t + "-layout%d" % layout
            gru.append(_mk("GRU", "t4-b2-i3-h2" + tag, 4, 2, 3, 2,
                           attrs={"linear_before_reset": lbr,
                                  "layout": layout}))
        gru.append(_mk("GRU", "hidden1" + t, 3, 2, 2, 1,
                       attrs={"linear_before_reset": lbr}))
        gru.append(_mk("GRU", "hidden16" + t, 2, 8, 16, 16,
                       attrs={"linear_before_reset": lbr}))
        gru.append(_mk("GRU", "zero-input" + t, 3, 2, 3, 2, zero_x=True,
                       attrs={"linear_before_reset": lbr}))
        gru.append(_mk("GRU", "init-h" + t, 3, 2, 3, 2,
                       h0=_states("GRU", 1, 2, 2, 0),
                       attrs={"linear_before_reset": lbr}))
        gru.append(_mk("GRU", "clip" + t, 3, 2, 3, 2, attrs={
            "clip": 0.3, "linear_before_reset": lbr}))
        gru.append(_mk("GRU", "bidirectional" + t, 3, 2, 3, 2, nd=2,
                       attrs={"direction": "bidirectional",
                              "linear_before_reset": lbr}))
        gru.append(_mk("GRU", "t1-single-step" + t, 1, 2, 3, 2,
                       attrs={"linear_before_reset": lbr}))

    # ---- LSTM ------------------------------------------------------
    for layout in (0, 1):
        tag = "-layout%d" % layout
        lstm.append(_mk("LSTM", "t4-b2-i3-h2" + tag, 4, 2, 3, 2,
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "hidden1" + tag, 3, 2, 2, 1,
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "hidden16" + tag, 2, 8, 16, 16,
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "zero-input" + tag, 3, 2, 3, 2, zero_x=True,
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "t1-single-step" + tag, 1, 2, 3, 2,
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "init-hc" + tag, 3, 2, 3, 2,
                        h0=_states("LSTM", 1, 2, 2, layout),
                        c0=_states("LSTM", 1, 2, 2, layout, 0.9),
                        attrs={"layout": layout}))
        lstm.append(_mk("LSTM", "peepholes" + tag, 3, 2, 3, 2,
                        attrs={"layout": layout},
                        P=rnd(rng, (1, 6), -0.5, 0.5)))
        lstm.append(_mk("LSTM", "clip" + tag, 3, 2, 3, 2, attrs={
            "clip": 0.5, "layout": layout}))
        lstm.append(_mk("LSTM", "bidirectional" + tag, 3, 2, 3, 2, nd=2,
                        attrs={"direction": "bidirectional",
                               "layout": layout}))
    lstm.append(_mk("LSTM", "reverse", 4, 2, 3, 2,
                    attrs={"direction": "reverse"}))
    lstm.append(_mk("LSTM", "relu-tanh", 3, 2, 3, 2,
                    attrs={"activations": ["Relu", "Tanh", "Tanh"]}))
    lstm.append(_mk("LSTM", "softsign-softplus", 3, 2, 3, 2,
                    attrs={"activations": ["Softsign", "Softplus",
                                           "Tanh"]}))
    lstm.append(_mk("LSTM", "hardsigmoid-hardsigmoid-tanh", 3, 2, 3, 2,
                    attrs={"activations": ["HardSigmoid", "HardSigmoid",
                                           "Tanh"],
                           "activation_alpha": [0.25, 0.3, 0.0],
                           "activation_beta": [0.6, 0.4, 0.0]}))
    lstm.append(_mk("LSTM", "zero-length", 0, 2, 3, 2,
                    note="an empty sequence: Y is empty, Y_h/Y_c are 0"))
    lstm[-1].n_out = 2
    _zk = lstm[-1].ref_override
    lstm[-1].ref_override = lambda: _zk()[:2]

    out["RNN"] = rnn
    out["GRU"] = gru
    out["LSTM"] = lstm
    return out


ALL = {}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    ALL.update(cases())
    only = [s for s in args.only.replace(",", " ").split() if s]
    ops = [o for o in ("RNN", "GRU", "LSTM") if not only or o in only]
    workdir = args.work_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "vc4rnn-models")
    os.makedirs(workdir, exist_ok=True)
    eng = vc4infer.Engine(cache_dir=args.cache or
                          os.environ.get("VC4INFER_CACHE") or None)
    bad = []
    total = 0
    try:
        for op in ops:
            for case in ALL[op]:
                total += 1
                status, msg = run_case(eng, case, workdir, args.verbose)
                ok = status in ("vc4", "refused-ok")
                print("%-4s %-6s %-34s %-12s %s%s"
                      % ("ok" if ok else "FAIL", op, case.name, status,
                         case.note, "" if ok else "  " + msg[:190]))
                if not ok:
                    bad.append((op, case.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    if bad:
        for op, name, status, msg in bad:
            print("  %-6s %-34s %-12s %s" % (op, name, status, msg[:170]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
