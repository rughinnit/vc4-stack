#!/usr/bin/env python3
"""check_rnn_seq.py -- a multi-step sequence against ``onnx.reference``.

``tests/test_onnx_rnn.py`` checks the attribute matrix against the ONNX spec
formula (the reference ignores direction/clip/activations).  This script does
the other half: one default-attribute graph per recurrent operator over a
longer sequence, compared value-by-value against ``onnx.reference``, and it
prints the launch count per timestep (the cost of the unrolled method).

    TMPDIR=$HOME/scratch/tmp python3 tests/check_rnn_seq.py
    python3 tests/check_rnn_seq.py --seq 16 --hidden 8

Exit status 0 only when every output is inside the matrix tolerance
``1e-4 + 1e-4*max(|ref|,1)``.
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding as vc4infer                  # noqa: E402
from vc4infer._onnxrun import Runner                       # noqa: E402

TOL_ABS, TOL_REL, OPSET, F32 = 1e-4, 1e-4, 22, TensorProto.FLOAT


def build(op, T, B, I, H, gates):
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (T, B, I)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, (1, gates * H, I)).astype(np.float32)
    R = rng.uniform(-0.5, 0.5, (1, gates * H, H)).astype(np.float32)
    Bv = rng.uniform(-0.3, 0.3, (1, 2 * gates * H)).astype(np.float32)
    ins = ["X", "W", "R", "B"]
    inits = [numpy_helper.from_array(a, n) for a, n in
             ((W, "W"), (R, "R"), (Bv, "B"))]
    # onnx.reference 1.17 never returns LSTM's Y_c (its _run returns
    # (Y, Y_h) whatever the node declares), so Y_c is covered by
    # tests/test_onnx_rnn.py against the specification formula instead
    outs = ["Y", "Y_h"]
    node = helper.make_node(op, ins, outs, name="n", hidden_size=H)
    g = helper.make_graph(
        [node], "t",
        [helper.make_tensor_value_info("X", F32, [T, B, I])],
        [helper.make_tensor_value_info(o, F32, None) for o in outs],
        initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    return m, X


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--input", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=6)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}
    eng = vc4infer.Engine(cache_dir=args.cache or
                          os.environ.get("VC4INFER_CACHE") or None)
    bad = 0
    try:
        for op in ("RNN", "GRU", "LSTM"):
            m, X = build(op, args.seq, args.batch, args.input, args.hidden,
                         gates[op])
            path = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                "rnnseq_%s.onnx" % op)
            onnx.save(m, path)
            ref = ReferenceEvaluator(m).run(None, {"X": X})
            r = Runner(path, eng, mode="hybrid", host_policy="device")
            got = r.run({"X": X}, profile=True)
            launches = sum(p["launches"] for p in r.profile)
            gpu = sum(p["gpu_ms"] for p in r.profile)
            worst_abs = worst_rel = 0.0
            over = 0
            for w, g in zip(ref, got):
                w = np.asarray(w, np.float32)
                g = np.asarray(g, np.float32)
                d = np.abs(g.astype(np.float64) - w.astype(np.float64))
                tol = TOL_ABS + TOL_REL * np.maximum(
                    np.abs(w.astype(np.float64)), 1.0)
                over += int(np.count_nonzero(d > tol))
                worst_abs = max(worst_abs, float(d.max()) if d.size else 0.0)
                rel = d / np.maximum(np.abs(w.astype(np.float64)), 1e-12)
                worst_rel = max(worst_rel,
                                float(rel[np.isfinite(rel)].max())
                                if rel.size else 0.0)
            nsteps = args.seq
            print("%-5s T=%d B=%d I=%d H=%d  launches %3d (%.1f/timestep)  "
                  "gpu %.2f ms  max abs %.3g  max rel %.3g  over tol %d"
                  % (op, args.seq, args.batch, args.input, args.hidden,
                     launches, launches / nsteps, gpu, worst_abs, worst_rel,
                     over))
            bad += over
    finally:
        eng.close()
    print("\n%s" % ("all outputs inside tolerance" if not bad
                    else "%d elements over tolerance" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
