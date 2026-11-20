#!/usr/bin/env python3
"""probe_haz - measure the QPU multiply write->read hazard on silicon.

Seeds one 16-lane vector (1, 1+1/16, ... 1+15/16, all exactly representable),
runs a named minimal dependency sequence from kgen.build_probe_seq, and
compares the raw result words with the IEEE model.  Because every value in the
probe is exact, a mismatch is a scheduling hazard, not rounding.

    VC4INFER_DAEMON=1 python3 infer/tests/probe_haz.py [seq ...]
"""
import os
import struct
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
from vc4infer import _binding as B          # noqa: E402


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def expected(seq, vals):
    head = seq.split("_")[0]
    t = [f32(v * 2.0) for v in vals]
    if head == "add":
        return [f32(a + 1.0) for a in t]
    return [f32(a * 3.0) for a in t]


def main(argv):
    seqs = argv[1:]
    if not seqs:
        seqs = (["acc_%d" % n for n in range(0, 6)] +
                ["bmov_%d" % n for n in range(0, 6)] +
                ["bmul_%d" % n for n in range(0, 6)] +
                ["amul_%d" % n for n in range(0, 6)] +
                ["add_%d" % n for n in range(0, 4)] +
                ["packmul_%d" % n for n in range(0, 4)] +
                ["packfadd_%d" % n for n in range(0, 4)])
    vals = [1.0 + k / 16.0 for k in range(16)]
    eng = B.Engine()
    a = eng.upload(np.array(vals, np.float32))
    o = eng.alloc(256)
    print("%-12s %-6s %s" % ("seq", "match", "first mismatch"))
    bad = 0
    for seq in seqs:
        eng.probe(seq, a, o, 16)
        got = [float(v) for v in eng.download(o, 256)[:16]]
        exp = expected(seq, vals)
        ok = all(g == e for g, e in zip(got, exp))
        detail = ""
        if not ok:
            bad += 1
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    detail = "lane %d got %.9g (%08x) want %.9g (%08x)" % (
                        i, g, struct.unpack("<I", struct.pack("<f", g))[0],
                        e, struct.unpack("<I", struct.pack("<f", e))[0])
        print("%-12s %-6s %s" % (seq, "OK" if ok else "BAD", detail))
    print("%d/%d sequences exact" % (len(seqs) - bad, len(seqs)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
