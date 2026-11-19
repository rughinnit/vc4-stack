#!/usr/bin/env python3
"""Bisect the ARB computation: optionally neuter the 'A rb base' add, then dump
ARB (A r24) through the store."""
import sys
sys.path.insert(0, "/home/raspi/work/engine-fast2/tools")
import kgen
from qenc import REGA, REGB

OUT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "dump"
M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
a = kgen.build_gemm10(M, N, K, 1)

if MODE == "noadd":
    i = next(i for i, d in enumerate(a.w) if d.get("note") == "A rb base")
    a.w[i]["add"] = dict(add_op=kgen.ADD_ADD, add_out=24, ws=0,
                         add_a=(REGA, 24), add_b=(kgen.IMM, 0), sf=0)
    a.w[i]["note"] = "A rb base (noadd)"

n = 0
for d in a.w:
    if d.get("note") in ("", "A rb base (noadd)") and d.get("add") \
            and d.get("mul") is None:
        ad = d["add"]
        if ad["add_op"] == kgen.ADD_OR and ad["add_a"] == (kgen.IMM, 0) \
                and ad["add_b"] == (kgen.IMM, 0):
            ws = ad["ws"]
            addr = ad["add_out"]
            if ws == 0:
                d["add"] = dict(add_op=kgen.ADD_ADD, add_out=addr, ws=0,
                                add_a=(REGA, 24), add_b=(kgen.IMM, 0), sf=0)
            else:
                d["add"] = dict(add_op=kgen.ADD_OR, add_out=addr, ws=1,
                                add_a=(REGB, 24), add_b=(REGB, 24), sf=0)
            n += 1
w = a.words()
open(OUT, "wb").write(b"".join(int(x).to_bytes(8, "little") for x in w))
print("%s: patched %d -> %s" % (MODE, n, OUT))
