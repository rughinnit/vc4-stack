#!/usr/bin/env python3
"""Patch the real gemm10's 32 accumulator-initialisation words so each C
accumulator = A r24 (ARB), then run the real kernel: the store dumps ARB per
lane."""
import sys
sys.path.insert(0, "/home/raspi/work/engine-fast2/tools")
import kgen
from qenc import REGA

OUT = sys.argv[1]
reg = int(sys.argv[2])
M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
a = kgen.build_gemm10(M, N, K, 1)
n = 0
for d in a.w:
    if d.get("note") == "" and d.get("add") and d.get("mul") is None:
        ad = d["add"]
        if ad["add_op"] == kgen.ADD_OR and ad["add_a"] == (kgen.IMM, 0) \
                and ad["add_b"] == (kgen.IMM, 0):
            ws = ad["ws"]
            d["add"] = dict(add_op=kgen.ADD_ADD, add_out=ad["add_out"], ws=ws,
                            add_a=(REGA, reg), add_b=(kgen.IMM, 0), sf=0)
            d["note"] = "PROBE ra%d" % reg
            n += 1
print("patched %d accumulator inits -> ra%d" % (n, reg))
w = a.words()
open(OUT, "wb").write(b"".join(int(x).to_bytes(8, "little") for x in w))
print("%d instructions -> %s" % (len(w), OUT))
