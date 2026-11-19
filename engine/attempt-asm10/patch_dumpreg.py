#!/usr/bin/env python3
"""Dump the kernel's AADR (A r20) or ARB (A r24) through the proven store, by
replacing the prime_a probe's 'A0 = A vector' copy."""
import sys
sys.path.insert(0, "/home/raspi/work/engine-fast2/tools")
import kgen
from qenc import REGA, NOP_ADDR

OUT = sys.argv[1]
reg = int(sys.argv[2])
M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
a = kgen.build_gemm10(M, N, K, 1, probe="prime_a")
i = next(i for i, d in enumerate(a.w) if d.get("note") == "A0 = A vector")
a.w[i]["add"]["add_a"] = (REGA, reg)
a.w[i]["add"]["add_b"] = (REGA, reg)
a.w[i]["note"] = "PROBE A0 = ra%d" % reg
w = a.words()
open(OUT, "wb").write(b"".join(int(x).to_bytes(8, "little") for x in w))
print("dumped ra%d -> %s (%d instructions)" % (reg, OUT, len(w)))
