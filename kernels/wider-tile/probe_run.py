#!/usr/bin/env python3
"""asm5 probe runner: launch one binary with GEMM uniforms, dump the arena,
and report the C region as (chunk,row,lane) identities.

  python3 probe_run.py <bin> <nqpu> <N> [dump_off dump_len]
No flock inside: the caller holds /tmp/vc4-gpu.lock.
"""
import os
import re
import struct
import subprocess
import sys

ASM = os.path.dirname(os.path.abspath(__file__))
QRUN = os.path.join(ASM, "qrun_gemm")
STATUS = "/home/raspi/vc4cl/tools/read_v3d_status"


def health():
    r = subprocess.run(["sudo", "-n", STATUS], capture_output=True, text=True)
    m = re.search(r"SRQCS\s*=\s*(0x[0-9a-f]+)", r.stdout)
    return "STATE: LIVE" in r.stdout, (m.group(1) if m else "?")


def main():
    binp = sys.argv[1]
    nqpu = int(sys.argv[2])
    N = int(sys.argv[3])
    doff = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x10000
    dlen = int(sys.argv[5], 0) if len(sys.argv) > 5 else 0x40000
    df = "/tmp/probe_dump.bin"
    env = ["QRUN_TMO_MS=30000", "QRUN_GEMM=%d" % N,
           "QRUN_DUMP=0x%x,0x%x" % (doff, dlen), "QRUN_DUMP_FILE=" + df,
           "QRUN_SAVE_A=/tmp/probe_a.bin", "QRUN_SAVE_B=/tmp/probe_b.bin"]
    cmd = ["timeout", "-k", "3", "120", "sudo", "-n", "env"] + env + \
          [QRUN, binp, str(nqpu), "1", "0", "0x10000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    live, srq = health()
    print("\n".join(l for l in out.splitlines()
                    if not l.startswith(("OUT", "AFTER", "arena:"))))
    if r.returncode != 0 or "TIMEOUT" in out or not live:
        print("ABORT rc=%d live=%s SRQCS=%s" % (r.returncode, live, srq))
        return 2
    raw = open(df, "rb").read()
    w = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    f = lambda x: struct.unpack("<f", struct.pack("<I", x))[0]

    def ident(c, r, lane):
        return 1000 * c + 10 * r + lane

    rev = {}
    for c in range(8):
        for rr in range(4):
            for L in range(16):
                rev[float(ident(c, rr, L))] = (c, rr, L)
    print("C region (rows 0..31, cols 0..19); cell = got->(want c,r,lane) or 'got' ")
    bad = 0
    for row in range(32):
        cells = []
        for col in range(20):
            v = f(w[row * N + col])
            key = v if v in rev else None
            got = rev.get(v)
            # expected: which chunk/row/lane *should* be here
            want = (row // 4, row % 4, col) if col < 16 else None
            if want and got == want:
                cells.append("  .  ")
            else:
                bad += 1
                if got:
                    cells.append("%d/%d/%d" % got)
                else:
                    cells.append("%g" % v if v == v else "nan")
        print("%2d: %s" % (row, " ".join("%7s" % c for c in cells)))
    print("cells wrong in rows0..31 cols0..15: %d/512" % bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
