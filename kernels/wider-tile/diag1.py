#!/usr/bin/env python3
"""asm5 diag1: run one GEMM launch, then report exactly what changed.

  python3 diag1.py <bin> <nqpu> <N> [tag]
Uses QRUN_CPUPANEL8=1 so the panel is valid without the packer, saves C and
the panel, and diffs both against float64 references computed here.
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


def rd(p, n):
    raw = open(p, "rb").read()
    return list(struct.unpack("<%dI" % n, raw[:4 * n]))


def fl(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def main():
    binp, nqpu, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "d1"
    a_f = "/tmp/%s_a.bin" % tag
    b_f = "/tmp/%s_b.bin" % tag
    c_f = "/tmp/%s_c.bin" % tag
    p_f = "/tmp/%s_p.bin" % tag
    env = ["QRUN_TMO_MS=15000", "QRUN_GEMM=%d" % N, "QRUN_CPUPANEL8=1",
           "QRUN_SAVE_A=" + a_f, "QRUN_SAVE_B=" + b_f,
           "QRUN_SAVE_C=" + c_f, "QRUN_SAVE_PANEL=" + p_f]
    cmd = ["timeout", "-k", "3", "200", "sudo", "-n", "env"] + env + \
          [QRUN, binp, str(nqpu), "1", "0", "0x10000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    live, srq = health()
    print("\n".join(l for l in out.splitlines()
                    if not l.startswith(("OUT", "AFTER", "arena:"))))
    if r.returncode != 0 or not live:
        print("ABORT rc=%d live=%s SRQCS=%s" % (r.returncode, live, srq))
        return 2
    nn = N * N
    Aw = rd(a_f, nn)
    Bw = rd(b_f, nn)
    Cw = rd(c_f, nn)
    Pw = rd(p_f, nn)
    A = [fl(x) for x in Aw]
    B = [fl(x) for x in Bw]

    # ---- panel: 8-row map, PANEL8[(rb*N+k)*8+r] = A[rb*8+r][k]
    rows = 8
    nrb = N // rows
    pref = [0.0] * nn
    for rb in range(nrb):
        for k in range(N):
            for rr in range(rows):
                pref[(rb * N + k) * rows + rr] = A[(rb * rows + rr) * N + k]
    pbad = [i for i in range(nn) if Pw[i] != struct.unpack("<I", struct.pack("<f", pref[i]))[0]]
    print("panel: %d/%d mismatches after the launch" % (len(pbad), nn))
    if pbad:
        # group by row block and k range
        groups = {}
        for i in pbad:
            rb, k, rr = i // (N * 8), (i // 8) % N, i % 8
            groups.setdefault(rb, []).append((k, rr))
        for rb in sorted(groups):
            ks = [k for k, rr in groups[rb]]
            rrs = sorted(set(rr for k, rr in groups[rb]))
            print("  panel rb=%d: %d cells, k=%d..%d, r=%s  first got %s want %s"
                  % (rb, len(groups[rb]), min(ks), max(ks), rrs,
                     " ".join("%.6g" % fl(Pw[(rb * N + min(ks)) * 8 + rr]) for rr in rrs),
                     " ".join("%.6g" % pref[(rb * N + min(ks)) * 8 + rr] for rr in rrs)))

    cbad = [i for i in range(nn) if Cw[i] != 0]
    print("C: %d/%d nonzero words (harness zeroed C before the launch)" % (len(cbad), nn))
    if cbad:
        print("  first nonzero at C[%d] = %.9g" % (cbad[0], fl(Cw[cbad[0]])))
        r0, c0 = cbad[0] // N, cbad[0] % N
        # float64 reference tile
        def ref(i, j):
            s = 0.0
            for k in range(N):
                s += A[i * N + k] * B[k * N + j]
            return s
        print("  ref tile row %d cols %d..%d: %s" % (r0, c0, c0 + 7,
              " ".join("%.6g" % ref(r0, c) for c in range(c0, c0 + 8))))
        print("  got tile row %d cols %d..%d: %s" % (r0, c0, c0 + 7,
              " ".join("%.6g" % fl(Cw[r0 * N + c]) for c in range(c0, c0 + 8))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
