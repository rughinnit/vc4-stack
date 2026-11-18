#!/usr/bin/env python3
"""Decode a DUMPREG arena dump (C file from g6.py) into the 32 dumped registers.

  python3 dumpreg6.py <c.bin> <a.bin> <b.bin> <N>

Layout: chunk ci = 2*j + c -> tile rows 4c..4c+3, cols 16j..16j+15.
Tile t (of N/8) covers rows 8t..8t+7, so tile 0 shows the first tile the QPU
stored and the last tile shows the last.  We print tile 0 and the last tile.
Also prints the k=0,1 references for the accumulator entries.
"""
import struct
import sys

NAMES = ["r0 A-vec", "r1 rot", "r5 bcast", "rb16 B k0j0",
         "rb17 B k1j0", "rb19 B k1j1", "ra17 B k1j2", "ra19 B k1j3",
         "r4 TMUpop", "r2 multmp", "ra28 BADR", "rb18 B k0j1",
         "ra20 AADR", "ra21 CADR", "ra22 CADR2", "ra23 KCNT",
         "ra24 T", "ra27 VPMQ", "rb20 BST", "rb21 NEG1",
         "ra0 C00", "ra2 C10", "ra4 C20", "ra6 C30",
         "rb0 C02", "rb2 C12", "rb4 C22", "rb6 C32",
         "ra1 C01", "ra3 C11", "rb1 C03", "rb3 C13"]


def rd(p, n):
    raw = open(p, "rb").read()
    return list(struct.unpack("<%dI" % n, raw[:4 * n]))


def fl(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def main():
    cf, af, bf, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nn = N * N
    C = rd(cf, nn)
    A = [fl(x) for x in rd(af, nn)]
    B = [fl(x) for x in rd(bf, nn)]
    ntile = N // 8
    for t in (0, ntile - 1):
        print("=== tile %d (C rows %d..%d) ===" % (t, 8 * t, 8 * t + 7))
        base = 8 * t * N
        for k in range(32):
            j, c = k // 2, k % 2
            row = 4 * c
            off = base + row * N + 16 * j
            vals = [fl(C[off + l]) for l in range(16)]
            print("%-12s row%d col%2d: %s" % (NAMES[k], row, 16 * j,
                                              " ".join("%.5g" % v for v in vals[:8])))
    # reference for the C accumulators, one k pair (k=0,1)
    print("=== reference C[r][c] = A[r][0]*B[0][c] + A[r][1]*B[1][c] ===")
    for (ri, name) in ((0, "ra0/ra1 (row0 j0/j1)"), (1, "ra2/ra3 (row1)"),
                       (2, "rb0/rb1 (row0 j2/j3)")):
        acc_row = ri if ri < 2 else 0
        if ri < 2:
            accs = [fl(C[acc_row * N + col]) for col in (0, 1)]
            print("%-22s got %s want %s" % (name, ["%.6g" % v for v in accs],
                  ["%.6g" % (A[acc_row * N] * B[col] + A[acc_row * N + 1] * B[N + col])
                   for col in (0, 1)]))
        else:
            accs = [fl(C[acc_row * N + 32 + col]) for col in (0, 1)]
            print("%-22s got %s want %s" % (name, ["%.6g" % v for v in accs],
                  ["%.6g" % (A[acc_row * N] * B[32 + col] + A[acc_row * N + 1] * B[N + 32 + col])
                   for col in (0, 1)]))
    # A vector: lane r should be A[r][k], lane 8+r = A[r][k+1] of row block 0
    av = [fl(C[l]) for l in range(16)]
    print("A vector (tile0 chunk0 row0): lanes %s" % " ".join("%.5g" % v for v in av))
    print("  want k=0: %s" % " ".join("%.5g" % A[r * N] for r in range(8)))
    print("  want k=1: %s" % " ".join("%.5g" % A[r * N + 1] for r in range(8)))
    bv = [fl(C[3 * N + l]) for l in range(16)]
    print("B k0 j0 (tile0 chunk0 row3): %s" % " ".join("%.5g" % v for v in bv[:8]))
    print("  want B[0][0..7]         : %s" % " ".join("%.5g" % B[l] for l in range(8)))


if __name__ == "__main__":
    main()
