#!/usr/bin/env python3
"""Audit an encoded gemm kernel for short read-after-write gaps.

For every instruction, list the A-file / B-file registers it writes and the
ones it reads; report any read at distance 1 (i.e. the next instruction) or
distance 0 of the same register in the same file.  Accumulators are reported
separately (they are known to be readable at distance 1).
"""
import struct
import sys

p = sys.argv[1]
raw = open(p, 'rb').read()
w = list(struct.unpack('<%dQ' % (len(raw) // 8), raw))


def fields(x):
    sig = (x >> 60) & 0xF
    if sig in (3, 15):        # end, branch
        return None
    ws = (x >> 44) & 1
    ao = (x >> 38) & 0x3F
    mo = (x >> 32) & 0x3F
    aop = (x >> 24) & 0x1F
    mop = (x >> 29) & 7
    ra = (x >> 18) & 0x3F
    rb = (x >> 12) & 0x3F
    aa, ab, ma, mb = (x >> 9) & 7, (x >> 6) & 7, (x >> 3) & 7, x & 7
    writes = []
    reads = []
    if sig == 14:             # ldi
        writes.append(((0 if ws == 0 else 1), ao))
        return writes, reads
    for op, out, port in ((aop, ao, 'a'), (mop, mo, 'm')):
        if not op:
            continue
        if 32 <= out <= 37:
            writes.append(('acc', out - 32))
        elif out < 32:
            writes.append((((0 if ws == 0 else 1) if port == 'a'
                            else (1 if ws == 0 else 0)), out))
    if sig != 13:                      # ALU immediate overrides raddr_b
        for mux, addr in ((aa, ra), (ab, rb)):
            if mux == 6 and addr < 32:
                reads.append((0, addr))
            elif mux == 7 and addr < 32:
                reads.append((1, addr))
    else:
        if aa == 6 and ra < 32:
            reads.append((0, ra))
    for mux in (ma, mb):
        pass
    return writes, reads


info = [fields(x) or ([], []) for x in w]
print('words=%d' % len(w))
bad = 0
for i in range(1, len(w)):
    for (f, r) in info[i][1]:
        if f == 'acc':
            continue
        for (f2, r2) in info[i - 1][0]:
            if f2 == f and r2 == r:
                print('  SHORT GAP word %d reads %s%d written by word %d'
                      % (i, 'ra' if f == 0 else 'rb', r, i - 1))
                bad += 1
print('short A/B-file read-after-write gaps: %d' % bad)
