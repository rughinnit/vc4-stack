#!/usr/bin/env python3
"""VPM control-word encoder/decoder, mirroring VC4C src/periphery/VPM.h bitfields
(which in turn mirror the Broadcom spec tables 32-37).

Every field is range-checked: these words are written straight into hardware
registers, and an out-of-range field silently overflows into its neighbour
(VPitch/NumberRows/RowLength are 4 bits each, so "16" must be encoded as 0).
"""


def _chk(name, v, lo, hi):
    if not (lo <= v <= hi):
        raise ValueError("%s=%d out of range %d..%d" % (name, v, lo, hi))
    return v


# ---- Table 36: VPM DMA load (memory -> VPM) basic setup ----------------------
def vdr_basic(row, vpitch=1, numrows=16, rowlen=16, vertical=0, mpitch=0, mode=0):
    """row: VPM row 0..63, vpitch: VPM rows to advance per loaded row,
    numrows/rowlen: 16 encoded as 0."""
    w = 1 << 31                                    # ID = 1
    w |= _chk('mode', mode, 0, 7) << 28
    w |= _chk('mpitch', mpitch, 0, 15) << 24
    w |= (_chk('rowlen', rowlen, 0, 16) % 16) << 20
    w |= (_chk('numrows', numrows, 0, 16) % 16) << 16
    w |= _chk('vpitch', vpitch, 0, 15) << 12
    w |= _chk('vertical', vertical, 0, 1) << 11
    w |= _chk('row', row, 0, 127) << 4              # {Y[6:0], X[3:0]}
    return w


def vdr_stride(pitch):
    return (9 << 28) | _chk('pitch', pitch, 0, 8191)


def dec_vdr(w):
    return dict(id=(w >> 31) & 1, mode=(w >> 28) & 7, mpitch=(w >> 24) & 15,
                rowlen=(w >> 20) & 15, numrows=(w >> 16) & 15,
                vpitch=(w >> 12) & 15, vertical=(w >> 11) & 1,
                row=(w >> 4) & 0x7F, col=w & 15)


def dec_vdr_stride(w):
    return dict(id=(w >> 28) & 15, pitch=w & 8191)


# ---- Table 32: generic VPM write (QPU -> VPM) -------------------------------
def vpw_generic(row, stride=1, size=2, horizontal=1, laned=0):
    w = 0
    w |= _chk('stride', stride, 0, 63) << 12
    w |= _chk('horizontal', horizontal, 0, 1) << 11
    w |= _chk('laned', laned, 0, 1) << 10
    w |= _chk('size', size, 0, 3) << 8
    w |= _chk('row', row, 0, 63)
    return w


# ---- Table 34: VPM DMA store (VPM -> memory) basic setup --------------------
def vdw_basic(row, depth, units, horizontal=1, col=0, mode=0):
    w = (2 << 30)                                   # ID = 2
    w |= _chk('units', units, 0, 127) << 23
    w |= _chk('depth', depth, 0, 127) << 16
    w |= _chk('horizontal', horizontal, 0, 1) << 14
    w |= _chk('row', row, 0, 63) << 7
    w |= _chk('col', col, 0, 15) << 3
    w |= _chk('mode', mode, 0, 7)
    return w


def vdw_stride(stride=0, blockmode=1):
    return (3 << 30) | (blockmode << 16) | _chk('stride', stride, 0, 0xFFFF)


def dec_vdw(w):
    return dict(id=(w >> 30) & 3, units=(w >> 23) & 0x7F, depth=(w >> 16) & 0x7F,
                horizontal=(w >> 14) & 1, row=(w >> 7) & 0x3F,
                col=(w >> 3) & 15, mode=w & 7)


def dec_vdw_stride(w):
    return dict(id=(w >> 30) & 3, blockmode=(w >> 16) & 1, stride=w & 0xFFFF)


if __name__ == '__main__':
    # cross-check the two constants that are already proven on silicon
    assert vdw_basic(0, 16, 4) == 0x82104000, hex(vdw_basic(0, 16, 4))
    assert vdw_basic(52, 4, 1) == 0x80845A00, hex(vdw_basic(52, 4, 1))
    assert vdw_stride() == 0xC0010000, hex(vdw_stride())
    assert vpw_generic(52, 1) == 0x1A34, hex(vpw_generic(52, 1))
    assert vpw_generic(0, 1) == 0x1A00, hex(vpw_generic(0, 1))
    print("proven constants reproduce")
    print("VDR basic row0 vp1 : 0x%08x %s" % (vdr_basic(0), dec_vdr(vdr_basic(0))))
    print("VDR basic row16 vp1: 0x%08x %s" % (vdr_basic(16), dec_vdr(vdr_basic(16))))
    print("VDR stride N=128   : 0x%08x %s" % (vdr_stride(512), dec_vdr_stride(vdr_stride(512))))
    print("VDW vert row0      : 0x%08x %s" % (vdw_basic(0, 16, 16, 0), dec_vdw(vdw_basic(0, 16, 16, 0))))
    print("VDW vert row16     : 0x%08x %s" % (vdw_basic(16, 16, 16, 0), dec_vdw(vdw_basic(16, 16, 16, 0))))
