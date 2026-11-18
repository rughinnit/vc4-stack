#!/usr/bin/env python3
"""Probe 3: which vertical-VDW field is the vector width?

VPM rows 0..15 = ramp  row*100 + word.  Three vertical stores, 512 B apart:
  block 0: Depth=8  Units=8
  block 1: Depth=16 Units=8
  block 2: Depth=8  Units=4
run_probe_vdw8c.py prints each 512 B block as floats.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *          # noqa: F401,F403
from qenc import _ops
from gen import Prog
import vpmwords as V

VARIANTS = [(8, 8), (16, 8), (8, 4)]


def build():
    p = Prog()

    def e(*ws):
        for w in ws:
            p.emit(_ops(**w) if isinstance(w, dict) else w)

    e(_ops(add_op=ADD_OR, add_out=16, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))
    e(_ops(add_op=ADD_OR, add_out=17, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))   # out base
    e(_ops(add_op=ADD_OR, add_out=18, add_a=areg(R_ELEM_NUM), add_b=areg(R_ELEM_NUM)))
    e(ldi(0x42C80000, 19, ws=1))                         # 100.0f in rb19
    e(nop(), nop())
    e(_ops(add_op=ADD_ITOF, add_out=32, add_a=areg(18), add_b=areg(18)))
    e(nop(), nop())
    e(ldi(0x1A00, R5, ws=1))
    e(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
    for r in range(16):
        if r:
            e(_ops(add_op=ADD_FADD, add_out=32, add_a=acc(0), add_b=(REGB, 19)))
            e(nop(), nop())
        e(_ops(add_op=ADD_OR, add_out=W_VPM, add_a=acc(0), add_b=acc(0)))
    e(ldi(512, 22, ws=1))
    e(_ops(add_op=ADD_OR, add_out=20, add_a=areg(17), add_b=areg(17)))
    for i, (dep, uni) in enumerate(VARIANTS):
        if i:
            e(_ops(add_op=ADD_ADD, add_out=20, add_a=areg(20), add_b=(REGB, 22)))
            e(nop(), nop())
        e(ldi(V.vdw_basic(0, dep, uni, horizontal=0), R5, ws=1))
        e(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
        e(ldi(V.vdw_stride(0), W_VPW_SETUP, ws=1))
        e(nop(), nop())
        e(_ops(add_op=ADD_OR, add_out=W_VPW_ADDR, ws=1, add_a=areg(20), add_b=areg(20)))
        e(_ops(add_op=ADD_OR, add_out=W_NOP, add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
        e(nop(), nop(), nop(), nop())
    e(end_program())
    e(nop(), nop(), nop())
    return p


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    print('probe_vdw8c.bin', len(build().write(os.path.join(ASM, 'probe_vdw8c.bin'))), 'instructions')
