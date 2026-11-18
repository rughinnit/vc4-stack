# asm5 results — no verified result; root causes narrowed, board blocked

Work tree `/home/raspi/work/asm5` (copy of `asm4`; `asm4` untouched).
Board: Pi 3B, stock clocks (V3D 300 MHz verified, `get_throttled=0x0`).
Every GPU run: `flock -w 2400 /tmp/vc4-gpu.lock` around the whole sequence (no
nested flock), `timeout`, `read_v3d_status` -> `STATE: LIVE`.

## Verdict: REJECT (nothing to accept yet)

No speed number is reported and the acceptance protocol was not addressed.  The
8x64 GEMM still does not produce output, and the session ended with the board
unable to run GPU work at all (its `libcrypto.so.3` page-cache copy is poisoned,
so `sudo` is dead).  Details and the exact recovery step are in `NOTES5.md` §5.

## What this run established

1. **The task's leading suspicion is disproved.**  `or vpw_addr, ra21, ra21`
   *does* set the VPM-DMA destination.  A dedicated store-only probe
   (`probe_store.py`, 277 words, no k loop) reads the same seven uniforms in the
   same order as `gemm4.py`, computes the same `4*qpu` / `4*qpu*128` values,
   runs the same 8-chunk store block with the *same* VPM row base reused for
   every chunk, and every one of the 512 words lands at the correct address
   with the correct value.  No delay, no nop and no `ldi` is needed after the
   address write; `u4` (the panel base) is not involved.
2. **The store block's encoding is not the problem.**  A field-level diff of
   `gemm4_128.bin`'s store block against the proven `rows4.bin` shows the same
   instruction shapes with the expected register numbers (`add vpw_setup, r5,
   ra27` / `or vpm, raX|rbX, ...` / `ldi vpw_setup, 0xc00101c0` / `or vpw_addr,
   ra21, ra21` / `or -, vpw_wait, vpw_wait`), and the RAM pitch is `N*4` with
   64 B written per row (the stride word is the *gap*), as `NOTES4.md` §3.1 says.
3. **New, clean measurement of the failure** (1 QPU, fresh boot,
   `QRUN_CPUPANEL8=1`): `exec_cycles = 994308` = 32 tiles, i.e. the store block
   executes its full count 32x, yet **C[0..16384] is pristine zero and the A
   panel is bit-exact (0/16384)**.  So at 1 QPU the store is *not* mis-addressed
   (asm4's "it lands on the panel" came from a 12-QPU run whose packer was never
   verified above 4 QPUs, and from a 12-QPU run taken while the board was
   already degrading).  Either the accumulators are zero or the VPM data writes
   are dropped.
4. **A concrete bug in the k-loop schedule, found by inspection of the encoded
   words (independent of the store).**  The broadcast is one row ahead of the
   MACs.  `rot_for(h,r) = (16-(r+8h))&15` lands row r in lane 0 (the proven
   convention, cross-checked against asm3's `mul_rot(1,0,14-m)`), but the loop
   rotates `rot_for(*nxt)` at `w0` and reps at `w1` of the *same* row:

       w0: v8min r1, r0, imm(48+rot_for(row r+1))
       w1: fmul j1 (reads the OLD r5 = row r, correct) | rep r1 -> r5 = row r+1
       w2/w4/w5: fmul j0/j3/j2 all read r5 = row r+1     <-- wrong row

   So 3 of the 4 MACs of every row multiply the *next* row's A value, and the
   prologue never reps row 0 at all, so the very first `fmul j1` of the whole
   loop uses whatever `r5` happened to hold.  Encoded proof: the loop's rot
   immediates are 63,62,...,48 (rows 1..7 then row 0 of the other half) and each
   rep sits one word after its rot.  This alone makes the result wrong; it is
   almost certainly why the "computes correctly internally / 1.32x" model in
   `NOTES4.md` was never confirmed by a measurement.
   Fix (needs one add slot re-arranged): rotate the *current* row at `w0`
   (`rot_for(h,r)`) and move `rep` into `w0`'s add slot so that all four fmuls
   of the row (`w1,w2,w4,w5`) read `r5` = row r; the A-file TMU copy that
   currently occupies `w0`'s add slot moves to `w6`'s free mul slot.  The
   prologue must also rep row 0 before the loop.
5. Still open: why nothing is written at all (item 3).  Two candidates, and a
   5-minute discriminating experiment for each (see `NOTES5.md` §3).

## Tooling left behind (all in asm5)

`probe_store.py` (store-only probe with `NCH`/`VROW`/`GAP`/`DPRD`/`BASESRC`/
`DSTEP` knobs), `probe_run.py` (launch + arena dump, no internal lock),
`diag1.py` (GEMM run + C/panel diff against float64 references),
`dumpw.py`, and a `SKIP=prime|loop|all` knob in `gemm4.py` for bisecting the
kernel body.

Harness/kernel changes made and compile-checked (**not run**): the harness now
passes `u7 = gpu_bus + out_off` (C base) and `u8 = gpu_bus`, and `gemm4.py`'s
`DBGSTORE` paths use `u7` instead of the hard-coded `0xde610000` that wrote into
unrelated DRAM (this is what broke the board, `NOTES5.md` §4).

## Numbers

| item | value |
|---|---|
| 8x64 GEMM, N=128, 1 QPU | ms / GFLOP/s **not measurable** (C all zero) |
| mismatches, best run | 16384/16384 (all zero) |
| clean runs | 0 |
| cycles/k (measured, 1 QPU) | 994308/(32 tiles x 64 k) = **485** cycles per k per QPU |

The last line is the only hardware number from this run: 485 cycles/k
against the modelled issue floor of 236 (118 words x 4 cycles per k pair / 2),
i.e. the loop is running at ~2x the modelled cycle count — consistent with the
TSTR/tile and store corrections still being unvalidated, and with the loop
being 118 words but not yet the limiting factor (the store adds ~190 cycles per
tile, i.e. ~3 per k, which does not explain 485).

## Keep / reject

**REJECT** the kernel as delivered.  Keep: the 8-row panel (verified), the
store-block encoding (now proven equivalent to `rows4`/asm3), `probe_store.py` /
`diag1.py`, and the `u7`/`u8` harness uniforms.  Reject: the k-loop broadcast
schedule (bug 4 above) and the store block's *context* until the VPM-write ->
DMA ordering question in item 5 is answered on silicon.
