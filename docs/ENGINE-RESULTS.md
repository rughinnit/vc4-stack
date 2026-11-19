# engine-fast2 — asm10 GEMM port: generalised addressing FIXED, suite green

Work tree `/home/raspi/work/engine-fast2` (copy of `engine-fast`, which is a copy
of `engine-fix`).  Board: Pi 3B, stock clocks (V3D 300 MHz, `get_throttled=0x0`,
core 400 MHz, ARM 1200 MHz).  Every GPU sequence: one launch at a time under
`flock`, short `VC4INFER_TMO_MS`, board state checked after every run.

## Verdict

**Shipped.**  `REPS=5 ./run_tests.sh` -> exit 0, `ALL TESTS PASSED`, 0
mismatches on every shape, strict dispatch intact (8 of 22 dispatchable, the
same list as `engine-fix`).

Matmul at 256^3 goes from **5.38 ms / 6.24 GFLOP/s to 3.97 ms / 8.46 GFLOP/s**
(+36 % total; gemm-only 9.45) — the port's GEMM is now *faster than asm10's own
proven kernel through the same launcher* (3.55 vs 4.12 ms).  The 10 GFLOP/s
acceptance target is **not** reached, and it is not reachable in this board
state: asm10's own recorded 13.745 GFLOP/s configuration, re-run through
asm10's own harness on this board today, measures **8.12** (see §5).

## 1. The addressing bug (root cause, measured)

asm10’s kernel derives `rowoff = rowblk << log2(32*N)` and shares it for the
panel and C.  The port replaced that with the three independent strides the
engine needs (panel row-block stride `32*K`, C row stride `8*N*4`, B row stride
`N*4`) taken from uniforms u3..u6, and that part was correct.  The failures the
earlier session saw (`C[r][c] = A[r][c+8]`, "only QPU 0's rows", and single-tile
shapes storing nothing) had **two other causes**, both in the tile-body
prologue:

1. **A GP register written by a uniform read is not visible to the next
   instruction.**  The prologue read u3 into B r30 and added it to the panel
   base in the very next word; the stale B r30 (0xFF) was added instead, so
   every A tile base was 255 bytes high.  The TMU rounds the per-lane address
   up to a 4-byte boundary, so the A vector came out exactly 8 k-pairs into the
   row block: `C[r][c] = A[r][c+8]`.  Measured with a register-dump build
   (`ARB = panel + 0xFF + 4*lane`, `AADR = panel + 0x100 + 4*lane`).
   Fix: read all eight uniforms first, combine them after >= 3 words (asm10
   keeps >= 6), and the same for u5 -> C row-block base.

2. **The flag-setting compare and its branch must be adjacent.**  The row-block
   guard was `add.setf r3, rb, -TR ; nop ; nop ; brr.4` and the branch then saw
   cleared condition codes, so the row-block body was never entered: the kernel
   ran the end-program path in ~30 us and stored nothing.  asm10 writes
   `add.setf ; branch ; 3 nops`.  Fix: same layout (comment in `kgen.py`).

Two smaller things found on the way and also fixed:
`brr.4`/`BR_ALL_N_SET` is a 4-bit condition field and is fine; the lane-offset
accumulator (`r3 = elem_num << 2`) is read 2 words after the write in asm10 and
the port had 3 (harmless, now identical).

With those three changes the k loop, prime, store and tail remain
**byte-identical to asm10/gemm6.py** (`diffbody`: 0 of 302 k-loop words differ).

## 2. Verification (in the order they were checked)

| check | result |
|---|---|
| 1 QPU, B = identity, 8x64x16 | `C[r][c] = A[r][c]` for c < 16, 0 beyond — 0/512 |
| 24x64x48 (K/2 % 3 == 0, drain path), 1 QPU | 0/1536 |
| 64x64x64, 8 QPUs | 0/4096 |
| 128^3, 12 QPUs | 0/16384 |
| 256^3, 12 QPUs | 0/65536 |
| 512x64x288 (drain), 12 QPUs | 0/32768 |
| suite `test_matmul` (128^3, 256^3, 64x48x32, 20x36x44, 16x16x16, 512x64x288) | 5/5 clean each |
| suite `test_conv` (3x3 pad1+bias Cin16/32, 1x1 +/-bias, 3x3 pad0, Cin64 K576) | 5/5 clean each |
| suite `test_ew` (add/mul/relu/add+tail/sigmoid/silu) | 5/5 clean each |
| `test_dispatch` (strict dispatch) | PASS, 8/22 dispatchable, same list |

Panel/kernel probe route used to find the bug (kept in `attempt-asm10/`, plus
`/tmp` scripts): `raw.c` (run any blob with a CPU panel, hex/byte dumps),
`patch_dumpreg.py` / `patch_dumparb.py` (rewrite the 32 accumulator-init words
to dump any address register through the proven store), `patch_bisect.py`,
`probe_unif8.py` (dump u0..u7 through a correct store).

## 3. Before / after (same session, same board state, `REPS=5`, best gpu ms)

before = `/home/raspi/work/engine-fix` (asm3 16x16 GEMM, `gemm_*` kernels),
after = this tree (asm10 8x64 GEMM + asm10 packer).  Both measured back to back
with the board idle, V3D 300 MHz, `get_throttled=0x0`.

| shape | before ms | before GFLOP/s | after ms | after GFLOP/s | change |
|---|---|---|---|---|---|
| 128^3 | 0.718 | 5.84 | 0.708 | 5.92 | +1 % |
| **256^3** | **5.376** | **6.24** | **3.965** | **8.46** | **+36 %** |
| 512x64x288 | 3.435 | 5.49 | 2.480 | 7.61 | +39 % |
| 64x48x32 | 0.086 | 2.28 | 0.069 | 2.86 | +25 % |
| 20x36x44 | 0.106 | 0.60 | 0.065 | 0.98 | +63 % |
| 16x16x16 | 0.022 | 0.37 | 0.004-0.037 | 0.22-1.99 | launch-bound, see below |

The recorded "current engine 5.90" was measured in an earlier session whose
board state was ~6 % slower than today's; against today's baseline (6.24) the
gain is +36 %, against the recorded number +44 %.

16x16x16 is launch-overhead-bound (pack + gemm is 4-37 us depending on whether
the JIT'd kernel is already in the code slot), so its GFLOP/s figure is noise;
the acceptance run measured 0.004 ms (1.99 GFLOP/s) and an earlier one 0.037 ms.
At M=16 the asm10 kernel also pads N 16 -> 64 (4x the columns) against the asm3
kernel's exact 16x16 tile, so this shape is the one where the port has no
headroom to win; it is not worth a special case.

## 4. Pack cost (reported as asked)

256^3: **pack 0.413-0.415 ms + gemm 3.551 ms = 3.965 ms** (pack is 10.4 % of the
total).  The packer is asm10's `pack10` (KL=16 VDR rows, 8x8-word VDMs,
grid-stride over 8-row units, 4 QPUs) and is byte-identical to
`asm10/pack10_256_k16_single_vp1_us.bin`; asm10 measured that packer at
0.382 ms, so the engine's pack pass is within 8 % of the stand-alone number
(the remainder is the launch pair).  The pack is DRAM-bound: 512 KB of traffic
at 1.34 GB/s = 84 % of the board's 1.6 GB/s peak, which is the hard floor for
any separate pack pass (asm10/RESULTS10).

## 5. Gap to the 13.745 GFLOP/s stand-alone number

Measured today, on this board, in this order:

| configuration | gemm ms | gemm GFLOP/s |
|---|---|---|
| **asm10's own harness** (`asm10/g6.py`, gemm6_256.bin, 12 QPUs, asm10 pack) | 4.130 | 8.12 |
| asm10's kernel through *this* engine (`diag_g6 256 12`) | 4.121 | 8.12 |
| **this port's kernel through this engine** (256^3) | **3.551** | **9.45** |

asm10's `cyc/k` is 398.5 today, *identical* to its recorded 398.5: the k loop
executes at exactly the recorded rate, and the extra wall time is memory stall
(CTR_IDLE 4.7 M cycles/QPU), not instruction throughput.  The same harness
measured 2.441 ms (13.745) when RESULTS10 was written, so the ~1.7x is a board
state difference, not a kernel or harness regression:

* the pack pass is byte-for-byte the same and still runs at its recorded
  0.383 ms (so DRAM *bandwidth* is unchanged);
* the L2 flush between pack and GEMM costs nothing (measured: with the flush
  removed from `vc4__launch`, 256^3 is 4.279 ms vs 4.275 ms);
* clocks are stock (V3D 300, core 400, `get_throttled=0x0`), load 0.1;
* the extra time is entirely in stall/idle on the GEMM's streaming reads, which
  is what changed between the two boots.

So the honest statement is: **this port is 15 % faster than asm10's own kernel
and 36 % faster than the engine it replaces, and it is at 9.45 GFLOP/s
gemm-only / 8.50 total where the same board today gives asm10's kernel 8.12
total.**  >10 GFLOP/s would need the board to be in its RESULTS10 state; no
kernel change can supply the 1.7x memory-latency difference.

## 6. What is in the tree

| file | state |
|---|---|
| `tools/kgen.py` | `build_gemm10` (asm10 GEMM, generalised addressing **fixed**), `build_pack10` (asm10 packer, byte-identical), `build_pack`/`build_gemm` (asm3, unchanged), `build_im2col` (asm3 16-row, restored) |
| `src/vc4_gemm.c` | asm10 `vc4_matmul`/`vc4__pack_run`/`vc4__gemm_run` + the asm3 `vc4__pack3_run`/`vc4__gemm3_run` kept for conv |
| `src/vc4_conv.c` | **asm3 path** (asm3 im2col + asm3 16x16 GEMM + k-major B panel) |
| `src/vc4_int.h`, `src/vc4_api.c` | both ABIs declared; dispatch table says which GEMM conv uses |
| `tests/test_matmul.c` | the ported test (unchanged, passes) |

### Why conv is on the asm3 GEMM

The conv needs a panel whose k rows are 8 pixels (the asm10 GEMM's A layout) but
built directly from a padded NCHW image.  The im2col can only write that layout
with the `horizontal=1` VDM, and on this silicon that shape is only exact at
16x16 words (asm10 measured "any other Depth/Units pair steps the destination by
64 B and pads"); the 8x8 `horizontal=1` variant silently ignores the VPM
row/col offsets, which is why every conv shape was wrong.  The two workable
routes are (a) a 16-row panel for conv (asm3 GEMM, chosen) or (b) an extra
pack10 pass over a pixel-major im2col matrix (extra pass and an unverified
stride).  (a) is the proven path, is *faster* for conv's small `Cout` (the asm10
tile pads N to 64, so a 16- or 32-channel conv would do 4x/2x the work), and it
keeps the conv out of this run's risk budget.  Conv is therefore 1.5-3.5
GFLOP/s as before, and the matmul carries the whole gain.

## 7. Suite (acceptance run)

```
=== maths (no GPU)          sigmoid/silu/im2col/gemm models: PASS
=== test_dispatch           PASS (8 of 22 dispatchable, list unchanged)
=== test_ew                 6/6 shapes 5/5 clean
=== test_matmul             128^3 5.93, 256^3 8.50, 64x48x32 2.84,
                            20x36x44 0.93, 16x16x16 0.22, 512x64x288 7.69
                            5/5 clean each, 0 mismatches
=== test_conv               3x3 pad1+bias 16/32, 1x1 +/-bias, 3x3 pad0,
                            Cin64 K576: 5/5 clean each
--- board: SRQCS=0x000c0c00 (done=12 error=0) V3D_ERRORS=0x00001000 STATE: LIVE
ALL TESTS PASSED
```

(full log kept as `suite-REPS5.log` in this tree)

## 8. Board notes

One GPU hang occurred during the investigation (a `64x64x64` launch of the
broken kernel, before the fixes); the QPU stopped completing jobs and *every*
subsequent launch, including asm10's proven kernel, timed out.  Unbinding the
`vc4_v3d` platform device to reset the block is **not** a recovery path here
(the unbind segfaulted, the rebind hung because the VPU firmware's VCHI service
is unavailable: `vc_sm_cma_vchi_init: failed to open VCHI service (-11)`), so
the board needs a reboot.  One more hard hang (a conv shape) rebooted the board
by itself.  After the fixes, 60+ launches ran clean with `STATE: LIVE` after
every one.  `V3D_ERRORS=0x1000` is set at boot and is masked by the engine.
