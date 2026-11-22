# Where YOLOv8n-160's QPU time goes, and what the GEMM's 13.7 GFLOP/s needs

This page records a kernel-efficiency pass over YOLOv8n at 160x160: the per-node
class split of QPU time, the split of a conv into its im2col panel and its
GEMM, the same split for a matmul, and the per-shape autotuner's silicon
numbers.  It also records the one measurement that decided several of the
questions below: the board's memory path runs in two states with identical
clock registers, and the GEMM moves 2.5x faster in one of them.

Everything here is kernel time.  A node's wall time in the engine profile
includes the socket round trip, the ARM-side copies and the Python dispatch;
its `gpu_ms` is the daemon's own kick-to-queue-idle time and is what the
tables use.

## 0. Conditions

| item | value |
|---|---|
| board | Raspberry Pi 3B, BCM2837, 12 QPUs, kernel 6.18 |
| clocks | core 400 MHz, V3D 300 MHz, `throttled=0x0` on the primary rows |
| ARM | capped at 1.0 GHz for this pass (`scaling_max_freq`), so wall numbers are ~20% above the 1.2 GHz record |
| backend | `vc4d`, one client, one private daemon socket, `--gpu-lock` |
| thermal | 45.6-49.4 C start, peak below 55 C; `tools/thermal-guard.sh --max 75` |
| model | `infer/packaging/vc4infer/data/models/yolov8n_160.onnx`, fixed 1x3x160x160 |
| passes | none (the engine default; `docs/OPTIMIZER.md` pass 8 is off by default) |

Tools: `infer/tools/onnx/profile_classes.py` for the class table and the conv
split, `infer/tests/bench_split` for the pack/GEMM split, `infer/build/bench_autotune`
for the tuner.

The daemon is a private one on its own socket so that the client library and
the daemon are built from the same tree; it takes the board GPU lock for every
launch.  The lock only serialises daemons that take it: a concurrent daemon
started without `--gpu-lock` is enough to lose a launch, and two timeouts and
two context resets were observed that way during this pass, none of them in a
tabulated run.  A full arena is also required (`--arena-mb 64` is enough for
this model) because a degraded 48 MiB arena fails a conv's scratch allocation.

## 1. QPU time by node class

YOLOv8n-160, 263 graph nodes, 832 launches, medians of three runs:

| class | QPU ms | share | launches |
|---|---:|---:|---:|
| Conv | 208.42 | 77.2% | 131 |
| Mul | 34.41 | 12.7% | 468 |
| Softmax | 6.89 | 2.6% | 40 |
| Concat | 6.34 | 2.3% | 60 |
| Reshape | 4.04 | 1.5% | 41 |
| Add | 2.79 | 1.0% | 38 |
| MaxPool | 2.36 | 0.9% | 9 |
| Split | 1.78 | 0.7% | 16 |
| Sigmoid | 1.02 | 0.4% | 14 |
| Resize | 0.71 | 0.3% | 2 |
| Slice / Transpose / Sub / Div | 1.19 | 0.4% | 13 |
| **total** | **269.96** | **100%** | **832** |

Wall time for the same runs is 643 ms, so QPU duty is 42%.  The rest of the
wall is the launch path, the client-side dispatch and the ARM-side stores and
pads; [`CPU-BREAKDOWN.md`](CPU-BREAKDOWN.md) accounts for it and it is not
counted here.

Two rows matter for kernel work:

* conv is 77% of QPU time: 64 conv calls, 131 launches.  Each call is an
  im2col panel launch and a GEMM launch, and three calls carry a third
  (`docs/OPTIMIZER.md` counts the same 64 rows and 128 of the launches);
* `Mul` is 468 launches for 34 ms, 0.073 ms each.  These are the graph-level
  SiLU multiplies and the C2f branch multiplies, run as one group per QPU per
  launch because a QPU that runs a second group in the same program hangs with
  the long sigmoid body (`src/vc4_ew.c`).  The epilogue pass moves the SiLU
  ones into the conv GEMM and takes the launch count to 365, but the QPU total
  does not move (641.9 ms without it, 636.6 with it, same boot): the same
  arithmetic on the same bytes, only fewer round trips.

### The board has two memory states

The same three runs on a boot 30 minutes earlier, same clock registers
(`core 400`, `V3D 300`), `throttled=0x50000` (an earlier under-voltage event,
nothing current), gave **641.9 ms** of QPU time, 2.4x the 270 ms above, with
the same launch count.  The conv class was 519.1 ms instead of 208.4 ms and a
256^3 GEMM ran at 3.63 GFLOP/s instead of 9.32.  Cycle counts and clocks were
unchanged.  The difference is the memory path: the 256^3 pack moves 512 KiB at 0.85 GB/s
on the slow boot and 1.17 GB/s on the fast one, and the GEMM's tile model
(9.3 MB of A, B and C traffic, section 3) gives 1.0 GB/s against 2.6 GB/s.
The GEMM is bound by those streaming reads.

Every number in this page therefore carries its boot; comparisons inside one
section are same-boot, and no absolute ms here should be compared with the
263 ms QPU baseline recorded in `OPTIMIZER.md` unless the GEMM rate matches
first.  `infer/tests/bench_split 256 256 256` is the two-minute check: 9-10
GFLOP/s is the fast state, 3.6 is the slow one.

## 2. Convolution: the panel is 23% of conv QPU

The NHWC path is im2col + GEMM (`src/vc4_nhwc.c`): a panel pass writes
`M x (K + 16)` floats in the GEMM's k-major layout, the bias folded as one
extra k block, and the asm3 GEMM (16x16 C tile, `bmode=1`) reads it.  Wrapping
`Runner._conv_nhwc` and reading `ctx->st.pack_ns` (the panel pass) and
`ctx->st.kernel_ns` (the GEMM) around each call splits it per layer:

| | ms | share of conv QPU |
|---|---:|---:|
| im2col panel (64 layers) | 45.75 | 22.7% |
| GEMM (64 layers) | 155.95 | 77.3% |
| conv total | 201.70 | 100% |

On the slow boot the same split was 54.6 / 449.4 ms (10.8% panel): the panel
is bandwidth-bound and moved less than the GEMM when the memory path
recovered, which is why its share rises on the fast state.

Per layer (`profile_classes.py --conv`, slow-boot state, ms; the ratio, not the
ms, is what transfers):

| layer | Cin | Cout | k | panel | GEMM |
|---|---:|---:|---:|---:|---:|
| /model.22/cv2.0/cv2.0.0 | 64 | 64 | 3 | 2.45 | 18.37 |
| /model.22/cv3.0/cv3.0.0 | 64 | 80 | 3 | 2.63 | 22.90 |
| /model.22/cv3.0/cv3.0.1 | 80 | 80 | 3 | 3.28 | 28.44 |
| /model.0 | 16 | 16 | 3 | 5.86 | 12.20 |
| /model.7 | 128 | 256 | 3 | 0.51 | 20.56 |
| /model.2/m.0/cv1 | 16 | 16 | 3 | 1.81 | 3.72 |
| /model.4/m.0/cv1 | 32 | 32 | 3 | 1.20 | 4.81 |
| /model.22/cv3.2/cv3.2.0 | 256 | 80 | 3 | 1.17 | 12.79 |

### A direct 3x3 kernel was not written, and here is the measurement

The panel is 23% of conv QPU, so a direct kernel could recover at most 46 ms
(17% of total QPU) and cannot recover more: the GEMM reads the panel once per
column block (N/16 times), and a direct path would read the padded image the
same number of times because the taps are k blocks.  What it would save is the
panel *write* plus its one image read, which is the 46 ms measured.

The naive direct A path costs more than it saves.  The GEMM's A load for one k
is a 64 B VDR: 16 lanes = 16 output pixels of one channel, which the panel
stores contiguously.  In the NHWC image those 16 pixels are `Cin*4` bytes
apart, so the same request becomes 16 separate 4 B accesses in 16 different
cache lines: 16x the transactions for 1/16 the useful bytes per line.  The
coalesced alternative is to do the panel's read-and-transpose inside the GEMM
(one 16x16 VDR plus a vertical VDW per k block), which is the panel pass's
work again, once per column block instead of once per layer.

So the panel is the right trade for `N >= 32`, and the remaining panel cost is
bandwidth at the same rate as everything else on this board.  The honest
description is that direct conv is worth at most 17% and needs a transposing A
path to avoid paying more than that; it is not in this pass.

### Where the conv arithmetic is actually wasted: 22% of M is padding

`Runner._conv_nhwc` pads each output row to 16 pixels (`op = align16(wo)`) and
the panel and GEMM run over `M = ho * op`.  At 160x160 that wastes:

| output width | op | M padding |
|---|---:|---:|
| 160 | 160 | 1.00 |
| 80 | 80 | 1.00 |
| 40 | 48 | 1.20 |
| 20 | 32 | 1.60 |
| 10 | 16 | 1.60 |
| 5 | 16 | 3.20 |

Summed over the 64 layers, `M` is 33,952 padded rows against 26,352 real
pixels, a factor of 1.29.  On the slow-boot conv that is 113 ms of the
637 ms total; on the fast boot it is roughly 45 ms.  The waste is a property
of the 16-pixel VDR block: the panel pass reads 16 pixels at one constant
pitch per tap, and a block that crosses a row boundary has no constant pitch.
Removing it needs either per-pixel VDRs (16 transactions per block, which the
panel pass cannot afford) or a per-block split at the row end, i.e. a second
im2col kernel with a variable-length segment inside a block.  That is the
largest single identified win and the next kernel to write.

## 3. GEMM: pack against kernel, at 128/256/512

`infer/tests/bench_split`, medians of five, same process and arena:

| shape | pack ms | GEMM ms | pack share | GEMM GFLOP/s | engine total |
|---|---:|---:|---:|---:|---:|
| 128^3 (slow boot) | 0.153 | 1.425 | 9.7% | 2.94 | 2.67 |
| 256^3 | 0.439 | 3.601 | 10.9% | **9.32** | 8.30 |
| 512^3 | 1.679 | 26.560 | 5.9% | **10.11** | 9.51 |
| 256^3 (slow boot) | 0.605 | 9.247 | 6.1% | 3.63 | 3.41 |
| 512^3 (slow boot) | 2.247 | 72.652 | 3.0% | 3.69 | 3.58 |

The packer is 6-11% of a matmul, so it is not the gap.  The engine's 8.34
GFLOP/s at 256^3 against the kernel's 9.32 is the pack pass and the launch,
and the gap to the recorded stand-alone 13.7 is not a code difference at all:

* the asm10 GEMM computes an 8x64 C tile with 32 live C vectors, so per tile it
  reads `K*64*4` bytes of B and `K*8*4` of A for `2*8*64*K` FLOPs.  That is
  **4 FLOP/byte**, and it is fixed by the register file: 32 accumulator
  vectors cannot hold a larger tile, and both a 16x16 and an 8x64 tile land on
  the same 4 FLOP/byte (16x16 is slightly better at 4.0, 8x64 at 3.5).
* at 9.32 GFLOP/s the kernel is moving `9.32/4 = 2.3 GB/s` of panel and B
  streams.  At 3.63 it moves 0.9 GB/s.  The packer on the same boot moves
  0.85-2.0 GB/s.  The GEMM is at the memory path's rate in both states, and
  its instruction schedule has 4x headroom: the 39-word k loop is 390 ns per
  tile-k at 400 MHz, i.e. 15.7 GFLOP/s of issue capacity.
* 13.745 GFLOP/s is 3.4 GB/s at 4 FLOP/byte, which is above this board's
  measured streaming rate.  That record was taken on a boot whose memory path
  served B from the L2; the same kernel re-run on the boot of the
  2026-11-19 record measured 8.12, and on the slow boot here 3.6, with
  identical cycle counts.
* the L2 is not doing that any more.  Enabling `L2CENA` alongside the
  `L2CCLR` the launch path writes (the sequence the kernel's
  `v3d_invalidate_l2c()` uses) changed 256^3 by 0.0% in direct mode, and a
  64-column B tile that fits the 128 KiB L2 12 times over still ran at the
  same 3.6 GFLOP/s as a 256 KiB one.  The reads are not being retained.

So the 8.5 vs 13.7 explanation, in the order the questions were asked: panel
packing 6-11%; K-loop overhead nil (cycle counts match the record); VPM
staging nil (the stall counters are on the streaming reads); QPU count already
12; unroll already the recorded 6-phase loop.  The gap is the board's memory
state, and the only code lever left is FLOP/byte, which needs a tile larger
than 32 accumulator vectors can hold.  A 16x32 tile (16 rows x 32 columns,
exactly 32 C vectors) would move 1.33x fewer bytes and needs the C rows to
live in the A-file registers that currently hold the addressing, so it is a
new schedule rather than a parameter.

## 4. Autotune: the record is not a gain at this board's noise level

`infer/build/bench_autotune 5`, cold (empty cache directory) and then against
the same directory.  `default` is the launch the executor runs without a
record; `tuned` is the same op after `vc4_autotune` chose and stored its
parameters.

Slow-boot cold search (13-14 candidate sets, 6.31-13.07 s wall):

| shape | chosen | default ms | tuned ms | gain |
|---|---|---:|---:|---:|
| 128^3 | pack_q=2 gemm_q=12 | 1.894 | 2.109 | **0.90x** |
| 256^3 | pack_q=2 gemm_q=3 | 11.540 | 11.073 | 1.04x |
| 512^3 | pack_q=4 gemm_q=3 | 79.960 | 76.653 | 1.04x |
| conv 32x32x32 3x3 | im2col_q=4 gemm_q=6 | 19.173 | 18.742 | 1.02x |

The same four from the record on that boot (0.01 s, no device work): 1.17x,
1.03x, 1.01x, 1.00x.  The 1.17x is the same launch as the default.

Fast-boot cold search (5.88-13.62 s wall):

| shape | chosen | default ms | tuned ms | gain |
|---|---|---:|---:|---:|
| 128^3 | pack_q=4 gemm_q=12 | 0.980 | 1.059 | **0.93x** |
| 256^3 | pack_q=2 gemm_q=12 | 4.833 | 5.870 | **0.82x** |
| 512^3 | pack_q=4 gemm_q=12 | 31.639 | 31.974 | 0.99x |
| conv 32x32x32 3x3 | im2col_q=4 gemm_q=6 | 14.225 | 18.515 | **0.77x** |

Three of four records made the shape slower.  In two of them the chosen GEMM
QPU count is the default (12) and only the packer changed, so the winner was
decided by a 1-4% search-time difference that did not survive re-measurement;
the conv case chose 6 QPUs, measured 13.788 ms at search time and 18.515 ms
when the shape was timed again.

The fix is in this pass: `vc4_autotune` now keeps the default unless the
winner beats it by more than `TUNE_ACCEPT_MARGIN` (3%), and records the
default otherwise.  A record can no longer make a shape slower than not
tuning it.  The gain the tuner can honestly claim on this board is therefore
1.00x for these shapes; the QPU counts that matter (12 GEMM QPUs, 4 pack) are
already the ladder's first candidate, and the search confirms that rather than
improving it.  The post-change cold and from-record numbers are the next
silicon step, because a private daemon and a concurrent daemon without
`--gpu-lock` collided and the context reset before the run could be repeated.

## 5. What is not in this pass

| candidate | measured reason it is not here |
|---|---|
| direct 3x3 conv | at most 46 ms (17%); the coalesced form repeats the panel's transpose per column block, the naive form is 16x the A transactions (section 2) |
| flat conv blocks (remove the align16 M padding) | 113 ms on the slow boot, 45 ms on the fast one, but needs an im2col kernel with a variable-length segment per 16-pixel block; the next kernel to write |
| 16x32 GEMM tile | 1.33x fewer bytes per FLOP, but the C rows would have to take the A-file registers the addressing uses: a new schedule |
| batching several ew groups per QPU | the long sigmoid/silu body reliably hangs a QPU on its second group at 12 QPUs (`src/vc4_ew.c`); the launch count it would save is host time, not QPU time |
| L2 enable on the launch path | no change to 256^3 (0.0%) and no change to a 64-column B tile that fits L2; the reads are not retained |

## 6. Gates and the next GPU step

No kernel body changed in this pass, so the element-level gates are unchanged
by construction; they still have to run on the board after the two tool files
and the autotune change:

1. `infer/tools/op_matrix.py` (152/152 expressible, 0 refused, 0 mismatch);
2. YOLOv8n-160 parity against the shipped reference at 5e-4 + 5e-4*max(|ref|,1);
3. `test_conv`, `test_matmul`, `test_nhwc`, `test_ew`;
4. the artifact round trip (`tests/test_artifact.py` and the `.vc4x` build);
5. `bench_autotune` cold and from record with the margin gate in place;
6. `bench_split` at 128/256/512, recorded with the boot state.

Reproduction, one client at a time:

```sh
cd ~/work/vc4-kernel
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 \
       VC4D_SOCKET=/run/vc4d-kernel.sock \
       VC4INFER_LIB=$PWD/infer/build/libvc4infer.so PYTHONPATH=$PWD/infer/python
tools/thermal-guard.sh --max 75 -- ./infer/tests/bench_split
tools/thermal-guard.sh --max 75 -- python3 infer/tools/onnx/profile_classes.py \
    infer/packaging/vc4infer/data/models/yolov8n_160.onnx --conv
```

`bench_split` no longer has to be built by hand: `infer/Makefile` builds it
with the rest of the tests.
