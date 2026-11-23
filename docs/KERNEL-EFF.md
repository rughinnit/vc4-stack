# Where YOLOv8n-160's QPU time goes, and what the GEMM's 13.7 GFLOP/s needs

This page records a kernel-efficiency pass over YOLOv8n at 160x160: the per-node
class split of QPU time, the split of a conv into its im2col panel and its
GEMM, the same split for a matmul, the per-shape autotuner's silicon numbers,
and one measurement hazard that decides how any of these numbers may be
compared: the memory path's stall time changes by up to 30% with the clock
registers unchanged and the instruction count per tile identical.

Everything here is kernel time.  A node's wall time in the engine profile
includes the socket round trip, the ARM-side copies and the Python dispatch;
its `gpu_ms` is the daemon's own kick-to-queue-idle time and is what the
tables use.

## 0. Conditions

| item | value |
|---|---|
| board | Raspberry Pi 3B, BCM2837, 12 QPUs, kernel 6.18.50 |
| clocks | `core_freq=400`, `core_freq_min=400`, `gpu_freq=300`, `sdram_freq=450`; `core` 400 MHz, `v3d` 300 MHz, `sdram` reports 166.7 MHz (all runs, both states) |
| ARM | capped at 1.0 GHz for this pass (`scaling_max_freq`) |
| thermal | 46-55 C, `throttled=0x0` in every tabulated run, fanless |
| backend | `vc4d`, one client, private socket, `--gpu-lock`, 64 MiB arena |
| passes | none for the class table (the engine default); the artifact row is `--opt all` |

Tools: `infer/tools/onnx/profile_classes.py` for the class table and the conv
split, `infer/tests/bench_split` for the pack/GEMM split and the cycle count,
`infer/build/bench_autotune` for the tuner.

## 1. QPU time by node class

YOLOv8n-160, medians of three runs, 831 launches:

| class | QPU ms | share | launches |
|---|---:|---:|---:|
| Conv | 210.37 | 76.8% | 131 |
| Mul | 35.91 | 13.1% | 467 |
| Softmax | 6.95 | 2.5% | 40 |
| Concat | 6.33 | 2.3% | 60 |
| Reshape | 4.25 | 1.6% | 41 |
| Add | 2.95 | 1.1% | 38 |
| MaxPool | 2.51 | 0.9% | 9 |
| Split | 1.58 | 0.6% | 15 |
| Sigmoid | 1.06 | 0.4% | 14 |
| Resize | 0.73 | 0.3% | 1 |
| Slice / Sub / Transpose / Div | 1.22 | 0.4% | 12 |
| **total** | **273.87** | **100%** | **831** |

Wall time is 584 ms, so QPU duty is 47%.  A second run of the same graph gave
832 launches and 274.7 ms; the one-launch difference is a tensor whose group
count lands exactly on a launch boundary.  The 263 ms baseline in
[`OPTIMIZER.md`](OPTIMIZER.md) is the same engine path measured when the
memory path was serving more of the streams (section 3).

Two rows matter for kernel work:

* conv is 77% of QPU time: 64 conv calls, 131 launches.  Each call is an
  im2col panel launch and a GEMM launch, and three calls carry a third;
* `Mul` is 467 launches for 36 ms, 0.077 ms each.  These are the graph-level
  SiLU multiplies and the C2f branch multiplies, run as one group per QPU per
  launch because a QPU that runs a second group in the same program hangs with
  the long sigmoid body (`src/vc4_ew.c`).

Folding the SiLU pair into the conv GEMM is worth more than the launch count
suggests: the artifact built with `--opt all` runs the same graph in **365
launches and 258.8-260.4 ms of QPU time** (wall 489-498 ms), 5% less QPU than
the default path and 16% less wall, because the fused epilogue does not write
the activation to DRAM and read it back for the multiply.  Parity is
unchanged (section 5).  The `Mul` row above is therefore a property of the
default path, not of the graph.

## 2. Convolution: the panel is 24% of conv QPU

The NHWC path is im2col + GEMM (`src/vc4_nhwc.c`): a panel pass writes
`M x (K + 16)` floats in the GEMM's k-major layout, the bias folded as one
extra k block, and the asm3 GEMM (16x16 C tile, `bmode=1`) reads it.  Wrapping
`Runner._conv_nhwc` and reading `ctx->st.pack_ns` (the panel pass) and
`ctx->st.kernel_ns` (the GEMM) around each call splits it:

| | ms | share of conv QPU |
|---|---:|---:|
| im2col panel (64 layers) | 49.89 | 24.2% |
| GEMM (64 layers) | 156.60 | 75.8% |
| conv total | 206.49 | 100% |

The panel share moves with the memory path (measured 22.7-24.2% across runs),
because it is bound by its strided source reads while the GEMM is bound by its
sequential ones.

Per layer (`profile_classes.py --conv`), the six largest:

| layer | Cin | Cout | k | panel ms | GEMM ms |
|---|---:|---:|---:|---:|---:|
| /model.7/conv | 128 | 256 | 3 | 1.37 | 10.37 |
| /model.22/cv3.0/cv3.0.1 | 80 | 80 | 3 | 2.57 | 8.17 |
| /model.22/cv3.0/cv3.0.0 | 64 | 80 | 3 | 2.07 | 6.59 |
| /model.0/conv | 16 | 16 | 3 | 4.96 | 3.15 |
| /model.22/cv3.2/cv3.2.0 | 256 | 80 | 3 | 1.18 | 6.39 |
| /model.22/cv2.0/cv2.0.1 | 64 | 64 | 3 | 2.45 | 5.96 |

### A direct 3x3 kernel was not written, and here is the measurement

The panel is 24% of conv QPU, so a direct kernel could recover at most 50 ms
(18% of total QPU), and it cannot recover more: the GEMM reads the panel once
per column block (N/16 times), and a direct path would read the padded image
the same number of times because the taps are k blocks.  What it saves is the
panel *write* plus the panel pass's one image read.

The naive direct A path costs more than it saves.  The GEMM's A load for one k
is a 64 B VDR: 16 lanes = 16 output pixels of one channel, which the panel
stores contiguously.  In the NHWC image those 16 pixels are `Cin*4` bytes
apart, so the same request becomes 16 separate 4 B accesses in 16 different
cache lines: 16x the transactions.  The coalesced alternative does the panel's
read-and-transpose inside the GEMM (one 16x16 VDR plus a vertical VDW per k
block), which is the panel pass's work again, once per column block instead of
once per layer.

### Where the conv arithmetic is wasted: 29% of M is padding

`Runner._conv_nhwc` pads each output row to 16 pixels (`op = align16(wo)`) and
the panel and GEMM run over `M = ho * op`.  At 160x160:

| output width | op | M padding |
|---|---:|---:|
| 160 | 160 | 1.00 |
| 80 | 80 | 1.00 |
| 40 | 48 | 1.20 |
| 20 | 32 | 1.60 |
| 10 | 16 | 1.60 |
| 5 | 16 | 3.20 |

Summed over the 64 layers, 33,952 padded rows against 26,352 real pixels, a
factor of 1.29.  That is 18% of conv QPU and 14% of the total in the table
above.  The waste is a property of the 16-pixel VDR block: the panel pass
reads 16 pixels at one constant pitch per tap, and a block that crosses a row
boundary has no constant pitch.

Removing it needs a panel pass whose 16-pixel block is assembled from four
4-pixel groups, each group being one VDR of four rows at the pixel pitch.  The
group base walks with the flat pixel index and a row-wrap add; no divide is
needed.  The GEMM then runs over `align16(ho*align4(wo))` rows and the output
tensor's pitch becomes `align4(wo)`.

The first version of that pass stored 16 k rows by 4 pixel words per group and
was rejected on silicon; the second fills the row kernel's own 16x16
transposing store from four 4-row VDRs and has passed its CPU-side proof.  The
silicon gate is the next step.  See section 7.

## 3. GEMM: pack against kernel, and the stall-time state

`infer/tests/bench_split`, medians of five, same process and arena:

| shape | pack ms | GEMM ms | pack share | GEMM GFLOP/s | cyc/tile-k |
|---|---:|---:|---:|---:|---:|
| 128^3 | 0.153 | 0.736 | 17.2% | 5.70 | 3395 |
| 256^3 | 0.444 | 3.648 | 10.9% | 9.20 | 3307 |
| 512^3 | 1.701 | 26.836 | 6.0% | 10.00 | 3269 |

`cyc/tile-k` is `gpu_cycles / ((M/8)*(N/64)*(K/16))`: V3D
`EXECUTION_CYCLES` summed over the QPUs, per 8x64 tile and one k block.  It is
the schedule's cost and it did not move in any run of this pass, fast or
slow: 3307 for 256^3 and 3269 for 512^3, every time.

The engine's 8.17 GFLOP/s at 256^3 against the kernel's 9.20 is the pack pass
(10.9%) and the launch.  The gap to the recorded stand-alone 13.7 is not a
code difference:

* the asm10 GEMM computes an 8x64 C tile with 32 live C vectors, so per tile it
  reads `K*64*4` bytes of B and `K*8*4` of A for `2*8*64*K` FLOPs: **4
  FLOP/byte**.  That is fixed by the register file; a 16x16 tile lands on the
  same 4.0 and an 8x64 on 3.5.
* at 9.2 GFLOP/s the kernel is moving about 2.3 GB/s of A and B streams; the
  packer on the same boot moves 1.17 GB/s of panel.  The 39-word k loop is
  390 ns per tile-k at 400 MHz, i.e. 15.7 GFLOP/s of issue capacity, so the
  kernel is not instruction-bound.
* 13.745 GFLOP/s is 3.4 GB/s at 4 FLOP/byte.  The same kernel re-run on the
  board's quieter days measured 8.12 with identical cycle counts, and 3.6 on
  the worst run here (section 4).  The difference is entirely stall time.

The L2 is not retaining these streams: enabling `L2CENA` alongside the
`L2CCLR` the launch path writes (the sequence the kernel's
`v3d_invalidate_l2c()` uses) changed 256^3 by 0.0% in direct mode, and a
64-column B tile that fits the 128 KiB L2 twelve times over ran at the same
rate as a 256 KiB one.

## 4. The stall-time state is intermittent and not a readable clock

The same launch, same parameters, same cycles per tile-k, measured 26.58 ms
and 34.17 ms at different moments, and the autotuner saw 28.58 ms and then
34.13 ms within ten seconds of its own search.  `docs/KERNELS.md` records the
caveat with the evidence; the summary is:

* `core` 400 MHz, `v3d` 300 MHz, `sdram` 166.7 MHz in every sample of both
  states; `firmware` config `core_freq=400`, `gpu_freq=300`, `sdram_freq=450`;
* `throttled=0x0`, 46-55 C, in every tabulated run; the slow observations
  were not hotter than the fast ones;
* `cyc/tile-k` identical (3269 for 512^3, 3307 for 256^3), so the program and
  the QPU issue rate are unchanged: the time is memory stall;
* twelve consecutive 512^3 runs over 15 s sat at 26.58-27.55 ms, and the
  inflated observations are isolated minutes apart, so it is not a warm-up or
  a ramp;
* a degraded daemon arena (48 MiB instead of 64 MiB) is a different failure
  (a conv scratch allocation fails), not a slowdown.

The remaining candidates are DRAM contention from another master (an ARM
kernel, the display pipe, a concurrent GPU client) or a firmware memory state
that no exposed register reports.  I did not reproduce it on demand: with one
client and the board idle it did not appear in a 15 s series.  The rule that
follows is in `KERNELS.md`: record `cyc/tile-k`, interleave A/B, and never
compare two absolute ms taken more than a minute apart without one.

## 5. Autotune: the record is 1.00x, and the before/after comparison drifts

`infer/build/bench_autotune 5`, cold (empty cache), with the 3% acceptance
margin in place:

| shape | search | chosen | default ms | tuned ms | gain |
|---|---:|---|---:|---:|---:|
| 128^3 | 5.12 s | pack_q=4 gemm_q=12 | 1.001 | 0.993 | 1.01x |
| 256^3 | 5.89 s | pack_q=2 gemm_q=11 | 4.250 | 4.202 | 1.01x |
| 512^3 | 9.78 s | pack_q=4 gemm_q=12 | 28.581 | 34.130 | **0.84x** |
| conv 32x32x32 3x3 | 8.13 s | im2col_q=4 gemm_q=9 | 12.912 | 10.104 | 1.28x |

The 512^3 row chose the *default* parameters (pack_q=4, gemm_q=12) and still
shows 0.84x: the same launch measured 28.581 ms and then 34.130 ms with the
same cycle count.  That is the section 4 state, not a record, and no
acceptance margin can fix a drift between the two measurements.  It is also
the reason the 1.28x conv row is not claimed as a win: at that amplitude the
drift and the gain are the same size, and the conv shape's 12-to-9 QPU change
needs an interleaved A/B before it means anything.

The margin still does its own job: a record that only tied the default in the
search is no longer written.  On the earlier boot the same benchmark produced
0.93x, 0.82x and 0.77x records, all of which are now 1.00x.  The honest
statement for these four shapes is that the tuner confirms the default QPU
counts (12 GEMM, 4 pack) and gains nothing measurable; the full table is in
`OPTIMIZER.md` pass 8.

The benchmark and the tuner both time the *whole op* (CLOCK_MONOTONIC around
the call), so a tuned record that reuses the same kernel is compared by a
measurement that includes the launch path.  An interleaved A/B/A sequence
inside `bench_autotune` is the fix; it is not in this pass.

## 6. Gates

Run on this boot with the private daemon, `REPS=5`, thermal guard:

| gate | result |
|---|---|
| `infer/tools/op_matrix.py` | **152/152** expressible, 0 refused, 0 mismatch, 0 hang, 0 invalid |
| YOLOv8n-160 parity, ONNX path | **0/44,100** over 5e-4, max abs 0.0204, max scaled 2.24e-4; QPU 273.7-274.7 ms, 832 launches |
| YOLOv8n-160 parity, `.vc4x` artifact (`--opt all`) | **0/44,100**, same max abs; QPU 258.8-260.4 ms, 365 launches |
| `test_conv` | 6 shapes x 5 reps clean, 2 unsupported forms refused |
| `test_matmul` | 8 shapes x 5 reps clean, 6 epilogue cases bit-identical to two launches, 2 silu refusals |
| `test_nhwc` | 31 shapes (conv, stride-2, dilation, grouped, depthwise, pool, upsample, concat, 3 epilogues) all pass |
| `test_ew` | 10 cases x 5 reps clean |
| artifact round trip | built with `--opt all`, loaded and validated, parity above; CPU container test `tests/test_artifact.py` passes |

Reproduction:

```sh
cd ~/work/vc4-kernel
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 \
       VC4D_SOCKET=/run/vc4d-kernel.sock \
       VC4INFER_LIB=$PWD/infer/build/libvc4infer.so PYTHONPATH=$PWD/infer/python
tools/thermal-guard.sh --max 75 -- ./infer/build/bench_split
tools/thermal-guard.sh --max 75 -- python3 infer/tools/onnx/profile_classes.py \
    infer/packaging/vc4infer/data/models/yolov8n_160.onnx --conv
```

## 7. The flat-panel im2col: closed by its silicon gate

**Closed.**  The second form of the flat panel -- a 16-pixel block whose 16
VPM rows are filled from four 4-row VDRs and stored with the row kernel's own
16 x 16 transposing store -- is **exact on one QPU and corrupt on two or
more**.  The store shape is not what failed: the QPU accepts this one where it
rejected the first form's 4-word store, with no `V3D_ERR_STAT`.  What has not
been shown to work is four loads filling the window ahead of one store.
`VC4_CONV_FLAT` refuses, the blob is named `im2col_nhwc_flat2_*`, and any
future attempt has to meet the constraint in "Constraint for any future
attempt" below.  Everything above the gate on this page is what that attempt
can reuse.

`kgen.build_im2col_nhwc_flat2(Cin, KH, KW, stride, dilation)` keeps the
16-pixel block and takes its pixels from the flat output plane:
`op = align4(wo)`, `M = align16(ho*op)`, four 4-pixel groups per block.  A
group of four pixels can never straddle a padded row, because the flat plane
is `ho` rows of exactly `align4(wo)` pixels.

The first form stored each group with a 16 k row by 4 pixel word transposing
VDW and was rejected on silicon (section below).  The second form uses no new
store shape at all: it fills the 16 VPM rows the row kernel's own 16 x 16
transposing store reads, from four 4-row VDRs at the pixel pitch.

| piece | shape | change from the row kernel |
|---|---|---|
| VDR per group | 4 rows (pixels) x 16 words (channels) at `stride*Cin_pitch*4` | 16 rows per block becomes four loads of 4, one per group; the 4-row VDR is the only new VPM shape |
| VDW per (tap, channel block) | 16 k rows x 16 pixel words, transposing, `vdw_stride(0)`, VPM row `16*q` | **unchanged**: the word the row kernel stores with |
| VPM rows | `16*q + 4*g + i` for group `g`, lane `i` | four groups fill one QPU's 16-row window exactly once per (tap, channel block) |
| group walk | add `4*stride*Cin_pitch*4`, counter, at zero add `stride*row_pitch - GPR*4*stride*Cin_pitch*4` | replaces `nx` blocks per row with a running flat index |
| uniforms | u0..u13: source, panel, strides, GPR, first-group offset, groups left, row counter | the GEMM, taps and channel loop are untouched |
| endblock | the real-group count u4 | blocks are contiguous per QPU, so the count ends the kernel; the last block's pad groups are still loaded and stored |

A block costs four group steps (three to reach its four groups, one at the end
to reach the next block's first group) and the walk is branch-free: the
counter decrement sets the flags and the row-wrap add and the counter reload
both ride on the zero flag.

### CPU-side proof (no GPU)

`infer/tests/check_im2col_flat.py` runs the host uniforms and the kernel's
block arithmetic in Python -- the same step/wrap sequence, the same
real-group count, the same per-QPU split -- and asserts, for every (pixel,
tap, channel block) and for two QPU splits plus one QPU:

* the VDR lane address is the pixel the flat index names;
* the group's four rows land on VPM rows `16*q + 4*g + i`, and the four groups
  cover the QPU's 16-row window exactly once;
* each VPM row, read from the padded image at the address the kernel computes,
  carries the 16 channels of pixel `4*g + i` -- the same content the
  row-padded kernel's single 16-row VDR puts there;
* applying the 16 x 16 transposing store to those rows (memory row = channel,
  words = pixels) gives, for every real output pixel, the same K values as the
  row-padded panel;
* the blob's own VPM literals: the store is the row kernel's 16 x 16
  transposing word with a zero stride, the load is four VPM rows, the
  channel-block back-step is present and neither refused shape appears.  The
  control words are generator constants, not model ones, so this is the part
  the model cannot see.

The group addresses are accumulated as the kernel accumulates them, and the
destination is a register accumulated per (tap, channel block), so a missing
advance shifts the tile instead of comparing equal.  Five mutations of the
generator each fail the check: the 4-word store, the 48-byte store gap, a
16-row load, the missing channel-block back-step, and the destination advance
placed only inside the channel loop.

This proof, and its destination model in particular, is what a future form
reuses: it is cheap, it is host-only, and it caught one of the two defects the
gate then confirmed.  It cannot see VPM request ordering, which is why it did
not catch the other.

15 shapes, 0 failures, the 13 of the first form plus 20x20 and 60x60:

| shape | M flat | M row | ratio | words flat2 | words row | delta |
|---|---:|---:|---:|---:|---:|---:|
| 160x160x16 k3 | 25,600 | 25,600 | 1.000 | 345 | 365 | -20 |
| 20x20x64 k3 | 400 | 640 | 0.625 | 412 | 365 | +47 |
| 10x10x32 k3 | 128 | 160 | 0.800 | 412 | 365 | +47 |
| 5x5x16 k3 | 48 | 80 | 0.600 | 345 | 365 | -20 |
| 13x7x16 k3 | 112 | 112 | 1.000 | 345 | 365 | -20 |
| 7x13x16 k3 | 112 | 208 | 0.538 | 345 | 365 | -20 |
| 3x1x16 k3 | 16 | 16 | 1.000 | 345 | 365 | -20 |
| 4x3x16 k3 | 16 | 48 | 0.333 | 345 | 365 | -20 |
| 20x20x16 k3 s2 | 128 | 160 | 0.800 | 345 | 365 | -20 |
| 20x20x16 k3 p0 | 368 | 576 | 0.639 | 345 | 365 | -20 |
| 20x20x16 k5 p2 | 400 | 640 | 0.625 | 751 | 995 | -244 |
| 20x20x16 k3 p2 d2 | 400 | 640 | 0.625 | 345 | 365 | -20 |
| 40x40x32 k1 | 1,600 | 1,920 | 0.833 | 150 | 119 | +31 |
| 20x20x16 k3 (op 20) | 400 | 640 | 0.625 | 345 | 365 | -20 |
| 60x60x16 k3 | 3,600 | 3,840 | 0.938 | 345 | 365 | -20 |

Rows per 16-pixel block: **16 panel rows, one 16x16 store**, over the same 16
VPM rows (four loads of 4) the row kernel fills with one load of 16.  Against
the refused first form (454 words for 3x3, 1,048 for 5x5, 180 for 1x1) the
blob shrinks by 109, 297 and 30 words.  The largest is 930 of the 2,048
instruction slot (5x5 with two channel blocks or more); beyond that the size
does not depend on `Cin`, because the channel loop is a branch: 512 channels
cost the same blob as 32.

Blob size is not what the panel costs, because the channel loop runs.  Per
channel-block iteration the flat2 body is 27 words against the row kernel's
15: 20 VPM command words (four load setups, four strides, four addresses, four
waits, one store setup, stride, address and wait) against 8, plus the
destination advance, the address update, counter, branch and its three delay
slots.  Per tap the row kernel spends 12 words rebuilding its tap base (the
padding the QPU's missing register forwarding forces) against the flat2
kernel's one 4-word group advance, so the two cross over between one and two
channel blocks:

| channel blocks per tap | flat2 words per tap | row words per tap | ratio |
|---:|---:|---:|---:|
| 1 | 25 | 27 | 0.93 |
| 2 | 59 | 42 | 1.40 |
| 4 | 113 | 72 | 1.57 |
| 32 | 869 | 492 | 1.77 |

(Per-tap counts are for every tap after the first; the first costs one
4-word advance less.)  The flat path runs 0.795 of the blocks, so at one
channel block the panel's instructions fall to 0.74 and at four or more they
rise to about 1.3-1.4, while its bytes and the GEMM's M fall to 0.795 in
every case.

### The first form's store was not legal

Recorded from the first form, run on the rebased tree (origin/main 54ddbd0)
with the packaged vc4d, one client, 45-48 C, `throttled=0x0`.  The blob
identity was checked before the result was believed: a fresh cache directory
held `im2col_nhwc2_16_5_5_s1_d1.bin` (995 words) for the default path and
`im2col_nhwc_flat_16_5_5_s1_d1.bin` (1,048 words) for the flag.  The probe was
a 16-channel, 5x5 image with a 5x5 kernel and pad 2: one image, three 16-pixel
blocks, 48 GEMM rows against 80.  Two store field combinations were tried:

| VDW fields (VPM block) | result |
|---|---|
| 16 rows x 4 words, transposing, 48 B gap | runs, no error, **max abs error 84.1** against the direct convolution (values of magnitude 20): the panel is transposed wrongly, the pixels land in the wrong words |
| 4 rows x 16 words, transposing, 48 B gap (the C store's own field values) | **launch error `V3D_ERR_STAT=0x10`**; SRQCS `qerr=0`, state LIVE, no timeout, no context reset, no reboot |

The conclusion is that the VPM-to-DMA store transposes a 16x16 block: the
panel's own store is 16/16 with a zero stride, the C store is 4/16 with
`horizontal=1`, and a 4-word-wide transposing store is either mis-laid-out or
rejected.  That is why the second form keeps the 16x16 store and moves the
flat schedule into the load side only.

### The gate: exact at one QPU, corrupt at two or more

Run on 5a2782d with the packaged vc4d, one client, ARM capped at 1.0 GHz,
`core` 400 MHz, `v3d` 300 MHz, 43-49 C, `throttled=0x0` in every run, under
`tools/thermal-guard.sh --max 75`.

**The store shape is accepted.**  The 5x5 probe returns `ok` in 0.15 ms with
no queue error, no timeout and no context reset, and loads
`im2col_nhwc_flat2_16_5_5_s1_d1` (751 words) with `gemm_48_16_416_b1_q3`
against the row path's 995-word blob and `gemm_80_16_416_b1_q5` -- **48 panel
rows where the row path uses 80**, read from the launched GEMM's name rather
than modelled.

**The one-QPU checkpoint.**  With the im2col restricted to one QPU (a test
build with `NHWC_IM2COL_QPU 1`) the same 5x5 probe through the engine matches
the CPU convolution to **9.8e-05, 0/400 over 5e-4** -- the same figure the row
path gets -- and panel dumps match the schedule word for word:
**13,600/13,600** words for the 16-channel 5x5 and **9,216/9,216** for an
8x8x16 3x3, with the VPM row content, the transposing store, the group walk
and the destination all as modelled.  That checkpoint is what says the block
schedule itself is sound; what it cannot see is more than one QPU.

**Two or more QPUs, the panel is corrupt.**  The same 5x5 probe reads max abs
14.8 at the committed four-QPU split, and the dumps show whole tiles
belonging to a later tap and, in a 1x1 case, a tile holding another QPU's
source address.  Each candidate was tested with a probe built for it:

* **not uniform staging.**  A kernel that stores the 16 uniform words it
  received back to memory shows every QPU getting its own block: source
  offset, panel offset, QPU index and counters all correct.
* **not a shared VPM window.**  Four QPUs given identical source and
  destination but distinct windows produce a correct tile, so a QPU's window
  is its own.
* **not the load or store drain.**  One extra VPM-load drain before the store
  and a second VPM-store wait after it change nothing.
* **not the uniform count.**  The proven row kernel with a 14th uniform read
  is exact across four QPUs.

**The surviving hypothesis** is that the store reads its window after later
loads have refilled it.  What this kernel does that no proven kernel in the
tree does is issue **four VPM loads ahead of one store** (the row im2col
issues one, the elementwise kernel none).  The overrun follows no fixed
per-QPU rule -- two taps per QPU at 16 channels, none then four then seven at
32, and another QPU's source address outright in the 1x1 case -- which is what
a store overtaken by a refill looks like, not a miscomputed address.  No
further window was spent on it: see "Why it is closed".

### Why it is closed

Even with the hazard fixed, the arithmetic is scoped to the wrong layers.  The
flat block only removes padding where `align4(wo) < align16(wo)`: nothing at
160 or 80 wide, 0.833 at 40, 0.625 at 20, 0.750 at 10.  The layers with
padding to remove are the head's 3x3s and `model.7`, and those are exactly the
layers with four to sixteen channel blocks, where the flat body costs 1.57 to
1.77 times the row kernel's words per tap (the table above) and the panel's
instructions do not fall.  The model's own numbers put the ceiling on the
gain at 210 ms of conv to 191 ms under an instruction bound (the modelled
table below), against the **20.1 ms per frame** at 160 that the conv pad
itself costs (`docs/CIVAC.md`, `docs/DAEMON-PIPE.md`).  Chasing a VPM
ordering hazard for a share of 19 ms on a subset of layers is the wrong
trade.

For reference, the model that motivated the form: over the model's 64 conv
layers the panel and GEMM row counts go 33,952 -> 26,992, a ratio of **0.795**
(20.5% fewer rows).  What that does to conv depends on what the panel pass is
bound by -- it is 24% of conv QPU, so the panel's 49 ms against the GEMM's
157 ms at 206 ms of conv:

| panel bound by | panel | GEMM | conv |
|---|---|---|---|
| bytes (0.795x) | 49 -> 39 ms | 157 -> 125 ms | 206 -> 164 ms |
| instructions, 1 channel block (0.74x) | 49 -> 36 ms | 157 -> 125 ms | 206 -> 161 ms |
| instructions, 4+ channel blocks (1.35x) | 49 -> 66 ms | 157 -> 125 ms | 206 -> 191 ms |

No row of that table was measured: the flat path never ran correctly on more
than one QPU, so its timings do not exist.

### Constraint for any future attempt

A block schedule that fills the block's 16 VPM rows from several loads may
not issue the store with more than one load in flight.  It must meet one of:

1. **one load in flight before the store** -- the loads are issued and
   retired one at a time, so the store's window cannot be refilled behind it;
   or
2. **a single 16-row load** where the block lies inside one output row --
   every layer whose `align4(wo)` is a multiple of 16, which is the whole
   160- and 80-wide part of this model -- which is the row kernel's own shape
   and needs no group walk at all.

The four-load form recorded here violates that constraint: it is exact at one
QPU and corrupt at two or more, which is why it is refused.  A form that meets
the constraint still has to pass its own gate on two QPUs before anything else
is measured; this page is the evidence that the store shape, the row fields
and the group walk are not what needs re-proving, and that a host model
without VPM ordering cannot close this class of defect.

### State of this pass

The flag still refuses.  `VC4_CONV_FLAT=1` prints
`refused: the flat panel is exact on one QPU and corrupt on two or more` and
serves the row-padded panel; the 5x5 probe with the flag set loads
`im2col_nhwc2_16_5_5_s1_d1` and `gemm_80_16_416_b1_q5` and matches the CPU to
9.8e-05, which is the row path doing the work.  The blob stays
`im2col_nhwc_flat2_*`, so a cache filled by the first form cannot be mistaken
for it.

Baseline on the same boot (5a2782d), same client and daemon, flag unset:
`selftest` **PASS 4/4**, YOLOv8n-160 **0/44,100 over 5e-4** (max abs 2.04e-02,
max rel 2.24e-04) in **832 launches, 597.0 ms wall**; `profile_classes.py
--conv` gives **wall 593.1 ms, QPU 267.4 ms, 831 launches, QPU duty 45 %**,
conv **205.9 ms** of which the panel is **46.0 ms (22.8 %)** and the GEMM
**156.0 ms**.  The largest convs are `model.7` (128->256 k3, 0.59 + 10.32 ms)
and the head's `cv3.0.1` (80->80 k3, 2.45 + 8.18 ms), `cv3.0.0` (64->80 k3,
1.94 + 6.61 ms) and `cv3.2.0` (256->80 k3, 1.20 + 6.43 ms).

**M and launches.**  The flat block reduces the GEMM's M (the probe: 80 rows
to 48; the model's 64 conv layers: 33,952 to 26,992, modelled) and does not
add a launch per conv: the im2col is still one launch per conv and the GEMM
one, with the im2col's QPU count the largest divisor of the block count up to
four.  The probe measured 3 launches either way (weight pack, im2col, GEMM).

## 8. What is not in this pass

| candidate | measured reason it is not here |
|---|---|
| direct 3x3 conv | at most 50 ms (18%); the coalesced form repeats the panel transpose per column block, the naive form is 16x the A transactions |
| flat conv blocks | closed by its gate: exact at one QPU, corrupt at two or more, and the arithmetic only helps the layers it cannot fix (section 7) |
| 16x32 GEMM tile | 1.33x fewer bytes per FLOP, but the C rows would have to take the A-file registers the addressing uses: a new schedule |
| batching several ew groups per QPU | the long sigmoid/silu body reliably hangs a QPU on its second group at 12 QPUs; the launches it would save are host time |
| L2 enable on the launch path | no change to 256^3 (0.0%) and none to a 64-column B tile that fits L2 |
