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
4-pixel groups, each group being one VDR of four rows at the pixel pitch and
one transposing VDW of 16 k rows by 4 pixel words at the panel's 64-byte k
pitch.  The group base walks with the flat pixel index and a row-wrap add; no
divide is needed.  The GEMM then runs over `align16(ho*align4(wo))` rows and
the output tensor's pitch becomes `align4(wo)`.

That kernel is written (`kgen.build_im2col_nhwc_flat`, selected by
`VC4_CONV_FLAT=1`) and has passed its CPU-side proof; the silicon gate is the
next step.  See section 7.

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

## 7. The flat-panel im2col: written, CPU-verified, silicon gate pending

`kgen.build_im2col_nhwc_flat(Cin, KH, KW, stride, dilation)` keeps the
16-pixel block and takes its pixels from the flat output plane:
`op = align4(wo)`, `M = align16(ho*op)`, four 4-pixel groups per block.

| piece | shape | change from the row kernel |
|---|---|---|
| VDR per group | 4 rows (pixels) x 16 words (channels) at `stride*Cin*4` | 16 rows per block becomes 4 per group; both are load shapes already in the tree |
| VDW per group | 16 k rows x 4 pixel words, 64-byte k pitch, `vdw_stride(48)` | the panel's transposing store with the word count cut to 4; the C store already runs the 4/16 combination |
| VPM rows | `16*q + 4*g`, q the QPU, g the group | four QPUs maximum, as before |
| group walk | add `4*stride*Cin*4`, counter, at zero add `row_adv - GPR*4*stride*Cin*4` | replaces `nx` blocks per row with a running flat index |
| uniforms | u0..u14: source, panel, strides, GPR, first-group offset, groups left, row counter | the GEMM, taps and channel loop are untouched |
| GIDX | none | blocks are contiguous per QPU, so the real-group count ends the kernel |

### CPU-side proof (no GPU)

`infer/tests/check_im2col_flat.py` re-runs the host uniforms and the kernel's
group arithmetic in Python and checks, for every (pixel, tap, channel block)
and for several QPU splits, that the VDR lane address is the pixel the flat
index names, the panel destination is the panel's `[k][16-pixel]` slot, and
the VPM row fields stay inside the 48-row window.  It then builds the panel
the flat schedule writes and the panel the row schedule writes and compares
them for every real output pixel.  13 shapes, 0 failures:

| shape | M flat | M row | ratio | words flat | words row |
|---|---:|---:|---:|---:|---:|
| 160x160x16 k3 | 25,600 | 25,600 | 1.000 | 454 | 365 |
| 20x20x64 k3 | 400 | 640 | 0.625 | 454 | 365 |
| 10x10x32 k3 | 128 | 160 | 0.800 | 454 | 365 |
| 5x5x16 k3 | 48 | 80 | 0.600 | 454 | 365 |
| 13x7x16 k3 | 112 | 112 | 1.000 | 454 | 365 |
| 7x13x16 k3 | 112 | 208 | 0.538 | 454 | 365 |
| 3x1x16 k3 | 16 | 16 | 1.000 | 454 | 365 |
| 4x3x16 k3 | 16 | 48 | 0.333 | 454 | 365 |
| 20x20x16 k3 s2 | 128 | 160 | 0.800 | 454 | 365 |
| 20x20x16 k3 p0 | 368 | 576 | 0.639 | 454 | 365 |
| 20x20x16 k5 p2 | 400 | 640 | 0.625 | 1,048 | 995 |
| 20x20x16 k3 p2 d2 | 400 | 640 | 0.625 | 454 | 365 |
| 40x40x32 k1 | 1,600 | 1,920 | 0.833 | 180 | 119 |

The instruction delta is +89 words for 3x3 (454 against 365), +53 for 5x5
(1,048 against 995) and +61 for 1x1; the largest blob is 1,048 of the 2,048
instruction slot.  The delta is per *block*, and the flat path runs 0.795 of
the blocks, so the panel's instruction count per output pixel still rises
(1.29 x for 3x3 at the model's mix) while its bytes fall by the same 0.795.

Modelled over the model's 64 conv layers, the panel and GEMM row counts go
33,952 -> 26,992, a ratio of **0.795** (20.5% fewer rows).  If the conv QPU
time scales with M, that is 210 ms -> 167 ms, about 43 ms of the 274 ms
total.  The panel pass also does four VDR/VDW commands per block instead of
one, so the real gain will be lower; that is what the silicon gate measures.

### Silicon gate (pending)

With the board handed back:

1. `VC4_CONV_FLAT=1` and `test_nhwc` (31 shapes, including stride-2,
   dilation, 5x5, grouped and the three epilogues): every shape must be
   clean, and the 4/16 VDW is new so the first run is a tiny-grid probe
   (one 5x5 image, `test_conv`'s 16x16 case).
2. YOLOv8n-160 parity at 5e-4 with `VC4_CONV_FLAT=1`, then the conv QPU ms
   and the launch count from `profile_classes.py --conv` against the same
   boot without the flag.
3. Only then flip the default; until then the env flag keeps both paths
   measurable and the shipped path unchanged.

## 8. What is not in this pass

| candidate | measured reason it is not here |
|---|---|
| direct 3x3 conv | at most 50 ms (18%); the coalesced form repeats the panel transpose per column block, the naive form is 16x the A transactions |
| flat conv blocks | written and CPU-verified (section 7); the silicon gate is pending, so the default path is unchanged |
| 16x32 GEMM tile | 1.33x fewer bytes per FLOP, but the C rows would have to take the A-file registers the addressing uses: a new schedule |
| batching several ew groups per QPU | the long sigmoid/silu body reliably hangs a QPU on its second group at 12 QPUs; the launches it would save are host time |
| L2 enable on the launch path | no change to 256^3 (0.0%) and none to a 64-column B tile that fits L2 |
