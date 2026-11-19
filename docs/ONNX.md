# ONNX on the VideoCore IV engine

`libvc4infer` is an **operator library**, not a graph runtime.  This document
describes the graph path added on top of it: a small ONNX exporter/runner in
`onnx/`, the two engine ops it needed (`vc4_maxpooln`, `vc4_softmax_fwd`), the
measured end-to-end numbers, and the honest coverage gap for YOLOv8n.

Everything that *computes* runs on the QPU.  The host only moves bytes
(padding, channel gathers, concat, nearest resize, boundary compaction) — the
same class of work the engine already does for its im2col layout.  An op or
attribute the engine cannot dispatch raises an error naming the node and its
ONNX op type; there is no CPU math fallback.

Hardware: Raspberry Pi 3B, BCM2837, VideoCore IV @ **stock** clocks
(core 400 MHz / V3D 300 MHz), kernel 6.18, engine through **vc4d**
(`VC4INFER_DAEMON=1`, unprivileged, one GPU user at a time).

Test date: 2026-11-19.  Engine tree: `engine/` at this commit.

## 1. Files

```
onnx/export_models.py   define + torch.onnx.export the fixed-shape models,
                        save models/<name>.onnx, .in.npy and the torch CPU
                        reference .out.npy
onnx/onnxrun.py         the ONNX executor (onnx package + ctypes binding)
onnx/verify.py          3 runs per model, compares with the torch reference,
                        prints per-node wall/GPU time and launch counts
onnx/export_yolo.py     real pretrained YOLOv8n -> ONNX (ultralytics)
onnx/yolo_report.py     static op/attribute/shape coverage + a dynamic run
onnx/tvshim/            name-only torchvision stand-in for the YOLO export
onnx/ref_onnx.py        independent numpy ONNX evaluator (the CPU reference)
onnx/yolo_bench.py      end-to-end benchmark + tolerance check + JSON report
onnx/make_refs.py       the fixed input + the numpy CPU reference (.npy)
onnx/torch_ref.py       the same input through torch's eager yolov8n (2nd ref)
```

Generated artifacts live in `onnx/models/` (not committed; regenerate with
`export_models.py`).  The engine additions are:

```
src/vc4_nhwc.c      vc4_conv_nhwc/pack_w  NHWC channel-tiled conv (+ pool/up2/copy)
src/vc4_convbwd.c   vc4_maxpooln          generic NxN / stride-S max pool
src/vc4_train.c     vc4_softmax_fwd       forward softmax (fused kernel 'p' probe)
tools/kgen.py       build_pooln           the generic pool kernel
tests/test_onnxops.c CPU-oracle check of both
vc4infer/_binding.py ctypes binding for both ops, vc4_get_stats, and a conv2d argtypes fix
```

## 2. Running a model

```sh
cd vc4-stack
source /home/raspi/work/torch-vc4/.venv/bin/activate   # torch CPU + numpy + onnx

python onnx/export_models.py
VC4INFER_DAEMON=1 python onnx/verify.py                # all three models
VC4INFER_DAEMON=1 python onnx/onnxrun.py onnx/models/modelB.onnx \
        --input onnx/models/modelB.in.npy --profile --repeat 3

# optional real YOLOv8n export + coverage report (downloads ~6 MB of weights)
python onnx/export_yolo.py                             # YOLO_IMGSZ=32 for 32x32
VC4INFER_DAEMON=1 python onnx/yolo_report.py onnx/models/yolov8n_640.onnx
```

The runner takes the GPU lock through vc4d; do **not** hold
`/tmp/vc4-gpu.lock` in the calling shell in daemon mode (vc4d takes it per
launch, and a second holder makes every launch time out).

`onnxrun.py` accepts fixed-shape fp32 models with batch 1 and one float input.
It evaluates the graph in ONNX topological order and keeps every activation in
the single 64 MiB arena.

## 3. Supported nodes

| ONNX node | implementation | constraints |
|---|---|---|
| `Conv` 3x3 / 1x1, stride 1, group 1 | `vc4_conv2d` (QPU im2col + QPU GEMM) | pads symmetric or pre-padded on host; `(H+2p)*align16(W+2p)*4 <= 8191` |
| `Conv` 3x3 stride 2 pad 1, group 1 | 4 phase images + one stride-1 `vc4_conv2d` | graph-level decomposition, 4x the MACs (see §6) |
| `Conv` 1x1 stride 2 pad 0 | phase (0,0) + `vc4_conv2d` 1x1 | 4x fewer outputs; no extra MACs |
| `Conv` depthwise (groups = Cin = Cout), stride 1 | block-diagonal dense weight + `vc4_conv2d` | Cout = Cin, stride 1; Cin x the MACs (see §6) |
| `Conv` 3x3/1x1 stride 1/2 (NHWC, default) | `vc4_conv_nhwc` + `vc4_conv_nhwc_pack_w` | Cin/Cout%16, `stride*Cin*4 <= 8191`; **no plane-size limit** |
| `Relu`, `Sigmoid` | `vc4_elementwise` | any layout |
| `Silu` | `vc4_elementwise` (`VC4_OP_SILU`) | any layout |
| `Add`, `Mul` | `vc4_elementwise` | equal shapes (layouts/pitches matched on the host) or a constant broadcast |
| `Sub`, `Div` | GPU `Mul(-1)` + `Add`; `Div` by a constant -> GPU `Mul` | see section 7.1 |
| `Split` | host byte gather | channel axis of a 4-D tensor |
| `Slice` | host gather | any axis, int64 starts/ends/axes/steps |
| `Transpose` | host permute | any permutation |
| `Softmax` | `vc4_softmax_fwd` | any axis (host permute to the last), <=16 classes, rows%8 |
| `Shape`, `Gather` | host int64 | static shapes |
| `Squeeze`, `Unsqueeze`, `Cast`, `Flatten` | view / type change | fp32 targets |
| `MaxPool` 2x2 stride 2 | `vc4_maxpooln` | native path is the general kernel |
| `MaxPool` kxk stride 1 (NHWC, SPPF) | `vc4_maxpool_nhwc` | k in {3,5}; the -inf border is materialised on the host |
| `MaxPool` NxN, any stride/pad | `vc4_maxpooln` (new QPU kernel) | border filled with `-inf` on host (bytes); output compacted |
| `Concat` | host byte concat | channel axis of NCHW only |
| `Resize` nearest | host byte gather | `mode=nearest`, `asymmetric` or `half_pixel`, integer or float scale |
| `MatMul` | `vc4_matmul` | 2-D only |
| `Gemm` | `vc4_matmul` + GPU `Add` for bias | `transA=0`, `alpha=1`, `beta∈{0,1}`; C of `[N]`, `[1,N]` or `[M,N]` |
| `Softmax` | `vc4_softmax_fwd` (new) | last axis, ≤16 classes, rows padded to 8 |
| `Flatten`, `Reshape`, `Identity` | view (no copy) | element count must match |
| `GlobalAveragePool` | `vc4_matmul` with a `1/HW` column | NCHW, batch 1 |

A `Mul(x, Sigmoid(x))` pair is rewritten into a single `Silu(x)` (see
7.1): YOLOv8's activation is exported as two nodes, and the engine has the
fused kernel.

Everything else fails loudly, e.g.:

```
onnxrun.UnsupportedNode: node '/model.9/Split' (Split): unsupported op Split
```

Inputs/outputs must be fp32.  `Expand`, `BatchNormalization`, `Pow`, `LRN`,
bilinear `Resize` and arbitrary `ConvTranspose` are not implemented;
`Div`/`Sub` only in the lowered forms of section 7.  A shape or attribute
combination outside the table above still raises `UnsupportedNode`.

## 4. Measured results

Torch CPU reference, tolerance `|got-ref| <= 1e-4 + 1e-4*max(|ref|,1)`.
A warm-up run compiles the QPU kernels (JIT, not counted); the three measured
runs follow.  "gpu" is the engine's pack+kernel time; "wall" includes the
daemon round trips, the ARM store and the byte-only layout work.

### modelA — 2x Conv3x3+ReLU, MaxPool2x2, Flatten, Linear, Softmax
input `1x3x32x32` → output `1x10`; 8 nodes, 38 launches.

| node | op | wall ms | gpu ms | launches |
|---|---|---:|---:|---:|
| /conv1/Conv | Conv 3→16 3x3 s1 | 6.11 | 2.25 | 3 |
| /Relu | Relu | 1.36 | 0.43 | 6 |
| /conv2/Conv | Conv 16→32 3x3 s1 | 6.66 | 2.94 | 3 |
| /Relu_1 | Relu | 2.09 | 0.79 | 11 |
| /pool/MaxPool | MaxPool 2x2 s2 | 2.85 | 0.78 | 11 |
| /Flatten | Flatten | 0.26 | 0 | 0 |
| /fc/Gemm | Gemm 8192→10 | 16.52 | 6.59 | 3 |
| /Softmax | Softmax 10 | 0.91 | 0.08 | 1 |

3 runs: **38.6 / 38.2 / 38.0 ms**; GPU 13.85 ms; max abs **4.84e-06**,
max rel (den=max(|ref|,1)) **4.84e-06**, **0 / 10 mismatches**.
First cold run after an empty kernel cache: 2.4 s (kgen JIT).

### modelB — YOLO-ish block
Conv 3x3 s2 → depthwise Conv 3x3 → MaxPool 5x5 s1 p2 → Concat → Resize 2x
nearest → Conv 1x1.  input `1x16x32x32` → output `1x16x32x32`; 9 nodes,
26 launches.

| node | op | wall ms | gpu ms | launches |
|---|---|---:|---:|---:|
| /conv_s2/Conv | Conv 16→32 3x3 s2 p1 (4 phases) | 9.86 | 3.86 | 3 |
| /Relu | Relu | 0.87 | 0.22 | 3 |
| /dw/Conv | depthwise 32→32 3x3 (dense expansion) | 4.69 | 2.04 | 3 |
| /Relu_1 | Relu | 0.88 | 0.24 | 3 |
| /pool5/MaxPool | MaxPool 5x5 s1 p2 | 2.57 | 0.78 | 11 |
| /Concat | channel concat 32+32 | 0.93 | 0 | 0 |
| /Constant | Constant | 0.42 | 0 | 0 |
| /Resize | nearest 2x | 5.05 | 0 | 0 |
| /head/Conv | Conv 1x1 64→16 | 4.95 | 0.86 | 3 |

3 runs: **35.8 / 33.4 / 32.3 ms**; GPU 7.99 ms; max abs **4.29e-06**,
max rel (den=max(|ref|,1)) **3.98e-06**, **0 / 16384 mismatches**.
First cold run: 1.9 s.

### modelC — Conv+ReLU, GlobalAveragePool, Flatten, Linear
input `1x16x8x8` → output `1x4`; 5 nodes, 9 launches.
3 runs **6.0 / 5.9 / 6.0 ms**; GPU 0.85 ms; max abs **4.06e-07**,
**0 / 4 mismatches**.

Per-node in `results.json`.  The wall/GPU gap is the per-launch daemon round
trip (~0.2–0.5 ms) plus the ARM-side layout copies; the GEMM in modelA is
dominated by the 64-column pad of a 10-wide output.

## 5. Engine additions

* `vc4_maxpooln(ctx, in, C,H,W, N,stride,pad, out,&Ho,&Wo)` — a straight-line
  4-output-row x 16-column QPU task; the host builds the `-inf` border
  (`tests/test_onnxops.c` checks 2x2/5x5/3x3 stride-2 cases exactly).
* `vc4_softmax_fwd(ctx, logits, rows, cols, out)` — rows % 8, cols ≤ 16.
  Uses the fused training kernel's `p` probe (same degree-5 `2^f` polynomial
  and 3x Newton reciprocal); the host pads to 16 classes with `-1e30` and
  compacts the result.  Measured error ≤ 3.4e-07.
* The Python binding (`vc4infer/_binding.py`, formerly `python/vc4infer.py`)
  gained `maxpooln`, `softmax_fwd`, `stats`, and a
  corrected `vc4_conv2d` ctypes signature (the old one had the wrong number
  and order of argument types, so `conv2d` could not be called from Python).
* The op table now lists `maxpool/NxNsS` as supported.

## 6. Why stride 2 and depthwise are decomposed

`vc4_conv2d` remains a **stride-1, group-1** kernel: its im2col VDR loads 16
consecutive pixels, so decimating on the QPU would need a different gather.
Rather than approximate or fall back to the CPU, the runner rewrites the node:

* **stride 2, 3x3, pad 1.**  Split the input into four parity phase images
  `P[a,b][i,j] = X[2i+a, 2j+b]` (a byte gather).  Each tap of the original
  conv belongs to exactly one phase and lands at a `dr,dq ∈ {-1,0}` offset, so
  a stride-1 3x3 conv over the concatenated 4*Cin-channel phase tensor with a
  phase-masked weight reproduces the result exactly.  The zeros waste 4x the
  MACs; every MAC still runs on the QPU.  modelB validates it to 4.3e-06.
* **depthwise.**  Expand to a dense `[Cout][Cin][3][3]` block-diagonal weight
  (byte copy + zero fill) and use the ordinary `vc4_conv2d`.  This costs Cin
  times the real MACs.  It is honest but not the right long-term kernel; a
  per-channel schedule is the obvious next step.

The alternative — a CPU convolution or a "close enough" substitute — is not
used anywhere.

## 7. YOLOv8n end to end at 160x160

The real Ultralytics YOLOv8n COCO checkpoint (3,151,904 parameters) exported
with `onnx/export_yolo.py` at **1x3x160x160** (opset 13, 12.4 MiB, 263 nodes)
now runs to completion on the QPUs and matches a CPU reference.

```
source /home/raspi/work/torch-vc4/.venv/bin/activate
cd vc4-stack/onnx
VC4INFER_DAEMON=1 python3 yolo_bench.py \
    /home/raspi/work/onnx/models/yolov8n_160.onnx \
    --input yolo160_in.npy --ref yolo160_ref.npy --repeat 3
```

### 7.1 What was added

**A channel-tiled conv path (the pitch limit is gone).**  `vc4_conv_nhwc`
(the NHWC plane developed on the YOLO work) is now part of the engine: QPU
`im2col_nhwc` over `[pixel][channel]` rows plus the QPU GEMM, output
`[pixel][channel]`.  The im2col VDR pitch is `stride*Cin*4` bytes, not the
whole plane, so a feature map no longer has to fit 8191 B - only `Cin` is
limited (`Cin <= 2047` at stride 1, `<= 1023` at stride 2).  `Cin`/`Cout` are
zero-padded to 16 channels (the first conv's 3 channels become 16, the DFL
conv's 1 output channel is computed as 16 and sliced).  The NCHW path is kept
only as a fallback for shapes NHWC does not take.

**Layout bookkeeping.**  A 4-D fp32 tensor carries `layout` (nchw/nhwc) *and*
the NHWC storage `pitch` (the conv writes `align16(wo)` columns).  Elementwise
ops are layout-agnostic; every op that depends on the channel axis converts on
the host.  Elementwise operands are matched on *pitch* as well as shape - two
NHWC tensors of equal logical shape can still store their pixels at different
strides, and a flat op over them silently mixes pixels (a real bug during
bring-up, found by the per-node check).

**The missing ONNX nodes** (`onnxrun.py`):

| node | implementation |
|---|---|
| `Split` (axis 1, 4-D) | host byte gather, keeps the source pitch |
| `Concat` axis 1 (4-D) and axis 1/-1 (3-D) | host byte concat |
| `Slice` | host gather, int64 starts/ends/axes/steps |
| `Div` | GPU `Mul` by the reciprocal of a *graph constant* (folded) |
| `Sub` | GPU `Mul(-1)` + GPU `Add`; `const - x` and `x - const` likewise |
| broadcast `Add`/`Mul` | constant operand expanded on the host (bytes), GPU op |
| `Softmax` any axis | host permute to the last axis, `vc4_softmax_fwd` (<= 16 classes, rows padded to 8), permute back - this is the DFL softmax on axis 1 |
| `Transpose` | host permute |
| `Shape`, `Gather` | int64 host tensors (static shapes) |
| `Squeeze`, `Unsqueeze`, `Cast`, `Flatten` | view / type change |

`Shape`/`Gather` and the int64 arithmetic of the head's dynamic `Reshape` are
the only things that compute on the ARM, and they compute *integers* (shape
bookkeeping), never activation values.

**A SiLU rewrite (the biggest single win).**  Every YOLO activation is two
ONNX nodes, `Sigmoid` then `Mul(x, sigmoid)`, which cost two elementwise
calls, two full passes over the tensor and two daemon round trips each - 949
of the 1145 launches at 160x160.  `Runner._find_silu` now finds that pattern
(only when the sigmoid output has exactly one consumer and is not a graph
output), skips the `Sigmoid` node and turns the `Mul` into one
`vc4_elementwise(VC4_OP_SILU)` launch: same `2^f` polynomial and 3x Newton
reciprocal as the sigmoid path, with the multiply folded into the same 16-vector
body.  57 of the 58 sigmoids fuse (the 58th is the head's confidence
`sigmoid`, whose consumer is a `Concat`); launches drop 1145 -> 678.
`VC4_NO_SILU_FUSE=1` restores the two-node path for an A/B.

The fused kernel's final multiply is a single `fmul` (the QPU truncates),
where the two-node path used the engine's correctly-rounded `mul`; the two
paths' final tensors differ by at most 4.7e-4 (mean 9.5e-7) and both are
inside the same tolerance (7.4).

**A graph memory planner.**  `libvc4infer` has no `free()`, so the runner
carves every activation out of one pool allocation and recycles it by ONNX
liveness (last consumer).  YOLOv8n's peak pool use at 160x160 is **26.6 MiB**
of a 40 MiB pool (12.3 MiB of that is the weights) instead of growing past the
64 MiB arena.

### 7.2 Engine bugs found and fixed on the way

* `vc4_conv_nhwc_pack_w` builds the B panel with **CPU writes** (a zero fill,
  then the bias rows).  Through vc4d a CPU-written persistent buffer must be
  flushed to DRAM before the QPUs read it; the launch path only tracks the
  per-op scratch.  Without the two `vc4__mark_dirty()` calls (one before the
  pack kernel, one after the bias fill, which happens *after* that launch) the
  GEMM read a stale panel: whole output-channel blocks lost their weights or
  their bias, data-dependently.  This was the last correctness bug.
* Kernel/program tables: a 263-node model needs ~100 distinct QPU blobs (one
  GEMM per conv shape), but the client table held 32 and vc4d held 64
  programs.  Both are 128 now (`engine/src/vc4_int.h`, `vc4d/src/vc4d.[ch]`,
  code slot 16 KiB, client data base 3 MiB).
* SPPF `MaxPool` pads with **-inf** in ONNX; the NHWC pool kernel pads with
  zero, which biases every border window that is entirely negative (SiLU
  outputs are).  Measured as a 0.11 absolute error.  The border is now
  materialised as -inf on the host (a constant fill, no arithmetic) and the
  kernel runs with pad 0.

### 7.3 Measured result (stock clocks, `VC4INFER_DAEMON=1`)

Input **1x3x160x160**, output **1x84x525** (P3 400 + P4 100 + P5 25 anchors),
263 ONNX nodes -> **206 dispatches** after the SiLU rewrite.

| | fused (default) | `VC4_NO_SILU_FUSE=1` |
|---|---:|---:|
| runs, paired A/B, 8 each, alternating | 619.2 - 628.6 ms | 708.8 - 743.8 ms |
| **median / mean** | **622.7 / 623.2 ms = 1.61 FPS** | 722.0 / 721.6 ms = 1.39 FPS |
| launches/image | **678** | 1145 |
| GPU time (pack + kernel) | **242.5 ms** (39 % of wall) | 278.0 ms (39 %) |
| node wall sum | 588.1 ms | 662.4 ms |
| arena / graph pool / peak | 64.0 / 40.0 / 26.6 MiB | same |
| elements over 5e-4 | 0 / 44100 | 0 / 44100 |

`onnx/ab160.sh` runs that pairing (it alternates the modes because this board's
wall time drifts a few percent between sessions - the same source of variance
`docs/CLOCK-VARIANCE.md` documents; core 400 MHz / V3D 300 MHz /
`get_throttled=0x0` in every run, and one session measured while another
agent's build was running came out at 711 ms for the same 678-launch graph).
Warm-up (JIT of ~100 kernels) is 900-1050 ms and is not counted.
`onnx/yolo160_bench.json` is one clean 3-run session of the fused graph on its
own (615.1 / 622.4 / 623.8 ms, 620.4 ms mean, 1.61 FPS, 678 launches, GPU
242.6 ms, 0/44100 at 5e-4).

Per op type (wall ms, GPU ms, launches):

| op | nodes | wall | gpu | launches |
|---|---:|---:|---:|---:|
| Conv | 64 | 413.8 | 201.9 | 128 |
| Mul (SiLU, fused) | 60 | 82.1 | 33.5 | 468 |
| Concat | 17 | 28.8 | 0 | 0 |
| Split | 8 | 12.0 | 0 | 0 |
| Reshape | 8 | 8.8 | 0 | 0 |
| Add | 9 | 7.7 | 2.7 | 38 |
| MaxPool | 3 | 7.6 | 1.7 | 3 |
| Softmax (DFL) | 1 | 7.6 | 1.6 | 22 |
| Constant | 24 | 6.4 | 0 | 0 |
| Resize | 2 | 3.7 | 0 | 0 |
| Sigmoid (head conf) | 1 | 2.9 | 1.0 | 14 |
| Sub | 2 | 2.3 | 0.3 | 4 |
| Transpose | 1 | 1.4 | 0 | 0 |
| Slice | 2 | 1.1 | 0 | 0 |
| Div | 2 | 0.8 | 0.1 | 1 |
| Shape, Gather | 2 | 0.3 | 0 | 0 |

The wall is now **39 % GPU**.  413.8 - 201.9 = 212 ms of the remaining wall is
inside the conv nodes and is not arithmetic; a traced build splits it into
pad-copy 0.4-4.0 ms (the stem's 1.6 MB NCHW->NHWC zero-padded copy is the 4 ms
one), the im2col launch 0.7-7.0 ms and the GEMM launch 1.0-3.5 ms per conv.
Each of those launches costs ~0.3-1 ms more wall than its GPU time (daemon
round trip + the ARM-side dirty-range clean), against ~0.1 ms for an
elementwise launch.  The other 468 elementwise launches cost 82.1 - 33.5 =
48.6 ms of wall, i.e. 0.10 ms of overhead each - already at the round-trip
floor.

### 7.4 Verification

Two independent CPU references evaluate the **same ONNX graph** on the Pi:

* `onnx/ref_onnx.py`, a numpy float32 evaluator with float64 accumulation
  (`make_refs.py` produces the input and this reference);
* the torch 2.15 eager forward of the same `yolov8n.pt` (`torch_ref.py`).

They agree with each other to **3.05e-4** absolute on this input.  Tolerance
`|got-ref| <= atol + rtol*max(|ref|,1)`, VC4 vs each reference:

| atol/rtol | vs numpy ref | vs torch ref | max abs | max rel |
|---|---:|---:|---:|---:|
| 1e-4 | 65 / 44100 (0.15 %) | 66 / 44100 (0.15 %) | 0.0204 | 2.2e-4 |
| **5e-4** | **0 / 44100** | **0 / 44100** | 0.0204 | 2.2e-4 |
| 1e-3 | 0 / 44100 | 0 / 44100 | 0.0204 | 2.2e-4 |

The residual is fp32 accumulation through 64 conv layers plus the 16-bin DFL
softmax weighted sum (the rows over 1e-4 are the four DFL box rows), not a
structural error.  With the SiLU rewrite disabled the same run gives 0.0201
max abs, 0/44100 at 5e-4: the rewrite is not what moves the error.

`onnx/yolo_bench.py` reproduces all of this and writes the JSON report
(`--atol/--rtol` select the tolerance; `onnx/yolo160_bench.json` is the
committed report).

### 7.5 320x320 fits with a bigger arena: 1419 ms/image

The 320x320 export (`1x84x2100`) needs a padded 6.6 MB stem plane, a 16.4 MB
im2col panel and ~4x the 160x160 activation working set, so it does not fit
the default 72 MiB daemon arena.  It does fit when the daemon serves 192 MiB
and the client maps it (`VC4INFER_ARENA_MB`, new), but the daemon can only
open that arena on a board where CMA is otherwise idle: `arena_open` tries
192 MiB, and with another 96 MiB arena still draining it silently falls back
to 96 MiB (`shrink[] = {1,1,1,2,4,8}`), which is *not* enough (`pool 69 MiB +
23 MiB scratch > 90 MiB usable`).  On a clean boot it opens (`CmaFree` 126 MiB
-> the 192 MiB heap plus 1 MiB left).

```
sudo systemctl stop vc4d
sudo /usr/local/sbin/vc4d --arena-mb 192 ... &
VC4_POOL_RESERVE_MB=40 VC4INFER_ARENA_MB=189 VC4INFER_DAEMON=1 \
  python3 yolo_bench.py models/yolov8n_320.onnx --input /tmp/yolo320_in.npy \
  --ref /tmp/yolo320_ref.npy --atol 5e-4 --rtol 5e-4
```

| | 160x160 | 320x320 |
|---|---:|---:|
| runs | 619.2 - 628.6 ms (paired, 8) | 1427.4 / 1409.8 ms |
| mean | 623.2 ms = 1.61 FPS | **1418.6 ms = 0.70 FPS** |
| launches | 678 | 1824 |
| GPU (pack + kernel) | 242.5 ms (39 %) | 598.6 ms (42 %) |
| arena / pool / peak | 64.0 / 40.0 / 26.6 MiB | 189.0 / 149.0 / 31.1 MiB |
| elements over tol at 5e-4 | 0 / 44100 | 1 / 176400 (max rel 5.07e-4) |
| max abs | 0.0204 | 0.0440 |

The single element over 5e-4 is a DFL box coordinate at 5.07e-4 relative; at
1e-3 it is 0/176400.  `VC4_POOL_RESERVE_MB=40` is needed because the default
24 MiB reserve leaves the stem's 23 MiB of per-conv scratch without room.

### 7.6 What did *not* work (measured)

**A conv epilogue is the obvious next step, and it does not fit the register
file.**  `build_gemm` (the 16x16 C-tile GEMM every conv uses) allocates all 32
A-file registers (r0-r15 = the C tile, r16-r31 = panel/C/B addresses, counters
and the k counter) and all 16 B-file registers (strides, lane bases, the `-1`
and `-Nqpu` constants).  The SiLU body needs 6 temporaries (t/f, n/2^n, the Horner
accumulator, the reciprocal, the x copy) and 12 constants; after the k loop
only r19, r27 and r31 are dead, so a store epilogue would have to either
re-allocate the whole verified kernel or round-trip C through VPM to free
r0-r15.  Both are kernel rewrites, not an edit, so the fusion was done in the
graph instead.  (A second option - fusing SiLU into the *im2col* of the next
conv - recomputes it once per tap: 9x the arithmetic at 3x3.)

**Batching elementwise groups per QPU is not reliable on this hardware.**  The
cheap way to cut the remaining 468 SiLU launches is to let each QPU run several
256-float groups in one program (the kernel already has the counter loop and
the round-robin addressing).  Measured, twice, on a stock board:

| groups/QPU | add, mul, relu | sigmoid, silu |
|---|---|---|
| 1 (status quo) | 5/5 clean, bit-exact | 5/5 clean |
| 2 | `add` corrupts the 2nd group of every QPU on the first launch (2048/4096 wrong, 1e37 garbage); later reps clean | V3D context timeout -> daemon dead, board needs a reboot |

Adding the VPM *read* wait (A-file r50, `vpm_ld_wait`) before the store did not
change either symptom; the store already issues the VPM *store* wait
(B-file r50) after every chunk.  So `docs/NUMERICS.md`'s "one group per QPU per
launch" stands, and the wedge left the V3D queue dead until the board was
rebooted.  This is why the launch count stops at 678 instead of ~200: the
remaining launches are SiLU passes over the activations, ~8 launches per
1.6 M-float pass at the kernel's 3072 floats/launch.

**The dirty-range clean is not the conv bottleneck.**  Building the launch path
with the per-op scratch clean disabled (`VC4INFER_NO_SCRATCH_CLEAN=1`) only
moved the 160x160 wall from 635 to 620 ms - and broke the output (25798/44100
wrong), which is why the blanket clean stays.

### 7.7 Remaining gaps

* Conv: 3x3/1x1 only, stride 1/2, symmetric pads, group 1 or depthwise
  (depthwise as a block-diagonal dense weight).  No dilation, no arbitrary
  kernels, no `ConvTranspose`.
* `Softmax` needs <= 16 classes (the DFL fits exactly); `Resize` is nearest
  only; `Slice`/`Concat`/`Split`/`Transpose`/`Reshape` are host byte work.
* batch 1, fp32 only, static shapes.
* 678 launches/image at 160x160 (was 1145).  The remaining ones are 128 conv
  (im2col + GEMM per node) and 468 SiLU, both limited by the "one group per QPU
  per launch" elementwise shape and by the two-launch conv.
* The verified run above uses `VC4INFER_DAEMON=1`.  Direct mode
  (`VC4INFER_DAEMON=0`, root, under `/tmp/vc4-gpu.lock`) gives the identical
  output, so the result is not mode-specific.

## 8. Limitations that matter for real models

* **Plane pitch limit.**  `(H+2p)*align16(W+2p)*4 <= 8191` bytes.  At
  640x640 the first YOLO conv needs 1.68 MB; at 64x64 it still needs 21 KB.
  Only feature maps up to roughly 32x32 fit the current im2col VDR pitch
  field.  Removing this is the single biggest blocker for real CNNs
  (channel tiling or an NHWC repack).
* **fp32 only**, no quantization, no dynamic shapes, batch 1.
* **Softmax** is limited to 16 classes by the fused kernel's 16-lane rows.
* **Concat/Resize** are host byte copies, so they cost a round trip but no
  arithmetic.  A real runner would do them in the arena (a memcpy kernel).
* **Launch overhead.**  Small layers are launch-bound: the elementwise path
  issues one launch per 12 groups of 256 floats, so a 32x32x32 ReLU is 11
  launches (~0.8 ms GPU, ~2 ms wall), and after the SiLU rewrite the 468
  remaining SiLU launches are 48.6 ms of wall for 33.5 ms of GPU.  The fix is
  more groups per QPU per launch, which this board's VPM/DMA path does not
  survive (7.6).
* The runner re-executes the graph from scratch each call; weights are
  uploaded once per process but there is no graph-level memory planner.
