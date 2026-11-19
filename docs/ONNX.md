# ONNX on the VideoCore IV engine

`libvc4infer` is an **operator library**, not a graph runtime.  This document
describes the graph path added on top of it: a small ONNX exporter/runner in
`infer/tools/onnx/`, the two engine ops it needed (`vc4_maxpooln`, `vc4_softmax_fwd`), the
measured end-to-end numbers, and the honest coverage gap for YOLOv8n.

In **strict mode** (the default) everything that *computes* runs on the QPU.
The host only moves bytes (padding, channel gathers, concat, nearest resize,
boundary compaction) — the same class of work the engine already does for its
im2col layout.  An op or attribute the engine cannot dispatch raises an error
naming the node and its ONNX op type; there is no CPU math fallback.  The
opt-in `mode="hybrid"` adds explicit host tiers; section 10 describes the
trade-off, and strict remains the default.

Hardware: Raspberry Pi 3B, BCM2837, VideoCore IV @ **stock** clocks
(core 400 MHz / V3D 300 MHz), kernel 6.18, engine through **vc4d**
(`VC4INFER_DAEMON=1`, unprivileged, one GPU user at a time).

Test date: 2026-11-19.  Engine tree: `infer/` at this commit.

## 1. Files

```
infer/tools/onnx/export_models.py      define + torch.onnx.export the fixed-shape
                        models, save models/<name>.onnx, .in.npy and the torch
                        CPU reference .out.npy
infer/tools/onnx/onnxrun.py            the ONNX executor (onnx + ctypes binding)
infer/tools/onnx/verify.py             3 runs per model, compares with the torch
                        reference, prints per-node wall/GPU time and launches
infer/tools/onnx/export_yolo.py        real pretrained YOLOv8n -> ONNX (ultralytics)
infer/tools/onnx/yolo_report.py        static op/attribute/shape coverage + a run
infer/tools/onnx/tvshim/               name-only torchvision stand-in (YOLO export)
infer/tools/onnx/ref_onnx.py           independent numpy ONNX evaluator (CPU reference)
infer/tools/onnx/yolo_bench.py         end-to-end benchmark + tolerance + JSON report
infer/tools/onnx/make_extra_models.py  dilated/grouped/batch/depthwise/stride-2
                        graphs used by the §8 verification
infer/tools/onnx/verify_extra.py       5 runs per graph vs ref_onnx, per-node time + MACs
                        (includes the bilinear Resize graph)
```

Generated artifacts live in `infer/tools/onnx/models/` (not committed; regenerate with
`export_models.py`).  The engine additions are:

```
infer/src/vc4_nhwc.c           vc4_conv_nhwc/pack_w  NHWC channel-tiled conv (+ pool/up2/copy)
training/src/vc4_convbwd.c     vc4_maxpooln          generic NxN / stride-S max pool
training/src/vc4_train.c       vc4_softmax_fwd       forward softmax (fused 'p' probe)
infer/tools/kgen.py            build_pooln           the generic pool kernel
infer/tests/test_onnxops.c     CPU-oracle check of both
infer/python/vc4infer/_binding.py ctypes binding for both ops, vc4_get_stats, and a conv2d argtypes fix
```

## 2. Running a model

```sh
cd vc4-stack
source /home/raspi/work/torch-vc4/.venv/bin/activate   # torch CPU + numpy + onnx
export VC4INFER_CACHE=$HOME/.cache/vc4infer/kcache      # writable JIT cache

python infer/tools/onnx/export_models.py
VC4INFER_DAEMON=1 python infer/tools/onnx/verify.py                # all three models
VC4INFER_DAEMON=1 python infer/tools/onnx/onnxrun.py infer/tools/onnx/models/modelB.onnx \
        --input infer/tools/onnx/models/modelB.in.npy --profile --repeat 3

# optional real YOLOv8n export + coverage report (downloads ~6 MB of weights)
python infer/tools/onnx/export_yolo.py                             # YOLO_IMGSZ=32 for 32x32
VC4INFER_DAEMON=1 python infer/tools/onnx/yolo_report.py infer/tools/onnx/models/yolov8n_640.onnx
```

The runner takes the GPU lock through vc4d; do **not** hold
`/tmp/vc4-gpu.lock` in the calling shell in daemon mode (vc4d takes it per
launch, and a second holder makes every launch time out).

`VC4INFER_CACHE` matters: the C library's built-in cache path is the installed
`/usr/lib/vc4infer/kcache`, which is root-owned once the `.deb` is installed, so
the kernel JIT cannot write there as an ordinary user.  The blocks below set a
per-user cache directory for that reason.

`onnxrun.py` accepts fixed-shape fp32 models with batch 1 and one float input.
It evaluates the graph in ONNX topological order and keeps every activation in
the single 64 MiB arena.

## 3. Supported nodes

| ONNX node | implementation | constraints |
|---|---|---|
| `Conv` (NHWC, default) | `vc4_conv_nhwc` / `_dil` / `_group` + `pack_w` | KH,KW in {1,3,5}, stride 1/2, dilation 1..8, symmetric pads; `Cin%16`, `K=Cin*KH*KW %16`; grouped per-group Cin/Cout/K %16; `stride*Cin*4 <= 8191`; batch N; **no plane-size limit** |
| `Conv` depthwise (groups = Cin = Cout) | block-diagonal dense weight + `vc4_conv_nhwc` | Cout = Cin, any kernel/stride the dense path takes; **Cin x the MACs** (native kernel not shipped, §8.5) |
| `Conv` (NCHW, legacy fallback) | `vc4_conv2d` (QPU im2col + GEMM) | non-square kernels/strides; stride 1, group 1, batch 1, dilation 1; `(H+2p)*align16(W+2p)*4 <= 8191` |
| `Relu`, `Sigmoid` | `vc4_elementwise` | any layout |
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
| `Resize` linear | 4 host gathers + GPU `Mul`/`Add` | `mode=linear`, `asymmetric` or `half_pixel`; the ARM gathers the four neighbours and builds the coordinate weights, the activation multiplies/adds run on the QPUs |
| `MatMul` | `vc4_matmul` | 2-D only |
| `Gemm` | `vc4_matmul` + GPU `Add` for bias | `transA=0`, `alpha=1`, `beta∈{0,1}`; C of `[N]`, `[1,N]` or `[M,N]` |
| `Softmax` | `vc4_softmax_fwd` (new) | last axis, ≤16 classes, rows padded to 8 |
| `Flatten`, `Reshape`, `Identity` | view (no copy) | element count must match |
| `GlobalAveragePool` | `vc4_matmul` with a `1/HW` column | NCHW, batch 1 |

Everything else fails loudly in strict mode, e.g.:

```
onnxrun.UnsupportedNode: node '/model.9/Split' (Split): unsupported op Split
```

Inputs/outputs must be fp32.  `Expand`, `BatchNormalization`, `Pow`, `LRN`,
arbitrary `ConvTranspose` is not implemented; bilinear `Resize` is a
GPU-MUL/ADD composition (§8.6);
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
* The Python binding (`infer/python/vc4infer/_binding.py`, formerly `python/vc4infer.py`)
  gained `maxpooln`, `softmax_fwd`, `stats`, and a corrected `vc4_conv2d`
  ctypes signature (the old one had the wrong number
  and order of argument types, so `conv2d` could not be called from Python).
* The op table now lists `maxpool/NxNsS` as supported.

## 6. Why the NCHW path decomposes stride 2 and depthwise

**Status note:** the default NHWC path no longer decomposes stride 2 - the
im2col gather decimates on the QPU (§8.1).  Depthwise is still the dense
expansion described here.  The text below documents the NCHW fallback, which
is only reached for shapes the NHWC kernels do not take.

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
with `infer/tools/onnx/export_yolo.py` at **1x3x160x160** (opset 13, 12.4 MiB, 263 nodes)
now runs to completion on the QPUs and matches a CPU reference.

```
source /home/raspi/work/torch-vc4/.venv/bin/activate
cd vc4-stack/infer/tools/onnx
export VC4INFER_CACHE=$HOME/.cache/vc4infer/kcache
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
  programs.  Both are 128 now (`infer/src/vc4_int.h`, `vc4d/src/vc4d.[ch]`,
  code slot 16 KiB, client data base 3 MiB).
* SPPF `MaxPool` pads with **-inf** in ONNX; the NHWC pool kernel pads with
  zero, which biases every border window that is entirely negative (SiLU
  outputs are).  Measured as a 0.11 absolute error.  The border is now
  materialised as -inf on the host (a constant fill, no arithmetic) and the
  kernel runs with pad 0.

### 7.3 Measured result (stock clocks, `VC4INFER_DAEMON=1`)

Input **1x3x160x160**, output **1x84x525** (P3 400 + P4 100 + P5 25 anchors),
263 nodes.

| | |
|---|---:|
| warm-up run (JIT of ~100 kernels) | 900 ms |
| runs | 710.1 / 722.0 / 728.5 ms |
| **mean** | **720.2 ms/image = 1.39 FPS** (best 710.1 ms = 1.41 FPS) |
| spread over 9 runs (3 sessions) | 708.6 - 728.5 ms (1.37 - 1.41 FPS) |
| launches/image | 1145 |
| GPU time (pack + kernel) | 279.8 ms (39% of wall) |
| node wall sum | 682.2 ms |
| arena / graph pool / peak | 64.0 / 40.0 / 26.6 MiB |

(JSON: `infer/tools/onnx/yolo160_bench.json`.  The per-op table below is from an earlier
equivalent run, 663.2 ms node wall sum at 277.3 ms GPU.)

Per op type (wall ms, GPU ms, launches):

| op | nodes | wall | gpu | launches |
|---|---:|---:|---:|---:|
| Conv | 64 | 408.4 | 201.6 | 128 |
| Sigmoid | 58 | 84.8 | 34.5 | 481 |
| Mul (SiLU) | 60 | 82.8 | 34.8 | 468 |
| Concat | 17 | 28.4 | 0 | 0 |
| Split | 8 | 11.5 | 0 | 0 |
| Reshape | 8 | 8.2 | 0 | 0 |
| Add | 9 | 7.8 | 2.7 | 38 |
| MaxPool | 3 | 7.7 | 1.7 | 3 |
| Softmax (DFL) | 1 | 7.6 | 1.6 | 22 |
| Constant | 24 | 6.3 | 0 | 0 |
| Resize | 2 | 3.8 | 0 | 0 |
| Sub | 2 | 2.4 | 0.3 | 4 |
| Transpose | 1 | 1.5 | 0 | 0 |
| Slice | 2 | 1.0 | 0 | 0 |
| Div | 2 | 0.8 | 0.1 | 1 |
| Shape, Gather | 2 | 0.3 | 0 | 0 |

The heaviest single nodes are the stem (31.4 ms wall / 8.0 ms GPU) and the
neck's 3x3 stride-2 convs (10-17 ms).  The elementwise SiLU pair costs 949 of
the 1145 launches: `vc4_elementwise` issues one launch per 12 groups of 256
floats, so a 20x20x128 SiLU is ~7 launches.  Batching those (or fusing
Sigmoid+Mul into the engine's existing `OP_SILU`) is the cheapest win.

### 7.4 Verification

Reference: the **same ONNX graph** evaluated on the Pi by `infer/tools/onnx/ref_onnx.py`,
an independent numpy float32 evaluator (float64 accumulation), cross-checked
against the torch CPU export-mode forward of the same `yolov8n.pt` (they agree
to 4.3e-4 absolute on this input).

Tolerance `|got-ref| <= atol + rtol*max(|ref|,1)`:

| atol/rtol | elements over tolerance | max abs | max rel |
|---|---:|---:|---:|
| 1e-4 | **99 / 44100 (0.22%)** | 0.0140 | 3.6e-4 |
| 5e-4 | **0 / 44100** | 0.0140 | 3.6e-4 |
| 1e-3 | 0 / 44100 | 0.0140 | 3.6e-4 |

All 99 elements at 1e-4 are in the four **DFL box** rows; the 80 class rows
(42000 elements) are inside 1e-4 everywhere.  The residual is fp32
accumulation through 64 conv layers plus the 16-bin DFL softmax weighted sum,
not a structural error.  The per-node check (each node evaluated from the
VC4's *own* inputs) is clean to <= 1e-4 relative everywhere.

`infer/tools/onnx/yolo_bench.py` reproduces all of this and writes the JSON report.

### 7.5 320x320: the memory wall this milestone removed

`yolov8n_320.onnx` (1x84x2100) exports fine and every node passes the
runner's checks, but the first conv already needs a 6.6 MB padded plane plus a
16.4 MB im2col panel of scratch, and the activation working set is ~4x the
160x160 one:

* 160x160: pool peak **26.6 MiB**, of which 12.3 MiB weights -> ~14 MiB of
  activations;
* 320x320: ~4x that -> ~56 MiB activations + 12.3 MiB weights = **~68 MiB**
  pool, plus >= 23 MiB of per-conv scratch = **~91 MiB**.

vc4d serves a 72 MiB arena (`--arena-mb 72`; the client maps 64 MiB), so
320x320 stops with `conv_nhwc: out of arena memory` on the first conv.  It
would fit with `--arena-mb 192` and a client that maps it (CMA has 256 MiB
total), a daemon restart away, but that was not needed for this milestone.
This is a memory limit, not a pitch or op limit.  **Resolution (§8.3):** the
client region is now settable with `VC4INFER_ARENA_MB` and the pool reserve
with `VC4_POOL_RESERVE_MB`; a 150 MiB vc4d arena plus a 145 MiB client region
(113 MiB pool, 33.4 MiB peak) runs 320x320 to completion with no kernel
change.

### 7.6 Remaining gaps (as of the 160x160 commit)

* Conv: stride/dilation/group coverage is described in §8; the native
  depthwise kernel is not shipped (dense expansion remains).
* `Softmax` needs <= 16 classes (the DFL fits exactly); `Resize` is nearest
  only; `Slice`/`Concat`/`Split`/`Transpose`/`Reshape` are host byte work.
* batch 1 at this commit (batch N landed in §8); fp32 only, static shapes.
* 1145 launches/image at 160x160: the wall/GPU ratio (2.4x) is launch and
  host-copy overhead, not arithmetic.
* The verified run above uses `VC4INFER_DAEMON=1`.  Direct mode
  (`VC4INFER_DAEMON=0`, root, under `/tmp/vc4-gpu.lock`) gives the identical
  output (same 99/44100, same max abs), so the result is not mode-specific.

## 8. Native conv variants, batch N and 320x320

This milestone removed three graph-level rewrites and one memory ceiling.  All
of it is still strict: an unsupported combination raises `UnsupportedNode`
with the node name, and every activation MAC runs on a QPU.

### 8.1 What changed in the engine

| capability | engine entry point | implementation |
|---|---|---|
| stride 2 (native) | `vc4_conv_nhwc` / `_dil` | the im2col VDR reads 16 **output** pixels whose sources are `stride` apart (`pitch = stride*Cin*4`), and the row advance is `stride` input rows.  No phase images, no extra MACs. |
| dilation | `vc4_conv_nhwc_dil` | `im2col_nhwc --dil d`: taps of one kernel row are `d` pixels apart and kernel rows `d` input rows apart.  Output `ho = (H+2p-d*(K-1)-1)/stride+1`. |
| grouped | `vc4_conv_nhwc_group` | one padded input, one activation panel reused by every group; each group runs im2col+GEMM and the GEMM writes **straight into its channel range** of the single `[M][Cout]` output.  The GEMM gained a C row stride (`vc4__gemm3_run_c`, kgen `gemm3 --cstride`).  Real grouped MACs. |
| batch N | all NHWC ops | `nimg` already existed in the kernels; the runner now allocates `N*M*Cout` and passes it.  Conv, pool, upsample, concat, elementwise and softmax share one launch across images. |
| bilinear `Resize` | runner (`_resize` + `_lerp4`) | four taps are host byte gathers, the four weights are host coordinate math, and `vc4_elementwise` MUL/ADD does every activation multiply/add on the QPUs (7 full-tensor passes). |
| 320x320 | runner + daemon | client region via `VC4INFER_ARENA_MB`, pool reserve via `VC4_POOL_RESERVE_MB`; the conv scratch (padded image + im2col panel) is 23 MiB at 320. |

Normative kernel constraints (all checked before a launch):

* `Cin % 16 == 0` for the whole conv (the runner zero-pads the first layer's 3
  channels and slices Cout back, so the DFL 1-channel conv is computed as 16).
* group 1: `Cout` is zero-padded to 16; `K = Cin*KH*KW` must be 16-aligned.
* grouped: `groups` must divide Cin and Cout, and **per group**
  `(Cin/groups) % 16 == 0`, `(Cout/groups) % 16 == 0`,
  `(Cin/groups*KH*KW) % 16 == 0`.  A group whose channel block is not a
  multiple of 16 raises; it is not silently padded (padding would mix groups).
* `stride in {1,2}`; `KH,KW in {1,3,5}`; `1 <= dilation <= 8`; symmetric pads
  only (`pads = [p,p,p,p]`).
* VDR pitch: `stride*Cin_total*4 <= 8191` bytes.  For grouped conv this is the
  **total** Cin (the group block is a contiguous sub-range of each pixel).
* output pitch `Wg = align16(wo)`; only columns `[0,wo)` are meaningful.
* depthwise (`group == Cin == Cout`) is still the verified block-diagonal
  dense expansion; it requires `Cout == Cin` (a channel multiplier raises).
  A native per-channel kernel was written and reached `st=0` on C=16/C=32
  standalone launches, but the full test still produced wrong planes and then
  poisoned the arena for later ops, so it is **not shipped** (see §8.5).

### 8.2 Measured (5 runs, numpy ONNX oracle)

`infer/tools/onnx/verify_extra.py` builds five fixed graphs with
`infer/tools/onnx/make_extra_models.py`, evaluates each with the independent numpy
evaluator (`ref_onnx.py`), then runs the ONNX graph on the QPUs five times.
Tolerance `|got-ref| <= 1e-4 + 1e-4*max(|ref|,1)`; **every** element is
checked.

| graph | op | output | mismatches | mean wall ms | GPU ms | launches | MACs now | MACs before | saved |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| batched | 2 images, Conv16→16 3x3 + ReLU + 1x1 | 2x16x20x20 | 0/12800 | 8.7 | 2.6 | 11 | 25.1 M | 25.1 M | launches: 2 images share 11 |
| grouped | Conv 64→64 g=4 3x3 | 1x64x16x16 | 0/16384 | 6.8 | 2.2 | 14 | 2.36 M | 9.44 M (dense) | 4x |
| dilated | Conv 16→16 3x3 d=2 p=2 | 1x16x24x24 | 0/9216 | 3.7 | 1.0 | 2 | 1.33 M | 5.31 M (dense d=1) | 4x |
| depthwise | Conv 32→32 g=32 3x3 | 1x32x16x16 | 0/8192 | 3.9 | 1.1 | 2 | 2.36 M | 2.36 M | 1x (dense kept) |
| stride-2 | Conv 32→32 3x3 s=2 p=1 | 1x32x16x16 | 0/8192 | 6.2 | 1.1 | 2 | 2.36 M | 9.44 M (4 phases) | 4x |
| bilinear | Conv 8→16 3x3 + Resize 2x linear | 1x16x24x24 | 0/9216 | 16.2 | 1.9 | 23 | 230 K (conv) + 64.5 K (lerp) | 0 (raised) | n/a |

The raw kernels have their own float64 CPU oracle in `infer/tests/test_nhwc.c`
(stride 1/2, batch 2, dilated 3x3 d2/d3, 5x5, grouped g2/g3/g4, depthwise):
`test_nhwc 5` reports **5/5 clean** for every case.  Kernel-level GPU times are
in §8.3.

### 8.3 320x320

The 160x160 peak was 26.6 MiB of pool (12.3 MiB of it weights) plus 6 MiB of
per-conv scratch.  320x320 is 4x the activations plus 23 MiB of scratch, so it
needs a bigger arena but **no kernel change**.  The client maps whatever the
daemon serves; the runner's reserve is configurable:

```sh
sudo systemctl stop vc4d
sudo /usr/local/sbin/vc4d --socket /run/vc4d.sock --arena-mb 150 \
     --group vc4 --drop-user nobody --max-timeout-ms 10000
cd vc4-stack/infer/tools/onnx
export VC4INFER_CACHE=$HOME/.cache/vc4infer/kcache
VC4INFER_ARENA_MB=145 VC4_POOL_RESERVE_MB=32 VC4INFER_DAEMON=1 \
  python3 yolo_bench.py yolov8n_320.onnx --input ... --ref ...
```

Measured (stock clocks, 145 MiB client region, 113 MiB graph pool, 33.4 MiB
peak; `|got-ref| <= 5e-4 + 5e-4*max(|ref|,1)`):

| | |
|---|---:|
| runs | 1593.8 / 1631.9 / 1482.7 ms |
| **mean** | **1569.5 ms/image = 0.64 FPS** (best 1482.7 = 0.67 FPS) |
| launches/image | 1824 |
| GPU time (pack + kernel) | 616.4 ms |
| arena / pool / peak | 145.0 / 113.0 / 33.4 MiB |
| accuracy | max abs 0.0440, max rel 5.07e-4, **1/176400** over 5e-4 |

The single element over 5e-4 is in the DFL box rows (the 80 class rows are
inside); it is the same fp32 accumulation floor the 160x160 run shows (65
elements over 1e-4, 0 over 5e-4).  Wall is dominated by 1824 daemon round trips and the host layout copies,
exactly as at 160x160.  For reference the same runner at 160x160 measures
**638.3 ms/image = 1.57 FPS** (best 617.9 = 1.62 FPS, 678 launches, GPU
243.4 ms, 64/40/27.8 MiB arena/pool/peak) with the same 12.3 MiB of weights:
4x the pixels costs 2.5x the wall (the GPU part scales ~2.5x, the launch
count 2.7x).

### 8.4 Constraint matrix and failure messages

Unsupported combinations raise before any launch.  Examples:

```
Conv /model.22/dfl/conv/Conv: NHWC needs Cin,K % 16 (16,16)   # K=16 OK
Conv x: grouped NHWC needs per-group Cin,Cout,K % 16 (8,16,72)
Conv x: groups 3 does not divide 64/64
Conv x: stride*Cin*4 = 12288 B > 8191 B
Conv x: non-square stride
MaxPool x: batch ...      # only the NCHW single-image op; NHWC is batched
GlobalAveragePool x: batch != 1
```

The NCHW legacy path (`vc4_conv2d`) is still batch 1 and dilation 1; it is
only reached for non-square kernels/strides, and says so when batch > 1.

### 8.5 What did not land

* **Native depthwise.**  A per-channel kernel (`kgen depthwise`) was generated
  and launched successfully on C=16 and C=32 (`st=0`, one launch, 637
  instructions), but the full `test_nhwc` run still compared a wrong output
  plane and then left later NHWC ops reading garbage, so it was removed from
  the library rather than shipped unverified.  Two findings are concrete and
  worth keeping: (1) the ALU has only **six readable accumulator registers**
  (`acc()` muxes 0..5); parking the per-block bias bus in "acc 9" silently
  aliased R1, the bias TMU read used a pixel as an address, and the QPU
  wedged; the bias must live in an A register (A15 was used); (2) the nine
  weight lines must be popped into the **B file** and copied to A16..A24 -
  an A-file pop into A27 followed by the next load wedged the TMU after the
  third load.  The verified dense block-diagonal expansion (Cin x the MACs)
  remains the dispatch.
* **On-device NMS.**  Not implemented.  The invariant that blocks the simple
  form is that greedy NMS's kept set is **data-dependent and sequential**:
  whether box j is suppressed depends on which earlier box survived, so a
  static one-task-per-box grid cannot do the suppression pass.  The two exact
  forms are (a) a persistent per-class QPU kernel with a serial loop over the
  kept boxes (a much larger piece of QPU assembly than any op here), or (b)
  materialise the N x N threshold matrix (17.6 MB at 2100 boxes) and run the
  greedy selection on the ARM - which still sorts the scores on the ARM and
  therefore breaks the strictness rule.  The runner has no CPU fallback, so a
  `NonMaxSuppression` node raises.  With no shipped GPU NMS there is no honest
  GPU-vs-host microbenchmark to report.

### 8.6 Bilinear resize (composition, no new kernel)

`mode=linear` (asymmetric or half_pixel) is implemented without a new QPU
kernel:

1. the host gathers the four neighbours (`X[y0,x0]`, `X[y0,x1]`, `X[y1,x0]`,
   `X[y1,x1]`) - byte movement only;
2. the host builds the four per-tap weight tensors
   `(1-wy)(1-wx), (1-wy)wx, wy(1-wx), wy*wx` from the static scale - this is
   coordinate math, never activation data;
3. seven `vc4_elementwise` passes (4 MUL + 3 ADD) do every activation
   multiply/add on the QPUs, with the intermediates recycled by the pool.

Measured on the `bilinear` graph (Conv 8->16 3x3 then a 2x linear resize of a
12x12 map): **5 runs, 0/9216 mismatches, max abs 5.25e-6**, 16.2 ms mean
wall, 1.9 ms GPU, 23 launches.  The working set is about 6x the output (four
taps + four weights + temporaries), so a large resize can exhaust the graph
pool; `Pool.alloc` raises `graph pool exhausted` with the exact byte count.
A dedicated bilinear kernel would cut both the launches and the memory; it is
the obvious next step.

## 9. Limitations that matter for real models

* **Plane pitch limit.**  `(H+2p)*align16(W+2p)*4 <= 8191` bytes.  At
  640x640 the first YOLO conv needs 1.68 MB; at 64x64 it still needs 21 KB.
  Only feature maps up to roughly 32x32 fit the current im2col VDR pitch
  field.  Removing this is the single biggest blocker for real CNNs
  (channel tiling or an NHWC repack).
* **fp32 only**, no quantization, no dynamic shapes.  Batch N is supported
  for conv/pool/elementwise/softmax graphs; `Gemm`/`GlobalAveragePool` remain
  batch 1 and say so.
* **Softmax** is limited to 16 classes by the fused kernel's 16-lane rows.
* **Concat/Resize** are host byte copies, so they cost a round trip but no
  arithmetic.  `Resize` is nearest or a Mul/Add composition for bilinear
  (§8.6); NMS is not shipped (§8.5).  A real runner would do nearest in the
  arena (a memcpy kernel).
* **Launch overhead.**  Small layers are launch-bound: the elementwise path
  issues one launch per 12 groups of 256 floats, so a 32x32x32 ReLU is 11
  launches (~0.8 ms GPU, ~2 ms wall).  Fusing or batching more groups per
  launch is the fix.
* The runner re-executes the graph from scratch each call; weights are
  uploaded once per process but there is no graph-level memory planner.

## 10. Host policy: the device invariant (default), `structural`, `any`

Strict mode is unchanged and remains the default (`Engine(path)`).  Hybrid mode
is opt-in (`mode="hybrid"`) and now defaults to `host_policy="device"`.

### 10.1 The invariant

Under `host_policy="device"` the host may do **only**:

* plan-time constant folding / shape resolution (`Shape`, `Size`, `Constant`,
  `ConstantOfShape`, `Gather` over shape tensors, and the integer shape
  arithmetic feeding them) - resolved when the plan is built, so they cost
  zero runtime work and never appear as executed nodes;
* orchestration: launch issuing, DMA/copy setup, daemon validation and memory
  allocation.

Every other node must execute on VC4.  There is no host escape:

* a numeric node with no kernel raises `HostNotAllowed(node, op)`;
* `If`/`Loop`/`Scan` raise `NeedsDeviceDispatch(node, op)` naming the missing
  device-side dispatch feature;
* `Engine.host_compute_nodes()` is the list of nodes that performed *runtime*
  CPU compute.  It is `[]` for a device-policy run and `plan()` asserts it.
* `Engine.profile()` reports `host_compute_nodes=0` explicitly.

"Slow on VC4" is not a reason to use the host.  Emulation on the QPU is
allowed; delegation is not.

### 10.2 Tiers

Every node is classified into exactly one tier, printed per node by
`vc4infer nodes -v` and returned by `Engine.node_backends()`:

| tier | members | device | structural | why |
|---|---|---|---|---|
| `vc4` | Conv, Relu, ... | runs | runs | the QPU path |
| `folded` | Shape/Size/Constant/ConstantOfShape and integer shape arithmetic | plan-time only | metadata (runtime) | resolved during planning; zero runtime cost |
| `host-meta` | metadata that could not be folded | **refused** (`HostNotAllowed`) | runs | runtime host metadata is still host compute |
| `host-control` | `If`, `Loop`, `Scan` | **refused** (`NeedsDeviceDispatch`) | runs | the executor is host-scheduled; the QPU has no device-side work queue |
| `host-unwritten` | `Erf`, `TopK`, `NonZero`, `Einsum`, `LSTM`, ... | **refused** (`HostNotAllowed`) | refused | a numeric operator with no VC4 kernel: DEBT |
| `unsupported-graph` | invalid domains/types, fp16/string tensors | refused | refused | cannot run on either backend |

`host-control` is architectural, not "slow": every branch/iteration decision
would be made by the ARM, and body tensors live in the caller's pool.  Making
it device-executable needs a persistent QPU kernel pulling work items
(including nested bodies) from a device-side queue so the QPU makes the branch
decision and the host is not in the loop per iteration.  That is a different
runtime, not another kernel.

`host_policy="structural"` is the previous behaviour: metadata and control flow
may run on the host.  `host_policy="any"` additionally runs `host-unwritten`
nodes in the numpy reference evaluator, for compatibility only.  Neither is a
performance mode, and both keep the debt visible.

### 10.3 Utilisation accounting

```sh
vc4infer util model.onnx [--imgsz N] [--reps N] [--json]
```

or `Engine.utilization(reps=3)`:

| field | meaning |
|---|---|
| `arithmetic_on_vc4` | share of the graph's static MACs executed by QPU nodes (Conv/Gemm/MatMul/GAP); 100% under `device` |
| `bytes_copied_by_host` | bytes of model/activation data the ARM moved (concat, resize, transpose, pad, layout copies), both directions, with a per-node/per-site breakdown |
| `qpu_duty` | QPU busy time / wall time from the engine's pack+kernel counters; also per node |
| `launches`, `launches_per_node` | launch count and launches per dispatched node |
| `launch_path_ms`, `launch_path_pct` | wall time minus QPU busy time, i.e. the ARM/daemon round-trip share |
| `host_compute_nodes` | runtime host-compute nodes; must be 0 under `device` |

Measured on YOLOv8n 160x160, stock clocks, `vc4infer util --reps 5` on a
quiet board:

| metric | value |
|---|---:|
| nodes | 233 vc4 + 30 folded |
| `host_compute_nodes` | **0** |
| `arithmetic_on_vc4` | **100.0%** (273,249,600 / 273,249,600 MACs) |
| `bytes_copied_by_host` | **0 B** (was 17,077,040; see 10.4) |
| `qpu_duty` | **58.9%** (320.4 / 543.5 ms) |
| launches | 784 (4.45 per dispatched node) |
| launch path | 223.1 ms (**41.1%** of wall) |

The duty cycle is limited by launches and the daemon round trip, **not by
arithmetic**: the graph is 100% QPU arithmetic but the QPU is idle while the
ARM issues the next launch.  The honest statement is "100% of MACs on VC4,
59% QPU duty at 160x160"; moving the last 17 MB of layout work onto the QPUs
moved 11 points of duty and 108 ms of wall time (10.4.3).

### 10.4 Host byte debt: what was eliminated

Per-run host byte movement on YOLOv8n 160x160:

| stage | bytes/run | note |
|---|---:|---|
| before this work | 38,410,632 | weight re-downloads + every layout copy |
| after caching weights/bias panels in host arrays | 17,077,040 | -21.3 MB/run |
| after `vc4_copy2d` is wired into every site | **0** | 17.1 MB moved by the QPUs instead |

For reference, the per-op debt that was eliminated (device policy, measured by
removing each site from `vc4infer util`):

| op | bytes/run removed | how |
|---|---:|---|
| Conv | 8,683,728 | 6,325,760 was `to_nhwc` downloading an NHWC tensor and discarding the copy; 1,945,600 is the first conv's NCHW 3ch -> NHWC 16ch prologue (one `copy2d_x`, the channel padding is the fill value); 412,368 is the DFL conv's `Cout=1` channel slice |
| Concat | 5,160,000 | pitched channel concat, one call per input, the block loop walks inputs wider than 16 channels per pixel |
| Split | 983,040 | channel-strided copy that keeps the source pitch (the old path also never charged its own 245 KB download) |
| Reshape | 826,080 | NHWC -> NCHW, compact and pitched forms, with the partial-word tail for 25/100/2100-wide extents |
| Resize | 471,040 | exact-2x NHWC `vc4_upsample2x_nhwc` |
| Softmax | 403,456 | the head's softmax consumes the transpose's NHWC output directly, so no layout copy is needed at all |
| Transpose | 268,800 | `[0,2,1,3]` through the block loop |
| MaxPool | 247,296 | `vc4_fill2d(-inf)` border + interior copy |
| Slice | 25,200 | last-axis runs |
| Mul | 8,400 | broadcast constant expanded once at plan time, cached like a weight panel |

#### 10.4.1 `vc4_copy2d`: the layout copy kernel (landed, verified)

The single primitive the debt needed is in the engine:
`vc4_copy2d` / `vc4_copy2d_n` / `vc4_copy2d_x` (`infer/src/vc4_gen.c`,
generated by `kops.build_copy2d`).  It is a 3-level affine copy:

```
row   (vpr vectors, `units` vectors per DMA store, `depth` words per vector)
  block (nblk blocks, sbstep/dbstep bytes apart in the two streams)
    vector: 16 consecutive destination words, source word per lane at
            src + lane*slane
```

so `slane = 4` is a contiguous copy and any other stride is a gather.  Lanes
`>= nlane` store the fill value (zero channel padding, the -inf MaxPool
border) and their source addresses are clamped to the vector base, so a
short row never reads past its buffer.  `units=1, depth=k` stores a k-word
tail exactly (the head's 525/2100/100-wide extents are not multiples of 16).
The VDW row stride and the chunk step are runtime registers, so one blob
serves every shape; the additional launch cost is reported in 10.4.3.

`infer/tests/test_copyops.c` (wired into `run_tests.sh`) verifies it against
independent CPU oracles written from the *semantics* of each layout op, and
every case is byte-exact (0 mismatches): contiguous 1050-float run (65.6
vectors, the head slice), pitched channel concat with W=5 and C=16+32+16 ->
C=64, pitched split 64ch -> 4x16ch, NHWC->NCHW for (16,5,5,pitch16),
(64,20,20,pitch20), (80,10,10,pitch16) and H=1/W=7, channel slice of a
[4][525] plane with 4-byte lanes, transpose [4][16][525] -> [16][4][525],
nearest 2x, the -inf MaxPool border, the first conv's NCHW 3ch -> NHWC 16ch
prologue and a 1 MiB bulk copy (1 MiB in 2.31 ms, 454 MB/s with units=4).

#### 10.4.2 The cacheable arena mapping, and the bug it caused

This is the mechanism behind every nondeterministic YOLOv8n output the layout
work produced, and it is worth stating on its own because it is not a layout
bug: **the client's arena mapping is cacheable**.  `vc4d` maps the arena
through `/dev/mem` with `O_SYNC` (uncached) and hands the client a dma-buf
from `/dev/dma_heap/linux,cma`; the client's mmmap of that dma-buf is a normal
write-back mapping (`vc4_dev_remote.c:66`).  Two host access patterns were
missing maintenance:

* **host write into a line the QPU has written.**  The clean that precedes
  every launch writes back *whole* cache lines, so the words of the line the
  host did not touch reverted to the copy the ARM had cached before the
  launch.  This needs no host read at all: a constant uploaded into a recycled
  pool offset is enough.
* **host read of arena bytes the QPU wrote.**  `pad_stream`, the
  elementwise/softmax copies into scratch, the conv pad image and the weight
  panel reads all `memcpy`d straight out of the arena and could serve a clean
  stale line.

Both are fixed in the client (the daemon cannot maintain another process's
D-cache, and its own `/dev/mem` view needs nothing): `vc4__write` and
`vc4__remote_upload` invalidate the range before writing, every arena->host
copy invalidates the source first (`vc4__invalidate_off`), `vc4_download`
keeps its CIVAC, and `vc4_invalidate_buf()` is the public form for zero-copy
users (`Engine.view()`).  `infer/tests/test_coherency.c` (in `run_tests.sh`)
fails if any of them loses it.

Measured before the fix, on the board (`/tmp/vc4t/coh*.c`):

| test | result |
|---|---|
| ARM reads a range the QPU just wrote, no maintenance | 400-600 of 4096 words stale (the ARM's cached copy) |
| the same read after 2 MiB of unrelated churn | 0-432 stale -> **it is the ARM D-cache**, not a late V3D L2 write-back |
| the same read after `DC CIVAC` | 0 stale |
| one host word + the pre-launch clean, then the QPU reads the line | the other 15 words came back as the *pre-QPU* values |

#### 10.4.3 Measured result of the device-side layout copies

YOLOv8n 1x3x160x160, stock clocks (core 400 MHz, V3D 300 MHz, `throttled=0`),
`VC4INFER_DAEMON=1`, board otherwise quiet (load 0.7-0.9, no other GPU user),
`vc4infer util --reps 5`:

| metric | before (main) | wired |
|---|---:|---:|
| `bytes_copied_by_host` | 17,077,040 | **0** |
| `host_compute_nodes` | 0 | **0** |
| `arithmetic_on_vc4` | 100.0% | **100.0%** |
| wall / FPS | 651.1 ms / 1.54 | **543.5 ms / 1.84** |
| `qpu_duty` | 46.6% (303.1 ms) | **58.9%** (320.4 ms) |
| launches | 678 | 784 |
| launch path | 348.0 ms (53.4%) | **223.1 ms (41.1%)** |

The device copies are not slower: they add 17.3 ms of QPU kernel time for
17.1 MB (~1.1 GB/s of copy throughput) and give back 125 ms of ARM/daemon
time, so the wall time improves by 108 ms (-17%) and FPS by 20%.

Output against `ref_onnx` (`|got-ref| <= 5e-4 + 5e-4*max(|ref|,1)`):
**0/44100**, max abs 0.0204, in **20/20 fresh processes and 20/20 repeats in
one process** (`/tmp/vc4t/det20.py`).  Before the coherency fix the same model
reported 12-80 elements over tolerance on a first run and ~22,700 on repeats,
varying per process.

#### 10.4.4 What the device paths refuse (named, never a host fallback)

These raise `UnsupportedNode` with the invariant in the message; they are not
widened to make a byte count smaller:

| op | refused | invariant that blocks it |
|---|---|---|
| Resize | any nearest other than exact 2x NHWC | the source index of an output pixel is `floor(x*in/out)`, not affine in the vector index (a pixel repeats every `C/16` vectors while the row step is a different constant); it needs a per-output gather table or a `--dup N` kernel variant |
| Transpose | any perm other than `[0,2,1,3]` on a 4-D NCHW tensor | the affine copy needs the vector to be 16 consecutive words of one stream and a single (row, block) decomposition; a general permute needs a rank-generic index table |
| Slice | two sliced axes, or a sliced axis followed by a non-full axis | the output row must be one contiguous run of the source |
| Concat | axes other than 1 on 4-D NHWC / 1 and -1 on 3-D | the destination vector is a pixel's 16-channel block |
| Conv / to_nhwc | an NHWC tensor needing a *different* channel padding | the lane count is a uniform, not a permutation |
| Softmax | more than 16 classes | the fused kernel's 16-lane rows (engine limit, unchanged) |

### 10.5 The per-node report and boundary copies

`node_backends()` returns `[(name, op, backend, reason)]` for every node
(actual verdict after a run, static partition before it).  `host_debt()`
returns only `host-unwritten`.  In `structural`/`any` mode, crossing the
boundary is counted per run (`host_to_vc4` + `vc4_to_host` tensors, plus the
byte total above), and the summary prints the tier split and the host wall-time
share.  Hybrid timings are never presented as VC4-only numbers.

### 10.6 Tests

`infer/tests/test_hybrid.py` covers:

* device policy: `hybrid_meta` folds Shape/Gather/Slice/Mul/Concat and runs
  with `host_compute_nodes == []`, output matching a full
  `ReferenceEvaluator` run;
* `Erf` and the mixed Split/Einsum/TopK/NonZero/LSTM model raise
  `HostNotAllowed` under `device` and are listed by `vc4infer nodes
  --host-policy device` (exit 4);
* `If` raises `NeedsDeviceDispatch` under `device` and runs as `host-control`
  under `structural`;
* YOLOv8n: `host_compute_nodes == 0`, `arithmetic_on_vc4 == 100%`, identical
  output to strict, and the utilisation/byte debt report above.


## 10. Generic QPU kernels for the numeric op long tail (WIP)

This section documents `infer/src/vc4_gen.c` and
`infer/python/vc4infer/kgen/kops.py`: the first slice of the "every numeric op
must be a VC4 kernel" milestone.  It is **partial and honest**: the
infrastructure and a verified subset landed, the transcendentals do not yet
produce correct numbers on the board, and nothing unverified is wired into the
executor (an op without a verified kernel still raises — there is no CPU path).

### 10.1 What was added

| piece | file | state |
|---|---|---|
| `uop` kernel family (25 unary ops, one QPU body each) | `kops.py`, riding `kgen.build_ew`'s verified harness | 7 ops verified, 18 not correct yet |
| `sel3` (where / tensor clip / prelu) | `kops.py` + `vc4_select3` | generated, not yet element-verified |
| `gather` (table-driven QPU copy: `out[o] = arena[tab[o]]`) | `kops.py` + `vc4_gather` | generated, not yet element-verified |
| `blit` (contiguous 256 B/group QPU copy) | `kops.py` + `vc4_blit` | generated, not yet element-verified |
| `reduce` (sum/mean/max/min/argmax/argmin over a contiguous run) | `kops.py` + `vc4_reduce` | generated, not yet element-verified |
| `cumsum` (16-lane Hillis-Steele prefix, 256-column blocks) | `kops.py` + `vc4_cumsum` | generated, not yet element-verified |

`kgen.build_ew` gained an `nvec` parameter (16 vectors per group for the short
bodies, 4 for the long ones, because a 16-vector group of a transcendental body
overflows the 16 KiB code slot) and a dispatch: `op >= 100` routes the per-vector
body to `kops.ew_body`, so the TMU priming, the VPM→DMA store and the counter
branch stay byte-identical to the element-wise kernel that has been verified
since the first milestone.

### 10.2 Measured, verified (stock clocks, `VC4INFER_DAEMON=1`)

`infer/tests/genops_probe.py <op>` against a numpy float64 reference,
tolerance `1e-4 + 1e-4*max(|ref|,1)`, 4096-element random + a hand-picked edge
vector:

| op | method | shapes | max abs | max rel | mismatches | launches | GPU ms |
|---|---|---|---:|---:|---:|---:|---:|
| `Abs` | 1 QPU AND (0x7fffffff) | 4096, 3 | 0 | 0 | 0 / 4099 | 2, 1 | 0.152, 0.083 |
| `Neg` | 1 QPU XOR (0x80000000) | 4096, 3 | 0 | 0 | 0 / 4099 | 2, 1 | 0.149, 0.080 |
| `Clip` (attrs) | FMAX/FMIN with the bounds in uniform registers | 4096, 5 | 0 | 0 | 0 / 4101 | 2, 1 | 0.147, 0.081 |
| `LeakyRelu` | max/min + one mul, alpha in a uniform | 4096, 5 | 5.96e-08 | 5.96e-08 | 0 / 4101 | 2, 1 | 0.149, 0.083 |
| `HardSigmoid` | mul + add + clamp, alpha/beta in uniforms | 4096, 5 | 5.96e-08 | 5.96e-08 | 0 / 4101 | 2, 1 | 0.151, 0.078 |
| `HardSwish` | as above + one mul | 4096, 5 | 4.77e-07 | 2.30e-07 | 0 / 4101 | 2, 1 | 0.149, 0.080 |
| `Sign` | two `> 0` masks, exact bit select | 4096 | 0 | 0 | 0 / 4096 | 2 | 0.149 |
| `Sign` | edge vector incl. `+0`/`-0` | 14 | 1.0 | 1.0 | 2 / 14 | 1 | 0.083 |

`Sign`'s two edge mismatches are the signed-zero cases: the kernel returns
`+0` for `-0.0` where numpy's `sign` returns `-0.0` (and the ONNX reference
returns `0`); the magnitude is exact.

The GPU cost is dominated by the launch (one launch per 4096 floats for the
16-vector ops, two for 4096 because 4096/4096 > 12 groups per launch); a
4096-float unary op costs ~0.15 ms of GPU time at stock clocks, i.e. the same
class as the existing `vc4_elementwise` ops.

### 10.3 What is not correct yet, with the exact symptom

The transcendental bodies are built as *range reduction + minimax polynomial +
exact integer scaling* (2^f degree-5 for exp/exp2, `m ∈ [1,2)` + `(m-1)/(m+1)`
series for log, `(1-z)/(1+z)` for tanh, A&S 7.1.26 for erf, Cody-Waite
quadrant reduction for sin/cos, all with the QPU's round-toward-zero FPU and
integer round-to-even where a true round is needed).  Their arithmetic is
verified in an offline emitter-level emulator that models the truncating FPU
(`kops` bodies reproduce numpy to <1e-7 for exp/log/softsign), but **on the
board the results are wrong**:

| op | board result | expected | note |
|---|---|---|---|
| `Exp` | correct at x=1 (2.7183), wrong at x=0.5 (0.324 vs 1.649), x=2.7 (113.6 vs 14.88) | — | per-lane, data-dependent |
| `Softsign` | 0 for every input | x/(1+\|x\|) | the `1/x` Newton core collapses to 0 |
| `Tanh`/`Erf`/`Softplus`/`Elu`/`Selu`/`Gelu`/`Sin`/`Cos`/`Log`/`Sqrt`/`Reciprocal`/`Floor`/`Ceil`/`Round` | wrong or zero | — | same class |

The isolation work so far:

* `abs`/`neg`/`sign`/`clip`/`leakyrelu`/`hardsigmoid`/`hardswish` are exact,
  so the harness, uniform reads, B-file constants, parameter uniforms, the
  `ADD_*`/`FADD`/`FMIN`/`FMAX`/`AND`/`OR`/`XOR` encodings and the store are
  all sound.
* `leakyrelu` with `alpha=0.5` gives exactly `-1.0` for `x=-2`, so `fmul` and
  the `mul → acc3 → mov` path the emitter uses is sound.
* Therefore the defect is inside the transcendental *core sequences*
  themselves (the magic-exponent trick, the Newton reciprocal, or the
  B-register staging the cores use), not in the kernel skeleton.  Two
  hypotheses were tested and rejected on the board: two extra nops after the
  bank-conflict staging write (made it worse: all lanes zero) and the
  `ONE`↔`E1` constant aliasing (fixed a real 3.5 % error but not the rest).

A stale-blob trap found on the way: `vc4__kernel` keys the JIT cache on the
kernel *name* only, so a blob generated by an earlier version of a generator is
reused forever.  `VC4INFER_CACHE=<fresh dir>` must be used when iterating on
`kops.py`; this cost two board reboots (the QPU can wedge on a bad kernel and
only a power cycle clears it, as `docs/ONNX.md` §3 already warns).

### 10.4 The host-byte debt (acceptance item 2) — not started

The `util` table quoted by the current milestone (`host_compute_nodes 0`,
`17,077,040 B` across 142 sites) is unchanged by this slice: the `gather` and
`blit` kernels that would remove it exist and generate, but they were not
element-verified in time, so they are **not** wired into the executor.  The
mapping from the table to the new kernels is nevertheless exact and does not
need new hardware work:

| op | bytes/run | path once `vc4_gather`/`vc4_blit` verify |
|---|---:|---|
| Conv | 8,683,728 | NHWC↔NCHW + input conversion: gather table per output element (4 B/elem overhead) or blit for the 64 B-aligned interior |
| Concat | 5,160,000 | `vc4_gather` with per-row tables (pitched channel concat) or `vc4_concat_nhwc` for C%16 |
| Split | 983,040 | gather, channel-strided |
| Reshape | 826,080 | gather (2-D transpose is exact; only the 2100-extent head needs the table) |
| Resize | 471,040 | gather (nearest), gather ×4 + GPU MUL/ADD (bilinear/cubic) |
| Softmax | 403,456 | gather (channel→last-axis permute) |
| Transpose | 268,800 | gather, arbitrary permutation |
| MaxPool | 247,296 | the `-inf` border: a 64 B-granular constant `fill` is already an argument of `vc4_gather` |
| Slice | 25,200 | gather |
| Mul | 8,400 | broadcast constant expansion: gather with a repeated index |

### 10.5 Next steps, in order

1. Fix the transcendental cores on the board.  The fastest instrument is the
   `emu2.py` style word-level emulator in this tree's scratch (decode the blob,
   run the ALU words with the truncating FPU) — it distinguishes a bad body from
   a bad encoding, which is the remaining ambiguity.
2. Element-verify `gather`, `blit`, `reduce`, `cumsum`, `sel3` (they need the
   same treatment `test_onnxops.c` gives `maxpooln`/`softmax_fwd`), then wire
   the layout table above into `_onnxrun.py` behind the documented
   `VC4_LAYOUT=vc4|host` switch, defaulting to `vc4`.
3. `NonZero`/`TopK` (bounded output + device-written count) on top of `reduce`
   and `gather`; then the NMS / fp64 / device-dispatch items.


## 11. Transcendentals: second method, probes, and where it still fails

Section 10 recorded that the magic-exponent method produced wrong numbers on the
board while an emitter-level emulator said the formula was right.  Following the
probe-first plan, the method was replaced and the FPU was characterised
directly.

### 11.1 The FPU flushes denormals (measured)

One input vector, one launch, existing verified kernels only:

| op | 1.4e-45 | -1.4e-45 | 1.1754944e-38 | -1.1754944e-38 | 0 / -0 |
|---|---|---|---|---|---|
| `Abs` (bit AND, no FPU) | 1.4e-45 | 1.4e-45 | 1.175e-38 | 1.175e-38 | 0 / 0 |
| `LeakyRelu` alpha=1 (max/min + fmul + fadd) | **0** | **0** | 1.175e-38 | -1.175e-38 | 0 / 0 |

So the datapath and the bitwise ALU ops preserve denormals, while the FPU
(`fmax`/`fmin`/`fmul`/`fadd`) flushes them to zero, including denormal
*operands* (the `max(x,0)`/`min(x,0)` pair already returns 0 for 1.4e-45).
Any design that produces or consumes a denormal therefore diverges from an
emulator that keeps them - this is one concrete cause of the section-10
mismatch, and it rules out the magic-exponent trick (which builds `2^n` by
integer bit insertion near the denormal boundary) for anything that can get
close to zero.

### 11.2 The methods now implemented (no floor, no magic, no denormals)

| op | method | measured offline accuracy | on board |
|---|---|---|---|
| `Exp` | halve by 2^8, degree-6 minimax on [-0.35,0.344], square 8 times | 4.9e-9 poly, ~1.3e-6 after squaring | **still wrong** (see 11.3) |
| `Tanh` | `sign(x)(1-t)/(1+t)`, `t = exp(-2|x|)` by the same halving scheme, reciprocal of `1+t` in [1,2] by a linear seed + 4 Newton steps | ~1e-7 | still wrong (zero) |
| `Log` | exponent and mantissa read with integer bit ops (exact), degree-9 polynomial in `m-1` on [0,1] | 1.5e-8 | still wrong (-inf) |
| `Sqrt` | exponent/mantissa split, odd-exponent fixup, degree-6 polynomial on [1,4], scale built by integer shift | 6.8e-5 rel | still wrong (zero) |
| `Reciprocal` | exponent split + [1,2] reciprocal (linear seed + 4 Newton) + `2^-e` by integer shift | ~1e-9 | still wrong |

Coefficients are Chebyshev fits rounded to fp32 (the hex bit patterns are in
`kops.STD`), and every mul-after-add on an accumulator is padded by two words
(the ADD→MUL forwarding gap this board has, see `build_softmax`'s note).

### 11.3 What the board still does, and the one clue that matters

`Exp` with 16 identical lanes and n=4:

| x | 0 | 0.5 | 1 | 2 | 2.7 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| got | 0 | 0.32436 | **2.7183** | 30.78 | 113.62 | 156.3 | 986.4 |
| ref | 1 | 1.6487 | 2.7183 | 7.3891 | 14.880 | 20.086 | 54.598 |

The result is identical - to the last printed digit - for the **old
magic-exponent body and the new halving+squaring body**, and identical across
all 16 lanes.  Since the two bodies share no instruction sequence (one uses
`ftoi`/integer shifts on `2^23`, the other only polynomial mul/add and eight
squarings), the number cannot be coming from the body.  Working backwards,
`got = p^256` gives `p = 1.003914` at x=1 (exactly `exp(1/256)`) but
`p = 1.013477` at x=2 (should be `1.007843`): the *core is exact at x=1 and
wrong elsewhere*, which is not how either implementation can fail.

The remaining suspects, in order, are (a) the operand the body actually reads
is not `E_AV` for this op (the harness copy `E_AV = acc4` is shared with the
TMU pop and the two-vector pipeline), and (b) `uop_nvec`'s group size and the
kernel's `nvec` disagree for `exp` (the C table is a hand-kept mirror of
`kops.HEAVY`; `exp` was added to `HEAVY` in this change, and a mismatch changes
which 64/256-float window each group covers).  Both are cheap to test with the
`probe_seq` instrument described in section 11.4: one 16-lane vector in, one
launch, raw results out, compared bit-for-bit with the model for the 16
boundary values (0, ±denormal, 1, 1±eps, 1e-30, 1e30, powers of two, the range
edges), which is exactly what `infer/tests/genops_probe.py` does not do today.

### 11.4 `probe_seq` (planned instrument)

A `probe` kernel kind taking `--seq N` and one 16-lane vector, writing raw
16-lane results, with `vc4_probe(ctx, seq, a, out, n)` in the engine.  Each
sequence is one bisectable step of a core (scale, poly, one squaring, the
exponent extract, the select).  Nothing is composed into a full op until the
sequence matches the bit-exact model on the 16 boundary inputs.  This is the
next concrete step; it did not fit in this change.

### 11.5 What is registered in the executor

Only the seven ops of section 10.2 plus the ONNX wiring of section 10 (now in
`_dispatch` and `coverage.py`, `vc4infer nodes` reports 7/7 vc4, host debt
none).  `Exp`/`Log`/`Sqrt`/`Tanh`/`Erf`/`Sin`/`Cos`/`Softplus`/`Gelu`/... keep
their kernels in `kops.py` but remain **unregistered**: a graph node using them
raises `UnsupportedNode`, never a wrong number and never a CPU path.

### 11.6 The `nvec` hypothesis, tested and ruled out

The leading hypothesis for the section-11.3 numbers was the 4-vector group path
(`_store4` + `_gshift` 8 + the VPMQ7/CSTEP staging) rather than the math: every
failing op was in `kops.HEAVY` (nvec=4) and every working op was at nvec=16.
Two experiments settle it:

| experiment | body | nvec | result |
|---|---|---:|---|
| `Abs` forced into `HEAVY` | 1 ALU word, no arithmetic | 4 | **exact** (2.7, -2.7, 0.5, -0.9 all bit-exact) |
| `Softsign` removed from `HEAVY` | `x * rcp(1+\|x\|)` | 16 | **0 for every lane**, same as at nvec=4 |

So the 4-vector store/step path is sound (`Abs` runs through it exactly) and a
failing body fails at both group sizes (`Softsign`).  The `nvec` hypothesis is
ruled out, and the section-11.3 observation that the old and new bodies agree
"to the last digit" says nothing about the store: both were measured at nvec=4.

What the failing bodies share, and the working ones do not: every working body
(`Abs`, `Neg`, `Sign`, `Clip`, `LeakyRelu`, `HardSigmoid`, `HardSwish`,
`Identity`, and the verified `Sigmoid`/`Silu`) feeds the arithmetic ALU only
from **`ldi` constants, uniforms and add-ALU results**.  Every failing body
(`Softsign`, `Exp`, `Log`, `Sqrt`, `Tanh`, `Reciprocal`, `Round`) feeds a
**MUL-ALU result back into a later MUL**, through either an accumulator
(`e.mul(ACC1, ACC1, ACC0)`) or a B register written by the mul's `mov`
(`core_rcp`'s Newton chain).  That is the next thing to probe: a two-op
sequence `t = fmul(x, 2); y = fmul(t, 3)` at nvec=16, compared bit-for-bit.
If it fails, chained products are the defect and every core must keep a chain
in one accumulator, or route it through the packed `fmul`+`fadd` word the
verified GEMM uses.

### 11.7 `Tanh` shipped by composition (no new kernel)

`tanh(x) = 2*sigmoid(2x) - 1`, where `sigmoid` is the **verified** element-wise
kernel (7.15e-07) and the scalings are the engine's correctly-rounded `MUL`
and a broadcast `ADD`.  Four GPU launches, no host arithmetic, no new kernel:

| graph | shape | max abs | max rel | mismatches |
|---|---|---:|---:|---:|
| `Tanh` | 1x3x8x8 | 6.41e-07 | 6.41e-07 | 0 / 192 |
| `Tanh` | 2x5x7x3 | 1.55e-06 | 1.55e-06 | 0 / 210 |
| `Tanh` | 1x16x4x4 (edge vector incl. 0, +-1e30, +-1e-30) | 0 | 0 | 0 / 256 |

Against `onnx.reference.ReferenceEvaluator`, tolerance `1e-4 + 1e-4*max(|ref|,1)`.
`genops_onnx.py` now covers 8 ops / 28 graphs / 4670 elements, 0 mismatches.
`vc4infer nodes` reports `Tanh` as a vc4 node.
