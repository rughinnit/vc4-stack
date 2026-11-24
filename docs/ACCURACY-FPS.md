# The latency–accuracy curve at 96, 128 and 160

This page answers a narrower question than [ACCURACY.md](ACCURACY.md) or
[BENCHMARKS.md](BENCHMARKS.md): **what does the same trained detector cost and
find at three input sizes the engine can actually run today**, and what would
have to be trained for the point that does not exist yet — a model *trained*
at 96 or 128 that still detects.

One caveat frames every number below.  The weights are the official
Ultralytics COCO `yolov8n.pt`, trained at 640×640.  Exporting the same graph at
96, 128 and 160 and feeding it a letterboxed image **degrades a 640 model by
resolution change**; it is not a 96-trained model and its accuracy is not what
a trained 96 model would reach.  Nothing here is an mAP: the image set is one
photograph, and what is measured is latency, engine parity, and which objects
survive at each size.

Conditions: Raspberry Pi 3B (BCM2837), Debian 13 arm64, kernel 6.18, core
400 MHz and V3D 300 MHz throughout (`tools/thermal-guard.sh --max 71` printed
`core 400000000 Hz` on every guard line; peaks 59.1–64.5 °C,
`throttled=0x20000` was already set before this work and no run moved the
clock).  Engine: `vc4infer` 1.2.10 from this tree (in-tree `libvc4infer.so.1`,
build `1017ea0`) through `vc4d` 1.2.9, unprivileged, one client at a time.
Optimiser: `VC4_OPT=all`.

## 1. The graph at three sizes

Exports are Ultralytics 8.4.215 / torch 2.15.0+cpu, opset 13, `simplify=False`,
batch 1, static shape, fp32 — the same recipe as the packaged
`yolov8n_160.onnx`.  The fresh 160 export returns **bit-identical output** to
that packaged model on the same input (max abs 0 over all 44,100 values,
different file bytes only), so the point below is the same model number that
[ACCURACY.md](ACCURACY.md) records.

| imgsz | convs | MACs | GFLOP | nodes executed | launches |
|---:|---:|---:|---:|---:|---:|
| 96 | 64 | 98.4 M | 0.197 | 138 | 462 |
| 128 | 64 | 174.9 M | 0.350 | 138 | 468 |
| 160 | 64 | 273.2 M | 0.546 | 138 | 536 |

The graph is the same 138 executed nodes at every size; only the planes grow.
Launches barely move (462 → 536) because most of them are not convs:

| imgsz | Conv | Concat | everything else | total |
|---:|---:|---:|---:|---:|
| 96 | 131 | 231 | 100 | 462 |
| 128 | 131 | 223 | 114 | 468 |
| 160 | 131 | 231 | 174 | 536 |

YOLOv8n's C2f blocks concatenate on a pitched NHWC plane, and on this engine
each `Concat` was a set of launches (231 of them) whatever the resolution.  That
is the first reason the cost does not scale with pixels.

**These counts are the pre-concat window.**  `Concat` now takes one launch per
input instead of one per 16-channel block, and a per-run node plan replaced the
per-frame dispatch rebuild: the same graph runs in **291 / 297 / 365** launches
at 96/128/160, at 315.7 / 348.2 / 432.5 ms and 3.16 / 2.86 / 2.31 FPS, with
output bit-identical ([CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)).  The tables in this
section stay as measured: the detections and `bus.jpg` boxes below were taken in
that window, and their wall column belongs to the same record.

## 2. The measured curve

`bus.jpg` letterboxed to each size with the package's own `Letterbox`, the same
array to ONNX Runtime and to the engine.  Three warm-up runs (kernel JIT not
counted), five timed runs, median.  One client; guard at 71 °C.

| imgsz | GFLOP | launches | QPU ms | node wall ms | **wall ms** | **FPS** | ORT CPU ms | × CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.197 | 462 | 179.8 | 330.8 | **353.9** | **2.83** | 44.0 | 8.0× |
| 128 | 0.350 | 468 | 204.0 | 360.5 | **384.9** | **2.60** | 59.4 | 6.5× |
| 160 | 0.546 | 536 | 262.7 | 439.6 | **466.4** | **2.14** | 89.1 | 5.2× |

"QPU ms" is the sum of per-node GPU time from the profile; "node wall" is the
sum of the per-node host wall, so wall minus node wall is between-node time.
Everything that is not QPU time — dispatch, packing, copies, between-node
time — is in `wall − QPU`: 174, 181 and 204 ms, or 0.38, 0.39 and 0.38 ms per
launch.

Doubling the pixels from 96 to 160 (2.78×) buys 1.32× wall time, not 2.78×.
QPU time grows 1.46×, the launch path is nearly flat.  Resolution is the
cheapest lever on this engine and the least effective one: at 96 the graph
already spends half its frame outside the QPU, on the path of its 462 launches.

### Arena

The default client arena is 64 MiB.  The planned pool and the measured
high-water mark are:

| imgsz | pool plan MiB | peak used MiB | arena mapped MiB |
|---:|---:|---:|---:|
| 96 | 31.7 | 25.1 | 64 |
| 128 | 35.5 | 25.6 | 64 |
| 160 | 41.6 | 26.3 | 64 |

All three points fit with more than 20 MiB to spare, so the arena is not the
constraint on this curve.  The 1.32× wall penalty from 96 to 160 is launch
count and conv work, not memory.

## 3. Engine parity at each point

Same input array, ONNX Runtime CPU (4 threads) as reference, tolerance
`|got − ref| ≤ 5e-4 + 5e-4·max(|ref|, 1)`.

| imgsz | elements | over 5e-4 | over 1e-4 | max abs | max rel where `\|ref\| ≥ 1` |
|---:|---:|---:|---:|---:|---:|
| 96 | 15,876 | **0** | 10 | 1.36e-2 | 3.81e-4 |
| 128 | 28,224 | **0** | 15 | 2.10e-2 | 3.13e-4 |
| 160 | 44,100 | **0** | 32 | 1.37e-2 | 3.19e-4 |

The 160 row is the tolerance result already in [ACCURACY.md](ACCURACY.md); the
96 and 128 rows extend it.  Decoded at confidence 0.25 with `decode_yolo` on
both outputs, mapped back to original pixels:

| imgsz | detections CPU / VC4 | same classes | max box diff | max score diff |
|---:|---|---|---:|---:|
| 96 | 3 / 3 | yes | 0.0028 px | 6.8e-5 |
| 128 | 4 / 4 | yes | 0.0043 px | 6.4e-5 |
| 160 | 4 / 4 | yes | 0.0016 px | 8.5e-5 |

The engine changes nothing that a box can see at any of the three sizes.  The
differences between resolutions below are the model's, not the engine's.

## 4. What each resolution finds

Confidence 0.25, boxes in original image pixels (810×1080 after the inverse
letterbox).  Both sides agree to the last printed digit, so one set is shown.

**160×160** — the reference point, four detections:

```
person  0.778  [ 39.02  409.15  228.73  901.73]
bus     0.703  [  1.83  226.87  800.57  758.83]
person  0.619  [678.45  424.49  810.00  865.76]
person  0.597  [206.19  391.52  353.17  863.16]
```

**128×128** — four detections, all four objects kept:

```
bus     0.849  [  0.00  229.50  801.84  728.40]
person  0.733  [ 43.55  401.83  224.58  899.68]
person  0.481  [678.76  402.93  810.00  872.77]
person  0.471  [197.55  371.27  349.11  869.45]
```

**96×96** — three detections; the right-hand pedestrian is gone:

```
person  0.545  [ 38.26  396.44  219.65  884.47]
person  0.480  [191.74  408.32  343.76  862.32]
bus     0.377  [ 29.29  232.05  810.00  751.16]
```

Matched back to the 160 set by class and IoU ≥ 0.5:

| imgsz | matched of 160's four | max coordinate delta | scores |
|---:|---:|---:|---|
| 96 | 3 / 4 | 27.5 px | bus 0.703 → 0.377; the missing person is the right-hand one, 0.597 at 160 |
| 128 | 4 / 4 | 30.4 px | bus 0.703 → 0.849; persons 0.778/0.619/0.597 → 0.733/0.481/0.471 |
| 160 | 4 / 4 | 0.002 px (engine vs CPU) | — |

What the 30 px delta is: at 128 the box that moves most is the bus, whose
bottom edge comes out at 728 px against 759 at 160 — 6 % of the box height,
on an object that is still found at a higher score.  The 96 point is worse in
kind, not degree: it loses the small right-hand person entirely and halves the
bus confidence.  On this image, 128 keeps the detection set and 96 does not.

This is one image and four objects.  It is evidence that resolution changes
which objects survive on these weights; it is not a recall or mAP estimate.

## 5. Budget arithmetic

Two measured constants from the rest of the documentation:

* **0.168 ms marginal launch.**  Removing 467 launches' worth of fused SiLU
  `Mul`s from YOLOv8n-160 saved 79 ms ([OPTIMIZER.md](OPTIMIZER.md) sec. "The
  activation epilogue").  This is the price of a launch the pipeline can delete.
* **3.3 GFLOP/s at detector conv shapes.**  The 8-conv 96×96 graph issues
  22.9 M MACs and spends 13.7 ms of QPU time ([VC4-MODEL.md](VC4-MODEL.md)
  sec. 2).  That is the conv rate at 12×12–48×48 feature maps, 2.6× below the
  8.44 GFLOP/s engine GEMM at 256³ and 4.2× below the hand-written 13.72
  GFLOP/s, because the tiles are short and the im2col panel is a real cost.

Priced at those two constants:

| imgsz | GFLOP | QPU ms | nominal GFLOP/s | arithmetic floor @3.3 | launches | launch floor @0.168 | floor sum | measured | measured / floor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.197 | 179.8 | 1.10 | 59.7 ms | 462 | 77.6 ms | 137.3 ms | 353.9 ms | 2.58× |
| 128 | 0.350 | 204.0 | 1.72 | 106.1 ms | 468 | 78.6 ms | 184.7 ms | 384.9 ms | 2.08× |
| 160 | 0.546 | 262.7 | 2.08 | 165.5 ms | 536 | 90.0 ms | 255.5 ms | 466.4 ms | 1.83× |

The launch column is the pre-concat count.  At the current 291 / 297 / 365 the
launch floor is 48.9 / 49.9 / 61.3 ms rather than 77.6 / 78.6 / 90.0, so
arithmetic is the larger half of the floor at every size, and pricing the
pre-concat measured wall against the current floor would put it at 1.9–2.9×
instead of 1.8–2.6×.  The `measured` column belongs to its own window;
[CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) carries the current wall and CPU set.

The measurement is 1.8–2.6× the floor, and the reason is the same at
every size: **0.168 ms is the marginal price of a launch, not the price of a
conv launch.**  A conv launch also sets up and submits its im2col panel; the
measured conv launch path is ~0.77 ms and a conv including its own QPU time is
~1.2 ms at 96 ([VC4-MODEL.md](VC4-MODEL.md) sec. 2).  YOLOv8n also does not
run at 3.3 GFLOP/s of *nominal* MACs: its depthwise convs are expanded to
block-diagonal dense weights, and Concat adds 231 copy launches, so the QPU
issues more work than the 0.197–0.546 GFLOP model counts.

For a graph built only from convs, the two prices give different budgets, and
the second is the one such a graph pays:

| term | 30 FPS (33.3 ms) | 60 FPS (16.7 ms) |
|---|---:|---:|
| 18 launches at the 0.168 ms marginal price | 3.0 ms | 3.0 ms |
| arithmetic left at 3.3 GFLOP/s | **0.100 GFLOP** | **0.045 GFLOP** |
| 18 conv launches at the measured 0.77 ms path | 13.9 ms | 13.9 ms |
| arithmetic left at 3.3 GFLOP/s | **0.064 GFLOP** | 0.009 GFLOP |

The measured fit (`T = 2.5 + 3.13 c` at 96) puts the 30 FPS point at nine
convs, 0.052 GFLOP, and the 8-conv 0.046 GFLOP graph at 27.5 ms / 36.3 FPS.
Pricing that graph at 3.3 GFLOP/s and its 18 conv launches at the measured
0.77 ms path gives 27.8 ms against 27.5 ms measured — the two constants and
the graph agree to 1 %, which is why 0.05 GFLOP is the design point for a
trained detector at 30 FPS.

YOLOv8n is nowhere near it: 0.197 GFLOP at 96, plus 231 Concat launches that
the student does not have.  **Resolution alone cannot turn YOLOv8n into a
30 FPS detector**; the smallest measured point is 2.83 FPS.

## 6. The point that does not exist: a trained 96/128 model

Nothing below was trained here.  No 96 or 128 model on a real dataset was
trained, for two independent reasons, both measured:

1. The board cannot train at a useful scale.  One forward+backward of the
   0.05 GFLOP student at 96, batch 4, four threads, is **299 ms per step**
   ([VC4-MODEL.md](VC4-MODEL.md) sec. 8).  COCO train2017 is 118,287 images, so
   an epoch is 29,571 steps: **2.46 h per epoch**, and ten epochs is a day.
   YOLOv8n itself is 8.4× more expensive: the fine-tune in sec. 7 measured
   **2.5 s per step**, i.e. 20.5 h per COCO epoch at batch 4 (23.0 h at the
   median 2.8 s, 33.7 h at the slowest measured 4.1 s).
2. The 0.05 GFLOP student's head has no loss in this tree.  It emits six
   logits per cell (objectness, cx, cy, w, h, class) from a plain 1×1 conv, not
   Ultralytics' `Detect`/DFL head, because the export of `Detect` contains
   `Split` and a `reg_max` softmax the engine refuses.  `yolo detect train`
   cannot consume it; the assignment and loss have to be written.

Training belongs on a workstation.  These are the exact commands.

### 6.1 Fine-tune YOLOv8n at 96 and 128

The checkpoint to start from is the same COCO 640 model measured above.  Fine-
tuning at the deployment size lets the loss see the resolution the engine will
run, which is what the 96/128 points in sec. 4 are missing.

```sh
# workstation, one CUDA device
python3 -m venv ~/venvs/yolo && . ~/venvs/yolo/bin/activate
pip install "ultralytics==8.4.215" onnx onnxruntime

cd ~/data
yolo detect train model=yolov8n.pt data=coco.yaml imgsz=96 \
  epochs=100 batch=64 device=0 workers=8 \
  rect=False cos_lr=True close_mosaic=10 patience=20 \
  project=runs name=yolov8n-96

yolo export model=runs/yolov8n-96/weights/best.pt format=onnx \
  imgsz=96 opset=13 simplify=False dynamic=False batch=1
```

Repeat with `imgsz=128` and `name=yolov8n-128`.  `rect=False` matters: the
board letterboxes to a square, and a rectangular training batch would optimise
a different preprocessing than the one deployed.  Do not enable `half` or
`int8`; the engine is fp32 only.

The exported file is the input to the same measurement used in sec. 2, on the
board:

```sh
cd ~/work/vc4-stack
export TMPDIR=$HOME/scratch/tmp VC4_OPT=all
export PYTHONPATH=$PWD/infer/python
tools/thermal-guard.sh --max 71 -- \
  ~/scratch/ortvenv/bin/python infer/tools/onnx/accfps_curve.py points \
    --model /path/yolov8n-96-best.onnx --tag 96 \
    --out $HOME/scratch/accfps/trained-96
```

Weights do not change conv cost, so the latency should land on the 353.9 ms of
sec. 2 within run-to-run spread; what changes is which detections appear.

### 6.2 Distil into the 0.05 GFLOP student

The student already exists as a graph and is already latency-measured; it has
no trained weights.  Build it the same way [VC4-MODEL.md](VC4-MODEL.md)
does:

```sh
python3 infer/tools/det/build.py vc4det-n --imgsz 96
# 8 convs, 22 compute nodes, 0.046 GFLOP, 96x96
# measured at random init: 27.5 ms, 18 launches, 36.3 FPS
```

Distillation recipe for the workstation:

* **teacher** — the fine-tuned `yolov8n-96` from 6.1 (same input size, so no
  scale mismatch).  A 160 teacher with a 96 student also works and gives a
  slightly stronger target; it costs one decode per image either way.
* **student** — `infer/tools/det/vc4det-n.yaml`, width 0.25, channels
  16/32/64, one stride-8 head, 12×12 cells, six logits per cell.
* **loss** — targets from the teacher's decoded predictions for the same
  letterboxed image: objectness BCE against teacher confidence, class BCE
  against the teacher's class probabilities (soft targets), box CIoU on
  (cx, cy, w, h) after a TaskAlignedAssigner-style assignment of teacher boxes
  to student cells.  Weights 7.5 box / 0.5 class, the Ultralytics defaults,
  plus 1.0 on objectness.  Optionally add an MSE on the 12×12 feature map.
* **schedule** — 300 epochs, AdamW, lr 2e-3 with 3-epoch warm-up and cosine
  decay, weight decay 5e-4, batch 64, no mosaic after epoch 290.  The student
  is small enough that an epoch is minutes, not hours, on one GPU.

The loop is the one piece not in this tree (sec. 6, point 2).  Its command
line, once written, is:

```sh
python infer/tools/det/distil.py \
  --teacher runs/yolov8n-96/weights/best.pt \
  --student infer/tools/det/vc4det-n.yaml --imgsz 96 \
  --data /datasets/coco --epochs 300 --batch 64 --device cuda:0 \
  --lambda-box 7.5 --lambda-cls 0.5 --lambda-obj 1.0
```

Then `build.py` exports the distilled checkpoint at 96, and the same
`accfps_curve.py --model ... --tag 96` command as 6.1 measures parity and
latency.  A trained student at 0.046 GFLOP is expected at ~27.5 ms / 36.3 FPS
because weights do not change the conv cost; the open question the workstation
run answers is accuracy, not speed.

## 7. The fine-tune that did run: coco8, three epochs

To show the pipeline end to end, `yolov8n.pt` was fine-tuned at 96 on
Ultralytics' `coco8` — **four training images, four validation images** — for
three epochs on this board's CPU.  The point is that train → export → engine →
parity works; the numbers are not a quality measurement.

| item | value |
|---|---|
| command | `yolo detect train model=yolov8n.pt data=coco8.yaml imgsz=96 epochs=3 batch=4 device=cpu workers=0` |
| steps per epoch | 1 (four images, batch 4) |
| step time | **2.5–4.1 s**, median 2.8 s (2.8, 2.5, 4.1 s); 2.5 s is the best step |
| validation | four images, ~1.0 s per pass |
| reported mAP50 / mAP50-95 | 0.202 / 0.134 — **meaningless at four images**; printed only because the validator ran |
| exported graph | 12,363 KiB, opset 13, 1×3×96×96, node for node the base export |
| engine latency | **354.1 ms, 2.82 FPS, 462 launches, 179.5 ms QPU** |
| engine parity | 0 / 15,876 over 5e-4, max abs 1.38e-2 |
| detections vs its own CPU | 3 / 3, same classes, scores within 7e-5 |

Three epochs on four images moved the weights by almost nothing: on `bus.jpg`
the fine-tuned model finds the same three objects as the base 96 export, with
the same scores to within 0.006 (person 0.541 vs 0.545, bus 0.383 vs 0.377).
That is the correct result for a demonstration run.  A fine-tune that changes
the detections needs a real dataset, i.e. 6.1.

One board-specific workaround: the board's `torchvision` 0.29.2 wheel does not
register operators against its custom `torch` 2.15.0+cpu build, so validation's
`torchvision.ops.nms` raises.  The run used a pure-torch NMS shim on
`PYTHONPATH` (outside the tree, under `~/scratch/accfps/tvshim-nms`).  A
workstation venv with stock torch needs no shim, and nothing in the training
step or the export depends on it.

## 8. Recommendation

**Best accuracy available at the best frame rate on this curve: 128×128,
384.9 ms, 2.60 FPS.**  It keeps all four detections and matches the 160 set at
IoU ≥ 0.5, for 31 ms (8.8 %) more than 96 — which loses the fourth object and
halves the bus score.  160×160 is the accuracy point (same four objects, the
bus box 30 px closer to the reference) at 21 % more latency; take it if 466 ms
is acceptable and not otherwise.

None of these is a usable frame rate in the real-time sense: 2.1–2.8 FPS is
more than an order of magnitude below a camera's 30 FPS.  The only measured
point at or above 30 FPS is the 0.046 GFLOP, 8-conv, 96×96 student at 27.5 ms
/ 36.3 FPS, and it has no trained weights.  So the recommendation splits:

* deploy today, accuracy first: **128×128 YOLOv8n, 2.60 FPS**;
* deploy at a usable frame rate: **the student at 96**, after the workstation
  recipe in 6.2 — 36.3 FPS is measured, accuracy is the work.

## 9. Reproduce

```sh
cd ~/work/vc4-stack
export TMPDIR=$HOME/scratch/tmp          # never /tmp on this board
export VC4_OPT=all
export PYTHONPATH=$PWD/infer/python

# 1. build the engine (in-tree library wins over the installed one)
tools/build-fast.sh vc4d -j2
tools/build-fast.sh infer -j3

# 2. export the three sizes (CPU-only, no device)
cd infer/tools/onnx
for s in 96 128 160; do
  YOLO_IMGSZ=$s YOLO_WEIGHTS=/path/to/yolov8n.pt \
    ~/venvs/yolo/bin/python export_yolo.py      # ultralytics + torch
done
mkdir -p ~/scratch/accfps/models
cp models/yolov8n_{96,128,160}.onnx ~/scratch/accfps/models/

# 3. CPU references (no device)
cd ~/work/vc4-stack
~/scratch/ortvenv/bin/python infer/tools/onnx/accfps_curve.py refs \
  --models ~/scratch/accfps/models --out ~/scratch/accfps/out \
  --image ~/scratch/refs/yolo/bus.jpg

# 4. the curve, one client, guarded
tools/thermal-guard.sh --max 71 -- \
  ~/scratch/ortvenv/bin/python infer/tools/onnx/accfps_curve.py points \
    --models ~/scratch/accfps/models --out ~/scratch/accfps/out \
    --image ~/scratch/refs/yolo/bus.jpg

# 5. join and match against the 160 point (no device)
~/scratch/ortvenv/bin/python infer/tools/onnx/accfps_curve.py report \
  --models ~/scratch/accfps/models --out ~/scratch/accfps/out
```

`points` prints the profiled run's launches, QPU time, node wall, arena and
parity per size and writes `point_<tag>.json`; `report` writes the IoU match
table.  A repeat of step 4 for 96 alone measured 353.7 ms, against the 353.9 ms
in sec. 2.  The raw rows behind every table are in
[`results/accuracy-fps.json`](results/accuracy-fps.json).
