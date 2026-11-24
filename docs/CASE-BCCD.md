# Case study: a trained blood-cell detector on the QPUs

A YOLO11n detector trained on the workstation (AMD RX 570, ROCm, gfx803) runs on
the board as a `.vc4x` artifact at **1.93 FPS / 519 ms** at 160x160, and its
detections on the two test images are **identical to ultralytics** on the same
checkpoint. This page records the whole chain and what of it is reproducible.

The second half of the page is the answer to what that artifact leaves open: at
96x96 the same model takes 419 ms (2.40 FPS) and at 160 it takes 519 ms and 438
launches. A graph built for
this engine instead of ported to it -- 8 convs, 0.046 GFLOP, 18 launches --
trains on the same data, scores **mAP50 0.596 / mAP50-95 0.426** on the val
split and runs at **40.0 FPS / 25.0 ms**. That graph now has weights.

## What the model is

| | |
|---|---|
| graph | yolo11n, 3 classes (RBC, WBC, Platelets) |
| training data | BCCD + Kaggle BCD merged, 1,071 images, 640 px, batch 8, 100 epochs |
| checkpoint used here | fine-tuned at 320 for 100 epochs (19.8 min on the RX 570), val mAP50 **0.904**, mAP50-95 **0.631** (P 0.814, R 0.920, 87 images, 1,137 instances) |
| size | 5.5 MB fp32 |

## The chain, in the order it was run

```sh
# workstation: fine-tune at the deployment size
source /mnt/Ext/env-tsl.sh
/mnt/Ext/venv-tsl2/bin/yolo detect train model=models/bccd_v11n_best.pt \
    data=configs/merged.yaml imgsz=320 epochs=100 batch=8 device=0 \
    workers=4 rect=False cos_lr=True close_mosaic=10 patience=25 seed=0 \
    project=runs name=bccd-v11n-320

# workstation: one ONNX per size the board will run (fp32, static, batch 1)
for s in 96 128 160 320; do
  /mnt/Ext/venv-tsl2/bin/yolo export \
      model=runs/detect/runs/bccd-v11n-320/weights/best.pt \
      format=onnx imgsz=$s opset=13 simplify=False dynamic=False batch=1
done

# board: check the graph before running it, then freeze it
vc4infer nodes bccd_ft_160.onnx                 # 355/355 supported, 0 unsupported
VC4_OPT=all vc4infer build bccd_ft_160.onnx -o bccd_ft_160.vc4x
vc4infer run bccd_ft_160.vc4x --imgsz 160 \
    --input BloodImage_00007.jpg --classes bccd_names.txt
```

## Measured, artifact path, ARM at 1.0 GHz, daemon 1.4.0 protocol 5

| imgsz | wall | FPS | QPU on VC4 | detections, two test images | ultralytics, same weights |
|---:|---:|---:|---|---|---|
96 | 419 ms | **2.40** | 100 % | 8 / 6 | 8 / 7 |
128 | 449 ms | 2.23 | 100 % | 12 / 12 | 14 / 16 |
160 | 519 ms | **1.93** | 100 % | **14 / 14** | **14 / 14** |
320 | 921 ms | 1.08 | 100 % | 15 / 21 | 16 / 19 |

At 160 the engine and ultralytics agree exactly. At 96, 128 and 320 they differ
by one or two detections because the engine decodes with its own NMS and its own
letterbox rather than ultralytics' post-processing.

**160 is the operating point**: exact agreement, twice the throughput of 320, and
far more cells than the 640-trained checkpoint found at that size (4 and 2).

## The defect this model exposed

Before the daemon's program table was enlarged, the same artifact ran at
**0.61 FPS**. This graph needs **133 distinct kernels**; the table held 128, so
the daemon evicted one per launch and the client re-uploaded it:

```
vc4d: evict prog slot 47 uid=1000 name=im2col_nhwc2_32_1_1_s1_d1 (LRU, 127 slots in use)
```

QPU duty was 17 % and 1,370 of the 1,651 ms were in the launch path. With 256
slots ([DAEMON.md](DAEMON.md), protocol 5) the same graph runs at 519 ms with
**zero evictions**, and the GPU time is unchanged (280.5 against 281.1 ms) - so
the entire 3x was daemon bookkeeping, not arithmetic.

Two things were also worth knowing at the time: the plain ONNX path refuses this
graph at 320 with `conv_nhwc_pads: out of arena memory` while the artifact runs,
and YOLO11 checkpoints carry `C3k2`, which ultralytics 8.1.47 cannot load, so the
fine-tune needed 8.4.x.

## Is it reproducible?

**Yes on the board side, given the model.** Every command above is the one that
was run, and the board results are repeatable within the run-to-run spread of
the other measurements in this repository: three reps, median, ARM pinned, fan
fitted, nothing else on the GPU.

**Not from this repository alone, on the training side.** Two inputs are not
here:

- the merged dataset (BCCD is MIT; the Kaggle BCD half carries Kaggle terms),
  so `configs/merged.yaml` cannot be shipped;
- a gfx803 workstation. The PyTorch 1.13.1 + ROCm 5.4.3 recipe is a separate
  project; on another GPU any current ultralytics and torch will do, and the
  export flags are the part that matters.

What *is* fully reproducible without the dataset is the engine side: take any
YOLO11n detect ONNX export at a fixed fp32 shape, run `vc4infer nodes` on it,
`build` an artifact, and measure with `vc4infer bench`. The numbers for a
different checkpoint will differ; the method and the parity check will not.

## The student: 0.046 GFLOP, trained

[VC4-MODEL.md](VC4-MODEL.md) measured an 8-conv, 22-node graph at 96x96 with
random weights and said plainly that it was not a detector: no loss, no weights,
no accuracy. This section is the same graph with weights, its accuracy on the
BCCD val split, and its measured speed on the board.

### What changed from the reference graph

Nothing in the backbone. The head's last conv widens from 6 to 8 logits, because
BCCD has three classes and the reference head carries one class in a single
channel:

| | reference (`vc4det-n.yaml`) | trained student (`vc4det-n3.yaml`) |
|---|---|---|
| convs | 8 | 8 |
| channels | 16/32/64 | 16/32/64 |
| input / head | 96x96, stride 8, 12x12 cells | same |
| logits per cell | 6: obj, cx, cy, w, h, class | 8: obj, cx, cy, w, h, 3 class scores |
| parameters (fused graph) | 81,286 | 81,352 |
| engine MACs / GFLOP | 22.93 M / 0.0459 | 22.93 M / 0.0459 |
| ONNX nodes (compute) | 26 (22) | 26 (22) from `build.py`, 22 (22) trained |
| launches on the board | 18 | 18 |

The four extra logits cost 66 parameters and no time: the conv kernel reads
16-channel blocks, so the logits conv is computed at Cout 16 either way and the
MAC count is identical. Measured on the board in one sitting, three runs of five
reps each:

| artifact | nodes | launches | median wall | QPU | FPS |
|---|---:|---:|---:|---:|---:|
| `vc4det-n_96` (reference, random init) | 26 | 18 | 25.1 ms | 13.3 ms | 39.8 |
| `vc4det-n3_96` (this head, random init) | 26 | 18 | 25.0 ms | 13.4 ms | 40.0 |
| `vc4det-n3_96` (trained) | 22 | 18 | **25.0 ms** | 13.4 ms | **40.0** |

The reference benches 25.1 ms here against the 27.5 ms recorded in VC4-MODEL.md
sec. 6: same launch count, same QPU time, different run-to-run clock and thermal
state. The trained export carries 22 nodes rather than 26, which is not a change
in computation either -- a freshly fused ultralytics graph has zero conv biases,
torch 1.13 aliases the identical zero tensors through four `Identity` nodes, and
trained weights have non-zero biases so the aliases disappear.

### How it was made trainable

`infer/tools/det/student.py` re-implements the graph by hand -- Conv2d +
BatchNorm + SiLU in the same order, same kernels (3x3 s2, 3x3 s1, 1x1), same
autopad -- and folds the BN exactly as `DetectionModel.fuse()` does before
export. `infer/tools/det/check_graph.py` turns "the ONNX is the specification"
into a test: it loads the eight conv weight/bias pairs out of the graph
`build.py` produced, wires them into the module, and compares the two graphs on
the same input.

```
architecture: match     max_abs_diff 1.5e-08
Conv 8, Sigmoid 7, Mul 7   engine MACs 22.93 M   81,352 params fused
```

(`check_graph.py` reports the module as 81,624 parameters because the training
form still carries BatchNorm; `build.py` reports both graphs after fusing.)

The head decode, used identically in training, in validation and on the board:
sigmoid on every logit, `cx, cy = (cell + p) * 8`, `w, h = p * 96` as fractions
of the canvas (no exponential to diverge), `score = obj * max(class)`. Boxes are
xyxy in the 96x96 letterboxed canvas -- the same frame ultralytics' own head
reports after its letterbox, which is what makes the two directly comparable.

### The training data, and a defect in it

`~/bccd/configs/merged.yaml`: 1,071 train / 87 val / 72 test images, 3 classes,
14,475 train instances (RBC 12,313, WBC 1,104, Platelets 1,058).

The merged set has a naming defect that matters to any training run on it.
`scripts/make_merged.py` copies the Kaggle BCD half as
`images/train/bcd_<name>.jpg` but writes its labels as `labels/train/<name>.txt`,
so **866 of the 1,071 train images have no label file whose stem matches**.
Ultralytics resolves labels by path substitution (`img2label_paths`), so those
866 images train as backgrounds:

```
images 1071   label paths that do not exist 866
Scanning datasets/merged/labels/train... 205 images, 866 backgrounds, 0 corrupt
```

The loader in `train_student.py` strips a leading lowercase dataset tag when the
exact stem is absent and reports the number of unlabelled images at startup
(`train=1071 (background 0)`). Two consequences are worth recording. The student
trains on all 1,071 images. The teacher's fine-tune (val mAP50 0.904) ran through
`yolo detect train` on the same yaml, so it effectively saw 205 labelled images
plus 866 backgrounds; its val numbers are unaffected, because the val and test
splits have matching stems, but its training set was a fifth of the intended
one.

### Distillation

Pseudo-labels from the fine-tuned YOLO11n
(`bccd-v11n-320/weights/best.pt`), run at 320, the size it was fine-tuned at,
not at the student's 96. Under one protocol (this repository's letterbox, decode
and AP code) the same checkpoint scores mAP50 0.878 at 320 and 0.409 at 96, so
the higher input is the stronger teacher.

```sh
python infer/tools/det/pseudo.py \
    --teacher runs/detect/runs/bccd-v11n-320/weights/best.pt \
    --images datasets/merged/images/train \
    --labels datasets/merged/labels/train \
    --out pseudo/train --imgsz 320 --conf 0.25 --iou 0.6
```

| | images | instances |
|---|---:|---:|
| ground truth in the train split | 1,071 | 14,475 |
| teacher detections at conf 0.25 | 1,071 | 4,146 |
| dropped, IoU >= 0.5 with a ground-truth box | | 2,658 |
| **kept as pseudo-labels** | 1,050 | **1,488** (RBC 1,427, PLT 59, WBC 2) |

The teacher contributes 1,488 boxes the labels do not have -- 10 % more objects,
almost all RBC, which is what a detector trained on a fifth of the set would be
expected to add. They are written as `cls cx cy w h conf p0 p1 p2` in
original-image coordinates so the trainer can letterbox them; `conf` and the
class vector ride along as soft targets.

Loss: task-aligned assignment (alpha 0.5, beta 6, topk 2 per ground-truth box,
one target per cell), CIoU on the decoded boxes at weight 7.5, BCE against the
teacher's class probabilities at 0.5, BCE objectness with the alignment metric as
the positive target at 1.0 -- the weights ACCURACY-FPS.md sec. 6.2 proposes.
AdamW 2e-3, cosine, 3-epoch warm-up, weight decay 5e-4, batch 64, 400 epochs,
fp32: this gfx803 part has no usable AMP path and the model has 81 k parameters,
so there is nothing to gain and a NaN to lose. Augmentation is 90-degree
rotations, mirroring, 0.9-1.2 zoom and mild brightness/contrast. One epoch is
2.7 s on the RX 570; the 400 take 18 minutes.

### Accuracy

Val split, 87 images, 1,137 instances, one protocol for every row: this
repository's letterbox to the stated size, the model's own decode, per-class NMS
at IoU 0.6, conf 0.001, pycocotools-style 101-point AP over IoU 0.50:0.05:0.95.

| model | input | mAP50 | mAP50-95 | AP50 RBC / WBC / PLT |
|---|---:|---:|---:|---|
| teacher, YOLO11n fine-tuned at 320 | 320 | 0.878 | 0.621 | 0.814 / 0.991 / 0.829 |
| teacher, same checkpoint | 96 | 0.409 | 0.231 | 0.363 / 0.844 / 0.019 |
| student, ground truth only | 96 | 0.601 | 0.427 | 0.805 / 0.998 / 0.000 |
| **student, ground truth + pseudo-labels** | 96 | **0.596** | **0.426** | 0.812 / 0.976 / 0.000 |

Pseudo-labels are worth nothing measurable on this dataset. The identical
schedule on ground truth alone scores 0.601 / 0.427 against 0.596 / 0.426 with
them -- a 0.005 difference in mAP50 and 0.001 in mAP50-95 on 87 images, inside
the run-to-run spread, and slightly in favour of the ground truth. The 1,488 extra boxes are 96 % RBC
and RBC AP50 is already 0.81, so the teacher had nothing to add to the class it
knows best, and the cells the labels miss are not cells the teacher finds. The
mechanism works -- the student trains on the teacher's probabilities rather than
one-hot targets, and its class loss sits at 0.073 instead of collapsing to
0.003 -- but it buys no accuracy here. What the pseudo-labels would have fixed
is the broken split: with the label-stem defect left in place they would have
been the only supervision for 866 of the 1,071 images. Fixing the labels was the
better repair.

Cross-check of the protocol against ultralytics' own validator, same checkpoint:
at 96 `yolo detect val` reports mAP50 0.376 with `rect=False` and 0.399 with
`rect=True` against 0.409 here; at 320 it reports 0.904 against 0.878. The
differences are the NMS IoU (0.6 here, 0.7 there) and `max_det`, not the boxes.

The student beats the teacher's own accuracy at the size the board can actually
serve: 0.596 against 0.409 mAP50, 0.426 against 0.231 mAP50-95. Against the
teacher at 320 (0.878 here, 0.904 under ultralytics' validator) it is 0.28
mAP50 behind, and that gap buys 16x the frame rate on the same silicon.

Platelets are the failure: AP50 0.000 against 83 val instances, and 0 predicted
on either test image. At 96 px a platelet is about 8 px across, which is one
stride-8 cell, so the grid cannot separate it from the RBC it sits on. The
teacher at 96 has the same problem (0.019); at 320 it reaches 0.829. If platelet
counts matter, the operating point has to be 128 or the head needs a second,
finer level, and both cost launches (VC4-MODEL.md sec. 7).

### Speed on the board

Artifact path, in-tree engine (`PYTHONPATH=infer/python`), `VC4_OPT=all`, daemon
protocol 5, ARM at 1.0 GHz, thermal guard, one client:

```sh
vc4infer nodes vc4det-n3_96.onnx        # 0 unsupported, 0 host debt
VC4_OPT=all vc4infer build vc4det-n3_96.onnx -o vc4det-n3_96.vc4x
vc4infer bench vc4det-n3_96.vc4x --imgsz 96 --reps 5
```

| input | wall | FPS | launches | QPU | QPU share | GFLOP issued | host bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | **25.0 ms** | **40.0** | **18** | 13.4 ms | 53.5 % | 0.046 | 0 |

All 22 nodes on the QPUs, 0 folded, 0 host-metadata, 0 host-compute, 0 bytes
copied. The per-run spread over the three runs was 25.0 / 25.0 / 25.0 ms median.

Detections on the two test images (`models/BloodImage_00007.jpg`,
`_00011.jpg`) with `models/bccd_names.txt`; both are in the test split, whose
labels carry 18 and 19 boxes:

| | 00007 | 00011 |
|---|---|---|
| ground truth | 18 (17 RBC, 1 WBC) | 19 (17 RBC, 1 WBC, 1 PLT) |
| YOLO11n at 96, conf 0.25 | 8 (7/1) | 7 (6/1) |
| YOLO11n at 160, conf 0.25 | 14 (12/1/1) | 14 (12/1/1) |
| **student at 96, conf 0.25** | **9 (8/1)** | **11 (10/1)** |
| **student at 96, conf 0.10** | **17 (16/1)** | **17 (16/1)** |

At the default conf 0.25 the student finds more cells than the teacher at the
same input size and fewer than at 160; at conf 0.10 it finds 17 of 18 and 17 of
19, missing the platelets that the grid cannot represent.

### The GPU this was trained on

One measured caveat, because it produced a wrong answer before it was found. On
this workstation (RX 570, gfx803, ROCm 5.4.3, torch 1.13.1) the **fused** form
of the graph -- plain `Conv2d(bias=True)` -- does not compute the same values on
the GPU twice:

```
trained student, Conv+BN (training form)   cpu vs gpu   1.2e-06
trained student, BN folded (export form)   cpu vs gpu   3.5e+00
same input, two consecutive GPU runs                    2.3e+00
```

The BatchNorm form is exact and is what trains; the folded form is what exports,
and the export runs on the CPU, so no number above is affected. But a teacher
scored by calling its `.pt` on the GPU at 96 reported mAP50 0.004 until the same
weights were run through ONNX Runtime on the CPU, which reported 0.409. Every
accuracy number in this section was produced on the CPU for that reason. Treat
`Conv2d(bias=True)` at small spatial sizes as suspect on this stack.

### Verdict

The student is a detector. At 96x96 it runs at **40.0 FPS with 18 launches** and
scores **mAP50 0.596 / mAP50-95 0.426** on the val split, against the deployed
YOLO11n's **2.40 FPS / 419 ms** and **0.409 / 0.231** at the same input, and
against the best point of the deployed model, 160 at **1.93 FPS**, whose
checkpoint scores 0.904 at 320 under ultralytics' validator. Sixteen times the
frame rate at the size both can run, for two thirds of that accuracy: that is
the trade this graph was built to make, and it holds with trained weights, not
just with the right node count.

What it is not: a three-class detector. Platelets are gone at 96 (AP50 0.000),
so in practice it counts RBC and WBC. It is also at the end of its accuracy
curve -- over the last 160 epochs mAP50 stayed between 0.595 and 0.601 while the
learning rate annealed, and the remaining error is the 12x12 grid against
objects 5-43 px wide, not the schedule.

Still not reproducible from this repository alone: the merged dataset, the
teacher checkpoint and a gfx803 workstation. What is reproducible given those is
the whole chain -- `pseudo.py`, `train_student.py`, `check_graph.py`,
`build.py`, `vc4infer nodes` / `build` / `bench` -- with the student's identity
to the reference graph checked numerically rather than asserted.
