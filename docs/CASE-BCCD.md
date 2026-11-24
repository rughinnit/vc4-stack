# Case study: a trained blood-cell detector on the QPUs

A YOLO11n detector trained on the workstation (AMD RX 570, ROCm, gfx803) runs on
the board as a `.vc4x` artifact at **1.93 FPS / 519 ms** at 160x160, and its
detections on the two test images are **identical to ultralytics** on the same
checkpoint. This page records the whole chain and what of it is reproducible.

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
