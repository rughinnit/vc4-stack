# ncnn vs the VC4 engine — speed and accuracy, same model

ncnn is the reference ARM CPU inference engine for this class of hardware, so it
is the right thing to be measured against. This is a real comparison: **the same
`yolov8n.pt`**, exported twice, run on the **same image**, with the **same
post-processing**.

## Method

| | |
|---|---|
| model | `yolov8n.pt` (Ultralytics COCO), 3.15 M params |
| ncnn build | `YOLO("yolov8n.pt").export(format="ncnn", imgsz=160)` → `model.ncnn.param` + `.bin` (202 layers), ncnn **1.0.20261020** cp313 aarch64 wheel |
| VC4 build | `vc4infer` 1.2.4, `yolov8n_160.onnx`, `managed` backend |
| input | `bus.jpg`, letterboxed to 160×160 by **our** `postprocess.Letterbox` (identical tensor to both) |
| post-processing | **our** `decode_yolo` on both raw outputs — so the comparison isolates arithmetic and speed, not decode |
| board | Pi 3B, ARM 1200 MHz, core pinned 400 MHz, `throttled=0x0`, quiet |

### Conversion pipeline — PNNX, not `onnx2ncnn`

The ncnn model was **not** converted from our ONNX file. Ultralytics'
`export_ncnn()` calls `torch2ncnn()`, which runs **PNNX** (pinned
`pnnx==20260526`; installed 20260526):

```
yolov8n.pt --(torch.jit.trace)--> torchscript --(pnnx)--> model.ncnn.param + model.ncnn.bin
```

The result is **154 layers**, of which **58 are `Swish`/`Sigmoid`**, against our
ONNX graph of 263 nodes (206 dispatches after our own graph-level SiLU fusion).
So ncnn also collapses the sigmoid×mul pair to a single `Swish` layer — it is
not winning on epilogue fusion, but on having no per-layer handoff at all.

**What this means for reading the comparison.** Both sides come from the same
`yolov8n.pt` checkpoint, so the weights and the architecture are identical, but
they travel through **two different conversion pipelines** (`torch.onnx` vs
PNNX). Part of the 3.2e-4 difference could therefore be graph transformation
rather than arithmetic. The strict arithmetic comparison is ONNX Runtime CPU vs
VC4 on the **same ONNX file** (`docs/ACCURACY.md`) — and that gives the same
3.2e-4 and the same identical detections, which is why the accuracy conclusion
holds either way.

So: this table compares **engines, including their graph optimisation**, not
kernels in isolation. That is the comparison a user makes when choosing one.

## Speed (median of 7 after 3 warm-ups, ARM confirmed at 1199–1200 MHz)

| backend | threads | ms | FPS | relative |
|---|---:|---:|---:|---:|
| **ncnn** | 4 | **74.8** | **13.4** | **1.00×** |
| ncnn | 2 | 78.3 | 12.8 | 1.05× |
| ncnn | 1 | 106.6 | 9.4 | 1.43× |
| ONNX Runtime CPU | default | 88.9 | 11.2 | 1.19× |
| **vc4infer (QPUs)** | — | **481.7** | **2.08** | **6.44×** |

**ncnn is 6.4× faster than our GPU path on this workload.** That is the honest
result and it is not a small margin.

Two earlier readings in this project were contention, not the engines: ncnn
722 ms and ORT 260 ms were both taken while other work held the board. With the
board quiet they are 74.8 and 88.9 ms. Any number quoted from this project now
carries the board state for exactly this reason (`docs/BENCHMARKS.md`).

## Accuracy (ncnn output vs VC4 output, same input)

| | |
|---|---:|
| raw output max abs | **1.34e-02** |
| raw output mean abs | 2.82e-05 |
| max relative where `|ref| ≥ 1` | **3.22e-04** |
| over `5e-4` | **0 / 44,100** |
| over `1e-4` | 32 / 44,100 |
| **detections** | **4 vs 4, same classes** |
| **max box difference** | **0.00 px** |
| max score difference | 8.8e-05 |

```
ncnn  person 0.7784  [ 25.8  60.6  53.9 133.6]      vc4  person 0.7784  [ 25.8  60.6  53.9 133.6]
ncnn  bus    0.7030  [ 20.3  33.6 138.6 112.4]      vc4  bus    0.7029  [ 20.3  33.6 138.6 112.4]
ncnn  person 0.6193  [120.5  62.9 140.8 128.3]      vc4  person 0.6192  [120.5  62.9 140.8 128.3]
ncnn  person 0.5971  [ 50.5  58.0  72.3 127.9]      vc4  person 0.5971  [ 50.5  58.0  72.3 127.9]
```

**Accuracy: equivalent.** Identical detections, boxes to the pixel, scores
within 1e-4. Both also agree with ONNX Runtime's CPU EP to the same degree
(`docs/ACCURACY.md`).

## Why the CPU wins here, and by how much

YOLOv8n at 160×160 is **273 M MACs**. The four A53s sustain about 9 GFLOP/s, so
the arithmetic floor is ~30 ms; ncnn reaches 74.8 ms (≈3.6 GFLOP/s effective).
Our GPU side spends:

| | ms |
|---|---:|
| QPU kernel time (measured) | ~258 |
| launch path, 784 launches × ~0.29 ms | ~222 |
| total | **~480** |

So the GPU is not losing on arithmetic (its GEMM sustains 8.47 GFLOP/s, faster
than the CPU's 9.24 at 256³ within measurement error). It loses because **this
graph is 784 tiny kernels**, and every one of them pays a per-launch cost that
ncnn simply does not have: ncnn keeps a whole graph in one process, one thread
pool, with no per-layer round trip.

The measured overhead breakdown is in `docs/BACKENDS.md` (the HAL is ~0.1 ms of
that ~0.29 ms; the rest is the Python runner and the engine's per-call setup).

## What this means for the project

1. **For small-model inference on this board, a tuned CPU engine is the right
   default.** Our stack's case is unprivileged GPU execution, strictness (no
   silent CPU fallback), training, and offloading work from the ARM — not raw
   FPS at 160×160.
2. **The gap is launch count, not the QPUs.** Reducing 784 launches (fusion,
   epilogues, batching) attacks the actual 222 ms; a faster kernel would not.
3. **The comparison is now reproducible**: `docs/NCNN.md` + the scripts
   (`infer/tests`-style, kept out of tree as one-off) describe how to rebuild it
   in ~2 minutes (`python -c "from ultralytics import YOLO; \
   YOLO('yolov8n.pt').export(format='ncnn', imgsz=160)"`).
4. **It also caught two real bugs** (now in `docs/BUGS.md`): the daemon does not
   reclaim arena regions from departed clients (G9), and the runner's pool
   silently floors at 8 MiB with an error that blames the model (G10).

## Operational notes found while doing this

- **Direct backend and a running `vc4d` cannot both drive the QPUs.** A direct
  launch while the daemon is active timed out
  (`conv_nhwc_dil: GPU timeout ... board may need a reboot`), while the daemon
  itself stayed healthy. Use `sudo systemctl stop vc4d` before benchmarking in
  `direct` mode, and restart it afterwards.
- **The daemon exhausts its arena after enough clients** (`vc4d: DENY op=ALLOC:
  arena exhausted`) even though no client is running; `systemctl restart vc4d`
  clears it. Same family as the 128-slot limit.
