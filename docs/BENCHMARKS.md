# Benchmarks and comparisons — the registry

Every number this project quotes, in one place, with the conditions it was taken
under. If a figure appears in another document and not here, this file wins.

**How to read a row.** A number is only comparable to another row with the same
**clock state**, **backend** and **board state**. Three things invalidated
measurements during this work and are now controlled:

1. **Core clock.** The firmware's DVFS cannot see QPU work issued through
   `vc4d`, and drops the core **400 → 250 MHz** mid-run. `/boot/firmware/config.txt`
   now pins `core_freq=400` and `core_freq_min=400` (the stock maximum, no
   voltage change). `docs/CLOCK-VARIANCE.md`.
2. **Backend.** `managed` (vc4d, ~0.1 ms/launch) vs `direct` (root, no
   per-launch cost) differ by **7 % on a GEMM and 16 % on YOLO**. `docs/BACKENDS.md`.
3. **Board load.** A concurrent build moves ARM numbers by 6–25 %. Every row
   below states the state it was taken in.
4. **Temperature — since 2026-11-20 the board is fanless.** Idle ~56 C, and a
   sustained QPU load reaches **73 C in under two minutes**; the soft limit is
   80 C, after which the firmware cuts the clock. A long run therefore drifts
   into a different regime mid-measurement, and *silently favours short runs*.
   Long commands are wrapped in `tools/thermal-guard.sh --max 72`, and any row
   taken after this date should carry its start/peak temperature. The ramp and
   the rules are in `docs/CLOCK-VARIANCE.md`.

Board: Raspberry Pi 3B, BCM2837, VideoCore IV (V3D), 12 QPUs, 6 TMUs,
905 MiB RAM, `6.18.50+rpt-rpi-v8`, kernel CMA 256 MiB. **Fanless since
2026-11-20** (~56 C idle, 73 C after ~2 min of QPU load): see rule 4.

---

## 1. Hardware envelope

| quantity | value | how |
|---|---|---|
| QPU peak, fp32 ALU only | **28.8 GFLOP/s** at 400 MHz | 12 QPUs × 16 lanes × 2 × (400 MHz / 4 clocks) |
| same, against Broadcom's 24 @ 250 MHz | consistent | `docs/XCHECK.md` |
| measured ALU-only ceiling | **28.36 GFLOP/s (98.5 %)**, 4.00 clocks/instruction | microbenchmark, `docs/NUMERICS.md` |
| issue rate | **1 instruction per 4 clocks**, 2 ops/word (add + mul ALU) | AG100-R + measured |
| L2 | 128 KB, GPU-only, outside the ARM MMU | Broadcom wiki, confirmed by the page-cache incident |
| VPM window | 48 rows at `VPMBASE=16` | measured |
| TMU | 2 outstanding requests per channel, ~20-instruction latency | measured + Anholt |

## 2. Dense GEMM, 256³ (the headline compute number)

| implementation | GFLOP/s | ms | conditions |
|---|---:|---:|---|
| hand-written `asm10` (GEMM only) | **13.72** | 3.35 | direct, 12 QPUs, clock pinned |
| hand-written `asm10` (total incl. packing) | **11.86–11.97** | 3.9 | same |
| engine `libvc4infer`, `direct` backend | **8.47** | 3.964 | root, no daemon |
| engine, `managed` backend (vc4d) | **8.44** | 4.264 | unprivileged |
| VC4CL (compiler path, `perf/combined`) | **3.085** | 11.0 | `managed`, N=256 |
| VC4CL, N=128 | 2.659 | — | `managed` |
| **CPU, 4× A53 (torch, 4 threads)** | **9.24** | 3.632 | board quiet |
| `torch_vc4` eager | 6.6 | 5.090 | includes Python dispatch |

At 128³ the daemon's per-launch cost is a much larger share (~30 % vs 7 %) —
that is what `docs/BACKENDS.md` is about. The engine's 8.47 vs asm10's 13.7 is
the engine's tile/packing strategy, not a layer tax.

## 3. YOLOv8n, real COCO weights — the progression

All at 160×160 unless stated, unprivileged `managed` backend, tolerance
`atol=rtol=5e-4`, compared against two independent CPU references.

| change | ms/image | FPS | launches | GPU ms | QPU duty | host bytes | over tol |
|---|---:|---:|---:|---:|---:|---:|---:|
| first end-to-end | 720.2 | 1.39 | 1145 | 279.8 | 39 % | — | 0/44,100 |
| SiLU pair fused at graph level | 622.7 | 1.61 | 678 | 242.5 | 39 % | — | 0/44,100 |
| native stride-2 / grouped / dilated / bilinear | 638.3 | 1.57 | 678 | 243.4 | 38 % | — | 0/44,100 |
| device-side layout copies (`copy2d`) | 542.2 | **1.84** | 784 | 320.3 | **59.1 %** | **0** | 0/44,100 |
| clock pinned at 400 MHz | 487.0 | **2.05** | 784 | 258.2 | 53.0 % | 0 | 0/44,100 |
| 1.2.4, `managed` | 492.3 | 2.03 | 784 | 258.5 | 52.5 % | 0 | 0/44,100 |
| 1.2.4, `direct` | **412.2** | **2.43** | 784 | 239.8 | 58.2 % | 0 | 0/44,100 |

Host byte movement on this model went **38,410,632 B → 17,077,040 B → 0 B**
(first measurement, after weight caching, after `copy2d`). The copies added
17.2 ms of QPU time for 17.1 MB (≈1.1 GB/s) and gave back ~126 ms of ARM time.

### 320×320

| state | ms/image | FPS | launches | QPU duty | over tol | recipe |
|---|---:|---:|---:|---:|---|---|
| first run (`--arena-mb 192`) | 1418.6 | 0.70 | 1824 | — | 1/176,400 | 192 MiB |
| native conv variants | 1569.5 | 0.64 | 1824 | — | 1/176,400 | `--arena-mb 150` |
| device copies, 150 MiB | **1323.1** | **0.76** | 1936 | 71.9 % | 1/176,400 | `vc4d --arena-mb 150`, `VC4INFER_ARENA_MB=145`, `VC4_POOL_RESERVE_MB=32` |

640×640: **does not fit** (working set + scratch ≈ 91 MiB against the arena, and
15 convs exceed the im2col pitch). No number is claimed.

### Small models (ONNX, shipped as `selftest`)

| model | launches | ms (first) | ms (1.2.3+) | accuracy |
|---|---:|---:|---:|---|
| modelA: conv/relu/pool/gemm/softmax 32×32 | 38 | 38.6 | 27.3–29.9 | max abs 4.84e-06, 0/10 |
| modelB: stride-2, depthwise, 5×5 pool, concat, resize | 26 → 19 | 35.8 | 18.9–21.3 | max abs 4.41e-06, 0/16,384 |
| modelC: conv/GAP/gemm 8×8 | 9 | 6.0 | 5.6–6.4 | max abs 4.06e-07, 0/4 |

Accuracy is its own document: [`ACCURACY.md`](ACCURACY.md) — VC4 vs ORT CPU on
the same image gives **identical detections (0.00 px)**, the raw tensor differing
by 3.2e-4 relative.

## 4. `torch_vc4` vs the CPU, per op

Quiet board, `torch.set_num_threads(4)`, median of 5 after 2 warm-ups, vc4 wall
includes the synchronous round trip (`torch-vc4/COMPLETENESS.md`).

| op | vc4 ms | CPU ms | ratio |
|---|---:|---:|---:|
| `mm` 32³ / 128³ / 256³ | 0.421 / 1.042 / **5.090** | 0.040 / 0.345 / **3.632** | 10× / 3.0× / **1.4×** |
| conv dense s1 16→16 k3 (3 sizes) | 1.565 / 10.066 / 9.486 | 0.077 / 2.450 / 3.209 | 20× / 4.1× / 3.0× |
| conv dense s2 (native NHWC) | 1.781 / 5.680 / 6.122 | 0.085 / 1.589 / 1.899 | 21× / 3.6× / 3.2× |
| conv depthwise (native NHWC) | 1.970 / 3.376 / 7.342 | 0.893 / 1.138 / 1.405 | 2.2× / 3.0× / 5.2× |
| `max_pool2d` 3×3 s2 | 1.32 | 0.087 | 15× |
| `adaptive_avg_pool2d` | 1.36 | 0.077 | 18× |
| `relu` 1024 / 65536 | 0.25 / 3.34 | 0.020 / 0.130 | 13× / 26× |
| `sigmoid` 1024 / 65536 | 0.28 / 4.87 | 0.045 / 0.935 | 6× / 5× |
| `softmax` 64×10 | 0.73 | 0.035 | 21× |
| `upsample` 2× 16×8×8 | 1.00 | 0.141 | 7× |
| `cat` 16+16 8×8 | 0.30 | 0.027 | 11× |
| `batch_norm` eval 16×16×16 | 4.00 | 0.076 | 53× |
| `sum` 1024 / 4096 | 2.18 / 5.25 | 0.039 / 0.029 | 55× / 182× |

The honest summary: **the eager backend is slower than the CPU at every size
measured**, by 1.4× on the largest GEMM and 5–180× on small ops. It exists so the
work happens on the GPU and the same code trains, not to win these rows.

## 5. Training

| quantity | value | conditions |
|---|---|---|
| MLP 64-32-10, SGD, 300 steps | **4.638 s = 64.7 steps/s**, 100 % train/test | `vc4train`, arena 64 MiB |
| CNN demo (conv+BN+relu+maxpool+FC+softmax-CE), 40 steps | loss **2.63994 → 0.005819**, final **3.564e-4 train / 3.784e-4 test** | batch 16 |
| step rate | **0.98–1.37 steps/s**, 16 images/step | quiet board |
| arena use | **7.84 MiB of 40 MiB** | |
| determinism | weights **bit-identical** across runs (`fnv` hashes), equal to a run two sessions earlier | |
| gradient checks | **46 checks ≤ 5.4e-6** norm-relative; **19/19** backward ops in the one-op-per-process bisect | |
| Adam state per parameter | **16 B** (4 weights + 4 grads + 8 m/v) | |
| fits with Adam, 64 MiB arena | ~**64-channel 64×64** conv nets | |
| does not fit | 128-channel 128×128 (**Adam state alone 113 MiB**); YOLOv8n (3.2 M params → 51 MB state) | |

## 6. Compiler and build

| measurement | value |
|---|---|
| VC4C branch `perf/spill-retry` | 130/130 kernels compile |
| `perf/product-acc` | kernel 517 → **453 words** (+15–17 %) |
| `perf/pipeline2` | **+24.6 %** at N=128 |
| `perf/pipeline3` / `perf/pipeline4` | +2 % / +1.2 % |
| `perf/combined` (installed) | **2.668 / 3.085 GFLOP/s** (N=128 / 256) |
| engine build, `-j4` vs `-j2` | **15.1 s vs 21.0 s**, 171 MiB peak, no swap |
| VC4C build, `-j2` vs `-j4` | **113.9 s vs 132.4 s**; `-j4` pushes **800 MiB swap** (slower) |
| ccache, 4 heavy C++ TUs | **113.9 s → 8.5 s** warm; cache only 8 MB |
| one kernel, `vc4c --bin` | **3.75 s**, 100 % of one core (single-threaded per kernel) |

## 7. Data movement

| quantity | value |
|---|---|
| `vc4_copy2d` bulk copy | **1 MiB in 2.31 ms = 454 MB/s** (`units=4`); 4.08 ms (`units=1`) |
| YOLO device copies added | 17.2 ms QPU for 17.1 MB (≈1.1 GB/s) |
| host bytes, YOLO 160 | **0** (was 17,077,040; before that 38,410,632) |
| device invariant | `host_compute_nodes 0`, `arithmetic_on_vc4 100 %` |

## 8. ONNX Runtime execution provider (160×160)

| backend | ms/image | FPS | over 5e-4 |
|---|---:|---:|---:|
| ORT + `vc4` provider | 699 | 1.43 | 0/44,100 |
| `vc4infer` native | 882 | 1.13 | 0/44,100 |
| ORT CPU EP | **260** | **3.85** | 0/44,100 |

Node split: **233 `vc4` + 30 folded + 0 `ort-cpu`** (fully claimed, 100 % of
MACs). Measured while the board was shared; the provider's value is moving work
off the ARM, not beating the CPU at this size, and the doc says so.

## 9. Backend overhead (the layered comparison)

| layer | cost | evidence |
|---|---:|---|
| packaging (`.deb`, apt, wheel, `-dev`) | **0 at run time** | install-time only |
| `libvc4infer` in-process | **no per-FLOP cost** | direct GEMM = the kernel's own 8.47 GFLOP/s |
| `vc4d` (the HAL) | **~0.1 ms per launch** | matmul 4.264 vs 3.964 ms (**7 %**); YOLO 492 vs 412 ms (**16 %**) |
| Python API (ctypes + numpy) | larger than the HAL on small ops | `mm 256³` 5.09 ms vs 3.97 ms from C |
| CLI | ~0.2 s of imports, once | |

Rule: **the overhead is per launch, not per FLOP.**

## 10. Clocks, thermal, power

| observation | value |
|---|---|
| sustained mixed load with the fan (10 min, 19 rebuilds + GEMM loop) | **89/89 samples at 400 MHz**, `throttled=0x0`, 52–61 °C |
| QPU-only sustained load, before the pin | core drops **400 → 250 MHz** after ~4 s, at 43 °C, 1.3125 V — **not thermal** |
| after `core_freq=400` + `core_freq_min=400` | **400 MHz in 5/5 samples** under a 400-rep load; 44.5–46.2 °C |
| effect on `mm 256³` | 3.73 → **8.47 GFLOP/s** |
| effect on YOLO 160 | 767 ms → **487 ms** |
| ARM contention (4 spinners) | VC4CL −25 %, asm10 −7 %, engine −8 % (32 % with pure spinners) |

## 11. Completeness (ONNX, opset 22) — measured 2026-11-21, feat/op-152

Generated by `infer/tools/op_matrix.py` on `vc4infer` 1.2.10 (source build of
this tree), stock clocks pinned at 400 MHz, fanless (peak 63.4 C, no
throttling).  Full table: [`COMPLETENESS.md`](COMPLETENESS.md).

| denominator | ops | **complete (vc4)** | general | vc4-limited | refused | mismatch | hang | folded | **pct_complete** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ops in opset 22 | 193 | **152** | 152 | 0 | **0** | **0** | **0** | 0 | **78.76 %** |
| not structurally host-only | **152** | **152** | 152 | 0 | **0** | **0** | **0** | 0 | **100.00 %** |

`complete` = every required form tested here ran within tolerance.  Nothing is
`folded`, `limited`, `unverified` or refused at the operator level.  The four
refused *cases* in the generated table are the declared edge forms
(`Concat` on axes other than 1, `MaxPool` `ceil_mode=1`, `Resize` `cubic`),
which are not required forms of those operators.

Four additions closed the last nine operators:

* `Slice` and `Transpose` fall back to a rank-generic index table plus
  `vc4_gather` (any axes, any non-zero step including negative, any perm, any
  rank) when the affine `copy2d` path does not cover the form;
* `Col2Im`, `GlobalLpPool` and `MaxRoiPool` are compared against an
  independent numpy oracle written from the ONNX definition, used only where
  `onnx.reference` 1.17 cannot evaluate the graph;
* `HannWindow`, `HammingWindow`, `BlackmanWindow` and `MelWeightMatrix` are
  QPU table generators now, not plan-time constants (index ramps as metadata,
  values by `SIN`/`COS`/compare/affine kernels).

**Refused: 0.**  Every required form of every operator in the expressible set
runs on the QPU.

## 12. Framework support effort (surveyed, not measured)

| framework | extension point | effort |
|---|---|---|
| ONNX | native executor | **available now** |
| PyTorch | `PrivateUse1` + `AutogradPrivateUse1` | **available now** (subset) |
| ONNX Runtime | EP C API | medium (weeks); Python provider shipped |
| ggml / llama.cpp | `ggml_backend_i` | medium |
| TFLite / LiteRT | `tflite_plugin_create_delegate` | medium |
| Apache TVM | Relax BYOC | large |
| JAX | PJRT C API | large |
| TensorFlow | StreamExecutor `SE_InitPlugin` | large |
| OpenVINO | `Core::register_plugin` | large |
| MXNet | custom-op only | not viable (`libarmpl` missing) |

## 13. What each change bought (one line each)

| change | measured effect |
|---|---|
| SiLU fusion at graph level | 1145 → 678 launches; 1.39 → 1.61 FPS |
| device-side layout copies | 17.1 MB host → 0; 1.57 → 1.84 FPS (and fixed the last determinism bug) |
| cacheable-arena coherency fix | 20/20 deterministic runs, previously 20–24,000 elements over |
| clock pin at stock 400 MHz | 1.84 → 2.05 FPS; GEMM 3.73 → 8.47 GFLOP/s |
| `direct` backend | 2.03 → 2.43 FPS |
| `/tmp` off tmpfs | 452 MB RAM returned; 1.84 → 2.09 FPS in that window |
| `-O3 -march=native` ("fast" package) | **within noise (1.4–1.7 % slower)**; the win is 401 prewarmed blobs, zero JIT |


## 14. YOLO matrix: six families × five sizes × three resolutions

Measured 2026-11-22 on the board in §1: 905 MiB RAM, core pinned at 400 MHz,
fanless, `vc4infer` 1.2.8 with `vc4d` 1.2.9.  The original six-family grid
covers **45 of its 90 nominal cells**.  Every one of those 45 has an ONNX
Runtime CPU baseline; one runs on the VC4 engine.  The other 45 ended in an
export memory limit, an arena refusal, or an unsupported operator, and each
record carries the exact first refusal.  With YOLO26 added below the nominal
total is **120** and coverage is **60 of 120**.  The machine-readable form is
[`yolo-matrix.json`](yolo-matrix.json).

**Models.**  `ultralytics` 8.4.215 provides v5u, v8, v9, v10, v11 and v12.
"YOLOv5" here is the Ultralytics `yolov5*u.pt` re-export; the standalone
`yolov5` package is not installed.  YOLOv9 was published as t/s/m/c/e, so the
n column uses `yolov9t`, the x column uses `yolov9e`, and the l column has no
checkpoint.  Sizes are the published n/s/m/l/x where they exist.

**Export.**  torch 2.15.0+cpu, opset 17, batch 1, fixed `imgsz`,
`simplify=False`, `device="cpu"`, one model at a time, `TMPDIR` on ext4.
The requested cgroup guard is inert on this kernel: `/proc/cmdline` contains
`cgroup_disable=memory`, root `cgroup.controllers` lists only
`cpuset cpu io pids`, and a 300 MB allocation inside
`systemd-run --scope --property=MemoryMax=100M` completed with no `memory.max`
in the scope.  Exports therefore ran under a per-process RSS watchdog
(`mem_watch.py`): 550 MiB for n/s at 160, 700 MiB for every other cell, and a
hard stop at 80 MiB of board-wide MemAvailable.  The watchdog kills only that
process group; rc=137 is recorded as `memory limit reached`, never as a crash.

**Plan.**  `Engine.plan()` against the default daemon (96 MiB served, 64 MiB
client arena).  A 150 MiB plan was attempted through a second `vc4d` on a
private socket: requested at `--arena-mb 150`, it started with a **37 MiB**
arena because CMA had 37 MiB free with the 96 MiB and 48 MiB daemons resident.
Stopping the system daemon is excluded by the run rules, so there is no
measured 150 MiB column.  Where the engine names a number ("about N MiB would
do"), that number is recorded as the need, not as a measured fit.

**Timing.**  Three warm-up and five timed images, the same fixed resize of the
Ultralytics `bus.jpg` sample on both backends, `VC4INFER_DAEMON=1`, managed
backend.  CPU is ONNX Runtime 1.32.0, `CPUExecutionProvider`,
`intra_op_num_threads=4`, `inter_op_num_threads=1`.  Every run is wrapped in
`tools/thermal-guard.sh --max 71`.  The first pass began with the SoC at 79 C
and the guard killed 26 CPU and 32 VC4 runs at the limit; those records were
discarded and every killed cell was re-run after cooling to 60 C.  No
thermal-killed number appears below.  The one surviving VC4 row started at
65.5 C and peaked at 67.1 C (`throttled=0x20000`, core 400 MHz).

**Accuracy.**  The CPU output on the fixed image is the reference.  `max abs`
is the largest elementwise difference over every graph output; `scaled` is the
largest `|VC4-CPU|/(1+|CPU|)`; `det` is boxes after a 0.25 score threshold and
class-wise NMS at 0.45, VC4/CPU; `box` is the largest matched corner delta in
pixels (same class, IoU > 0.5).  **COCO mAP was not evaluated**; this is parity
with the CPU graph, not detection quality.

### Export

| version | size | weights | 160 s / MiB | 320 s / MiB | 640 s / MiB |
|---|---|---|---|---|---|
| v5 | n | yolov5nu.pt | 29 / 10.2 | 29 / 10.2 | 32 / 10.3 |
| v5 | s | yolov5su.pt | 32 / 34.9 | 25 / 35.0 | 32 / 35.1 |
| v5 | m | yolov5mu.pt | 89 / 95.8 | — | — |
| v5 | l | yolov5lu.pt | — | — | — |
| v5 | x | yolov5xu.pt | — | — | — |
| v8 | n | yolov8n.pt | 28 / 12.1 | 19 / 12.1 | 22 / 12.2 |
| v8 | s | yolov8s.pt | 43 / 42.6 | 30 / 42.6 | 38 / 42.8 |
| v8 | m | yolov8m.pt | — | — | — |
| v8 | l | yolov8l.pt | — | — | — |
| v8 | x | yolov8x.pt | — | — | — |
| v9 | n | yolov9t.pt | 32 / 8.2 | 26 / 8.2 | 30 / 8.3 |
| v9 | s | yolov9s.pt | 46 / 27.6 | 36 / 27.6 | 44 / 27.8 |
| v9 | m | yolov9m.pt | 76 / 76.7 | 65 / 76.7 | — |
| v9 | l | none | — | — | — |
| v9 | x | yolov9e.pt | — | — | — |
| v10 | n | yolov10n.pt | 23 / 8.8 | 20 / 8.9 | 24 / 9.0 |
| v10 | s | yolov10s.pt | 32 / 27.7 | 28 / 27.8 | 38 / 27.9 |
| v10 | m | yolov10m.pt | 64 / 58.7 | 49 / 58.7 | — |
| v10 | l | yolov10l.pt | — | — | — |
| v10 | x | yolov10x.pt | — | — | — |
| v11 | n | yolo11n.pt | 24 / 10.1 | 21 / 10.1 | 25 / 10.2 |
| v11 | s | yolo11s.pt | 36 / 36.1 | 34 / 36.1 | 39 / 36.2 |
| v11 | m | yolo11m.pt | 92 / 76.7 | 68 / 76.8 | — |
| v11 | l | yolo11l.pt | — | — | — |
| v11 | x | yolo11x.pt | — | — | — |
| v12 | n | yolo12n.pt | 29 / 10.0 | 25 / 10.0 | 35 / 10.2 |
| v12 | s | yolo12s.pt | 41 / 35.5 | 34 / 35.5 | 47 / 35.6 |
| v12 | m | yolo12m.pt | 85 / 77.1 | 72 / 77.1 | — |
| v12 | l | yolo12l.pt | — | — | — |
| v12 | x | yolo12x.pt | — | — | — |

### Fit

| version | size | 160 | 320 | 640 |
|---|---|---|---|---|
| v5 | n | unsupported op | needs 95 > 64 | needs 292 > 64 |
| v5 | s | needs 96 > 64 | needs 146 > 64 | needs 348 > 64 |
| v5 | m | needs 218 > 64 | — | — |
| v5 | l | — | — | — |
| v5 | x | — | — | — |
| v8 | n | fits 64 (peak 42) | fits 64 (peak 85) | needs 126 > 64 |
| v8 | s | needs 100 > 64 | needs 119 > 64 | needs 194 > 64 |
| v8 | m | — | — | — |
| v8 | l | — | — | — |
| v8 | x | — | — | — |
| v9 | n | fits 64 (peak 20) | fits 64 (peak 29) | needs 119 > 64 |
| v9 | s | fits 64 (peak 63) | needs 89 > 64 | needs 164 > 64 |
| v9 | m | needs 172 > 64 | needs 191 > 64 | — |
| v9 | l | — | — | — |
| v9 | x | — | — | — |
| v10 | n | unsupported op | unsupported op | needs 141 > 64 |
| v10 | s | needs 209 > 64 | needs 226 > 64 | needs 301 > 64 |
| v10 | m | needs 263 > 64 | needs 283 > 64 | — |
| v10 | l | — | — | — |
| v10 | x | — | — | — |
| v11 | n | unsupported op | unsupported op | needs 127 > 64 |
| v11 | s | needs 103 > 64 | needs 122 > 64 | needs 197 > 64 |
| v11 | m | needs 201 > 64 | needs 222 > 64 | — |
| v11 | l | — | — | — |
| v11 | x | — | — | — |
| v12 | n | unsupported op | unsupported op | needs 141 > 64 |
| v12 | s | needs 161 > 64 | needs 180 > 64 | needs 255 > 64 |
| v12 | m | needs 298 > 64 | needs 326 > 64 | — |
| v12 | l | — | — | — |
| v12 | x | — | — | — |

The need is the engine's own floor (weights, packed conv panels,
largest tensor) plus its scratch reserve and 8 MiB; `fits` means the
static plan accepted the graph at the 64 MiB client arena.

### Performance

#### 160 × 160

| version | size | CPU ms | CPU FPS | VC4 ms | VC4 FPS | ratio | launches | error | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v5 | n | 103.0 | 9.71 | — | — | — | — | — | unsupported on VC4 |
| v5 | s | 237.4 | 4.21 | — | — | — | — | — | does not fit (needs 96 MiB) |
| v5 | m | 555.0 | 1.80 | — | — | — | — | — | does not fit (needs 218 MiB) |
| v5 | l | — | — | — | — | — | — | — | not exportable (memory limit) |
| v5 | x | — | — | — | — | — | — | — | not exportable (memory limit) |
| v8 | n | 108.2 | 9.24 | 521.0 | 1.92 | 4.8× | 802 | 1.5e-02 / 3.1e-04; 4/4 det; box 0.00 px | ok (64 MiB arena) |
| v8 | s | 272.2 | 3.67 | — | — | — | — | — | does not fit (needs 100 MiB) |
| v8 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v8 | l | — | — | — | — | — | — | — | not exportable (memory limit) |
| v8 | x | — | — | — | — | — | — | — | not exportable (memory limit) |
| v9 | n | 114.3 | 8.75 | — | — | — | — | — | unsupported on VC4 |
| v9 | s | 283.5 | 3.53 | — | — | — | — | — | does not fit (runtime arena) |
| v9 | m | 659.8 | 1.52 | — | — | — | — | — | does not fit (needs 172 MiB) |
| v9 | l | — | — | — | — | — | — | — | not exportable (no upstream l weights) |
| v9 | x | — | — | — | — | — | — | — | not exportable (memory limit) |
| v10 | n | 85.7 | 11.67 | — | — | — | — | — | unsupported on VC4 |
| v10 | s | 223.4 | 4.48 | — | — | — | — | — | does not fit (needs 209 MiB) |
| v10 | m | 508.7 | 1.97 | — | — | — | — | — | does not fit (needs 263 MiB) |
| v10 | l | — | — | — | — | — | — | — | not exportable (memory limit) |
| v10 | x | — | — | — | — | — | — | — | not exportable (memory limit) |
| v11 | n | 93.3 | 10.71 | — | — | — | — | — | unsupported on VC4 |
| v11 | s | 386.0 | 2.59 | — | — | — | — | — | does not fit (needs 103 MiB) |
| v11 | m | 486.8 | 2.05 | — | — | — | — | — | does not fit (needs 201 MiB) |
| v11 | l | — | — | — | — | — | — | — | not exportable (memory limit) |
| v11 | x | — | — | — | — | — | — | — | not exportable (memory limit) |
| v12 | n | 99.2 | 10.08 | — | — | — | — | — | unsupported on VC4 |
| v12 | s | 383.5 | 2.61 | — | — | — | — | — | does not fit (needs 161 MiB) |
| v12 | m | 598.3 | 1.67 | — | — | — | — | — | does not fit (needs 298 MiB) |
| v12 | l | — | — | — | — | — | — | — | not exportable (memory limit) |
| v12 | x | — | — | — | — | — | — | — | not exportable (memory limit) |

#### 320 × 320

| version | size | CPU ms | CPU FPS | VC4 ms | VC4 FPS | ratio | launches | error | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v5 | n | 274.5 | 3.64 | — | — | — | — | — | does not fit (needs 95 MiB) |
| v5 | s | 764.3 | 1.31 | — | — | — | — | — | does not fit (needs 146 MiB) |
| v5 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v5 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v5 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v8 | n | 349.2 | 2.86 | — | — | — | — | — | does not fit (runtime arena) |
| v8 | s | 783.4 | 1.28 | — | — | — | — | — | does not fit (needs 119 MiB) |
| v8 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v8 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v8 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v9 | n | 339.0 | 2.95 | — | — | — | — | — | unsupported on VC4 |
| v9 | s | 987.7 | 1.01 | — | — | — | — | — | does not fit (needs 89 MiB) |
| v9 | m | 1982.8 | 0.50 | — | — | — | — | — | does not fit (needs 191 MiB) |
| v9 | l | — | — | — | — | — | — | — | not exportable (no upstream l weights) |
| v9 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v10 | n | 303.5 | 3.30 | — | — | — | — | — | unsupported on VC4 |
| v10 | s | 756.7 | 1.32 | — | — | — | — | — | does not fit (needs 226 MiB) |
| v10 | m | 1989.0 | 0.50 | — | — | — | — | — | does not fit (needs 283 MiB) |
| v10 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v10 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v11 | n | 309.8 | 3.23 | — | — | — | — | — | unsupported on VC4 |
| v11 | s | 748.7 | 1.34 | — | — | — | — | — | does not fit (needs 122 MiB) |
| v11 | m | 2043.4 | 0.49 | — | — | — | — | — | does not fit (needs 222 MiB) |
| v11 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v11 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v12 | n | 323.7 | 3.09 | — | — | — | — | — | unsupported on VC4 |
| v12 | s | 812.0 | 1.23 | — | — | — | — | — | does not fit (needs 180 MiB) |
| v12 | m | 2144.0 | 0.47 | — | — | — | — | — | does not fit (needs 326 MiB) |
| v12 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v12 | x | — | — | — | — | — | — | — | not attempted (optional) |

#### 640 × 640

| version | size | CPU ms | CPU FPS | VC4 ms | VC4 FPS | ratio | launches | error | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v5 | n | 1207.0 | 0.83 | — | — | — | — | — | does not fit (needs 292 MiB) |
| v5 | s | 2598.0 | 0.38 | — | — | — | — | — | does not fit (needs 348 MiB) |
| v5 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v5 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v5 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v8 | n | 1330.3 | 0.75 | — | — | — | — | — | does not fit (needs 126 MiB) |
| v8 | s | 3224.5 | 0.31 | — | — | — | — | — | does not fit (needs 194 MiB) |
| v8 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v8 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v8 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v9 | n | 1480.3 | 0.68 | — | — | — | — | — | does not fit (needs 119 MiB) |
| v9 | s | 3593.6 | 0.28 | — | — | — | — | — | does not fit (needs 164 MiB) |
| v9 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v9 | l | — | — | — | — | — | — | — | not exportable (no upstream l weights) |
| v9 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v10 | n | 1164.6 | 0.86 | — | — | — | — | — | does not fit (needs 141 MiB) |
| v10 | s | 2937.0 | 0.34 | — | — | — | — | — | does not fit (needs 301 MiB) |
| v10 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v10 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v10 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v11 | n | 1132.6 | 0.88 | — | — | — | — | — | does not fit (needs 127 MiB) |
| v11 | s | 2840.8 | 0.35 | — | — | — | — | — | does not fit (needs 197 MiB) |
| v11 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v11 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v11 | x | — | — | — | — | — | — | — | not attempted (optional) |
| v12 | n | 1361.1 | 0.73 | — | — | — | — | — | does not fit (needs 141 MiB) |
| v12 | s | 3152.6 | 0.32 | — | — | — | — | — | does not fit (needs 255 MiB) |
| v12 | m | — | — | — | — | — | — | — | not exportable (memory limit) |
| v12 | l | — | — | — | — | — | — | — | not attempted (optional) |
| v12 | x | — | — | — | — | — | — | — | not attempted (optional) |

### Status per cell

#### 160 × 160

| version | n | s | m | l | x |
|---|---|---|---|---|---|
| v5 | unsupported | no fit (needs 96) | no fit (needs 218) | — | — |
| v8 | 4.8× CPU | no fit (needs 100) | — | — | — |
| v9 | unsupported | no fit (runtime) | no fit (needs 172) | — | — |
| v10 | unsupported | no fit (needs 209) | no fit (needs 263) | — | — |
| v11 | unsupported | no fit (needs 103) | no fit (needs 201) | — | — |
| v12 | unsupported | no fit (needs 161) | no fit (needs 298) | — | — |

#### 320 × 320

| version | n | s | m | l | x |
|---|---|---|---|---|---|
| v5 | no fit (needs 95) | no fit (needs 146) | — | — | — |
| v8 | no fit (runtime) | no fit (needs 119) | — | — | — |
| v9 | unsupported | no fit (needs 89) | no fit (needs 191) | — | — |
| v10 | unsupported | no fit (needs 226) | no fit (needs 283) | — | — |
| v11 | unsupported | no fit (needs 122) | no fit (needs 222) | — | — |
| v12 | unsupported | no fit (needs 180) | no fit (needs 326) | — | — |

#### 640 × 640

| version | n | s | m | l | x |
|---|---|---|---|---|---|
| v5 | no fit (needs 292) | no fit (needs 348) | — | — | — |
| v8 | no fit (needs 126) | no fit (needs 194) | — | — | — |
| v9 | no fit (needs 119) | no fit (needs 164) | — | — | — |
| v10 | no fit (needs 141) | no fit (needs 301) | — | — | — |
| v11 | no fit (needs 127) | no fit (needs 197) | — | — | — |
| v12 | no fit (needs 141) | no fit (needs 255) | — | — | — |

### What this says

**One cell runs, and it is slower than the CPU.**  Of 45 exported graphs, only
YOLOv8n at 160×160 completes on the QPU: 521.0 ms, 1.92 FPS, 802 launches,
with the QPU busy 50 % of the wall.  The same graph on four A53 cores takes
108.2 ms (9.24 FPS), so the VC4 is **4.8× slower**.  Parity is exact enough:
4/4 detections, largest matched box delta 0.0002 px, max scaled difference
3.1e-4 (the 1.5e-2 max absolute is the logit range, which is why the scaled
number is the meaningful one).

**The arena is the main hardware limit, and the gap is large.**  35 exported
cells hit the 64 MiB client arena: 33 were refused at plan time, and 2
(v8n320, v9s160) passed the plan and then died inside a kernel with `out of
arena memory`, so the planner's clamp is optimistic at 64 MiB.  The measured
needs range from 89 MiB to 348 MiB.  The worst are v5s640 **348 MiB (5.4× the
arena)**, v12m320 326 MiB (5.1×), v10s640 301 MiB, and v12m160 298 MiB.
Even the cheapest refusals are close to double the arena:
v8s160 needs 100 MiB, v8n640 126 MiB, v9n640 119 MiB.  A 150 MiB arena would
cover the n column at 640 and most of the s column at 160, but it cannot be
mapped on this boot because CMA is occupied by the resident daemons.

**Nine cells are op-limited, not size-limited.**  v5's 6×6 stride-2 stem
conv is unsupported; v9's Slice form needs a compact NCHW source; v10, v11 and
v12 use `Split` on axis 2 in their attention blocks.  These are visible at
160×160, where the arena is not in the way.  A larger board would not change
them; they need kernel or graph-rewrite work.  A larger arena would only move
the first refusal to the same operator.

**l and x are export-limited, not engine-limited.**  The largest successful
exports peaked at 439 MiB RSS for n, 597 MiB for s and 627 MiB for m.  Every
l and x attempt was killed by the watchdog (peak 486-610 MiB, or the 80 MiB
board floor), so no l/x ONNX graph exists to plan against.  On a board with
more RAM those cells would at least be measurable; on this one the answer says
nothing about whether the kernel could run them.

**The CPU fallback is the practical path at this size.**  All 45 exported
cells have a CPU number, from 85.7 ms (v10n160, 11.7 FPS) to 3594 ms
(v9s640, 0.28 FPS).  At 640 the fallback is 0.28-0.86 FPS across the six
families.

**What I did not measure.**  The 150 MiB arena column, the 22 l/x cells at
320 and 640 after every 160 l/x export hit the memory limit, and COCO mAP.
The 22 are marked `not attempted (optional)` in the JSON rather than given an
invented status.


### YOLO26 (added 2026-11-23)

YOLO26 is the current Ultralytics family, and `ultralytics` 8.4.215 loads
`yolo26n.pt` through `yolo26x.pt` and exports both heads.  With it the nominal
total becomes **120 cells**: seven families × five sizes × three resolutions
(105), plus the 15 NMS-free variants of the YOLO26 cells.  **60 of 120 have an
ONNX graph**: the 45 rows above plus 15 YOLO26 graphs.  No existing row was
re-measured or edited; the 90-row grid above keeps its own 45-of-90 count.

**Heads.**  `default` is `export(nms=None)`, the one-to-many head with raw
output `[1,84,N]` and external NMS.  `nmsfree` is `export(nms=False)`, the
end-to-end one-to-one head with top-k, output `[1,300,6]`.  The graphs differ
in the final blocks; the backbone is shared.  Both used opset 17, batch 1,
fixed `imgsz`, `simplify=False`, and the 700 MiB watchdog with the 80 MiB
board floor described above.

**Coverage.**  n and s exported at all three resolutions in both heads.  m is
asymmetric because the memory ceiling, not the graph, decided each cell:
`default` at 320 and `nmsfree` at 160 and 640.  l and x failed in both heads at
every resolution.  The asymmetry is repeatable in the logs but it is a
one-attempt-per-cell result, so it says the export is marginal at m, not that
one head is intrinsically cheaper.

#### Export

| size | head | 160 s / MiB | 320 s / MiB | 640 s / MiB |
|---|---|---|---|---|
| n | default | 36 / 9.3 | 32 / 9.3 | 34 / 9.4 |
| n | nmsfree | 24 / 9.3 | 22 / 9.3 | 25 / 9.4 |
| s | default | 35 / 36.3 | 31 / 36.3 | 38 / 36.5 |
| s | nmsfree | 31 / 36.3 | 36 / 36.4 | 42 / 36.5 |
| m | default | — | 67 / 78.0 | — |
| m | nmsfree | 66 / 78.0 | — | 95 / 78.1 |
| l | default | — | — | — |
| l | nmsfree | — | — | — |
| x | default | — | — | — |
| x | nmsfree | — | — | — |

#### Fit at 64 MiB

| size | head | 160 | 320 | 640 |
|---|---|---|---|---|
| n | default | unsupported op | unsupported op | needs 126 > 64 |
| n | nmsfree | unsupported op | unsupported op | needs 126 > 64 |
| s | default | needs 106 > 64 | needs 125 > 64 | needs 200 > 64 |
| s | nmsfree | needs 106 > 64 | needs 125 > 64 | needs 200 > 64 |
| m | default | — | needs 227 > 64 | — |
| m | nmsfree | needs 206 > 64 | — | needs 310 > 64 |
| l | default | — | — | — |
| l | nmsfree | — | — | — |
| x | default | — | — | — |
| x | nmsfree | — | — | — |

#### Performance

| size | head | imgsz | CPU ms | CPU FPS | VC4 ms | VC4 FPS | ratio | launches | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| n | default | 160 | 86.6 | 11.55 | — | — | — | — | unsupported on VC4 |
| n | nmsfree | 160 | 82.0 | 12.20 | — | — | — | — | unsupported on VC4 |
| n | default | 320 | 254.9 | 3.92 | — | — | — | — | unsupported on VC4 |
| n | nmsfree | 320 | 258.9 | 3.86 | — | — | — | — | unsupported on VC4 |
| n | default | 640 | 1019.9 | 0.98 | — | — | — | — | does not fit (needs 126 MiB) |
| n | nmsfree | 640 | 1090.2 | 0.92 | — | — | — | — | does not fit (needs 126 MiB) |
| s | default | 160 | 216.9 | 4.61 | — | — | — | — | does not fit (needs 106 MiB) |
| s | nmsfree | 160 | 210.2 | 4.76 | — | — | — | — | does not fit (needs 106 MiB) |
| s | default | 320 | 720.0 | 1.39 | — | — | — | — | does not fit (needs 125 MiB) |
| s | nmsfree | 320 | 795.4 | 1.26 | — | — | — | — | does not fit (needs 125 MiB) |
| s | default | 640 | 2842.4 | 0.35 | — | — | — | — | does not fit (needs 200 MiB) |
| s | nmsfree | 640 | 2761.9 | 0.36 | — | — | — | — | does not fit (needs 200 MiB) |
| m | default | 160 | — | — | — | — | — | — | not exportable (memory limit) |
| m | nmsfree | 160 | 577.5 | 1.73 | — | — | — | — | does not fit (needs 206 MiB) |
| m | default | 320 | 2020.2 | 0.50 | — | — | — | — | does not fit (needs 227 MiB) |
| m | nmsfree | 320 | — | — | — | — | — | — | not exportable (memory limit) |
| m | default | 640 | — | — | — | — | — | — | not exportable (memory limit) |
| m | nmsfree | 640 | 6772.8 | 0.15 | — | — | — | — | does not fit (needs 310 MiB) |
| l | default | 160 | — | — | — | — | — | — | not exportable (memory limit) |
| l | nmsfree | 160 | — | — | — | — | — | — | not exportable (memory limit) |
| l | default | 320 | — | — | — | — | — | — | not exportable (memory limit) |
| l | nmsfree | 320 | — | — | — | — | — | — | not exportable (memory limit) |
| l | default | 640 | — | — | — | — | — | — | not exportable (memory limit) |
| l | nmsfree | 640 | — | — | — | — | — | — | not exportable (memory limit) |
| x | default | 160 | — | — | — | — | — | — | not exportable (memory limit) |
| x | nmsfree | 160 | — | — | — | — | — | — | not exportable (memory limit) |
| x | default | 320 | — | — | — | — | — | — | not exportable (memory limit) |
| x | nmsfree | 320 | — | — | — | — | — | — | not exportable (memory limit) |
| x | default | 640 | — | — | — | — | — | — | not exportable (memory limit) |
| x | nmsfree | 640 | — | — | — | — | — | — | not exportable (memory limit) |

#### Status per cell

`default`:

| size | 160 | 320 | 640 |
|---|---|---|---|
| n | unsupported | unsupported | no fit (needs 126) |
| s | no fit (needs 106) | no fit (needs 125) | no fit (needs 200) |
| m | — | no fit (needs 227) | — |
| l | — | — | — |
| x | — | — | — |

`nmsfree`:

| size | 160 | 320 | 640 |
|---|---|---|---|
| n | unsupported | unsupported | no fit (needs 126) |
| s | no fit (needs 106) | no fit (needs 125) | no fit (needs 200) |
| m | no fit (needs 206) | — | no fit (needs 310) |
| l | — | — | — |
| x | — | — | — |

#### What this adds

**YOLO26 is faster on the CPU than the older families at the same size, and no
closer to the QPU.**  The n cell is 82.0 ms at 160 (12.20 FPS), against
108.2 ms for v8n on the same image, and 1019.9 ms at 640 (0.98 FPS).  The m
NMS-free graph at 640 is the slowest CPU cell in the whole matrix at 6772.8 ms
(0.15 FPS).  No YOLO26 cell reaches the VC4 engine, so there is no ratio to
report.

**The first refusal is the same attention block as v10-v12.**  At 160 and 320
the planner stops at `Split` on axis 2 and `MatMul` with a rank-4 operand in
`/model.10/m/m.0/attn`; this is the C2PSA block, not the YOLO26 head.  The
arena is the first refusal for every s and m cell and for n at 640: needs are
106 MiB (s160), 125 (s320), 200 (s640), 206 (m160 nmsfree), 227 (m320 default),
310 (m640 nmsfree) and 126 (n640), all against 64.

**The NMS-free head's two `ConstantOfShape` nodes are now folded at plan
time.**  They are constant anchor tensors, so they no longer appear as a
refusal (the 2026-11-22 fold); the shared Split/MatMul remains the only
operator refusal in the head.  `TopK`, `GatherElements`, `Mod`, `Equal`,
`ReduceMax`, `Where` and `Expand` are in the graph but this build does not
list them as unsupported, so their runtime behaviour is **unmeasured**: every
cell stops at the attention block or at the arena before reaching them.
Fixing Split/MatMul would expose that head to a second round of operator work.

**l and x remain export-limited.**  Both heads die at the watchdog at every
resolution; the observed peaks were 494-602 MiB.  As with the six families
above, that is a 905 MiB RAM limit, not an engine verdict.

**What I did not measure.**  Any VC4 execution for YOLO26, COCO mAP for either
head, and the runtime cost of TopK, GatherElements and the rest of the
end-to-end tail, which no cell reaches.

## 15. Models beyond YOLO (measured 2026-11-22)

Fifteen graphs outside the YOLO family, same protocol: nine ImageNet backbones
from the ONNX backend test corpus, the three shipped `modelA/B/C` graphs, and
three ONNX Model Zoo models (super-resolution, style transfer, int8
segmentation).  Full tables, exact refusal strings and the corpus caveat are in
[`MODELS.md`](MODELS.md).  Conditions: engine 1.2.10, managed, 96 MiB client
arena, ONNX Runtime 1.32.0 on four threads, 3 warm-ups + 5 timed, guard
`--max 71`; starts 56.9-60.7 C, peaks 59.1-70.4 C, core 400 MHz.  The re-run
of the three fixed models used `infer/build/libvc4infer.so.1`; the installed
copy predates the autotune entry point.

| group | graphs | run on VC4 | headline |
|---|---:|---:|---|
| shipped small models | 3 | **3** | 29.8 / 17.2 / 6.2 ms; CPU 2.4 / 1.8 / 0.2 ms (12.4x / 9.7x / 30x); max abs 6.9e-06 / 3.8e-06 / 2.6e-07 |
| ImageNet corpus, 224x224 | 9 | **1** | SqueezeNet runs: 2892.7 ms VC4, 92.1 ms CPU, 1000/1000 outputs bit-identical; 3 stop on the arena (104-135 MiB), 4 on weight tensors of 144-392 MiB, 1 on grouped-conv alignment |
| Model Zoo, other tasks | 3 | 0 | super-resolution needs 147 MiB of conv scratch (its `Cin=1` conv now plans 12/12); style transfer needs 540 MiB of 9x9-conv scratch; segmentation is int8 (`QLinearConv`) |

The corpus graphs are topology-only: every weight is a `ConstantOfShape` node
with the constant value 0.02.  The engine folds those nodes at plan time, so
every corpus row is the unmodified file; SqueezeNet runs end to end and its 39
folded tensors are byte-identical to `benchmodels/materialise.py`.  The other
small graphs are planned as published and stop at their own constraint
(arena, grouped-conv alignment).  `modelA/B/C` and SqueezeNet are the non-YOLO
models that execute on the QPUs.

The 152/152 operator result does not extend to model coverage.  The blockers
are the arena (104-603 MiB against 96, including single weight tensors of
144-392 MiB that cannot be materialised into it), conv kernels restricted to
1x1/3x3/5x5 (the corpus stems are 7x7 and 11x11; style transfer is 9x9),
grouped-conv 16-alignment, and int8 - not missing operators.
