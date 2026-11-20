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
3. **Board load.** a concurrent build moves ARM numbers by 6–25 %. Every row
   below states the state it was taken in.

Board: Raspberry Pi 3B, BCM2837, VideoCore IV (V3D), 12 QPUs, 6 TMUs,
905 MiB RAM, `6.18.50+rpt-rpi-v8`, kernel CMA 256 MiB.

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

## 11. Completeness (ONNX, opset 22) — measured 2026-11-20, merged tree

Generated by `infer/tools/op_matrix.py` on `vc4infer` 1.2.4 with `ops-norm2` and
`ops-trig2` merged, stock clocks pinned at 400 MHz, quiet board. Full table:
[`COMPLETENESS.md`](COMPLETENESS.md).

| denominator | ops | **complete (vc4)** | vc4-limited | refused | mismatch | hang | folded | **pct_complete** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all ops in opset 22 | 193 | **104** | 2 | 42 | 0 | 0 | 3 | **53.89 %** |
| not structurally host-only | **152** | **104** | 2 | 42 | 0 | 0 | 3 | **68.42 %** |

`folded` (3 window generators) is resolved at plan time and never counted as
vc4. Two denominators are always reported; `pct_expressible` (69.74 %) counts
the general case rather than every attribute variant.

The 42 remaining refusals, by kind:

| kind | operators |
|---|---|
| cheap gaps, not hardware limits | `Sum`, `Max`, `Min`, `Mean` (variadic element-wise), `Div` (non-constant divisor), `Cast` (general), `LpNormalization` |
| gather/scatter family | `Gather`, `GatherElements`, `GatherND`, `ScatterElements`, `ScatterND`, `NonZero`, `TopK`, `Unique`, `ReverseSequence`, `CenterCropPad`, `DepthToSpace` |
| geometry / detection | `ConvTranspose`, `Col2Im`, `DeformConv`, `GridSample`, `RoiAlign`, `MaxRoiPool`, `NonMaxSuppression`, `Det`, `Einsum`, `AffineGrid` |
| recurrent | `LSTM`, `GRU`, `RNN` |
| signal / random / quantise | `DFT`, `STFT`, `MelWeightMatrix`, `RandomNormal(LIKE)`, `RandomUniform(LIKE)`, `Bernoulli`, `Multinomial`, `QuantizeLinear`, `DynamicQuantizeLinear` |

Previous measurement for comparison: **40/152 (26.3 %)** before the reductions /
comparisons / shape block, the 17 pooling-norm ops and the 9 inverse-hyperbolic
ops landed.

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
