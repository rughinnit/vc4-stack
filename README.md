# vc4-stack — ONNX, PyTorch, JAX and TensorFlow on the Raspberry Pi GPU

Runs neural networks on the **VideoCore IV** QPUs of a Raspberry Pi 3B — the GPU
that shipped with the board in 2016 — with the ARM cores doing only control and
byte movement.  There is no CPU arithmetic fallback: an operator either executes
on the QPUs or is refused by name.

**A vertically integrated compute and inference stack for the Raspberry Pi's
VideoCore IV** — runtime, kernels, graph optimiser, framework adapters and
packages — built and measured on the board, where the vendor shipped only an
OpenGL driver.

| measured on the board | |
|---|---|
| ONNX operators, expressible set | **152/152**, 0 refused, 0 mismatched |
| YOLOv8n 160x160, every op on the QPUs | **481.7 ms / 2.08 FPS** (ncnn on the four A53 cores: 74.8 ms) |
| GEMM, hand-written QPU kernel | **13.7 GFLOP/s** at 256 cubed |
| adapters | PyTorch 244 checks, JAX 25 VHLO ops, LiteRT 44 checks, and TensorFlow, ONNX Runtime, ggml, TVM — one engine behind all of them |

Every number is measured on the board at stock clocks (core 400 MHz / V3D
300 MHz); the conditions are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), the
open items in [`docs/STATUS.md`](docs/STATUS.md).

## Start here

New to the stack?  [`docs/TUTORIAL.md`](docs/TUTORIAL.md) takes a stock Pi to a
running YOLOv8n and a trained `torch_vc4` model, and
[`docs/STATUS.md`](docs/STATUS.md) says what is done, limited and open.

## Install

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/install.sh | sudo sh
```

One command: signed apt source, the whole stack, the `vc4d` helper, group
membership and a verification pass.  `--fast` builds for speed on the board,
`--torch` adds the PyTorch backend, `--pin-clocks` floors the core at its stock
400 MHz, `--uninstall` removes it; apt by hand, the pip index and the package
list are in [`docs/INSTALL.md`](docs/INSTALL.md).  At run time choose the
**managed** backend (unprivileged, ~0.1 ms/launch, the default) or **direct**
(root, no launch overhead: 410 vs 481.7 ms) — [`docs/BACKENDS.md`](docs/BACKENDS.md).

## Hardware

Verified on **Raspberry Pi 3 Model B** (BCM2837, VideoCore IV, 12 QPUs, 6 TMUs),
arm64, Debian 13, kernel 6.18 — the board behind every number in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| board | SoC / GPU | state | note |
|---|---|---|---|
| Pi 3 Model B | BCM2837 / VideoCore IV | **tested** | the reference board |
| Pi 3 B+, 3 A+, CM3, CM3+ | BCM2837(B0) / VideoCore IV | expected, untested | same QPU and `vc4` driver path |
| Pi Zero 2 W | BCM2710A1 / VideoCore IV | expected, untested | 512 MB limits the arena to the small models |
| Pi 2 Model B v1.2 | BCM2837 / VideoCore IV | expected, untested | the v1.1 board is 32-bit only |
| Pi 4, 400, CM4 | VideoCore VI | **not compatible** | different 3D block; the QPU path here does not apply |
| Pi 5, CM5 | VideoCore VII | **not compatible** | same reason |
| Pi 1, Zero, CM1, Pi 2 v1.1 | ARMv6/ARMv7 | **unverified** | no armhf build is shipped, and I do not have a 32-bit VideoCore IV board to test on, so compatibility is unknown rather than measured |

Requirements: the `vc4` DRM driver bound to the V3D block, CMA sized for both
arenas (96 MiB default), `dma-buf`, systemd for `vc4d`, 64-bit userspace, no
overclock (core 400 MHz, V3D 300 MHz, both stock).  The 32-bit boards are
**unmeasured**, not disproved: a claim would need an armhf build, a probe of the
V3D identification and QPU geometry on the board.

## Frameworks

The engine is shared; each framework is an adapter with its own entry point.
All of them talk to `vc4d` by default, so nothing below needs root.

| framework | entry point | detail |
|---|---|---|
| ONNX | `vc4infer run model.onnx --input x.npy --imgsz 160` | [`docs/ONNX.md`](docs/ONNX.md) |
| ONNX, Python | `import vc4infer; eng = vc4infer.Engine("model.onnx"); eng.run({"x": a})` | [`docs/API.md`](docs/API.md) |
| PyTorch | `import torch_vc4; t = t.to("vc4")` — the device is a real `PrivateUse1` | [`torch/README.md`](torch/README.md) |
| JAX | `JAX_PLATFORMS=vc4` with the PJRT plugin registered (`jax_plugins.vc4` or `PJRT_NAMES_AND_LIBRARY_PATHS=vc4:/path/libpjrt_vc4.so`) | [`docs/PJRT.md`](docs/PJRT.md) |
| TensorFlow | `load_pluggable_device_library("tf/build/libtf_vc4_plugin.so")`, then `with tf.device("/device:VCIV:0"):` | [`tf/README.md`](tf/README.md) |
| LiteRT | `Interpreter(model, experimental_delegates=[load_delegate("litert/build/libvc4_delegate.so")])` | [`litert/README.md`](litert/README.md) |
| ONNX Runtime | `import vc4infer.ort; InferenceSession(m, providers=["vc4"])` | [`docs/ONNXRUNTIME.md`](docs/ONNXRUNTIME.md) |
| ggml / llama.cpp | `GGML_BACKEND_PATH=libggml-vc4.so`, loaded by `ggml_backend_load_all` | [`docs/GGML.md`](docs/GGML.md) |
| TVM | `relax.ext.vc4` with the pattern table, then `FuseOpsByPattern` | [`tvm_byoc/README.md`](tvm_byoc/README.md) |

Two environment variables matter everywhere: `VC4INFER_DAEMON=1` (the packages'
default) selects the unprivileged daemon, `VC4INFER_KGEN` points at the kernel
generator in a checkout.  A model outside a framework's table fails and names
the operator; it does not run on the ARM instead.

## What is measured

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — the registry: hardware envelope,
  GEMM, YOLOv8n at 160/320/640, per-op against the CPU, training, clocks.
- [`docs/COMPLETENESS.md`](docs/COMPLETENESS.md) — the per-operator ONNX table:
  152/152 of the expressible set, 152/193 of opset 22, 0 refused.
- [`docs/MODELS.md`](docs/MODELS.md) — coverage beyond YOLO: fifteen graphs
  against ONNX Runtime, which plan, run or are refused, and why.
- [`docs/STATUS.md`](docs/STATUS.md) — one row per workstream, its state, and
  the file or test behind that state.
- [`docs/ACCURACY-FPS.md`](docs/ACCURACY-FPS.md) — YOLOv8n at 96/128/160 with
  COCO weights: 2.83 / 2.60 / 2.14 FPS, the detections that survive each size,
  and the launch and FLOP budget behind the curve.
- [`docs/TINY-LLM.md`](docs/TINY-LLM.md) — TinyStories-1M end to end: 378 static
  nodes, none refused, 1.06 tokens/s at 16 tokens, and the two engine defects
  that keep its output from matching ONNX Runtime.

Accuracy is in [`docs/ACCURACY.md`](docs/ACCURACY.md) (identical detections to
0.00 px against the CPU); the ncnn head-to-head (6.4×, same detections) in
[`docs/NCNN.md`](docs/NCNN.md); every bug and cause in [`docs/BUGS.md`](docs/BUGS.md);
raw logs in [`docs/results/`](docs/results) and [`docs/logs/`](docs/logs).

## Documentation

| document | contents |
|---|---|
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | start here: prerequisites, install, verify, YOLO three ways, PyTorch backend, ONNX export, measured performance |
| [`docs/INSTALL.md`](docs/INSTALL.md) | the one-liner, the flags, the apt source by hand, the packages, the speed build, pip, uninstall |
| [`docs/API.md`](docs/API.md) | `vc4infer` and `torch_vc4` reference: `Engine`/`Device`/`Detector`, error types, `host_policy`, op table |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | every failure mode we hit, with symptom, cause and fix |
| [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) | the repository family, what lives where, the old-path map, the two execution paths |
| [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) | the inference optimiser: pass pipeline, correctness gates, per-shape autotuning, plan accounting |
| [`docs/VC4-MODEL.md`](docs/VC4-MODEL.md) | a detector built to the measured launch and FLOP budget, and the 30/60 FPS gaps |
| [`docs/KERNELS.md`](docs/KERNELS.md) | the QPU kernels, the store defects behind them, how each is verified |
| [`docs/NUMERICS.md`](docs/NUMERICS.md) | the truncating FPU, denormal flushing, 3-word dependent chains, register and field rules |
| [`docs/BUGS.md`](docs/BUGS.md) | every bug with root cause, fix, verification and commit, plus the open ones |

The full index is [`docs/README.md`](docs/README.md); this is the main repository
(engine, kernels, training, docs) and issues belong here; the other repositories
are components or the package channel — [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md).

## Scope and honest limits

- Inference: a fixed model family at low resolution, fp32 only.  int8 has no
  exact packed byte MAC (int8 MAC/s == fp32) — [`docs/int8-RESULTS.md`](docs/int8-RESULTS.md).
- Training: a toy trainer only.  No fp16/bf16, no FMA, 76 MiB device memory, and
  the board browns out under sustained load — [`docs/TRAINING.md`](docs/TRAINING.md).
- Not CUDA/ROCm parity: no tensor cores, no fp16, no cuBLAS/cuDNN, no autograd
  library — [`docs/POSITIONING.md`](docs/POSITIONING.md).
- Single operators, not whole graphs: the 152/152 figure says nothing about
  fusion, layout propagation or allocation — [`docs/STATUS.md`](docs/STATUS.md).
- Every reported number carries its conditions, a run log and a mismatch count —
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/results/`](docs/results).

## Licence

Licence: [LICENSE](LICENSE), GPL-3.0-or-later; third-party components and the
AGPL-3.0 YOLOv8n checkpoint in [NOTICE](NOTICE); external sources in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
