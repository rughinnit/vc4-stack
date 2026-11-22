# vc4-stack — ONNX, PyTorch, JAX and TensorFlow on the Raspberry Pi GPU

Runs neural networks on the **VideoCore IV** QPUs of a Raspberry Pi 3B — the GPU
that shipped with the board in 2016 — with the ARM cores doing only control and
byte movement.  There is no CPU arithmetic fallback: an operator either executes
on the QPUs or is refused by name.

| measured on the board | |
|---|---|
| ONNX operators, expressible set | **152/152**, 0 refused, 0 mismatched |
| YOLOv8n 160x160, every op on the QPUs | **481.7 ms / 2.08 FPS** (ncnn on the four A53 cores: 74.8 ms) |
| GEMM, hand-written QPU kernel | **13.7 GFLOP/s** at 256 cubed |
| PyTorch backend | 244 completeness checks, autograd for its whole op table |
| JAX (PJRT) | 25 VHLO operations, `jax.grad` stays on the device |
| LiteRT (external delegate) | 44 checks on silicon, claimed nodes on the QPUs |

Every number is measured on the board at stock clocks; the conditions are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), and what is still open is in
[`docs/STATUS.md`](docs/STATUS.md).

Target: Raspberry Pi 3B, Broadcom VideoCore IV (V3D), Debian 13 arm64, **stock
clocks** (core 400 MHz / V3D 300 MHz).  Everything that computes runs on the
VC4 QPUs.  The ARM cores run the control plane only (allocation, I/O, byte
movement); there is no CPU arithmetic fallback anywhere in this stack.

## Compatible hardware

Verified on **Raspberry Pi 3 Model B** (BCM2837, VideoCore IV, 12 QPUs, 6 TMUs),
arm64, Debian 13, kernel 6.18 — that is the board behind every number in
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

Requirements, all of them: a kernel with the `vc4` DRM driver bound to the V3D
block, CMA large enough for the daemon arena plus the client's (96 MiB
default), `dma-buf`, systemd for `vc4d`, and 64-bit userspace. No overclock is
used or needed: core 400 MHz, V3D 300 MHz, both stock.

I do not have a 32-bit VideoCore IV board (Pi 1, Zero, CM1, Pi 2 v1.1)
to test on, so those boards are **unmeasured**, not disproved: a claim
would need an armhf build and a probe of the V3D identification and QPU
geometry on the board itself.

## Install

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/install.sh | sudo sh
```

One command: signed apt source, the whole stack, the `vc4d` helper, group
membership, and a verification pass. At run time you choose the **managed**
backend (unprivileged, ~0.1 ms/launch, the default) or **direct** (root,
no launch overhead: 410 ms vs 481.7 ms on YOLOv8n) — see
[`docs/BACKENDS.md`](docs/BACKENDS.md). `--fast` builds for speed on the board,
`--torch` adds the PyTorch backend, `--pin-clocks` floors the core at its stock
400 MHz, `--uninstall` removes it. Details: [`docs/INSTALL.md`](docs/INSTALL.md).

## This is the main repository

The whole VideoCore IV project is organised around this repo: the engine, the
kernels, the training core, the measurements and the documentation are here, and
issues belong here. The other repositories are either components or the package
channel.

| repository | role |
|---|---|
| **`vc4-stack`** (this repo) | **main**: inference engine, QPU kernels, training, PyTorch backend (`torch/`), docs, measurements |
| [`vc4-apt`](https://github.com/rughinnit/vc4-apt) | the only install source: signed apt repo + pip index (`apt install vc4-toolkit`) |
| [`torch-vc4`](https://github.com/rughinnit/torch-vc4) | **mirror** of `torch/`; the source of truth is this repository |
| [`vc4top`](https://github.com/rughinnit/vc4top) | GPU monitor TUI + the `vc4mon` DKMS module |
| [`VC4C`](https://github.com/rughinnit/VC4C) | OpenCL C → QPU compiler (fork; performance branches) |
| [`VC4CL`](https://github.com/rughinnit/VC4CL) | OpenCL runtime (fork; Debian 13 + unprivileged fixes) |
| [`VC4CLStdLib`](https://github.com/rughinnit/VC4CLStdLib) | GPU-side OpenCL standard library (fork) |

Every measured number — GEMM, YOLOv8n, per-op vs CPU, training, builds,
clocks, completeness — is in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), and
every bug with its cause, fix and verification is in
[`docs/BUGS.md`](docs/BUGS.md) (the open ones are listed there too).
Accuracy — including VC4 vs the CPU on the same image, where the detections are
identical to 0.00 px — is in [`docs/ACCURACY.md`](docs/ACCURACY.md), and the
head-to-head against **ncnn** (6.4× faster than our GPU path at 160×160, with
identical detections) is in [`docs/NCNN.md`](docs/NCNN.md).

Full table, the two execution paths and a "which repo do I open" guide:
[`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md).

**Current state:** [`docs/STATUS.md`](docs/STATUS.md) — what is done,
limited, open or in progress, and the evidence behind each row.

## Start here

New to the stack?  Follow the tutorial, in this order:

1. [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — stock Pi to a running YOLOv8n and a
   trained `torch_vc4` model; every command was run on the board or is
   explicitly marked as not re-run.
2. [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — the failure modes we
   actually hit, with symptom, cause and fix.
3. [`docs/API.md`](docs/API.md) — `vc4infer` and `torch_vc4` reference, error
   types, `host_policy`, op table.
4. [`docs/README.md`](docs/README.md) — the full documentation index.

The short version: `sudo apt install vc4infer`, make sure `vc4d` is running and
your user is in the `vc4` group, then
`vc4infer run model.onnx --input img.jpg --imgsz 160`.  The client never opens
`/dev/mem`; the privileged helper is `vc4d`.

## Where each product lives

| path | what it is | start here | status |
|---|---|---|---|
| `infer/` | **libvc4infer + vc4infer** — the engine (dma-buf arena, kernel JIT/cache, strict op registry, matmul/conv/elementwise), the ONNX executor, the Python API, the CLI, and the `.deb`/wheel builders | [`infer/README.md`](infer/README.md), [`docs/VC4INFER.md`](docs/VC4INFER.md) | active — vc4infer **1.2.1**; YOLOv8n 160x160 runs end to end on the QPUs at **2.09 FPS** (stock clocks, quiet board) |
| `torch/` | **torch_vc4** — the PyTorch `PrivateUse1` backend: compiled extension over the same engine and `vc4d`, the aten subset, autograd, the op table | [`torch/README.md`](torch/README.md), [`torch/COMPLETENESS.md`](torch/COMPLETENESS.md) | feature-complete for its declared subset — 48 registration checks, 244 completeness checks on silicon |
| `kernels/` | hand-written QPU kernels: `wider-tile/asm6..asm10`, each with its NOTES/RESULTS and generator, plus the `attempt-asm10` history | [`kernels/wider-tile/`](kernels/wider-tile) | verified, frozen (results in `kernels/wider-tile/RESULTS*.md`) |
| `training/` | backward kernels (`vc4_train.c`, `vc4_convbwd.c`) and the MLP/conv trainer with finite-difference checks | [`training/README.md`](training/README.md) | toy scale; gradients verified |
| `tools/` | `build-fast.sh` (memory-aware `-j` + ccache), `qpu_disasm.py`, `xcheck/` (four independent encoders), `gput.sh` (locked GPU run + board check) | script headers | in use |
| `vc4d/` | privileged helper daemon (protocol v2): `/dev/mem` and V3D dispatch in one small root process that drops capabilities, so the engine runs unprivileged | [`vc4d/README.md`](vc4d/README.md) | stable; 82/82 hostile cases |
| `docs/` | the single source of documentation | [`docs/README.md`](docs/README.md) | — |

`infer/` is one product directory: `src/` and `include/` are the C library,
`python/vc4infer/` is the installable package, `cli/` holds the `/usr/bin/vc4infer`
entry point, and `tests/`, `examples/`, `packaging/` and `pyproject.toml` belong
to the same product.  The backward kernels are compiled into the same library
from `training/src/`.

## Install without the install script

The script above performs exactly these steps; run them by hand if you prefer:

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/key.asc \
  | sudo gpg --dearmor -o /etc/apt/keyrings/vc4-apt.gpg
echo "deb [signed-by=/etc/apt/keyrings/vc4-apt.gpg] https://rughinnit.github.io/vc4-apt stable main" \
  | sudo tee /etc/apt/sources.list.d/vc4-apt.list
sudo apt update
sudo apt install vc4-toolkit
```

`vc4-toolkit` is the single install command: `vc4d` (daemon), `vc4infer`
(engine, Python API, CLI), `vc4infer-dev` (header + pkg-config + CMake),
`vc4-query` (the `deviceQuery` analogue), `vc4-smi` (the `nvidia-smi`
analogue), `vc4train` (training demo) and `vc4top`/`vc4top-dkms`.

```sh
vc4-query             # what did I just install?
vc4-smi               # is the GPU healthy?
vc4infer selftest     # 4/4 bundled models
```

For the speed-tuned rebuild of the same stack (`-O3 -march=native`, a
prewarmed read-only kernel cache and a tuned daemon), install
`vc4-toolkit-fast` instead.  Both paths, the pip index (`vc4infer`,
`torch_vc4`) and the framework support table are in
[docs/INSTALL.md](docs/INSTALL.md).

## Build and run

```sh
make -C infer -j4                            # libvc4infer.a/.so + all tests
python3 infer/tools/check_math.py            # GPU-free: kernel formulas vs Python

infer/run_tests.sh                           # full serialised suite (root, /dev/mem)
REPS=5 VC4INFER_DAEMON=1 infer/run_tests.sh  # same, unprivileged through vc4d

infer/cli/vc4infer nodes infer/packaging/vc4infer/data/models/modelA.onnx
infer/packaging/vc4infer/build_deb.sh        # -> dist/vc4infer_<ver>_arm64.deb
infer/packaging/build_wheel.sh               # -> dist/vc4infer-<ver>-py3-none-any.whl

make -C torch                                # -> dist/torch_vc4-<ver>-cp313-...whl
```

`make -C torch` (equivalently `packaging/build_torch_wheel.sh`) builds the
PyTorch backend wheel into the same shared `dist/` from `torch/`, against this
checkout's `infer/` and `vc4d/` builds.  It needs an interpreter with torch
`2.15.0+cpu` installed and refuses to run with a message naming what is
missing; `make -C infer` does not depend on it (nor on torch being present).
`packaging/publish_apt.sh` copies `dist/torch_vc4-*.whl` into `vc4-apt/pip/`
under its unchanged name.

**One GPU user at a time.**  Two concurrent workloads, or a wild kernel, can
hang the V3D hard enough to need a power cycle.  In daemon mode `vc4d` takes
`/tmp/vc4-gpu.lock` per launch; a client must **not** hold that lock itself
(the daemon answers `E_BUSY` and the engine retries for ~60 s).  The direct-root
test suite takes the lock itself and checks `STATE: LIVE` after every stage.
See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) section 4.

Builds on this board are memory-bound, not CPU-bound.  The engine is safe at
`-j4` (171 MiB peak tree RSS, no swap); the VC4C compiler is not, because one
LLVM-weight C++ translation unit peaks at ~400 MiB, so `-j4` pushes 800 MiB
into swap and ends up *slower* than `-j2`.  `tools/build-fast.sh` picks the
largest job count that fits in `MemAvailable`, enables ccache when it is
installed, and prints the chosen jobs, memory, elapsed time and cache hit
rate.  Full matrix and numbers: [docs/BUILD.md](docs/BUILD.md).

```sh
tools/build-fast.sh infer --clean
BUILD_FAST_TARGET=VC4CC tools/build-fast.sh /home/raspi/work/combined/build
```

## Working with the frameworks

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

Two environment variables matter everywhere: `VC4INFER_DAEMON=1` (the default
in the packages) selects the unprivileged daemon, and `VC4INFER_KGEN` points at
the kernel generator when running from a checkout rather than an installed
package.  A model that uses an operator outside a framework's table fails and
names it; it does not run on the ARM instead.

## How this compares to CUDA and ROCm

The same layers, at the scale one board can measure:

| layer | NVIDIA | AMD | this project |
|---|---|---|---|
compute language and compiler | CUDA C++ / NVCC | HIP / HIP-Clang | OpenCL C through `VC4C` (fork) |
compute runtime | CUDA runtime | HIP runtime | `VC4CL` (fork), plus `vc4d` for unprivileged launches |
math kernels | cuBLAS, cuDNN | rocBLAS, MIOpen | hand-written kernels in `libvc4infer` and `kgen` |
inference optimiser | TensorRT | MIGraphX | the `vc4infer` planner, with the pass pipeline in [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) |
framework plugins | Torch-TensorRT, TF-TRT | MIOpen integrations | `torch_vc4`, PJRT, TF PluggableDevice, LiteRT delegate, ORT provider, ggml backend, TVM BYOC |
profiling and telemetry | Nsight | rocprof | `vc4top`, `vc4-smi`, `vc4mon` counters |
distribution | CUDA toolkit | ROCm repositories | `vc4-apt`, signed apt and pip |

What is ours rather than a port: the hand-written QPU kernels and the generator
that emits them, the executor and its numerics model (the FPU truncates, the VPM
and DMA rows disagree, division needed a sign-preserving reciprocal), the
unprivileged daemon and arena, and the measurement record.

What is deliberately not attempted: a new language, a compiler written from
scratch (VC4C is forked), tensor cores, unified memory, asynchronous copy
engines, atomics, multi-GPU, or CUDA parity. The structural ceiling is
28.8 GFLOP/s; a hand-written GEMM reaches 13.7.

## How optimised each layer is

Not one layer is at the structural peak, and the peak itself assumes every slot
issues two ALU operations with no memory stall:

| layer | current | ceiling | blocker |
|---|---|---|---|
| ALU microbenchmark | 28.36 GFLOP/s (**98.5%** of 28.8) | 28.8 | none; the one near-optimal measurement |
| coalesced memory loop | 16.42 (57%) | 28.8 | L2 and VPM bandwidth |
| hand-written GEMM | **13.7** stand-alone, **8.5-9.5** through the engine (30-47%) | 28.8 | register pressure, VPM rows, no cross-tile software pipelining |
| convolution | im2col + GEMM | a direct-conv kernel would skip the materialisation | not written yet |
| elementwise and unary | launch-bound | - | one launch per node today |
| VC4C compiler | **3.09** (10.7%) | 28.8 | register allocation and scheduling; 4.4x behind hand-written code |
| runtime (`vc4d`) | 0.1 ms per launch, **802 launches = 222 ms** | - | no batching, no persistent queue |
| graph layer | 481.7 ms for YOLOv8n 160 | 74.8 ms (ncnn) | no fusion, layout propagation or memory reuse yet |
| libraries | shape coverage, no autotuning yet | - | the autotune track has just started |

Why 100% is the wrong frame: this block has no public ISA documentation, so
every behaviour was established by measurement — the FPU truncates, the VPM and
DMA row fields disagree, a small immediate in the A slot wedges the QPU. Each
optimisation has to be discovered and then proven on silicon. That is why the
ALU microbenchmark reaches 98.5% while a real GEMM reaches 30-47%: the
arithmetic can be saturated, the memory path cannot.

The gaps are being attacked in this order: graph fusion and layout
([`docs/OPTIMIZER.md`](docs/OPTIMIZER.md)), per-shape kernel autotuning, launch
batching or a persistent queue, then VC4C target passes.

## Training and model development on the board

Inference is the broad path; training exists but is capacity-bound.  What runs
where:

| framework | train on the QPUs? | why |
|---|---|---|
| PyTorch | **yes** | autograd dispatches backward to the QPUs and the optimiser state lives in the arena; MLP 64.7 steps/s, CNN 0.98-1.37 steps/s |
| JAX | **yes, within the 25-op subset** | `jax.grad` runs on the device and the update ops are in the subset |
| TensorFlow | no | host `GradientTape` needs kernels that are not registered yet (18 today) |
| LiteRT, ggml, ONNX, ONNX Runtime, TVM | no | they are inference runtimes by design |

The constraint is memory: fp32 weights, gradients and Adam moments are **16 bytes
per parameter** before activations.  A 96 MiB client arena holds roughly 6 M
parameters with no activations at all, and there are always activations.  Two
measured cases: a 128-channel 128x128 CNN needs **113 MiB of Adam state alone**
and does not fit; YOLOv8n's optimiser state is about **51 MB**, which with
160x160 activations is outside what the arena can serve.  Numbers and conditions:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) section 5.

The workflow that works:

1. develop and train on a workstation in PyTorch or JAX;
2. export ONNX, or keep the PyTorch checkpoint and use `torch_vc4` on the board;
3. deploy on the Pi through the engine, LiteRT or ONNX Runtime;
4. use on-board training for what fits: small heads, calibration, and the bundled
   `vc4train` demo (a 64-32-10 MLP), not YOLO-scale training.

Capacity, not missing kernels, sets that boundary: a larger arena would change
the training case first, and that is a different board (VideoCore VI/VII, which
this stack does not target).

## Measured ceilings (stock 300/400 MHz, no overclock)

| workload | GFLOP/s | % of 28.8 structural peak |
|---|---|---|
| ALU-only microbenchmark | 28.36 | 98.5% |
| coalesced memory-fed loop | 16.42 | 57% |
| hand-written GEMM (verified) | 9.48 | 33% |
| compiled VC4CL path | 3.09 | 10.7% |

Structural peak: 12 QPUs x 16 lanes x 2 ALU ops x 1/4 instruction per clock x 300 MHz.

The engine's public GEMM measures 8.5 GFLOP/s through `vc4d`; the hand-written
`asm10` kernel's recorded stand-alone ceiling is 13.7 GFLOP/s.  See
[`docs/ENGINE-RESULTS.md`](docs/ENGINE-RESULTS.md) and
[`docs/TUTORIAL.md`](docs/TUTORIAL.md) section 7.3 for the provenance.

## Scope and honest limits

- Inference: a fixed model family at low resolution, fp32 only.  int8 has no
  exact packed byte MAC on this ISA (int8 MAC/s == fp32), so it is not a
  performance shortcut.
- Training: a toy trainer only. No fp16/bf16, no FMA, 76 MiB device memory, and the
  board browns out under sustained load. General training is out of scope.
- Not CUDA/ROCm parity: no tensor cores, no fp16, no cuBLAS/cuDNN, no autograd library.

Every reported number must have a run log and a mismatch count.  Raw logs live in
[`docs/results/`](docs/results) and [`docs/logs/`](docs/logs).

## Layout map (for older links)

| was | now |
|---|---|
| `engine/` (sources, Makefile, tests, tools, blobs, kcaches) | `infer/` |
| `engine/attempt-asm10/` | `kernels/attempt-asm10/` |
| `engine/RESULTS*.md`, `engine/*.log` | `docs/results/`, `docs/logs/` |
| `onnx/` (exporter, verifier, coverage, bench, executor wrapper) | `infer/tools/onnx/` |
| `vc4infer/` (Python package) | `infer/python/vc4infer/` |
| `pyproject.toml` | `infer/pyproject.toml` |
| `cli/`, `examples/`, `packaging/` | `infer/cli/`, `infer/examples/`, `infer/packaging/` |
| `engine/src/vc4_train.c`, `vc4_convbwd.c`, training tests | `training/src/`, `training/tests/` |
| the `torch-vc4` repository (backend, tests, `COMPLETENESS.md`) | `torch/` (history grafted; the repository is a mirror) |

## References and licence

External sources (Broadcom guide, reverse-engineering wikis, V3DLib/QPULib,
py-videocore, anholt/vc4-gpu-tools, VC4C/VC4CL) are listed with URLs and status in
[docs/REFERENCES.md](docs/REFERENCES.md).  Licence: [LICENSE](LICENSE)
(GPL-3.0-or-later); third-party components and the AGPL-3.0 YOLOv8n checkpoint are
listed in [NOTICE](NOTICE).
