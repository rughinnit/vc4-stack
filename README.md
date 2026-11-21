# vc4-stack — VideoCore IV compute stack

Target: Raspberry Pi 3B, Broadcom VideoCore IV (V3D), Debian 13 arm64, **stock
clocks** (core 400 MHz / V3D 300 MHz).  Everything that computes runs on the
VC4 QPUs.  The ARM cores run the control plane only (allocation, I/O, byte
movement); there is no CPU arithmetic fallback anywhere in this stack.

## Install

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/install.sh | sudo sh
```

One command: signed apt source, the whole stack, the `vc4d` helper, group
membership, and a verification pass. At run time you choose the **managed**
backend (unprivileged, ~0.1 ms/launch, the default) or **direct** (root,
no launch overhead: 410 ms vs 488 ms on YOLOv8n) — see
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
| `vc4-apt` | the only install source: signed apt repo + pip index (`apt install vc4-toolkit`) |
| `torch-vc4` | **mirror** of `torch/`; the source of truth is this repository |
| `vc4top` | GPU monitor TUI + the `vc4mon` DKMS module |
| `VC4C` | OpenCL C → QPU compiler (fork; performance branches) |
| `VC4CL` | OpenCL runtime (fork; Debian 13 + unprivileged fixes) |
| `VC4CLStdLib` | GPU-side OpenCL standard library (fork) |

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

## Install

The stack installs from a signed apt repository, the way CUDA or TensorRT do:

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
