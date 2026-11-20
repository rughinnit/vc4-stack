# vc4-stack — VideoCore IV compute stack

Target: Raspberry Pi 3B, Broadcom VideoCore IV (V3D), Debian 13 arm64, **stock
clocks** (core 400 MHz / V3D 300 MHz).  Everything that computes runs on the
VC4 QPUs.  The ARM cores run the control plane only (allocation, I/O, byte
movement); there is no CPU arithmetic fallback anywhere in this stack.

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

## Build and run

```sh
make -C infer -j4                            # libvc4infer.a/.so + all tests
python3 infer/tools/check_math.py            # GPU-free: kernel formulas vs Python

infer/run_tests.sh                           # full serialised suite (root, /dev/mem)
REPS=5 VC4INFER_DAEMON=1 infer/run_tests.sh  # same, unprivileged through vc4d

infer/cli/vc4infer nodes infer/packaging/vc4infer/data/models/modelA.onnx
infer/packaging/vc4infer/build_deb.sh        # -> dist/vc4infer_<ver>_arm64.deb
infer/packaging/build_wheel.sh               # -> dist/vc4infer-<ver>-py3-none-any.whl
```

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

## References and licence

External sources (Broadcom guide, reverse-engineering wikis, V3DLib/QPULib,
py-videocore, anholt/vc4-gpu-tools, VC4C/VC4CL) are listed with URLs and status in
[docs/REFERENCES.md](docs/REFERENCES.md).  Licence: [LICENSE](LICENSE)
(GPL-3.0-or-later); third-party components and the AGPL-3.0 YOLOv8n checkpoint are
listed in [NOTICE](NOTICE).
