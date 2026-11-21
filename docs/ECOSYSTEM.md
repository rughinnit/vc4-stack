# The VC4 project family — what lives where

**`vc4-stack` is the main repository.** It is the entry point, the place the
measurements and documentation live, and the place to open an issue. Everything
else is either an artifact source (packages), a component that is published from
here, or a fork of an upstream project that this stack improves.

| repository | provenance | what it is | state | artifacts |
|---|---|---|---|---|
| **`rughinnit/vc4-stack`** | this stack | **MAIN.** The stack: inference engine (`libvc4infer` + `vc4infer`), QPU kernels, training core, ONNX runner and ORT provider, `vc4d` helper, PyTorch `PrivateUse1` backend (`torch/`), docs, tools | **v1.2.10**, suite green, YOLOv8n 160 at **481.7 ms / 2.08 FPS** | nothing published directly; packages come from `vc4-apt` |
| `rughinnit/vc4-apt` | this stack | **The only installation source.** Signed apt repository + PEP 503 pip index (`https://rughinnit.github.io/vc4-apt`) | live, key `E4CB3BA0…`, `InRelease` signed | `vc4infer`, `vc4infer-dev`, `vc4-toolkit`, `vc4-toolkit-fast`, `vc4d`, `vc4train`, `vc4-query`, `vc4-smi`, `vc4top`, `vc4top-dkms`, `torch_vc4` wheel |
| `rughinnit/torch-vc4` | this stack | **Mirror** of `vc4-stack/torch/`. The backend's history was grafted into `vc4-stack` under `torch/`; this repository is kept so the published wheel's metadata and old links keep resolving, and it is not a second source tree | source of truth is `vc4-stack/torch/` (`torch/COMPLETENESS.md`); the mirror follows it without carrying new work | `torch_vc4-0.3.0-cp313-…whl` via `vc4-apt` (built from `vc4-stack`) |
| `rughinnit/vc4top` | fork (extended) | GPU monitor TUI + the **`vc4mon`** read-only DKMS module for unprivileged counters | **1.0.5**, 78-register parity with DRM debugfs | `vc4top`, `vc4top-dkms` via `vc4-apt` |
| `rughinnit/VC4C` | fork of `doe300/VC4C` | OpenCL C → QPU compiler. Our performance work lives on branches, with a combined branch as the recommended one | `perf/combined` (+ `perf/spill-retry`, `perf/product-acc`, `perf/pipeline2/3/4`); 130/130 kernels compile | feeds VC4CL |
| `rughinnit/VC4CL` | fork of `doe300/VC4CL` | OpenCL runtime. Debian 13 fixes, unprivileged operation, the M21 GEMM investigation | branches `fix/debian13-*`, `docs/m21-gemm-findings`; **3.085 GFLOP/s** with the improved compiler | `.deb` for VC4CL |
| `rughinnit/VC4CLStdLib` | fork of `doe300/VC4CLStdLib` | GPU-side OpenCL standard library used by VC4C/VC4CL | unmodified upstream plus the toolchain notes | — |

## How they fit together

Two execution paths and one distribution channel:

```
                        vc4-stack   (main: docs, kernels, measurements)
                             |
        +--------------------+---------------------+
        |                    |                     |
   our own path        the OpenCL path        the products
        |                    |                     |
  vc4d  (privileged     VC4C ──> VC4CL ──>    vc4infer   (ONNX engine, CLI,
  helper)  +  libvc4infer      VC4CLStdLib    |          Python API, ORT provider)
        |                          |          torch_vc4  (PyTorch backend)
  QPU kernels  <---+               |          vc4top      (TUI + vc4mon DKMS)
                   |               |
                   +─────── vc4-apt (signed apt + pip) ──── every .deb and wheel
```

* **Our own path** (`vc4d` + `libvc4infer` + hand-written QPU kernels) is the
  fastest and the one the products use: 13.7 GFLOP/s hand-written GEMM,
  8.47 GFLOP/s engine GEMM, 487 ms YOLOv8n at 160.
* **The OpenCL path** (VC4C → VC4CL) is the compiler route; its improvements
  are measured and reported in `vc4-stack` (`docs/ENGINE-RESULTS.md`,
  `docs/BUILD.md`).
* **`vc4-apt`** is the single install source. Nothing is distributed from the
  other repositories.

## Which repository should I open?

| I want to… | go to |
|---|---|
| install and run something | `vc4-apt` (`apt install vc4-toolkit`), documented in `vc4-stack/docs/INSTALL.md` |
| understand how it works / read measurements | **`vc4-stack`** (`docs/TUTORIAL.md` first) |
| report a bug or a wrong number | **`vc4-stack`** |
| use PyTorch on the GPU | **`vc4-stack/torch/`** (`torch_vc4`; the `torch-vc4` repository is a mirror) |
| change the OpenCL compiler or runtime | `VC4C` / `VC4CL` branches |
| monitor the GPU | `vc4top` |

The state of every workstream, with the file or test behind it, is in
[STATUS.md](STATUS.md).

## Rules this family follows

1. **One main repo.** Plans, measurements, tolerances, known limitations and
   the framework adapters live in `vc4-stack` (`infer/`, `torch/`, `litert/`,
   `pjrt/`, `tf/`, `ggml/`, `tvm_byoc/`); the other repos link back here rather
   than keeping their own version of the truth.  `torch-vc4` is a mirror of
   `torch/`, not a second source tree.
2. **One distribution channel.** Every published artifact is built and signed
   through `vc4-apt`. No repository ships a binary that `vc4-apt` does not.
3. **Forks stay forks.** `VC4C`, `VC4CL` and `VC4CLStdLib` keep their upstream
   remotes (`doe300/…`) and our work lives on named branches with a `rughinnit`
   remote, so upstream can be merged without rewriting history.
4. **Numbers carry their conditions.** Every performance figure states the clock
   (pinned at the stock 400 MHz), the board state and the tolerance, because the
   firmware can silently halve the core clock (`docs/CLOCK-VARIANCE.md`).
