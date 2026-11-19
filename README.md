# vc4-stack — VC4 compute stack: kernels, engine, training

Target: Raspberry Pi 3B, Broadcom VideoCore IV (V3D), Debian 13 arm64, stock clocks.

Everything here runs on the VC4 QPUs for arithmetic. The ARM cores run the control
plane only (framework, allocation, I/O); there is no CPU compute fallback.

## Layout

- `kernels/`  hand-written QPU kernels and the verified harnesses
              (GEMM 16x16 packed-A, wider tile, int8 experiments)
- `engine/`   libvc4infer: dma-buf arena, kernel upload/cache, strict op registry,
              matmul, conv2d (im2col), elementwise. Unsupported ops raise.
              `VC4INFER_DAEMON=1` runs it unprivileged through `vc4d`
              (`VC4INFER_DAEMON=1`, see `docs/DAEMON.md`).
- `onnx/`     ONNX graph path: fixed-shape exporter, executor (onnx + ctypes),
              three-model verification; supported nodes, measured numbers and
              the YOLOv8n coverage gap in `docs/ONNX.md`
- `training/` forward+backward kernels and a toy trainer (MLP/conv) demonstrating
              gradient computation on the VC4 data plane
- `vc4infer/` the installable product: Python engine API (`Engine`, `Device`),
              `postprocess`, the YOLO adapter, the CLI and the optional
              Ultralytics backend; docs in `docs/VC4INFER.md` and
              `docs/PYTHON-API.md`
- `cli/vc4infer` the `/usr/bin/vc4infer` entry point (thin over the package)
- `examples/` Engine, Detector, bench and Ultralytics examples
- `packaging/` the `.deb` and wheel builders plus the signed apt/pub publisher
- `docs/`     measurements, ceilings, coverage tables, per-track results
- `vc4d/`     privileged helper daemon (protocol v2): /dev/mem and V3D
              dispatch in one small root process that drops capabilities, so
              the engine can run as an ordinary user (`docs/DAEMON.md`)
- `tools/`    QPU encoder/launcher, verification harnesses,
              `qpu_disasm.py` (blob disassembler that fails on unknown opcodes)
              and `build-fast.sh` (memory-aware parallel builds with ccache)

## Building

Builds on this board are memory-bound, not CPU-bound.  The engine is safe at
`-j4` (171 MiB peak tree RSS, no swap); the VC4C compiler is not, because one
LLVM-weight C++ translation unit peaks at ~400 MiB, so `-j4` pushes 800 MiB
into swap and ends up *slower* than `-j2`.  `tools/build-fast.sh` picks the
largest job count that fits in `MemAvailable`, enables ccache when it is
installed, and prints the chosen jobs, memory, elapsed time and cache hit
rate.  Full matrix and numbers: [docs/BUILD.md](docs/BUILD.md).

```sh
tools/build-fast.sh /home/raspi/work/engine-daemon --clean
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

## Scope and honest limits

- Inference: a fixed model family at low resolution; int8 is the performance path.
- Training: a toy trainer only. No fp16/bf16, no FMA, 76 MiB device memory, and the
  board browns out under sustained load. General training is out of scope.
- Not CUDA/ROCm parity: no tensor cores, no fp16, no cuBLAS/cuDNN, no autograd library.

Every reported number must have a run log and a mismatch count.

## References

All external sources (Broadcom guide, reverse-engineering wikis, V3DLib, anholt/vc4-gpu-tools, VC4C/VC4CL upstream) are listed in [docs/REFERENCES.md](docs/REFERENCES.md).

## Documentation

Full index: [docs/README.md](docs/README.md). Includes references, decoder cross-validation, numerics, clock variance, training and conv results, the vc4d security model, the read-only vc4mon kernel interface, and the V3DLib report.

Verification matrix: [docs/SOURCES-VERIFICATION.md](docs/SOURCES-VERIFICATION.md). Findings catalogue: [docs/FINDINGS.md](docs/FINDINGS.md).
