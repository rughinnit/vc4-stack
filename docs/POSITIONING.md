# Where this stack sits: alongside CUDA and ROCm, and against the structural peak

The project is small enough to compare layer by layer with the two mature GPU
compute stacks, and it publishes how far each of its own layers is from what the
hardware could do.  This page holds both comparisons; the front-page summary
keeps only the headline numbers.

## 1. The same layers, at the scale one board can measure

| layer | NVIDIA | AMD | this project |
|---|---|---|---|
| compute language and compiler | CUDA C++ / NVCC | HIP / HIP-Clang | OpenCL C through `VC4C` (fork) |
| compute runtime | CUDA runtime | HIP runtime | `VC4CL` (fork), plus `vc4d` for unprivileged launches |
| math kernels | cuBLAS, cuDNN | rocBLAS, MIOpen | hand-written kernels in `libvc4infer` and `kgen` |
| inference optimiser | TensorRT | MIGraphX | the `vc4infer` planner, with the pass pipeline in [`OPTIMIZER.md`](OPTIMIZER.md) |
| framework plugins | Torch-TensorRT, TF-TRT | MIOpen integrations | `torch_vc4`, PJRT, TF PluggableDevice, LiteRT delegate, ORT provider, ggml backend, TVM BYOC |
| profiling and telemetry | Nsight | rocprof | `vc4top`, `vc4-smi`, `vc4mon` counters |
| distribution | CUDA toolkit | ROCm repositories | `vc4-apt`, signed apt and pip |

What is ours rather than a port: the hand-written QPU kernels and the generator
that emits them, the executor and its numerics model (the FPU truncates, the VPM
and DMA rows disagree, division needed a sign-preserving reciprocal), the
unprivileged daemon and arena, and the measurement record.

What is deliberately not attempted: a new language, a compiler written from
scratch (VC4C is forked), tensor cores, unified memory, asynchronous copy
engines, atomics, multi-GPU, or CUDA parity.  The structural ceiling is
28.8 GFLOP/s; a hand-written GEMM reaches 13.7.

## 2. How optimised each layer is

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
DMA row fields disagree, a small immediate in the A slot wedges the QPU.  Each
optimisation has to be discovered and then proven on silicon.  That is why the
ALU microbenchmark reaches 98.5% while a real GEMM reaches 30-47%: the
arithmetic can be saturated, the memory path cannot.

The gaps are being attacked in this order: graph fusion and layout
([`OPTIMIZER.md`](OPTIMIZER.md)), per-shape kernel autotuning, launch batching or
a persistent queue, then VC4C target passes.  What the pass pipeline does today,
including the autotuning cache and its measured cost, is in
[`OPTIMIZER.md`](OPTIMIZER.md); what still blocks each row is in
[`STATUS.md`](STATUS.md).

## 3. Measured ceilings (stock 300/400 MHz, no overclock)

| workload | GFLOP/s | % of 28.8 structural peak |
|---|---|---|
| ALU-only microbenchmark | 28.36 | 98.5% |
| coalesced memory-fed loop | 16.42 | 57% |
| hand-written GEMM (verified) | 9.48 | 33% |
| compiled VC4CL path | 3.09 | 10.7% |

Structural peak: 12 QPUs x 16 lanes x 2 ALU ops x 1/4 instruction per clock x
300 MHz.

The engine's public GEMM measures 8.5 GFLOP/s through `vc4d`; the hand-written
`asm10` kernel's recorded stand-alone ceiling is 13.7 GFLOP/s.  See
[`ENGINE-RESULTS.md`](ENGINE-RESULTS.md) and [`TUTORIAL.md`](TUTORIAL.md)
section 7.3 for the provenance.

The same numbers, with their run conditions, are in
[`BENCHMARKS.md`](BENCHMARKS.md) section 2 for the GEMM rows and section 10 for
the clocks and thermal envelope; the kernel that holds the stand-alone record is
documented in [`KERNELS.md`](KERNELS.md).

## 4. The gap a user feels: the graph layer

Against the CUDA and ROCm rows above, the gap a user feels is the graph layer: 481.7 ms for YOLOv8n 160 against 74.8 ms for ncnn on the same board and
the same image, with identical detections.  That measurement, its attribution
(launch count rather than arithmetic) and what would close it are in
[`NCNN.md`](NCNN.md).
