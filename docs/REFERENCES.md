# External references, and what they confirm

## Broadcom VideoCore IV 3D Architecture Reference Guide (AG100-R, 2013)

- VPM: 16 words wide x 64 rows, horizontal/vertical vector access, VPM setup fields
  (STRIDE, HORIZ, LANED, SIZE, ADDR), VDW store with UNITS/DEPTH/BLOCKMODE/STRIDE.
- Vertical 32-bit vectors must start on a row multiple of 16.
- V3D_VPMBASE reserves VPM for user programs in 256-byte units.
- QPU register map (VPM_READ/WRITE, VPMVCD_RD/WR_SETUP, VPM_ST_BUSY/WAIT).
Source: VC4C `doc/VideoCoreIV-AG100-R.pdf` (checked in).

## hermanhermitage/videocoreiv (docs, tinyasm, IDA plugin, bare-metal samples)

- Overview page: GPU "24 Gigaflops (3x4x8x250MHz)".
  * CONFIRMS our structural model: 12 QPUs x 16 lanes x 2 ALU ops x 1/4 instr/clock.
    250 MHz -> 24 GFLOP/s, 300 MHz -> 28.8 GFLOP/s, 500 MHz -> 48 GFLOP/s.
  * Confirms 1 instruction per 4 clocks with dual-issue add+mul.
- Memory hierarchy: L2 is 128 KB, 4-way, dedicated to the GPU, and OUTSIDE the ARM MMU;
  ARM latencies L1 7 ns, L2 hit 68 ns, SDRAM 128 ns (page hit) / 167 ns (miss).
  * Explains why DMA buffers must be addressed by bus/physical address (dma-buf + /dev/mem),
    and why a stray DMA address can overwrite any page (the page-cache corruption incident).
- Performance notes (VPU/patents): FP operations pipeline over three cycles, stall if either
  of the two following instructions references the result.
  * Matches the 3-word dependent-chain rule measured for QPU accumulators.
- Tooling: `tinyasm`, IDA plugin, plus `itszor/vc4-toolchain` (GNU binutils with vc4).
  * Use for bit-exact cross-validation of our encoder (`qenc.py`).
- Wiki pages to mine: MMIO Register map, Register Documentation, Kernels under Linux,
  Programmers Manual.

## hermanhermitage/videocoreiv-qpu

- QPU instruction-set documentation: rotations, pack/unpack, small immediates, signalling.
  * Our empirical findings to check against it: rotate amounts are 49-63 (v8min imm 48 is not
    a rotation); rep with ws=1 spans 16 lanes, ws=0 is per-quad; v8muld is lossy.

## axizo-pi/V3DLib (MIT, v0.9.2)

- C++ DSL + compiler + runtime for vc4/vc6/vc7; examples Matrix, NN, HeatMap, Mandelbrot.
- Requires headless execution and root (/dev/mem); needs bcm_host (Debian 13 workaround).
- Documented vc4 issues: shared memory not freed on fatal error (power cycle required);
  TMU-read + DMA-write to overlapping memory invalidates data.

## doe300/VC4CL and doe300/VC4C

- The OpenCL runtime and compiler this work extends; our accepted branches and the M21
  report live in forks under rughinnit.

## Open questions worth a second source

1. Rotate amount range and the meaning of v8min immediate 48.
2. `rep` write-swap semantics (per-quad vs full lane).
3. Whether the TMU truly allows only 2 outstanding requests per channel (measured 2).
4. v8muld rounding rule (measured round(a*b/255), lossy).
