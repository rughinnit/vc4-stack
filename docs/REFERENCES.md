# References

Every external source used by this work, with what it was used for and its status.
"Confirmed" means a fact from it was checked against our measurements or code.

## Primary hardware documentation

| source | URL | used for | status |
|---|---|---|---|
| Broadcom VideoCore IV 3D Architecture Reference Guide (AG100-R) | checked in at `VC4C/doc/VideoCoreIV-AG100-R.pdf` | VPM layout and setup fields, VDW store (UNITS/DEPTH/BLOCKMODE/STRIDE), V3D_VPMBASE, QPU register map | confirmed |
| VideoCore IV Programmers Manual (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/VideoCore-IV-Programmers-Manual | instruction-set background | referenced |
| MMIO Register map (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/MMIO-Register-map | register offsets for telemetry and the helper design | referenced |
| Register Documentation (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/Register-Documentation | register semantics | referenced |
| VideoCore IV kernels under Linux (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/VideoCore-IV-Kernels-under-Linux | how QPU kernels are launched from Linux without a driver | referenced |
| BCM2835 Overview (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/VideoCore-IV---BCM2835-Overview | **24 GFLOP/s at 250 MHz (3x4x8x250)** confirms 1 instr / 4 clocks and dual issue; L2 is 128 KB, GPU-only, outside the ARM MMU; L1 7 ns / L2 68 ns / SDRAM 128-167 ns | confirmed |
| Performance Considerations (wiki) | https://github.com/hermanhermitage/videocoreiv/wiki/VideoCore-IV-Performance-Considerations | FP pipeline over three cycles, stall if either of the next two instructions references the result; matches our 3-word dependent chains | confirmed |

## Repository and tooling sources

| source | URL | used for | status |
|---|---|---|---|
| hermanhermitage/videocoreiv | https://github.com/hermanhermitage/videocoreiv | `videocoreiv.arch`, `tinyasm`, IDA plugin, bare-metal samples; docs above | in use (xcheck) |
| hermanhermitage/videocoreiv-qpu | https://github.com/hermanhermitage/videocoreiv-qpu | QPU instruction-set documentation (rotations, pack/unpack, small immediates) | in use (xcheck) |
| itszor/vc4-toolchain | https://github.com/itszor/vc4-toolchain | GNU binutils port with a vc4 assembler/disassembler for byte-level cross-validation | in use (xcheck) |
| anholt/vc4-gpu-tools | https://github.com/anholt/vc4-gpu-tools | Broadcom-copyrighted `vc4_qpu_defines.h` (opcode/register enums), `vc4_packet.h`, `vc4_qpu_disasm.c` disassembler, `vc4_dump_hang_state.c` hang parser | in use (xcheck) |
| Raspberry Pi firmware/userland (`bcm_host`) | bookworm `libraspberrypi-dev` packages (see `README.debian13.md`) | required by VC4CL and V3DLib on Debian 13, which dropped it | in use |
| Mesa `vc4` gallium driver | https://gitlab.freedesktop.org/mesa/mesa (`src/gallium/drivers/vc4`) | reference for the CL submission UAPI and QPU encoding | referenced |
| Linux `drm/vc4` | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/vc4 | kernel-side reference for the read-only telemetry interface and a future compute UAPI | referenced (design) |
| Linux `drm/v3d` | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/v3d | model for a BO/submit/fence UAPI (used by Pi 4/5) | referenced (design) |

## Alternative frameworks

| source | URL | used for | status |
|---|---|---|---|
| axizo-pi/V3DLib | https://github.com/axizo-pi/V3DLib | MIT C++ DSL/compiler/runtime for vc4/vc6/vc7; `Examples/Matrix`, `Examples/NN`; Debian 13 vc4 install notes | benchmarked |
| wimrijnders/V3DLib (upstream author account) | https://github.com/wimrijnders/V3DLib | original project behind the axizo-pi re-upload | referenced |
| mn416/QPULib | https://github.com/mn416/QPULib | predecessor of V3DLib; VC4 QPU programming | referenced |
| mn416/py-videocore | https://github.com/mn416/py-videocore | Python QPU programming; considered for the hand-written path | referenced |

## Upstream runtime and compiler

| source | URL | used for | status |
|---|---|---|---|
| doe300/VC4CL | https://github.com/doe300/VC4CL | OpenCL 1.2 runtime for VideoCore IV; our fork carries the M21 report | forked / documented |
| doe300/VC4C | https://github.com/doe300/VC4C | OpenCL C compiler for VC4; our accepted improvements live here | forked / patched |
| doe300/VC4CLStdLib | https://github.com/doe300/VC4CLStdLib | GPU-side OpenCL standard library | used |

## Our repositories

| repo | branch | contents |
|---|---|---|
| https://github.com/rughinnit/vc4-stack | main | hand-written kernels (asm3-asm10), engine, training, docs |
| https://github.com/rughinnit/VC4C | `perf/combined`, `perf/product-acc`, `perf/pipeline2/3/4`, `perf/spill-retry`, `perf/postra-pack`, `feat/lane-block-stores` | compiler improvements; accepted and rejected branches |
| https://github.com/rughinnit/VC4CL | `docs/m21-gemm-findings` | M21 report, asm3 harness, measurements |
| https://github.com/rughinnit/vc4top | `master`, `gh-pages` | TUI; `.deb` and signed apt repository |

## Claims that cite these sources

| claim | source |
|---|---|
| 28.8 GFLOP/s at 300 MHz (1 instr / 4 clocks, 2 ops/instruction, 12 QPUs x 16 lanes) | BCM2835 Overview (24 GFLOP/s at 250 MHz) + our measurement |
| L2 128 KB, GPU-only, outside the ARM MMU, hence physical DMA addressing | BCM2835 Overview |
| FP results take 3 cycles; two consumers must not follow immediately | Performance Considerations (patent quote) + our 3-word chains |
| VPM/VDW store semantics (DEPTH=4, UNITS, stride as gap) | AG100-R + our probes |
| Rotate range and `rep` ws semantics | under test in `docs/XCHECK.md` (anholt/vc4-gpu-tools + tinyasm) |
