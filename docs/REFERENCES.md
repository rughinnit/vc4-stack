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
| hermanhermitage/videocoreiv | https://github.com/hermanhermitage/videocoreiv | `videocoreiv.arch` and `tinyasm` are the **32-bit scalar VPU**, not the QPU; IDA plugin, bare-metal samples; docs above | reviewed; not a QPU oracle |
| hermanhermitage/videocoreiv-qpu | https://github.com/hermanhermitage/videocoreiv-qpu | QPU instruction-set documentation (rotations, pack/unpack, small immediates) | in use (xcheck) |
| itszor/vc4-toolchain | https://github.com/itszor/vc4-toolchain | GNU binutils port for the **32-bit scalar VPU** (ELF32, `fadd$alu32cond`), not the 64-bit QPU word; `gas` builds but is not a QPU oracle | reviewed in `docs/XCHECK.md`; not used |
| maazl/vc4asm | https://github.com/maazl/vc4asm | independent QPU macro assembler/disassembler with constraint checker; source of the `r5rep`/`r5quad` names and the rotate/constant aliasing notes | in use (xcheck) |
| **lnstadrum, "Towards GPU-accelerated image classification on low-end hardware"** | **https://medium.com/analytics-vidhya/towards-gpu-accelerated-image-classification-on-low-end-hardware-ec592e125ad9** | **Analytics Vidhya, 2021-04-20: 225 K-parameter, 120-class GLSL classifier on Pi Zero W at 595 ms (670 M MAC/s), 328 ms on Pi 3 B+; documents the GLES 2.0 limits (RGBA8 0..1 textures, <=25 texels, <=8 samplers) and a co-designed grouped-conv + channel-shuffle network. Corroborates our launch/CPU-supply bound and the CPU-beats-GPU-at-160 result. See `docs/MEDIUM-CLASSIFICATION.md`** | **read in full (Wayback snapshot, 2026-11-20); numbers quoted, no runtime here (GL path, 8-bit model, not ours)** |
| **Seneral/VC4CV** | **https://github.com/Seneral/VC4CV** | **MIT-licensed VC4 computer-vision framework: 19 hand-written QPU programs (5x5 min/max stencil, tiled blit, blob mask), `vc4.qinc` VPM/DMA bit-layout macros, and the "ZeroCompat" camera-buffer erratum. Cross-checks our VDW 16-bit stride and the TMU scheduling rule; candidate register-sliding-window stencil for pooling. See `docs/VC4CV.md`** | **reviewed (readme + vc4.qinc + 5x5_minopt.asm); 16-bit VDW stride confirmed; not runnable here (legacy mailbox/VCSM, fb/camera target)** |
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
| Rotate codes: 48 = rotate by r5, 49..63 = rotate 1..15; 48..63 alias constants -16..-1 | confirmed in `docs/XCHECK.md` (vc4asm, qpuasm, AG100-R addendum) |
| `rep` ws semantics: ws=1 = `r5rep` (16 lanes), ws=0 = `r5quad` (per quad) | confirmed in `docs/XCHECK.md` (vc4asm parser tables, qpuasm tables, AG100-R addendum) |

## Emma Anholt — author of the open VC4/V3D stack

Profile and resume: https://www.anholt.net/resume.html · https://github.com/anholt

She built the open source Linux graphics stack for VideoCore IV at Broadcom
(2014-2019), then the initial V3D driver for the Pi 4, and is now at Igalia.

| work | URL | relevance to this project |
|---|---|---|
| **`drm/vc4` kernel driver** (command execution, memory management, HDMI/DPI/DSI display, bcm2835 platform drivers) | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/vc4 | THE reference for the root-removal design. VC4 has no MMU between the 3D engine and main memory, so the driver validates shader programs in the kernel; that is the model a safe compute UAPI must follow. |
| **Mesa `vc4` Gallium driver** | https://docs.mesa3d.org/drivers/vc4.html · https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/gallium/drivers/vc4 | Reference QPU compiler: `vc4_qpu.c/h`, `vc4_qpu_emit.c`, `vc4_qpu_schedule.c`, `vc4_qpu_validate.c`, `vc4_register_allocate.c`, `vc4_qir_*` | 
| **`anholt/mesa` (archived)** | https://github.com/anholt/mesa | historical vc4/v3d branches; the repo itself says use GitLab |
| **`anholt/vc4-gpu-tools`** | https://github.com/anholt/vc4-gpu-tools | Broadcom QPU enums, packet headers, disassembler, hang-state parser; used for encoder cross-validation |
| **V3D driver (Pi 4)** | Mesa `src/gallium/drivers/v3d` + `drivers/gpu/drm/v3d` | model for a BO/submit/fence UAPI with an MMU |
| **`anholt/twivc4` — "This Week in VC4"** | https://github.com/anholt/twivc4 · https://anholt.livejournal.com | developer notes: TMU latency ~20 QPU instructions, cooperative hyperthreading only for fragment shaders, delay-slot/TLB_Z scheduling, dmabuf fencing |
| **`anholt/libepoxy`** | https://github.com/anholt/libepoxy | GL dispatch library (not used here) |
| **`anholt/hash_table`**, **`anholt/drm-v4l2-test`** | https://github.com/anholt/hash_table · https://github.com/anholt/drm-v4l2-test | small C utilities; dmabuf sharing demo (context for our dma-buf use) |
| **bcm2835 platform drivers** | kernel `drivers/clk/bcm`, `drivers/soc/bcm`, `drivers/firmware/raspberrypi.c`, dmaengine/spi/i2c/uart/gpio/mmc/thermal | the platform plumbing a compute UAPI would have to cooperate with |

### What her work confirms for us

- **No MMU on VC4** between 3D/QPU and shared main memory. She solved the security
  problem with kernel-side validation of idiomatic shader assembly. That is exactly the
  argument for a validated helper/UAPI rather than granting `/dev/mem`.
- **TMU latency ~20 QPU instructions** (her blog), consistent with our measured 2-deep
  TMU limit and the need for 3-phase pipelining to cover it.
- **Cooperative thread switching exists only for threadable fragment shaders**, which is
  why our compute kernels cannot hide latency with a second thread.
- The Mesa `vc4` compiler is the mature version of what VC4C does; its scheduler and
  register allocator are the reference for any further compiler work.

## QPULib and py-videocore (checked 2026-11-19)

### mn416/QPULib — https://github.com/mn416/QPULib

- 439 stars, C++, v0.1.0, **experimental and no longer under development** (last push 2020-12).
- The direct predecessor of V3DLib: a C++ embedded DSL compiled at runtime to QPU programs,
  with interpreter/emulator backends and worked examples (Euclid, 3D rotation with unrolling
  and non-blocking loads, 2D convolution/heat transfer) plus **performance sections** in its
  README.
- Runs on the Pi ARM and offloads to the QPUs; needs root and the same QPU ownership as
  V3DLib (KMS/`QPU_ENABLE` conflict applies).
- **Use:** historical reference and technique source (unrolling, non-blocking loads, multi-QPU
  dispatch), and a second correctness oracle. Not a runtime candidate (superseded by V3DLib,
  same blockers).

### Idein/py-videocore — https://github.com/Idein/py-videocore

- 813 stars, Python, **MIT**, actively maintained (last push 2024-03).
- A Python internal DSL for QPU assembly: host and GPU code in one script, no ahead-of-time
  compilation. Package modules: `assembler.py`, `encoding.py`, `driver.py`, `mailbox.py`,
  `vinstr.py`, `checker.py`, `v3d.py` (V3D support for newer Pis).
- Examples include **`sgemm.py` and `sgemm_1thread.py`** — a full SGEMM in the DSL, the most
  directly comparable third-party implementation to our hand-written GEMM.
- `driver.py` allocates through **VCSM** (`rpi_vcsm.VCSM`) and talks to `/dev/vcio`; on this
  kernel the legacy VCSM/VCHI path is dead (`vc_sm_cma_vchi_init: failed to open VCHI service`),
  the same blocker V3DLib hit. So it is not usable as a runtime here as-is.
- **Use:** its `encoding.py`/`assembler.py` are a **fourth independent QPU encoder** (after
  vc4asm, qpuasm, vc4dis) and a good candidate to extend `docs/XCHECK.md`; `sgemm.py` is a
  reference implementation for comparison; the DSL is a model for ergonomic kernel writing.
- Related: `Idein/py-videocore6` targets the Pi 4 V3D.

## Pete Warden (jetpacapp) — QPU assembler and GPU-accelerated CNN on VC4

| work | link | what it is / relevance |
|---|---|---|
| **`jetpacapp/qpu-asm`** | https://github.com/jetpacapp/qpu-asm | Macro assembler for the VideoCore IV QPU, with macros/functions and DMA-controller macros. It is the direct ancestor of **`maazl/vc4asm`**, one of the four independent assemblers used in `docs/XCHECK.md`. Our encoder validation chain runs through his work. |
| **`jetpacapp/DeepBeliefSDK`** | https://github.com/jetpacapp/DeepBeliefSDK | Jetpac's CNN inference framework (AlexNet architecture from Krizhevsky et al.) for iOS/Android/Linux/OSX/**Raspberry Pi**. The Pi build uses the **QPUs for a large part of the calculation** and must run as root. Reported: **~5 s per image on a stock Pi, ~3 s with a boosted GPU clock** (2014). |
| "Deep learning on the Raspberry Pi!" | https://petewarden.com/2014/06/09/deep-learning-on-the-raspberry-pi | Write-up of the port; explains the GPU offload and the open-sourced QPU/DMA macros. |
| "How to optimize Raspberry Pi code using its GPU" | https://petewarden.com/2014/08/07/how-to-optimize-raspberry-pi-code-using-its-gpu | Practical QPU optimisation notes; builds on Eman's assembler work. |
| Raspberry Pi Foundation news | "More QPU magic from Pete Warden" and "New QPU macro assembler" | The Foundation's own summary: his Deep Belief port provided an **accelerated GEMM matrix-multiply on the QPUs**. |

### Why this matters here

- **Historical precedent for exactly our goal:** a full CNN ran on the VC4 QPUs in 2014 with a
  QPU GEMM, at ~5 s/image (~3 s with a GPU overclock) on a stock Pi. That is the same order as
  our measured band for a synthetic 160x160 forward (202.6 ms) and our estimate for small models;
  DeepBelief's network is AlexNet-scale, far larger than YOLOv8n's per-image work at low resolution.
- **Toolchain lineage:** qpu-asm -> vc4asm -> our cross-validation oracles. Our encoder is checked
  against the descendents of his assembler.
- **Caveat:** the DeepBelief Pi library is shipped as a prebuilt 32-bit Raspbian `.so`
  (`RaspberryPiLibrary`), with the network implementation not open; it is not directly runnable on
  our arm64 trixie board and cannot be used as a component. The value is the assembler, the
  technique notes, and the historical baseline.
