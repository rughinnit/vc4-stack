# Source verification matrix

Every external source given or used, what we took from it, and how we checked it.
Status: **confirmed** = reproduced by measurement or code; **corrected** = our earlier
reading was wrong and fixed; **partial** = verified only in part; **blocked** = could not
be exercised on this board.

## Links provided by the user

| source | claim used | verification | evidence | status |
|---|---|---|---|---|
| axizo-pi/V3DLib | C++ DSL/compiler/runtime for vc4; Matrix and NN examples | built on this board (2 build fixes); `Examples/Matrix` compared with CPU reference in interpreter **and** emulator backends | identical results, bit-identical between backends; `docs/V3DLIB.md` | **confirmed (correctness)** |
| axizo-pi/V3DLib | usable runtime here | `QPU_ENABLE` fails under KMS; `gpu_mem=76M` too small (32 MB heap request); runtime unbind Oopses on 6.18 | `docs/V3DLIB.md` blockers 1 and 2 | **blocked** |
| hermanhermitage/videocoreiv | documentation, tinyasm, IDA plugin, bare-metal samples | tinyasm/`vc4-toolchain` assemble **scalar VPU** code, not QPU (`fadd r0,r1,r2` -> ELF32 `0f02c800`) | `docs/XCHECK.md` | **corrected (not a QPU oracle)** |
| .../wiki/BCM2835-Overview | "24 GFLOP/s (3x4x8x250MHz)" | predicts 28.8 GFLOP/s at 300 MHz; we measured 28.36 ALU-only (98.5%) and 4.00 clocks/instruction | `docs/XCHECK.md`, `docs/NUMERICS.md` | **confirmed** |
| .../wiki/BCM2835-Overview | L2 = 128 KB, GPU-only, outside the ARM MMU | explains why DMA buffers must be bus-addressed, and why a stray DMA can overwrite any page (we corrupted libcrypto's page cache) | incident recorded in `docs/NUMERICS.md` / VC4D.md | **confirmed** |
| .../wiki/Performance-Considerations | FP ops pipeline over 3 cycles; a stall if either of the two following instructions reads the result | matches our measured 3-word dependent chains (add-ALU result to mul-ALU, A/B file RAW >= 2) | `docs/NUMERICS.md` | **confirmed** |
| anholt/vc4-gpu-tools | authoritative QPU enums, disassembler, hang parser | 5,329 words from our kernels decoded with **0 unknown opcodes**; 653 gemm6 words field-checked, 0 mismatches; rotate A/B isolate | `docs/XCHECK.md`, `tools/qpu_disasm.py` | **confirmed** |
| anholt/vc4-gpu-tools | register layout for telemetry | `vc4mon` 78 registers identical to `drm/vc4`'s `/sys/kernel/debug/dri/0/v3d_regs` | `docs/VC4MON.md` | **confirmed** |
| Anholt drm/vc4 + Mesa vc4 | no MMU; kernel-side validation model; reference compiler | informs our helper/UAPI design; QPU emit/schedule/validate/RA consulted | `docs/REFERENCES.md` | **referenced (not executed)** |
| Anholt TWIVC4 blog | TMU latency ~20 QPU instructions; threading only for fragment shaders | consistent with our 2-outstanding TMU limit and the 3-phase loop that removed the stall | `docs/REFERENCES.md`, asm9 results | **consistent** |
| mn416/QPULib | dormant predecessor of V3DLib; examples and perf notes | not built; technique reference only (same KMS/root constraints as V3DLib) | `docs/REFERENCES.md` | **referenced** |
| Idein/py-videocore | independent Python QPU encoder and an SGEMM example | encoding cross-check run as a **fourth oracle**; SGEMM not run because its VCSM path is dead on 6.18 | `docs/XCHECK.md` (fourth encoder section) | **partial (encoder), blocked (runtime)** |
| **Medium: lnstadrum, GPU-accelerated classification on low-end hardware** | **GLES 2.0 path on VC4: 595 ms on Zero W / 328 ms on Pi 3 B+; RGBA8 0..1 activations; <=25 texels and <=8 samplers per shader; 8-bit activations reach 72 % top-1** | **read the whole article (Medium 403s; used the Wayback capture of 2024-12-05); checked its two structural claims against our measurements** | **its Pi 3 B+ 328 ms at 400 M MAC = 1.22 GMAC/s vs our 273 M MAC in 478-520 ms = 525-570 MMAC/s (not like-for-like: theirs is 8-bit and co-designed); their "CPU must keep the GPU supplied" matches our 59 % QPU duty and 41 % launch path; `docs/MEDIUM-CLASSIFICATION.md`** | **confirmed (claims and arithmetic); not reproducible here (GL path, quantised model)** |
| **capstone-engine/capstone** | **"a general multi-architecture disassembler could serve as a QPU encoding oracle"** | **fetched the repository page; read the supported-architecture list and the licence** | **no VideoCore IV / QPU target in the list (ARM, AArch64, Alpha, ARC, BPF, EVM, LoongArch, HPPA, M68K, M680X, Mips, MOS65XX, PPC, RISC-V, SH, Sparc, SystemZ, TMS320C64X, TriCore, WASM, XCore, x86, Xtensa) — cannot decode a QPU word; our oracle stays `qpu_disasm.py` + anholt + vc4asm** | **rejected (not applicable)** |
| **Seneral/VC4CV** | **MIT VC4 CV framework; QPU 5x5 stencil programs, `vc4.qinc` VPM/DMA macros** | **fetched the repo README, the `qpu_programs/` listing, `vc4.qinc` and `qpu_mask_tiled_5x5_minopt.asm`; checked two hardware claims against our code** | **`vdw_setup_1` says the VPM stride field is 16 bits, not the documented 13 — our `vpmwords.vdw_stride()` already validates to `0xFFFF`; the stencil uses register-held column minima with `v8min` on byte lanes, unlike our im2col path; `docs/VC4CV.md`** | **confirmed (claims); blocked (runtime — mailbox/VCSM QPUs, fb/camera target)** |
| jetpacapp/DeepBeliefSDK | CNN inference on the Pi using the GPUs, ~5 s/image (3 s boosted), root | historical baseline only; Pi library is a prebuilt 32-bit Raspbian `.so`, not runnable on arm64 trixie | `docs/REFERENCES.md` | **referenced (baseline)** |
| jetpacapp/qpu-asm | QPU macro assembler + DMA macros; ancestor of `maazl/vc4asm` | our encoder is validated against vc4asm, i.e. against his lineage | `docs/XCHECK.md` | **confirmed (lineage)** |
| petewarden.com posts (2014) | GPU optimisation notes; Deep Belief port with QPU GEMM | technique/baseline reference; numbers consistent with our measured band | `docs/REFERENCES.md` | **referenced** |

## Other sources used

| source | claim | verification | status |
|---|---|---|---|
| Broadcom AG100-R | VPM/VDW setup fields, VPMBASE units, register map | probes: DEPTH=4 store, stride-as-gap, VPM 48-row window, field encodings | **confirmed** |
| maazl/vc4asm | canonical rotate encoding | 33/33 forms bit-exact after our `qenc` fix | **confirmed** |
| doe300/VC4C | compiler baseline | our accepted branches; 130/130 corpus compiles | **confirmed** |
| doe300/VC4CL | OpenCL runtime | installed and gated with our compiler (3.085 GFLOP/s) | **confirmed** |
| Mesa drm/v3d (Pi 4) | BO/submit/fence UAPI model | design reference for the read-only/compute interface | **referenced** |
