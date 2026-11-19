# V3DLib on this board — build, correctness, blockers, verdict

Source: https://github.com/axizo-pi/V3DLib (MIT, v0.9.2, `main` @ `aab3820`, 2026-11-16).
Platform: Pi 3B Rev 1.2, Debian 13 trixie, kernel 6.18.50+rpt-rpi-v8, GCC 14.2, CMake 3.31.
Work tree: `/home/raspi/work/v3dlib`. Board restored afterwards: `/dev/dri/{card0,renderD128}`,
`vc4_v3d` bound, stock 400/300 MHz, `gpu_mem=76M`, config.txt unchanged.

## Verdict

**REJECT as a runtime backend. BORROW selectively (offline interpreter/emulator, techniques).**
It cannot execute on this board without removing KMS and raising gpu_mem, which would break the
existing VC4CL/DRM stack; and its generic Matrix kernel is indicatively far slower than ours.

## Build

There is no top-level CMakeLists; it is a hand-written Makefile requiring `BASE` and three externals.
- `extern/userland` -> libvcsm.a, libbcm_host.so: OK
- `extern/mesa` -> libmesa.a: OK (~2 min)
- sibling `CmdParameter`: upstream only has v0.4.6 while `script/install.sh` demanded 0.4.7

Local fixes required (both in the work tree, uncommitted):
| file | change | reason |
|---|---|---|
| `Lib/global/basics.h` | `#include <cstdint>` | GCC 14 no longer gets uint64_t transitively |
| `script/install.sh` | expected 0.4.6 | upstream version gate aborts |

Built: libv3dlib.a, Matrix, Hello, ID, runTests, and NN (RNN/LSTM/GRU). `Examples/NN/Tools/Try`
fails to build (hardcoded `/home/wim/...` include path, missing `global/global.h`); unrelated.

## Correctness (CPU backends)

`Examples/Matrix` has no checker, so N=48 deterministic inputs were compared across modes:
| mode | time | vs CPU reference |
|---|---|---|
| `-k=cpu -r=interpreter` | ~0.1 s/rep | reference |
| `-r=interpreter` | 0.816 s | exact match |
| `-r=emulator` (compiled vc4 code) | 1.643 s | exact match, bit-identical to interpreter |

So the vc4 code generator and kernel semantics are correct. These are CPU-bound, not performance.

`runTests`: 9 cases, 5 passed, 4 failed, 64 skipped; all four failures are
`FATAL: Failed to allocate vc4 shared memory.` (environment, not logic). `RNN -r=interpreter`
aborts on `Unknown ALU opcode` (an interpreter bug unrelated to the GPU).

## Why there is no GPU number

### Blocker 1 — KMS owns the QPU/V3D block
`Hello` as root: `FATAL: Unable to enable QPUs. Check your firmware is latest.`
`vc4_v3d` is bound to 3fc00000.v3d; mailbox `QPU_ENABLE` (tag 0x30012) fails while
`dtoverlay=vc4-kms-v3d` is active. The documented fix is to remove the overlay and reboot,
which removes `/dev/dri/renderD128` and breaks VC4CL.

A runtime unbind was tried instead: the kernel **Oopsed in `vc4_irq_uninstall`** during
`component_unbind` (NULL deref, headless device), and rebind hung in D-state. Reboot recovered.
Conclusion: no safe runtime handover on kernel 6.18; config.txt was left unchanged.

### Blocker 2 — gpu_mem=76M is too small
V3DLib requests a 32 MB heap for its 8 MB default (`BufferObject::alloc_mem` multiplies by 4),
and `-m=` cannot lower it because the default heap is allocated before the setting is applied
(confirmed with gdb). Direct mailbox probe: 1/4/8/16 MB succeed; 24/32/48 MB fail.
Its install doc requires `gpu_mem=128`.

Also noted: `vc_sm_cma_vchi_init: failed to open VCHI service (-11)` — the legacy VCSM/VCHI path
is dead on this kernel/firmware; the vc4 mailbox path itself works.

## Performance comparison

| implementation | N=128 | N=256 |
|---|---|---|
| hand-written GEMM (ours) | - | 13.72 GEMM-only / 11.97 total |
| engine matmul (ours) | - | 8.49 |
| VC4CL compiled (ours) | 2.66 | 3.085 |
| ALU-only ceiling (ours) | - | 28.36 |
| V3DLib Matrix | not measurable | not measurable |

Upstream ships only a seconds-vs-dimension plot (10 iterations, 12 QPUs); at N=256 that is
~0.57 s, i.e. indicatively <= 0.6 GFLOP/s. This is NOT a measurement and is not entered in any
results table.

## Integration assessment

- Needs root, `/dev/mem`, `/dev/vcio`, `libbcm_host`, `libvcsm`, and **exclusive** QPU ownership.
- Memory: firmware mailbox MEM_ALLOC -> MEM_LOCK -> `/dev/mem` mmap; TMU reads via L2, DMA writes,
  no coherency (hence its documented TMU/DMA overlap corruption).
- Language: C++ embedded DSL (QPULib lineage), compiled at runtime, with interpreter/emulator.
- Cannot coexist with VC4CL, the `vc4d` daemon or concurrent GPU users.

Useful path: use its interpreter/emulator as an offline correctness oracle and borrow techniques
(TMU prefetch, 16-lane dot products, DSL kernel generation). MIT licence permits it.

## To obtain a real V3DLib number

A dedicated Pi 3 (or explicit permission to reconfigure this one) with:
```
#dtoverlay=vc4-kms-v3d
gpu_mem=128
```
then reboot and run `Matrix -d=128/-d=256 -nmax`. VC4CL is unavailable during that window.
