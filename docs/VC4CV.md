# VC4CV — VideoCore IV Computer Vision framework (Seneral)

| | |
|---|---|
| source | <https://github.com/Seneral/VC4CV> |
| licence | **MIT** (`LICENSE` in the repo root) |
| size / activity | 18 commits, 18 stars, 3 forks, 2 open issues (as of 2026-11-20) |
| language | C++ (host/GL) + QPU assembly (vc4asm syntax) |
| fetched for this review | repo page + `README.md`, `qpu_programs/` listing, `qpu_programs/vc4.qinc`, `qpu_programs/qpu_mask_tiled_5x5_minopt.asm` (2026-11-20) |
| status here | **documented, not integrated** — cannot run on this stack as-is (see §5); two techniques are candidates (§6) |

## 1. What it is

"VideoCore IV Computer Vision framework and examples for the RaspberryPi Zero —
GL- and QPU-based." It is a **computer-vision** project, not an ML one: camera
frame capture, colour-space conversion, thresholding, blob detection and a 5×5
min/max stencil, implemented twice — once with OpenGL ES 2.0 shaders, once in
hand-written QPU assembly.

Layout (from the repo):

```
camera/          camera capture (mmal/vc path)
gl/              GL ES 2.0 host code
gl_blobs/        two-pass blob-detection shaders + CPU connected components
gl_shaders/      the shader sources
qpu/             host-side QPU launch plumbing
qpu_programs/    19 .asm programs + vc4.qinc (macro/bit-layout header)
main_gl.cpp  main_gl_blobs.cpp  main_qpu.cpp   Commands.md   CMakeLists.txt
```

QPU programs: `qpu_fb_pattern`, `qpu_debug_full/tiled`, `qpu_blit_full/tiled`,
`qpu_mask_full`, `qpu_mask_tiled` (+ `1x1`, `1x1_optimized`, `1x5`,
`1x5_organized`, `5x5`, `5x5_blob`, `5x5_blobopt`, `5x5_blobwrite`,
`5x5_memory`, `5x5_min`, `5x5_minopt`), `qpu_free_mutex`, `qpu_test_tiled`.
They are assembled with <https://github.com/maazl/vc4asm> (already one of our
encoder cross-check oracles), which is what the `.qinc` include and the
`ldtmu0`/`t0s`/`v8min` syntax imply.

## 2. Why it is relevant to this project

It is an **independent** VC4 QPU codebase that solves a problem we did not
solve that way: neighbourhood stencils (5×5 min/max) held in registers and
shifted with byte-lane operations, rather than our im2col + GEMM formulation.
It is also a second pair of eyes on the VPM/DMA bit layouts, and it documents a
SoC erratum we had not seen.

## 3. What it corroborates (checked against our code/measurements)

| claim in VC4CV | our equivalent | verdict |
|---|---|---|
| `vdw_setup_1`: "VPM supports **16 bit** stride rather than 13 as documented" | `infer/tools/vpmwords.py:71` — `vdw_stride()` validates `stride` to `0xFFFF`, not `0x1FFF` | **confirmed independently**; our encoder already relies on it |
| TMU loads are register-scheduled (`ldtmu0` early, value read later via `t0s`) | our TMU rule: 2 outstanding requests per channel, ~20-instruction latency | consistent |
| VPM write must wait for the previous VDW (`read vw_wait` before `mov vw_setup`) | our launch path does the same (`vw_wait` before re-arming) | consistent |
| GL path costs "tens of milliseconds" for a nontrivial shader at 1640×1232; raw QPU is "milliseconds" | our own finding that per-launch overhead, not arithmetic, dominates at small sizes (`docs/CLOCK-VARIANCE.md`, `docs/BUILD.md`) | consistent |

## 4. The bug worth knowing about: "ZeroCompat"

Documented in the README (issue #1): on the Zero, **TMU reads from buffers that
come directly out of the camera pipeline are unreliable** — with or without
parallel QPU execution — while the same read works if the frame is first copied
to another buffer, and VDR (VPM DMA) works on the camera buffer directly. Their
workaround (in `Seneral/MakersVR`) is a separate QPU program that copies the
frame with VDR+VDW before any TMU access.

We do not read a camera pipeline, so this does not affect us today. It matters
as a **general caution**: TMU results can depend on the provenance of the
buffer, not only on its contents — which is the same class of surprise as the
cacheable-arena coherency bug we fixed in `docs/ONNX.md` §10.4.2. If a future
capture path feeds the QPUs directly, this is the first thing to test.

## 5. Why it cannot run here as-is

* It enables the QPUs through the **legacy mailbox/VCSM** path. On this board
  (`6.18.50+rpt-rpi-v8`, KMS) VCSM is dead (`vc_sm_cma_vchi_init: failed to
  open VCHI service`) and `QPU_ENABLE` is refused — the same blockers recorded
  for V3DLib and py-videocore in `docs/V3DLIB.md` and `docs/SOURCES-VERIFICATION.md`.
* It targets the **framebuffer and camera pipeline** (`/dev/fb0`, mmal), not a
  compute arena, so its launch and memory model is the opposite of `vc4d`'s.
* Programs are written for **vc4asm**; our launcher takes encoded 64-bit words
  from `kgen`/`qenc`. Feeding theirs through `vc4d` would mean assembling with
  `vc4asm`, extracting the words, and replacing their framebuffer/camera
  uniforms with arena addresses — possible, but only worth doing for a
  technique we actually want (§6).

**No code has been copied.** MIT would permit it with attribution; the value
here is the ideas and the bit-layout cross-check, not the source.

## 6. Candidate techniques (ranked)

1. **Register sliding-window stencil.** `qpu_mask_tiled_5x5_minopt.asm` keeps a
   window of column-minima in registers and shifts them with `shr`/`shl` +
   `v8min` on **byte lanes**, producing a 5×5 min filter at 4 pixels per
   instruction group. Our `vc4_maxpooln` is a straightforward 4-row × 16-column
   walk; a window-shift formulation could cut the QPU time for 3×3/5×5 pooling
   and small conv kernels substantially. Worth prototyping against our current
   `maxpooln` on 5×5/stride-1 (the SPPF case in YOLOv8n).
2. **Byte-lane arithmetic (`v8min`/`v8adds`, `.8af/.8bf` swizzles).** We use
   32-bit lanes almost exclusively. For 8-bit quantised or mask-producing work
   (thresholding, NMS masks, int8 storage) this is 4× the lanes.
3. **Semaphore/mutex pattern** (`qpu_free_mutex.asm`, `sacq`/`srel`): a way to
   coordinate multiple QPU programs without a host round trip. Relevant to the
   "batch several nodes per launch" idea that currently fails for elementwise
   work.
4. **`vc4.qinc` as a third independent encoder oracle.** Its `vdr_setup_*`,
   `vdw_setup_*`, `vpm_setup`, `dma_*` and `h*/v*` macros encode the same
   fields as our `vpmwords.py`; diffing the two would catch a field-width error
   the way `vc4asm` and `anholt/vc4-gpu-tools` already did for opcodes.

## 7. Status

Documented and cited; not integrated; no measurements taken from it. The
stencil technique in §6.1 is the only item I would schedule, and only against a
measured comparison with the existing kernel — consistent with how every other
kernel in this repository was accepted.
