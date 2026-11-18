# libvc4infer 0.1.0

An inference engine layer for the Raspberry Pi's VideoCore IV GPU (V3D, Pi 3B,
stock clocks), built on top of the hand-written, element-verified QPU GEMM from
`/home/raspi/work/asm3`.

It is a **C library** (`libvc4infer`) plus tests, a kernel generator and a
Python/ctypes binding.  Everything GPU-side is a QPU kernel; the CPU only moves
bytes (padding, layout), never multiply-accumulate.  Any op, shape, layout or
dtype that is not implemented returns `VC4_ERR_UNSUPPORTED` — there is no CPU
compute fallback anywhere.

```
include/vc4infer.h      public API
src/vc4_dev.c           dma-buf arena, /dev/mem, V3D launch, L2/SLC handling
src/vc4_kern.c          kernel blob cache + JIT, the asm3 two-launch rule
src/vc4_gemm.c          vc4_matmul: QPU pack -> verified QPU GEMM
src/vc4_conv.c          vc4_conv2d: QPU im2col -> GEMM (bias folded as a k term)
src/vc4_ew.c            add/mul/relu/sigmoid/silu elementwise kernels
src/vc4_api.c           the op registry (strict dispatch) + coverage table
tools/kgen.py           QPU kernel generator (reuses the asm3 encoder)
tests/                  CPU-oracle verification, no fallback paths
python/vc4infer.py      ctypes binding with the same API
run_tests.sh            serialised, health-checked test runner
```

## Build and run

```sh
make                       # libvc4infer.a, libvc4infer.so, tests
mkdir -p kcache             # generated kernel blobs live here
sudo ./build/test_matmul 5  # 5 runs, every element checked
sudo ./build/test_conv 5
sudo ./build/test_ew 5
./run_tests.sh              # all of the above, with the GPU lock + health checks
```

The tests take `/tmp/vc4-gpu.lock` themselves and print a board state after
each stage.  **One GPU user at a time**: two concurrent workloads (or a wild
kernel) can hang the V3D hard enough to require a power cycle.

## API surface

```c
vc4_status vc4_init(vc4_ctx **ctx, const char *cache_dir, size_t arena_bytes);
void       vc4_shutdown(vc4_ctx *ctx);
size_t     vc4_arena_bytes / vc4_arena_used;
vc4_status vc4_alloc(vc4_ctx*, size_t, vc4_buf*);          /* persistent   */
vc4_status vc4_upload / vc4_download / vc4_host_ptr / vc4_bus_addr;

vc4_status vc4_matmul (ctx, A, B, C, M, N, K, layout);
vc4_status vc4_conv2d (ctx, in, Cin, H, W, wt, Cout, KH, KW, bias,
                       stride, pad, out, &HO, &WO);
vc4_status vc4_elementwise(ctx, op, a, b, out, n);
vc4_status vc4_relu / vc4_sigmoid / vc4_silu (ctx, a, out, n);

const vc4_op_info *vc4_supported_ops(size_t *n);           /* coverage table */
int  vc4_op_supported(vc4_op);
void vc4_print_supported_ops(FILE*);
void vc4_get_stats(ctx, vc4_stats*);   /* launches, pack/kernel ns, FLOPs */
```

Buffers are `{off, bytes}` inside **one** dma-buf arena (64 MiB by default,
falling back to 32/16/8 MiB if CMA is tight).  Persistent buffers are bump
allocated; each op also uses a scratch region that is reset when it returns.

## How the GEMM works

`vc4_matmul` is two QPU launches in one process against one arena:

1. **pack** — `tools/kgen.py`'s panel builder: per 16x16 tile, a VDR (VPM DMA
   load, memory pitch = K\*4) followed by a *vertical* VDW, which the hardware
   writes out transposed as 16 consecutive 64 B vectors, i.e. the k-major panel
   `PANEL[(rb*K + k)*16 + r] = A[rb*16 + r][k]`.  No ALU data movement.
2. **gemm** — the asm3 kernel: 16x16 C tile per QPU request, 16 lanes = 16
   columns, A read from the panel (one 64 B line per k), B row-major, packed
   `fmul`+`fadd`, hardware rotate+replicate broadcast, 39-word k loop,
   VPM->DMA store of 4 chunks of 4 rows.

The k-loop body is emitted by `emit_kloop()` and is **bit-identical** to
`asm3/gemm2_128_d3_v1.bin` (39 words, same branch offset; checked during
development and reproducible with `tools/kgen.py gemm ...` + a 39-word
compare).  That is the correctness anchor of this library: the part that
computes is the part that was verified.

Generalising square N x N to arbitrary M x N x K changed only addressing:
the tile loop became a two-level (row block x column block) loop, every
stride/offset is a uniform chosen by the ARM, and the row-block loop has a
compile-time bound plus an in-loop guard (asm3's own pattern), so a corrupted
register cannot turn it into an unbounded DMA march.

## The two-launch rule

asm3 found that overwriting the code area and launching again can execute the
*previous* program, because the QPU instruction fetch goes through the L2 and a
clean line at the same address shadows the new code.  This library gives every
distinct kernel blob its own code area and its own uniform area
(`VC4_CODE_SLOT` / `VC4_UNI_SLOT` in `src/vc4_int.h`) and invalidates
`SLCACTL` before every launch.  Kernels are cached by shape in `kcache/`
(generated on first use by `tools/kgen.py`; `VC4INFER_NO_JIT=1` makes a cache
miss an error instead).

Every QPU kernel also obeys the asm3 rule that a TMU *pop* never shares an
instruction word with a write to a TMU address register (that shared word is
the 12-QPU corruption root-caused in `asm3/NOTES.md`).

## Writing QPU code safely

`tools/kgen.py` tracks every register write and read it emits and inserts nops
so a dependent read is never closer than three words (the QPU has no
forwarding), except inside the verified k loop and store, which keep asm3's
proven schedule verbatim.  Kernel loop bounds are compile-time constants, not
uniforms, wherever they drive DMA.  Both rules exist because a wrong counter
in a panel builder turns a 64-tile transpose into an unbounded VPM DMA march
over physical memory — see `RESULTS.md`.

## Op coverage (strict dispatch)

`vc4_print_supported_ops()` prints the table; `src/vc4_api.c` is the single
source of truth.  Supported: `matmul`, `conv2d/3x3`, `conv2d/1x1`, `add`,
`mul`, `relu`, `sigmoid`, `silu`.  Explicitly unsupported (and listed as such,
so the table is an honest coverage statement): `conv2d/5x5`,
`conv2d/stride2`, `conv2d/depthwise`, `conv2d/grouped`, `matmul/f16`,
`batched-matmul`, `maxpool`, `softmax`, `layernorm`, `concat/slice`,
`transpose/permute`, `upsample`, `quantize/dequantize`.

## conv2d

`vc4_conv2d` never materialises an im2col matrix:

1. the padded NCHW image is built in the arena (zero fill + one `memcpy` per
   row: layout, not arithmetic);
2. a QPU kernel builds the packed A panel directly from it.  For one row block
   (16 output pixels, contiguous in the padded image) and one filter tap, a
   single VDR loads 16 channels x 16 pixels with the pitch set to the padded
   plane size, and the vertical VDW writes the transposed 1 KB block
   `PANEL[rb][tap*Cin + ci_block*16 .. +15][0..15]`;
3. the weights `[Cout][Cin*KH*KW]` are transposed into a k-major B panel by the
   same pack kernel;
4. the GEMM runs with `bmode=1` (B from the packed panel).  `bias` is folded in
   as one extra k term (A row = 1.0, B row = bias), so even the bias add is on
   the GPU.

Constraints (from the VDR's 13-bit pitch field and the 16-wide panel tiles):
`stride == 1`, `KH == KW in {1,3}`, `Cin % 16 == 0`, `Cout % 16 == 0`, and
`(H + 2*pad) * align16(W + 2*pad) * 4 <= 8191` bytes of padded plane.  Taps are
unrolled at generation time, so a 5x5 is a generator change, not a new kernel.
The final store into NCHW `[Cout][HO][WO]` is a strided gather on the ARM
(bytes only).

## Elementwise

One iteration = 16 vectors = 256 floats per QPU: 16 TMU gathers per operand
(16 lanes = 16 consecutive floats = one 64 B line), the arithmetic, then the
verified 4-chunk VPM->DMA store with stride 0.  `add`/`mul`/`relu` are exact
IEEE fp32.  `sigmoid` is `1/(1+exp(-x))` with `exp(-x) = 2^(-x*log2 e)`, a
degree-5 minimax polynomial for `2^f` on [-0.5, 0.5] and a bit-trick seed plus
three Newton steps for the reciprocal; `silu = x*sigmoid(x)`.  Measured error
is printed by `tests/test_ew.c`.

## Python binding

```python
import vc4infer, numpy as np
with vc4infer.Engine() as eng:
    a = eng.upload(np.random.rand(256,256).astype(np.float32))
    b = eng.upload(np.random.rand(256,256).astype(np.float32))
    c = eng.alloc(256*256*4)
    eng.matmul(a, b, c, 256, 256, 256)
    print(eng.download(c, 256*256*4).reshape(256,256))
```

## PyTorch `PrivateUse1` integration path (not built here)

The engine is deliberately the *backend* only: a thin C++ extension would own
the tensor plumbing.  The path that keeps strict dispatch intact:

1. **Register the backend.** `torch.utils.rename_privateuse1_backend("vc4")`
   then `torch.utils.generate_methods_for_privateuse1_backend()` gives
   `torch.device("vc4:0")` and `Tensor.to("vc4")`.
2. **Allocator.** Implement `at::Allocator` over the single `vc4_ctx` arena.
   `data_ptr` is an opaque handle; store the arena *offset* and let the kernels
   recover it (`off = ptr - arena_bus`).  Nothing else may dereference it —
   which is exactly what strict dispatch guarantees.
3. **Hooks.** `PrivateUse1HooksInterface` must report
   `hasPrimaryContext() == true`, `getDefaultGenerator` throwing, and **no
   `DeviceGuard` fallback** to CPU.
4. **Kernels.** `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` for the subset the
   engine implements: `mm`, `addmm`, `convolution`, `add.Tensor`, `mul.Tensor`,
   `relu`, `sigmoid`, `silu`.  Each one calls `vc4_matmul` / `vc4_conv2d` /
   `vc4_elementwise` and translates `VC4_ERR_UNSUPPORTED` into a thrown
   `c10::Error`.  NCHW in, NCHW out (the engine wants row-major fp32).
5. **No fallback.** Do *not* register `CompositeImplicitAutograd` for these ops
   (that would decompose to CPU kernels) and do not install a catch-all
   fallback key.  An op outside the list then fails loudly with
   `NotImplementedError: Could not run 'aten::...' with arguments from the
   'vc4' backend`, which is the behaviour we want while coverage grows.
6. **Inference only.** No autograd kernels: run under `torch.inference_mode()`,
   with weights pre-packed once (the pack kernel costs ~10 % of a matmul).

## Limits worth knowing

- fp32 only, row-major only, `Cin`/`Cout` multiples of 16 for conv.
- conv is stride-1; large feature maps hit the VDR pitch limit (padded plane
  <= 8191 bytes), which the natural next step removes by tiling channels or by
  repacking to NHWC.
- one process per arena; the arena is not shared between processes.
- the library needs `/dev/mem` (root) and `/dev/dma_heap/linux,cma`.
