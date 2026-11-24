# ggml on the VC4 QPUs

`libggml-vc4.so` is a ggml backend for the Raspberry Pi 3B's VideoCore IV
(V3D) QPUs.  It implements ggml's backend ABI (`ggml_backend_i` +
`ggml_backend_buffer_i` + `ggml_backend_device_i` + `ggml_backend_reg_i`) and
lowers each graph node to the `libvc4infer` engine (`infer/include/vc4infer.h`):
hand-written QPU kernels, one dma-buf arena, strict dispatch.

It is a **device backend, not a CPU fallback**: `supports_op` lists exactly the
nodes it can run on the QPUs, so `ggml_backend_sched` partitions everything
else to the CPU backend explicitly.  The fallback list is printed by the test
and recorded below.

What it is **not**: a way to run a large model.  The arena is 64 MiB by default
(the daemon on the reference board serves 96 MiB), fp32 only, and the engine's
GEMM panel for a weight matrix is `N*K*4` bytes.  See §8 for the arithmetic.

---

## 1. What was built, and against which ggml

| | |
|---|---|
| ggml source | `vendor/llama.cpp` inside the `llama-cpp-python 0.3.37` sdist |
| llama.cpp commit | **`a956a852c9c8bc1fb07206364838ddef299178c0`** (2026-11-14, "Fix ctz64 on 32bit (#32273)") |
| ggml version | **1.0.0** (llama.cpp `LLAMA_VERSION 1.0.0-dev`) |
| `GGML_BACKEND_API_VERSION` | **3** |
| toolchain | gcc 14.2.0 (Debian), cmake 3.31.6, Cortex-A53 at 400 MHz (pinned, not changed) |

The sdist is missing `ggml/ggml.pc.in` (it is not in its `MANIFEST`), so cmake
fails at `configure_file`; a minimal upstream-shaped `ggml.pc.in` was written
back into the vendored tree before configuring.  That is the only modification
to the vendored ggml (the build reports commit `a956a85-dirty` because of it).

CPU-only ggml shared library:

```sh
cd ~/scratch/src/llama_cpp_python-0.3.37/vendor/llama.cpp
cmake -S ggml -B ~/scratch/build/ggml-shared \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON \
      -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ~/scratch/build/ggml-shared -j3
# -> libggml.so, libggml-base.so, libggml-cpu.so (no CUDA/Vulkan/Metal/OpenCL)
```

`GGML_NATIVE=ON` selected `-mcpu=cortex-a53+crc+nodotprod+noi8mm+nosve+nosme`
(dotprod/SVE/SME probes all failed, FMA succeeded).  OpenMP is on; the CPU
fallback measured below uses 4 threads.

The backend lives in `ggml/` of this repository:

```sh
cd ~/work/vc4-ggml/ggml
cmake -S . -B build          # finds ../infer/build/libvc4infer.so
cmake --build build -j3
# build/libggml-vc4.so    the GGML_BACKEND_DL plugin (ggml_backend_init + score)
# build/libvc4ggml.so     compiled-in registration for the ctypes harness
# build/test_ggml_vc4     (a) tiny graph, (b) MUL_MAT, (c) llama-family block
# build/test_dl_load      loads libggml-vc4.so through GGML_BACKEND_PATH
```

`libvc4infer.so` is built from `../infer` of this tree (`make -C infer
build/libvc4infer.so`, which also needs `make -C vc4d`).  Everything runs
unprivileged through `vc4d`:

```sh
export VC4INFER_DAEMON=1
export VC4INFER_CACHE=$HOME/scratch/kcache
```

The first use of a shape JIT-generates its kernel blob into `VC4INFER_CACHE`
(python `kgen`, ~0.3–1 s per blob).  That is a one-time cost per cache; the
test's first scheduled run was 274 ms (warm cache) after the MUL_MAT shapes had
been generated.

---

## 2. Registration, both ways

**Dynamic** (`GGML_BACKEND_DL_IMPL`), the way llama.cpp loads an out-of-tree
backend:

```
$ GGML_BACKEND_PATH=$PWD/libggml-vc4.so ./test_dl_load
load_backend: loaded VC4 backend from .../libggml-vc4.so
devices after ggml_backend_load_all():
  [0] CPU      type=0  free=905 MiB total=905 MiB  CPU
  [1] VC4      type=1  free=38 MiB total=64 MiB  VideoCore IV QPUs (V3D, Raspberry Pi 3B) via vc4infer
add graph on VC4: mismatch=0 (n=1024), buffer 12288 B
RESULT: PASS
```

`ggml_backend_score()` returns 40 when `/run/vc4d.sock` (or `/dev/mem`) is
present and 0 otherwise, so ggml skips the library on a board without a VC4.
`ggml_backend_load_all()` resolves `GGML_BACKEND_PATH` and dlopens it; the
plugin links `libggml.so` by soname, so it uses the host's ggml rather than a
private copy.

**Static**: `ggml_vc4_register()` calls `ggml_backend_register()` with the same
registry object.  `libvc4ggml.so` is compiled with the backend linked in (no
`GGML_BACKEND_DL`) and is used by `tests/ctypes_test.py`, which loads it with
`ctypes`, runs all three tests and prints their report:

```
$ VC4INFER_DAEMON=1 python3 tests/ctypes_test.py
... RESULT: PASS   ({"ok": true, ...})
```

Neither path requires a llama.cpp build; the same registry is what a
llama-cpp-python `_llama` extension would register if it linked
`libggml-vc4.a`.

---

## 3. The vtable that was implemented

Compiled against `ggml/src/ggml-backend-impl.h` from the commit above, so the
struct layouts are the tree's, not a guess.  `GGML_BACKEND_API_VERSION` is 3.

`ggml_backend_i` (16 slots, the optional ones `NULL`):

| slot | implementation |
|---|---|
| `get_name` | `"VC4"` |
| `free` | `vc4_shutdown()`, release the arena |
| `graph_compute` | walks `cgraph->nodes`, lowers each op, `GGML_STATUS_FAILED` on anything unsupported |
| `set_tensor_async`, `get_tensor_async`, `set_tensor_2d_async`, `get_tensor_2d_async`, `cpy_tensor_async`, `synchronize`, `graph_plan_*`, `event_record`, `event_wait`, `graph_optimize` | `NULL` (synchronous backend) |

`ggml_backend_buffer_i` (14 slots):

| slot | implementation |
|---|---|
| `free_buffer` | return the region to the pool free list (coalescing) |
| `get_base` | arena mapping + pool offset |
| `memset_tensor` | `dc civac`, `memset`, `dc civac` |
| `set_tensor` | `dc civac`, `memcpy`, `dc civac` |
| `get_tensor` | `dc civac`, `memcpy` |
| `clear` | same over the whole region |
| `init_tensor`, `set/get_tensor_2d`, `cpy_tensor`, `reset`, `set/get_tensor_n`, `cpy_tensor_n` | `NULL` (ggml falls back to per-tensor copies) |

The buffer is one region of the engine's dma-buf arena.  Its mapping is
cacheable, so every host access is bracketed by `dc civac` — exactly the
discipline `vc4_upload`/`vc4_download` use.  `is_host` is deliberately
**false**: the memory is host-addressable, but declaring it a device buffer
routes every access through the calls above and makes the scheduler insert
explicit host↔device copies.

`ggml_backend_buffer_type_i`: `alloc_buffer` (from a first-fit free list over a
reserved pool), `get_alignment` = 64, `get_max_size` = the pool, `is_host` =
false.  `alloc_buffer_n`, `get_alloc_size*` are `NULL`.

`ggml_backend_device_i`: name `VC4`, description, memory (pool/arena),
type `GGML_BACKEND_DEVICE_TYPE_GPU`, `init_backend`, `get_buffer_type`,
`supports_op`, `supports_buft`; no host buffer, no `buffer_from_host_ptr`, no
events, **no `offload_op`** (so the scheduler never forces a node onto VC4).

`ggml_backend_reg_i`: name `VC4`, one device, `get_proc_address` = `NULL`.

GLUE: one arena per process (vc4d hands one dma-buf to a client and `vc4_init`
maps it once), so a second `init_backend` is refused with a clear message.

### Arena layout

```
vc4d arena 96 MiB ── client takes 64 MiB (VC4INFER_ARENA_MB, default 64)
  │  0x000000 .. 0x600000   engine code/uniform slots (VC4_DATA_BASE)
  ├─ tensor pool      min(48 MiB, 60% of arena) = 38 MiB
  │    └─ op scratch   min(8 MiB, pool/4) = 8 MiB   → 30 MiB for ggml buffers
  └─ engine op scratch (GEMM panels, im2col, padding) grows above the pool:
       free = arena − pool = 26 MiB in the default configuration
```

`VC4GGML_ARENA_MB` and `VC4GGML_SCRATCH_MB` override the two sizes.  The
`VC4` device reports `free = pool` (30 MiB), `total = arena` (64 MiB).

---

## 4. Operation table

"Device" = runs entirely on the QPUs.  "Fallback" = `supports_op` returns
false, `ggml_backend_sched` keeps it on the CPU and inserts the copies.  Every
device row is a composition of element-verified engine kernels; no op does
multiply-accumulate on the ARM, and the only host work is byte movement and
integer index arithmetic.

| ggml op | VC4 | lowering | notes |
|---|:--:|---|---|
| `ADD` | ✅ | `vc4_elementwise(ADD)` | broadcast of `src1` is a host byte expansion into arena scratch |
| `SUB` | ✅ | `NEG(src1)`, then `ADD` | the public API has no sub kernel (`VC4_OP_SUB` is enumerated but not routed) |
| `MUL` | ✅ | `vc4_elementwise(MUL)` | broadcast as above |
| `DIV` | ✅ | `RECIPROCAL(src1)`, then `MUL` | |
| `SQR` | ✅ | `vc4_elementwise(MUL, x, x)` | |
| `UNARY ABS/NEG` | ✅ | `vc4_unary(ABS/NEG)` | |
| `UNARY RELU/SIGMOID/SILU` | ✅ | `vc4_relu/sigmoid/silu` | SILU added because llama's FFN uses it and the kernel is verified |
| `UNARY TANH` | ✅ | `vc4_unary(TANH)` | |
| `UNARY GELU` | ✅ | `vc4_unary(GELU)` | tanh approximation, the same body as ggml's table |
| `UNARY GELU_ERF` | ✅ | `vc4_unary(GELU_ERF)` | |
| `MUL_MAT` | ✅ | `vc4_matmul_bt` | fp32, both ggml layouts: contiguous `B [N][K]` and a transposed view of `[K][N]` (the latter runs `vc4_matmul` then `vc4_transpose`) |
| `RMS_NORM` | ✅ | `MUL`, `vc4_matmul` against ones, `AFFINE`, `RSQRT`, lane broadcast, `MUL` | 6 launches; no `vc4_reduce` (refused on this board) |
| `NORM` | ✅ | `vc4_layernorm_fwd` with gamma=1, beta=0 | 1 call; the caller applies the weight |
| `SOFT_MAX` | ✅ | `vc4_softmax` | mask (`src1`), sinks (`src2`) and `max_bias` fall back; `scale` is an `AFFINE` pre-pass; cols ≤ 16 uses the fused kernel |
| `GET_ROWS` | ✅ | one `vc4_copy2d_x` per index | no `vc4_gather` (it wedges the QPU); indices are read as integers, rows are copied by the QPUs |
| `CPY`/`CONT`/`DUP` | ✅ | `vc4_copy2d_x` lane runs | both sides must be contiguous f32 with equal element count |
| `NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE` | ✅ (no-op) | — | metadata |
| `ROPE`, `ROPE_BACK`, `FLASH_ATTN_EXT`, `MUL_MAT_ID`, `GET_ROWS_BACK`, `SET_ROWS`, `SCALE`, `CONCAT`, `SUM`, `SUM_ROWS`, `OUT_PROD`, `IM2COL`, `CONV_*`, `PAD`, `ARGSORT`, `TOP_K`, … | ❌ | CPU | anything not listed above |
| quantised types, f16, bf16, i8 tensors | ❌ | CPU | the engine is fp32-only |

`MUL_MAT` details:

* contiguous `B [N][K]` (the layout `ggml_mul_mat(weight, x)` always has):
  `vc4_matmul_bt(A = activations, B = weight)`.  Needs `ne1 % 16 == 0` and
  `K % 16 == 0`; a token count `M` that is not a multiple of 16 is padded in
  arena scratch (zero rows, copy the real rows back).  `M = 1` decode works
  and computes 16 rows.
* transposed view `B [K][N]` (logical `[K,N]`, `K×N` row-major memory, which
  ggml's own CPU kernel rejects): `vc4_matmul` then `vc4_transpose`, because
  ggml's `dst` is the transpose of the engine's `C`.  Needs 16-aligned
  `M0,N1,K` and `N1 % 64 == 0`.
* The engine's `matmul_bt` builds both k-major panels per call
  (`(M+N)*K*4` bytes of engine scratch), which is why the per-call memory cost
  is the limiter for large weight matrices.

---

## 5. Measured results

Board: Pi 3B, core clock pinned 400 MHz (`vcgencmd measure_clock core` =
400000000 throughout), `throttled=0x0`, fanless.  All runs wrapped in
`tools/thermal-guard.sh --max 74`.  Timing is the median of 7 graph computes
after one warm-up; both backends run the same graph and data.

### (a) tiny graph `relu(add(a,b)*a)`, n = 4096

| | VC4 | ggml CPU (4 threads) |
|---|---:|---:|
| time | 0.829 ms | 0.047 ms |
| mismatches | **0** | — |
| max abs error | **0** (bit-exact) | — |

Three launches of 4096 floats are launch-bound: ~0.28 ms per launch through
`vc4d` dominates.  Start 62.8 °C, peak 62.8 °C.

### (b) `MUL_MAT`, fp32

`VQR` = VC4 GFLOP/s, `CPU` = ggml-cpu GFLOP/s (4× Cortex-A53).

contiguous `B [N][K]` (`ggml_mul_mat(weight, x)`, the llama layout):

| M0,N1,K | mismatch | VC4 ms | CPU ms | VC4 GFLOP/s | CPU GFLOP/s |
|---|---:|---:|---:|---:|---:|
| 64³ | 0 | 0.541 | 0.142 | 0.97 | 3.70 |
| 128³ | 0 | 1.178 | 0.865 | 3.56 | 4.85 |
| 256³ | 0 | **5.706** | **6.586** | **5.88** | **5.10** |
| 48,96,80 (non-square) | 0 | 0.664 | 0.195 | 1.11 | 3.77 |

transposed-view `B [K][N]` (host reference; ggml-cpu cannot run this layout):

| M0,N1,K | mismatch | VC4 ms | VC4 GFLOP/s |
|---|---:|---:|---:|
| 64³ | 0 | 0.477 | 1.10 |
| 128³ | 0 | 1.111 | 3.77 |
| 256³ | 0 | **4.644** | **7.22** |
| 48,128,80 | 0 | 0.535 | 1.84 |

Largest absolute error at 256³: `8.1e-6` (BT) / `7.4e-6` (view); relative error
where `|ref| > 1e-3` is ≤ `2.3e-3`.  The non-square row exists so a swapped
operand role cannot pass: the reference is `dst[i0 + i1*M0] = Σ_k A[k,i0]·B[k,i1]`.

**The honest crossover**: ggml's CPU backend wins below ~200³; the VC4 wins
from ~256³ (1.15× here) and the gap grows with size because the cost is per
launch and per panel, not per element.  This is the same 5.9–7.3 GFLOP/s class
the engine reports elsewhere (`docs/BENCHMARKS.md`), not a new ceiling.

### (c) llama-family block, end to end

A random-weight llama block: `GET_ROWS` embedding → RMS_NORM → Q/K/V →
ROPE → scores → SOFT_MAX → attention value → Wo → residual → RMS_NORM →
gate/up → SILU → elementwise mul → down → residual → RMS_NORM → output →
softmax.  `tok=64, embd=64, ff=128, vocab=256`, **0.28 MiB of weights**, all
weights in the VC4 buffer, run through `ggml_backend_sched` with
`{VC4, CPU}`:

```
nodes VC4=21 CPU=2, 3 splits
on VC4:     GET_ROWS, RMS_NORM, MUL_MAT, RESHAPE, SOFT_MAX, ADD, UNARY
fell back:  ROPE, RESHAPE
accuracy vs CPU reference: mismatch=0 max_abs=8.01e-08 max_rel=1.11e-05
VC4+sched: 298.92 ms first, 28.47 ms best
CPU reference (4 threads): 3.09 ms
```

Start 62.8 °C, peak 62.8 °C, end 62.3 °C.  The first run includes 18 kernel
JITs (cold `kcache` in effect for the new shapes); with the cache warm the
first run was ~300 ms in earlier trials.

Why the VC4 loses on a 0.28 MiB block: 21 nodes become ~60 launches
(GET_ROWS alone is one launch per token; RMS_NORM is 6; the generic softmax is
4).  At ~0.12–0.3 ms per launch through `vc4d` that is the whole 28 ms.  A
decode step (`tok=1`) is the same number of launches for 64× less work; a real
model must amortise them over larger matrices, which is exactly what the §8
memory ceiling prevents on this board.  This is reported as measured, not as a
recommendation.

No real LLM was run: see §8 for why one cannot fit, and what would be needed.

---

## 6. What falls back, and how to see it

The test iterates `cgraph->nodes` after `ggml_backend_sched_graph_compute()` and
asks `ggml_backend_sched_get_tensor_backend()` for each node; it prints the
distinct op names on each side.  For the block above: `ROPE` (not implemented)
and the `RESHAPE` the scheduler placed next to it on the CPU (ggml keeps a
`RESHAPE` on the backend of the tensor it reshapes in this split).

For a real llama.cpp graph the expected fallbacks are `ROPE` (all variants),
`FLASH_ATTN_EXT` (llama.cpp chooses the classic attention path when the device
does not support it), `MUL_MAT_ID` (MoE), `SET_ROWS`/`GET_ROWS_BACK`,
`SCALE`, `CONCAT`, and any `CPY` with a permuted destination (KV-cache writes
are views with `nb[1] > row bytes`).  Those are CPU nodes; ggml inserts the
host↔device copies, which on this backend go through `set_tensor`/`get_tensor`
(`dc civac` + `memcpy`).

`VC4GGML_TRACE=1` prints one line per node with its time:

```
vc4: [  4] MUL_MAT        q                         0.712 ms
vc4: [  5] ROPE           q_rope                    0.031 ms   (CPU split)
```

---

## 7. Why some engine ops are avoided

Three engine entry points are not used deliberately, because they are refused
or wrong on this silicon (see `docs/OPS-REDUCE.md`, `infer/python/vc4infer/`):

* `vc4_reduce` — refused (`VC4_ERR_UNSUPPORTED`, "walks a fixed 256-output
  group … seeds r4 instead of r0").  Row sums (RMS_NORM) are a GEMM against a
  constant ones matrix instead: `vc4_matmul(x, ones[cols][16])`, the first
  column of each row is the sum.
* `vc4_gather` — wedges the QPU.  `GET_ROWS` copies each selected row with
  `vc4_copy2d_x` (one launch per index).  This is why `GET_ROWS` is the most
  launch-hungry node in the llama block.
* `vc4_cumsum`, `vc4_select3` — not used (wedge / wrong values).

`vc4_matmul`'s internal host padding path is also avoided: it copies with the
ARM without invalidating the cacheable mapping, and a QPU-written line can be
stale (measured in the ONNX runner).  The backend pads in its own scratch after
an explicit `civac`, or requires shapes that need no padding.

---

## 8. The limit: model size vs arena

This is the part that decides what can run, so it is stated as arithmetic
rather than as a demo.

* Client arena: 64 MiB by default (`VC4INFER_ARENA_MB`); the reference daemon
  is started with `--arena-mb 96`.  The backend reserves
  `min(48 MiB, 60% of arena)` and keeps 8 MiB of that as op scratch, so the
  **ggml tensor pool is 30 MiB** and the engine has 26 MiB of its own scratch
  for GEMM panels.
* Weights are fp32.  A 1 M-parameter model is 4 MiB, 4 M is 16 MiB, 7.5 M is
  the whole pool with no room for activations.  A 3 B-parameter model is
  **12 GiB** of fp32 weights — 400× the pool.  Even an int8 3 B model (3 GiB)
  is 100× the pool, and this backend has no quantised types at all.
* Per-MUL_MAT the engine needs `(N + M_pad) * K * 4` bytes of panel plus the
  padded C tile.  For a 512×512 weight that is 1 MiB; 1024×1024 is 4 MiB;
  4096×4096 is 64 MiB, which alone exceeds the 26 MiB engine scratch.  In
  other words the largest single weight matrix this board can multiply is
  roughly 1024×1024, and that leaves nothing else.
* Practical ceiling with this backend: **a few million fp32 parameters**
  (roughly ≤ 4 M with activations), i.e. a toy/embedding-scale transformer or
  a random-weight graph for validating the lowering — exactly what §5(c) is.
  A real 7 B/3 B/1 B LLM does not fit, and this document does not claim one
  was run.

What would change the ceiling: quantised ggml types (q4_0/q8_0) with an engine
dequant or a QPU dot kernel (the kernels are fp32-only today), keeping weights
on the host and streaming tiles (the daemon's per-launch cost would dominate),
and a bigger CMA/arena.  None of that is implemented here.

---

## 9. Reproduction

```sh
# ggml (once)
cd ~/scratch/src/llama_cpp_python-0.3.37/vendor/llama.cpp
cmake -S ggml -B ~/scratch/build/ggml-shared -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON \
      -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ~/scratch/build/ggml-shared -j3

# engine + backend (from this repository)
make -C vc4d -j3
make -C infer build/libvc4infer.so -j3
cmake -S ggml -B ggml/build && cmake --build ggml/build -j3

# tests (unprivileged through vc4d; one GPU user at a time)
export VC4INFER_DAEMON=1 VC4INFER_CACHE=$HOME/scratch/kcache
cd ggml/build
./test_ggml_vc4                 # (a) + (b) + (c)
./test_ggml_vc4 matmul          # just the matrix sweep
GGML_BACKEND_PATH=$PWD/libggml-vc4.so ./test_dl_load
python3 ../tests/ctypes_test.py
```

The engine's own suite is unchanged and still passes with this backend in the
tree: `cd infer && REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh`.
