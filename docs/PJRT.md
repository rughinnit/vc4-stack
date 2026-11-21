# JAX on the VideoCore IV QPUs: the PJRT plugin

`libpjrt_vc4.so` is a PJRT C API plugin.  jaxlib loads it, `jax.devices()`
lists a `vc4` device, and `jax.jit` / `jax.grad` programs run their fp32
primitives on the QPUs through `libvc4infer`.  The plugin is in `pjrt/`.

    pjrt/
      src/vc4_pjrt.c          PJRT ABI, client/device/buffer/executable, executor
      src/stablehlo.c         MLIR bytecode reader and VHLO lowering
      src/stablehlo.h         lowered instruction/table format
      third_party/pjrt_c_api.h  vendored XLA header (provenance below)
      python/jax_plugins/vc4/__init__.py   jax_plugins registration
      tools/pjrt_decode.c     dumps a lowered program (development aid)
      tools/decode_mlir.py    Python model of the reader (development aid)
      tests/test_pjrt.py      device, subset, gradient and refusal checks

## 1. Build, load, register

The lowering uses `vc4_gather`, `vc4_reduce`, `vc4_select3`, `vc4_matmul` and
`vc4_conv2d`; build the engine from this checkout first (the installed
`vc4infer` 1.2.8 predates the commit that ungated the first three).

    make -C infer
    make -C pjrt      # links infer/build when it exists, else the installed lib
    make -C pjrt decode        # pjrt_decode: reader and lowering, no engine
    make -C pjrt test-lower PYTHON=~/scratch/jaxvenv/bin/python3
    # either registration path works:
    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4 python3 -c 'import jax; print(jax.devices())'
    PJRT_NAMES_AND_LIBRARY_PATHS=vc4:$PWD/pjrt/build/libpjrt_vc4.so \
        JAX_PLATFORMS=vc4 python3 -c 'import jax; print(jax.devices())'

Both print `[Vc4Device(id=0)]`.  `GetPjrtApi()` returns the `PJRT_Api` table;
the plugin sets `pjrt_api_version` to **0.118** and implements the entries a
single-device client needs (client, topology, device and memory descriptions,
events, buffer transfer, compile/load/execute, executable metadata).

The header is vendored because the jaxlib 0.12.1 wheel ships only
`xla/ffi/api`, not `xla/pjrt/c/pjrt_c_api.h`.  It is the unmodified Apache-2.0
header from XLA commit `97b8433b3ed82b78b742f751d2c7f510e5d78d25`, the commit
pinned by jax 0.12.1's `MODULE.bazel`; sha256
`8c54c8e030ea88fcba627dd83286e09edbc65aaa34f7aced36da51a48fe83328`.
Struct layouts and the section/attribute encodings were taken from that
header and from MLIR's `BytecodeReader`/`BuiltinDialectBytecode.td` and
StableHLO's `VhloBytecode.cpp`; nothing was copied from a plugin tutorial.

## 2. What jaxlib actually sends

`PJRT_Client_Compile` is called with `format = "mlir"` and a **StableHLO
portable artifact**: MLIR bytecode version 6 using the `vhlo` (versioned
StableHLO) dialect, producer `StableHLO_v1.19.0`.  This is not text.  There is
no MLIR library on the board, so `stablehlo.c` implements the encodings that
the emitter produces:

* the nine top-level sections, the string table, the dialect/op-name tables,
  and the attribute/type offset table;
* the builtin attribute and type encodings used here (Integer, Float, String,
  Array, Dictionary, TypeAttr, DenseArray, DenseTypedElements; Integer,
  Float, RankedTensor, Tuple, Function types);
* the VHLO attribute and type encodings (ArrayV1, BooleanV1, DictionaryV1,
  FloatV1, IntegerV1, StringV1, TensorV1, TypeV1; RankedTensorV1, FunctionV1,
  TupleV1, scalar types);
* the operation mask, properties, results, operands and nested regions,
  including isolated regions encoded as inline IR sections;
* operation properties in ODS order (alphabetical for the VHLO ops used).

Unregistered `sdy.mesh` sharding annotations (added when more than one JAX
backend is registered) are consumed and ignored: they do not change the
value, and every op that carries them still arrives as a VHLO op.

## 3. Execution model

One `PJRT_Buffer` is one 64-byte aligned host allocation; its contents are
the device value.  On the first compile the plugin initialises a `vc4_ctx`
(through `vc4d` when `/run/vc4d.sock` exists), decodes the module, and lowers
it to a linear instruction list over **one arena slab per executable**:

* every SSA value gets a fixed 64-byte-aligned slot;
* constants and layout tables are built and uploaded once, at compile time;
* reshape and f32→f32 convert are aliases (the storage is already dense
  row-major);
* `Execute` uploads the arguments into their slots, runs the instruction
  list, downloads the result slots, and returns fresh buffers.  No value
  changes representation between kernels.

Layout is a QPU gather: `broadcast_in_dim`, `transpose`, `slice`, `pad`,
`reverse`, and the operand permutations a `dot_general` or a non-trailing
`reduce` needs become `vc4_gather` with an int32 table of arena byte offsets
computed at compile time.  `vc4_gather` is one launch per op, including its
padded tail (the kernel pads into scratch itself).  A `pad` puts its padding
value in the gather's fill slot, which is the one runtime value a table can
carry; `gather_v1` tables hold the values of a static index tensor.

Integer values never reach a kernel.  An expression whose leaves are all
`constant_v1` -- the index arithmetic JAX emits around a gather, a bounds
mask, an `iota` ramp -- is folded at compile time and used to build the
tables.  The fold reads no argument and no kernel result, so it is not host
arithmetic on device data.

Three instruction kinds drive more than one kernel launch: `VC4SL_DOT_N` runs
one `vc4_matmul` per batch element, `VC4SL_CONV` runs one `vc4_conv2d` per
image (im2col gather, weight pack, GEMM), and `VC4SL_REDUCE` carries the
reduction op (`SUM`, `MAX`, `MIN`) instead of hard-coding a sum.

## 4. Op subset

Everything else returns `UNIMPLEMENTED` with the operation name in the
message.  There is no CPU arithmetic fallback anywhere.

| VHLO op | lowering | launches |
|---|---|---|
| `add_v1`, `multiply_v1` | `vc4_elementwise(ADD/MUL)` | 1 |
| `subtract_v1` | `vc4_unary(NEG)` + `vc4_elementwise(ADD)` | 2 |
| `negate_v1` | `vc4_unary(NEG)` | 1 |
| `maximum_v1`, `minimum_v1` | `vc4_elementwise(CMP_GE/CMP_LE)` + `vc4_select3(WHERE)` | 2 |
| `relu_v1` (accepted; JAX emits `maximum` instead) | `vc4_relu` | 1 |
| `compare_v1` (GT/GE/LT/LE/EQ) | `vc4_elementwise(CMP_*)` | 1 |
| `select_v1` | `vc4_select3(WHERE)` | 1 |
| `dot_general_v1/v2/v3`, any rank, any number of batching and contracting dimensions | layout gathers to `[batch][M][K]` and `[batch][K][N]` where the operands are not already in that order, then one `vc4_matmul` per batch element | 2 per batch element, +1 per permuted operand |
| `reduce_v1` with `add`, `maximum` or `minimum`, any axes | layout gather to put the reduced axes last (skipped when they already are), then `vc4_reduce(SUM/MAX/MIN)` | 1, +1 when the axes are not trailing |
| `broadcast_in_dim_v1` | compile-time index table, `vc4_gather` | 1 |
| `transpose_v1` | compile-time permutation table, `vc4_gather` | 1 |
| `slice_v1`, any rank, any axes, static strides | start/limit/stride table, `vc4_gather` | 1 |
| `pad_v1`, edge and interior padding | table with a fill entry per padding slot, `vc4_gather` | 1 |
| `reverse_v1` (what a negative-step slice becomes) | table, `vc4_gather` | 1 |
| `gather_v1/v2` with a static index tensor, no operand batching dimensions | table, `vc4_gather` | 1 |
| `iota_v1` | integer ramp materialised at compile time and uploaded | 0 |
| `convolution_v1`, NCHW/OIHW/NCHW, 3x3 or 1x1, stride 1, symmetric padding, `Cin`/`Cout` a multiple of 16 | `vc4_conv2d` per image | 3 per image (im2col, weight pack, GEMM) |
| `reshape_v1` | alias | 0 |
| `convert_v1` f32→f32, and a constant integer converted to f32 | alias / compile-time constant | 0 |
| `constant_v1` | uploaded once at compile | 0 |
| `call_v1`, `return_v1`, `func_v1`, builtin `module` | inlined / structural | — |

Twenty-five VHLO operation names are accepted (the rows above, counting
`func_v1`, `call_v1` and `return_v1`, and not counting `relu_v1`, which no
producer emits).  The version suffix is not part of the match: `gather_v1`
and `gather_v2` both lower, and their property lists differ only in the
batching-dimension fields.

Refused with a named op and the offending attribute or shape: `while_v1` and
`if_v1`, `divide_v1` (so `mean` is `reduce` + `divide`, and the divide is
refused), a `reduce` whose body is not `add`/`maximum`/`minimum` (for example
`multiply_v1` for `prod`), a `reduce` whose initial value is not that op's
identity, a `gather` whose index operand is not a compile-time constant or
that carries operand batching dimensions, a `pad` whose padding value is not
a compile-time constant, a `slice` with bounds outside its axis or a
non-positive stride, a `convolution` with stride other than 1, dilation,
groups, window reversal, a non-NCHW dimension order, a non-square or larger
than 3x3 kernel, asymmetric padding, a padded image beyond the 8191-byte VDR
pitch, or channels that are not a multiple of 16, any non-fp32 compute, and
any predicate or integer result returned to the host.  Each message states
the value that was refused, for example

    convolution_v1: 3 input channels are not a multiple of 16
    convolution_v1: window_strides [2,2] are outside the executed subset ([1,1])
    reduce_v1: the reduction body multiply_v1 is outside the executed subset
        (add, maximum, minimum)
    gather_v2: the indices are not a compile-time constant, so there is no
        static gather window



## 5. Measurements

Board: Raspberry Pi 3B, stock clocks (core 400 MHz, `throttled=0x0` for this
run; `0x20000` is a latched soft-limit flag), fanless, run under
`tools/thermal-guard.sh --max 71`, start 64.5 °C, peak 68.2 °C.  jax/jaxlib
0.12.1, cp313; engine built from this checkout (`libvc4infer`), through
`vc4d`.

The test suite (`pjrt/tests/test_pjrt.py`) runs every case on the vc4 device
and on JAX's CPU backend in one process and compares element by element.
Tolerance is the project's 5e-4 (`atol = rtol`); "bit-neq" counts values that
are not bit-identical (the GEMM's summation order differs).

| case | max abs diff | over tol | bit-neq |
|---|---:|---:|---:|
| add | 1.19e-07 | 0 / 6 | 3 |
| multiply | 0 | 0 / 6 | 0 |
| subtract | 5.96e-08 | 0 / 6 | 1 |
| negate | 0 | 0 / 6 | 0 |
| relu | 0 | 0 / 6 | 0 |
| maximum | 0 | 0 / 6 | 0 |
| dot (2×3 · 3×4) | 5.96e-08 | 0 / 8 | 5 |
| dot, lhs contracting dim 0 | 5.96e-08 | 0 / 8 | 5 |
| dot, rhs contracting dim 1 | 1.19e-07 | 0 / 8 | 5 |
| reshape | 0 | 0 / 6 | 0 |
| broadcast_in_dim (2×3×1 → 2×3×4) | 0 | 0 / 24 | 0 |
| sum (all axes) | 1.19e-07 | 0 / 1 | 1 |
| sum (axis 1) | 1.19e-07 | 0 / 2 | 2 |
| mlp_forward | 4.77e-07 | 0 / 1 | 1 |
| **grad_x** | 4.77e-07 | 0 / 6 | 6 |
| **grad_w** | 0 | 0 / 12 | 0 |

That table predates the ops added below: `iota` and `slice` are accepted now,
and the refusal checks that used them were replaced by value comparisons that
have not run on silicon yet.

Per call, warm, median of 20, including Python and transfer overhead:

| program | wall | QPU kernels | launches |
|---|---:|---:|---:|
| `sum(relu(x @ w))`, forward | 2.56 ms | — | 6 |
| `jax.grad(...)` wrt x | 3.11 ms | 0.68 ms | 11 |
| same forward on JAX CPU | 1.38 ms | — | — |
| same gradient on JAX CPU | 1.36 ms | — | — |

At these shapes the QPU path is slower than the CPU: a handful of launches,
the upload/download of every parameter and result, and the compile-time
tables dominate 18 MACs.  The crossover is a matter of size, not of the
plugin; the engine's own numbers (a 256³ GEMM at 8.5 GFLOP/s, YOLOv8n at
2 FPS) are the scale at which the QPUs win.

### 5.1 What the lowering checks measure

`pjrt/tests/test_lower.py` needs no QPU.  It exports each program with
`jax.export` -- the same portable artifact the compile path receives -- and
runs `pjrt/build/pjrt_decode` on it (source in `pjrt/tools`), which links the
reader and the lowering and nothing else.  Thirty-three checks, all passing on the current tree:

* the instruction list of every new op, so a case that starts or stops
  lowering is visible;
* the reduction geometry (`outer`, `red`, the reduction op) and the GEMM and
  convolution geometry (M, N, K, batches; Cin, H, W, Cout, kernel, padding,
  images);
* the exact source element of every output element for `slice`, `reverse`,
  `pad` and `gather`, computed in numpy by applying the same operation to an
  `arange` tensor, and for `pad` the fill entries as -1.

That is a check of the lowering, not of the kernels: it proves the tables and
the geometry say what they claim, and says nothing about the values the QPUs
produce.  Those need the serialized pass below.

### 5.2 The serialized GPU pass (not yet run)

The board was busy with another run while this change was made, so no kernel
was launched.  The pass to run, on an idle board:

    cd ~/work/vc4-pjrt2
    make -C infer -j2                    # gather/reduce/select3/conv2d kernels
    make -C pjrt
    TMPDIR=$HOME/scratch/tmp PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4,cpu \
      VC4PJRT_LOG=$HOME/scratch/logs/pjrt-ops.log \
      tools/thermal-guard.sh --max 71 -- \
      ~/scratch/jaxvenv/bin/python3 pjrt/tests/test_pjrt.py

`test_pjrt.py` runs every case on both backends, compares element by element
at the project tolerance (5e-4) and prints the launches the plugin logged.
Expected per case: `over=0` of the element count, and these instruction and
launch counts, which come from the lowered program (exact) and from the
launch table above (predicted):

| case | instructions | launches |
|---|---|---:|
| `slice`, `slice_step`, `slice_rank3` | gather | 1 |
| `reverse` (negative step) | gather | 1 |
| `pad`, `pad_interior` | gather with fill | 1 |
| `iota` | none | 0 |
| `gather_axis0`, `gather_2d` | gather | 1 |
| `take` | gather, gather, select | 3 |
| `sum_leading`, `max_leading` | gather, reduce | 2 |
| `min_trailing`, `max_last` | reduce | 1 |
| `sum_nontrailing` | gather, reduce | 2 |
| `dot_batched` (2 x 3x4 · 4x3) | gather, batched GEMM | 5 |
| `dot_rank3` (2x3x4 · 4x4) | GEMM | 2 |
| `conv_3x3` (1x16x5x5, 16 filters) | conv | 3 |
| `conv_1x1` (1x16x5x5, 16 filters) | conv | 3 |

The refusals are checked in the same run and must print `refused (names op)`:
`reduce_product`, `divide`, `conv_channels`, `conv_stride`, `while_loop`.

Bit-exactness is not expected from the GEMM, convolution or reduction paths:
their summation order differs from XLA:CPU's.  A pass is `ALL PJRT TESTS
PASSED` with every case at `over=0`.

## 6. Does the backward pass stay on the device?

Yes for this function.  `jax.grad(lambda x: sum(relu(x @ w)))` compiles to
`dot_general`, `broadcast_in_dim`, `compare GT`, `select`, a transposed
`dot_general`, `reduce add`, and a scalar `add`.  Every one of those has a
lowering in the table above; the backward pass runs as 9 lowered
instructions / 11 QPU launches with no CPU arithmetic, and `grad_x`/`grad_w`
match the CPU backend (0/6 and 0/12 over tolerance).  That is the property
JAX buys: the gradient graph is just more primitives, so the plugin needed no
derivative code.

What would break is a backward pass that emits a primitive outside the
subset.  `iota`, `slice`, convolutions and `while` loops do; they are refused
by name rather than silently computed on the ARM.

## 7. Why a full JAX device is a compiler backend

PJRT looks like a device plugin but its compile entry point is a *compiler*
entry point.  jaxlib hands over a serialised StableHLO module and expects an
executable; it does not hand over a list of calls.  Supporting JAX generally
therefore means:

* a StableHLO reader (this plugin has the bytecode subset JAX emits for small
  programs; the general format is larger);
* layout assignment for every op;
* fusion and memory planning, or an interpreter with the dispatch cost seen
  above;
* implementations for conv, gather/scatter, sort, RNG, reductions with
  arbitrary axes, integer and predicate dtypes, control flow, and the whole
  `stablehlo` op set;
* a real allocator: this plugin bump-allocates one slab per executable in the
  engine arena and never frees it before client shutdown.

The subset above is an executor, not a backend.  It is enough to demonstrate
that `jit`, `grad` and device transfers work, and it gives a measured baseline
for the launch and transfer costs a real backend would have to hide.

## 8. Reproduction

    make -C infer
    make -C pjrt
    make -C pjrt test-lower PYTHON=~/scratch/jaxvenv/bin/python3   # no QPU
    cd /home/raspi/work/vc4-pjrt
    export TMPDIR=$HOME/scratch/tmp
    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4,cpu \
      VC4PJRT_LOG=$HOME/scratch/logs/pjrt.log \
      tools/thermal-guard.sh --max 71 -- \
      ~/scratch/jaxvenv/bin/python3 pjrt/tests/test_pjrt.py

`VC4PJRT_LOG` records each compile (`format`, size) and each execution
(instructions, launches, QPU kernel nanoseconds).  `VC4PJRT_DUMP_DIR` writes
every module jaxlib sends, which is how the bytecode reader was developed;
`pjrt/tools/pjrt_decode` and `pjrt/tools/decode_mlir.py` print the lowered
program and the bytecode model respectively.

The engine suite is independent of the plugin and does not build or load it:
`cd infer && REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh`.
