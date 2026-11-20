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

The lowering uses `vc4_gather`, `vc4_reduce` and `vc4_select3`; build the
engine from this checkout first (the installed `vc4infer` 1.2.8 predates the
commit that ungated them).

    make -C infer
    make -C pjrt      # links infer/build when it exists, else the installed lib
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

Layout is a QPU gather: `broadcast_in_dim`, `transpose`, and the operand
transpose a `dot_general` needs become `vc4_gather` with an int32 table of
arena byte offsets computed at compile time.  `vc4_gather` is one launch per
op, including its padded tail (the kernel pads into scratch itself).

## 4. Op subset

Everything else returns `UNIMPLEMENTED` with the operation name in the
message.  There is no CPU arithmetic fallback anywhere.

| VHLO op | lowering | launches |
|---|---|---|
| `add_v1`, `multiply_v1` | `vc4_elementwise(ADD/MUL)` | 1 |
| `subtract_v1` | `vc4_unary(NEG)` + `vc4_elementwise(ADD)` | 2 |
| `negate_v1` | `vc4_unary(NEG)` | 1 |
| `maximum_v1` | `vc4_elementwise(CMP_GE)` + `vc4_select3(WHERE)` | 2 |
| `relu_v1` (accepted; JAX emits `maximum` instead) | `vc4_relu` | 1 |
| `compare_v1` (GT/GE/LT/LE/EQ) | `vc4_elementwise(CMP_*)` | 1 |
| `select_v1` | `vc4_select3(WHERE)` | 1 |
| `dot_general_v1/v2`, rank 2, no batching, one contracting dim each | `vc4_matmul`; a contracting dim 0 on either side becomes a `vc4_gather`, then `vc4_matmul` | 2 (pack + GEMM), +1 if transposed |
| `reduce_v1` with `add`, trailing contiguous axes, zero init | `vc4_reduce(SUM)` | 1 |
| `broadcast_in_dim_v1` | compile-time index table, `vc4_gather` | 1 |
| `transpose_v1` | compile-time permutation table, `vc4_gather` | 1 |
| `reshape_v1` | alias | 0 |
| `convert_v1` f32→f32 | alias | 0 |
| `constant_v1` | uploaded once at compile | 0 |
| `call_v1`, `return_v1`, `func_v1`, builtin `module` | inlined / structural | — |

Eighteen VHLO operation names are accepted (the rows above, counting
`func_v1`, `call_v1` and `return_v1`, and not counting `relu_v1`, which no
producer emits).

Refused with a named op, by design: `iota_v1`, `slice_v1`, `gather_v1`,
`convolution_v1`, `while_v1`/`if_v1`, any `dot_general` with batching
dimensions or rank > 2, any `reduce` that is not a trailing contiguous sum
with a zero initial value, any non-fp32 compute, and any predicate result
returned to the host.

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

`iota` and `slice` are refused and the error names the op.

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
    cd /home/raspi/work/vc4-pjrt
    export TMPDIR=$HOME/scratch/tmp
    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4,cpu \
      VC4PJRT_LOG=$HOME/scratch/logs/pjrt.log \
      tools/thermal-guard.sh --max 71 -- python3 pjrt/tests/test_pjrt.py

`VC4PJRT_LOG` records each compile (`format`, size) and each execution
(instructions, launches, QPU kernel nanoseconds).  `VC4PJRT_DUMP_DIR` writes
every module jaxlib sends, which is how the bytecode reader was developed;
`pjrt/tools/pjrt_decode` and `pjrt/tools/decode_mlir.py` print the lowered
program and the bytecode model respectively.

The engine suite is independent of the plugin and does not build or load it:
`cd infer && REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh`.
