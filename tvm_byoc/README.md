# TVM Relax on the VC4 QPUs — the BYOC vertical slice

`docs/FRAMEWORKS.md` §3 row 6 lists Apache TVM as the widest front-end door
(TF, Keras, MXNet and PyTorch through Relax importers) and the largest single
piece of work, because a BYOC backend is a compiler backend.  This directory
is the first vertical slice of that backend: one composite is partitioned,
lowered by a code generator, and executed by a runtime module that drives
`libvc4infer`.  It is deliberately narrow; the limits section states exactly
what is not there.

The 0.27 wheel has no `tvm/relay` package, so `relay.ext.<name>`,
`relay.op.get` and Relay's BYOC pipeline do not exist on this board.  The path
below is the Relax one:

```
Relax module ──FuseOpsByPattern──▶ composite function (Codegen="vc4")
                                        │
                            relax.ext.vc4 (codegen.py)
                                        │  JSON descriptor + module call
                                        ▼
                    libvc4_tvm.so: runtime.VC4JSONRuntimeCreate
                                        │
                    record │ host │ vc4  (runtime_vc4.cc)
                                        ▼
                      vc4_matmul → vc4_elementwise(ADD) on the QPU
```

| piece | file | what it does |
|---|---|---|
| pattern table | `vc4/patterns.py` | `add(matmul(x, w), const)` → `vc4.matmul_bias_add`, with a check on dtype, static shapes and the bias form |
| codegen | `vc4/codegen.py` | `relax.ext.vc4`: grouped function → descriptor + module; `build_for_vc4()` = partition + `RunCodegen` |
| runtime module | `vc4/runtime_vc4.cc` | the JSON runtime; `record`, `host` and `vc4` modes |
| loader | `vc4/runtime.py` | dlopens the module library and calls `runtime.VC4JSONRuntimeCreate` |
| build | `vc4/build.sh` | compiles `libvc4_tvm.so` against the installed TVM and, when present, `libvc4infer` |
| CPU checks | `tests/test_byoc.py` | pattern, descriptor, dispatch and numerics without touching the QPU |
| silicon pass | `tests/silicon_check.py` | the same graph in `vc4` mode, only with `--vc4` |

## The claimed composite

```
C = matmul(A, B) + bias        A: [M][K]   B: [K][N]   bias: (), (N,), (1, N) or (M, N)
```

Both engine ops exist and are element-verified in the stack (`docs/KERNELS.md`,
`docs/BENCHMARKS.md` §2): `vc4_matmul` (pads M/N/K internally, so M=4 works)
and `vc4_elementwise` with `VC4_OP_ADD`.  The check therefore rejects only what
the engine cannot take:

* anything but fp32 (the QPU FPU is fp32; there is no fp16 path);
* symbolic M, K or N (the descriptor, and the engine plan, are static);
* a bias that is a runtime parameter or a shape the engine cannot fold into a
  same-shape add, and a bias larger than `MAX_EMBED_ELEMS = 4096`.

Rejected groups are simply not claimed.  Under `FuseOpsByPattern` they stay in
the host graph and run in Relax, which is the normal BYOC fallback; the codegen
never sees them, so it cannot silently mis-lower them.

`FuseOpsByPattern(..., annotate_codegen=True)` is used without
`MergeCompositeFunctions`: the latter wraps a whole *chain* of composites in one
`Codegen` function, which needs a module that models the data flow between the
groups.  With `annotate_codegen=True` every composite is its own module call,
which is what the engine executes here (two adjacent `matmul + bias` layers
become two modules and two launches sets).

## The descriptor

The codegen emits one JSON document per composite; `VC4_TVM_DUMP_DIR` writes it
to disk.  For `M=4, K=16, N=8` and a row bias it is:

```json
{
  "version": 1, "backend": "vc4",
  "symbol": "fused_relax_matmul_relax_add1_vc4", "mode": "record",
  "inputs": [{"name": "x", "role": "x", "shape": [4, 16], "dtype": "float32"},
             {"name": "w", "role": "w", "shape": [16, 8], "dtype": "float32"}],
  "outputs": [{"name": "out", "shape": [4, 8], "dtype": "float32"}],
  "nodes": [{
    "op": "vc4.fused_matmul_add", "name": "...matmul_bias_add",
    "inputs": ["x", "w"],
    "attrs": {"M": 4, "N": 8, "K": 16, "bias_mode": 2, "bias_shape": [8],
              "bias_data": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
              "pattern": "vc4.matmul_bias_add", "dtype": "float32"}
  }]
}
```

`bias_mode` is `1` scalar, `2` row (`bias[n]`), `3` full (`bias[m][n]`); the
values are the folded constant, so the module has no constant inputs and the
bias is an operand of the compiled artifact.  The descriptor is the whole
contract: symbol, call signature, and per node the shapes and the constant.

The module is an `ffi::ModuleObj` with kind `vc4_json`, not a subclass of the
TVM-internal `JSONRuntimeBase`: the wheel ships
`src/runtime/extra/contrib/json/json_runtime.h` but not the headers it includes
(`tvm/support/io.h`, `support/bytes_io.h`), so that base class cannot be
compiled against an installed wheel.  The runtime keeps the same model (a
symbol, a tensor list, a node list) and implements `SaveToBytes` /
`ffi.Module.load_from_bytes.vc4_json` for serialisation.

## Runtime modes

`VC4_TVM_MODE` overrides the descriptor's `mode`, so an artifact compiled once
can be checked on the CPU and then executed on the board.

| mode | what it does | touches the device |
|---|---|---|
| `record` | logs the call (shapes, byte counts, FNV-1a of the operands) and fills the output with NaN | no |
| `host` | same log, then computes the fused op on the ARM as a plumbing reference | no |
| `vc4` | `vc4_init` once, per-node arena buffers, `vc4_upload`, `vc4_matmul`, `vc4_elementwise(ADD)`, `vc4_download`, `vc4_get_stats` | yes |

`record` is the descriptor default on purpose: a module that was built for a
CPU-side check cannot return plausible numbers by accident.  `host` exists to
prove the partition, the descriptor and the call signature are correct on the
CPU; `vc4` is the only mode that opens the engine.  The log is one JSON object
per line on `$VC4_TVM_LOG` (stderr by default).

The `vc4` path maps one node to two engine launches: the GEMM writes an
intermediate tile and the elementwise ADD folds the bias.  The bias is expanded
to `[M][N]` once, at first use, and then lives in the arena; the per-node input
and output buffers are allocated once per module instance.  Inputs and outputs
are host tensors and `vc4_upload`/`vc4_download` is the only data movement, so
no value changes on the ARM.

## Build

```
cd ~/work/vc4-tvm
export TMPDIR=$HOME/scratch/tmp
TVM_PYTHON=/home/raspi/scratch/tmp/fw/tvm/bin/python tvm_byoc/vc4/build.sh
```

The script locates the TVM headers and `libtvm_runtime.so` from the Python that
imports `tvm` (`TVM_PYTHON`, default `python3`), adds `-DVC4_TVM_ENGINE` when
`/usr/include/vc4infer.h` is installed, and writes `tvm_byoc/build/libvc4_tvm.so`.
`--no-engine` builds the record/host modes only; `VC4_TVM_LIB` points the loader
at a different artifact.

## CPU checks (no QPU)

```
export TMPDIR=$HOME/scratch/tmp
/home/raspi/scratch/tmp/fw/tvm/bin/python tvm_byoc/tests/test_byoc.py
```

Nine checks, all on the CPU: the partition and its refusals, the descriptor
text, `host` numerics against both Relax-on-llvm and numpy, `record` leaving
NaN, the three bias forms, two adjacent layers producing two module calls, and
`readelf`/`nm` confirming the engine symbols are linked into the module.

Measured on this board (2026-11-21, `tvm 0.27.0.post1`, no device use):

```
9/9 checks passed
max|host - numpy|        9.5e-07   (4x16x8, row bias)
max|host - relax(llvm)|  9.5e-07
```

## Silicon pass (the serialized run)

The board runs one GPU client at a time.  Check that nothing else is running
(`ps -eo cmd | grep -E 'vc4|onnx'`, `systemctl is-active vc4d`) and then:

```
cd ~/work/vc4-tvm
export TMPDIR=$HOME/scratch/tmp
export VC4_TVM_LOG=$HOME/scratch/logs/tvm-byoc-silicon.log
/home/raspi/scratch/tmp/fw/tvm/bin/python tvm_byoc/tests/silicon_check.py --vc4
```

Expected for `--m 4 --k 16 --n 8` (the default):

```
shape M=4 K=16 N=8, bias 8 row floats
calls recorded: 1  engine reports: 1
  call mode=vc4 M=4 N=8 K=16 x_fnv1a=<same as the host pass> w_fnv1a=<same>
  engine launches=3 flops=1056
max|vc4 - relax cpu| < 1e-4
max|vc4 - numpy|     < 1e-4
NaN in output: False
PASS
```

`launches=3` is the packer, the GEMM and the ADD; `flops=1056` is
`2·M·N·K + M·N`.  The script refuses to start without `--vc4`, and it exits 2
when `VC4_TVM_MODE` would override the descriptor.

## What still blocks a fused graph on the QPU

Nothing in this slice needs another kernel: the two ops exist, the module
compiles and links against `libvc4infer.so.1`, and the CPU path executes the
call through the same descriptor that the device path consumes.  The remaining
gaps are scope, not stubs:

* **One composite.** `conv2d + relu` is the obvious second pattern
  (`vc4_conv2d` + `vc4_relu`), and the engine's NCHW conv needs
  `Cin % 16 == 0`, `Cout % 16 == 0`, `KH = KW ∈ {1, 3}` and `stride == 1`.
  It is not registered yet, so conv graphs stay on the host.
* **No chains.** Every module call is one composite.  Fusing a chain into one
  device graph (the point of BYOC for launch-bound graphs, see
  `docs/BENCHMARKS.md` on the 41% launch path) needs a multi-node descriptor
  with intermediate tensors, i.e. a module that models the data flow rather than
  one op.
* **Constants are folded, up to 4096 elements.** Larger weights must become
  module constants through `get_const_vars`/`__init_<symbol>`; the descriptor
  and the runtime have the hook, the codegen raises instead of embedding them.
* **Static shapes only.** `M`, `K` and `N` are in the descriptor and the engine
  plan; a symbolic matmul is not claimed.  Dynamic shapes are a documented
  structural limit (`docs/FRAMEWORKS.md` §6).
* **fp32 only**, like the rest of the engine.
* **No autodiff lowering yet.** The `Gradient` pass would emit `matmul`,
  `matmul_bt`, `relu_bwd` and bias gradients; only the forward `matmul + bias`
  is claimed, so training graphs fall back to the host.
* **One arena.** The runtime allocates per module instance and never returns
  buffers; a long-lived service would need a plan-time allocator.

The 0.27 wheel also matters for scope: without `tvm/relay` there is no Relay
BYOC and no `relay.op`, so every other backend recipe in the TVM tree (including
`relax.backend.contrib.example_npu`, which has a pattern table but no codegen)
is Relax-only here.
