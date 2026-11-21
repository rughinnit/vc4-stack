# Status — is the VC4 stack done?

No. This page is the single answer: one row per workstream, the state it is
in, the file or test behind that state, and the measurement that would move
it. `tools/status_check.py` fails when a row leaves the allowed states, when
a `done` row loses its evidence, when a cited path disappears, or when the
completeness numbers below stop matching `docs/completeness.json`.

States: 9 done, 4 limit, 4 open, 0 in progress.

| state | meaning |
|---|---|
| `done` | measured on silicon; the evidence is a file with the number or a named test |
| `limit` | a documented hardware or structural boundary, with the reason |
| `open` | not done; the next concrete step is named |
| `in progress` | a branch exists; the branch and what it still needs are named |

A declared subset that ships is not the same claim as a framework that runs
general models; the rows keep those apart.  "Measured" means the number is in
the cited file, under the conditions stated there: stock 400 MHz core, managed
backend unless noted, fanless, board quiet.

## Workstreams

| workstream | state | evidence | what would change the state |
|---|---|---|---|
| Engine and QPU kernels | done | `docs/BENCHMARKS.md` §2: hand-written GEMM 13.72 GFLOP/s and engine GEMM 8.47 GFLOP/s at 256³; `docs/BUGS.md` A18-A19 and B4: gather, reduce, cumsum and select3 element-verified on silicon, 0 bad; `docs/KERNELS.md` §4: the store defects behind the last of those failures | A measured regression of the engine GEMM, or an element check failing on the QPU |
| ONNX completeness | done | `docs/completeness.json`: 152/152 expressible, 152/193 opset 22, 0 refused, 0 folded, 41 structurally excluded; `docs/COMPLETENESS.md` is the generated table (also [below](#completeness)) | A refused or mismatching required form in a new sweep, or a new opset adding an expressible operator |
| Daemon (`vc4d`) | done | `docs/DAEMON.md`: +0.309 ms, 7.8% wall at 256³ and 30% at 128³; `docs/VC4D.md`: 90/90 hostile cases rejected; `docs/BUGS.md` B9-B15 name the test that fails before each fix | A launch-path regression, or any of B9-B15 re-opening |
| Training | limit | `docs/BENCHMARKS.md` §5: MLP 64.7 steps/s, CNN 0.98-1.37 steps/s, Adam state 16 B/param; 128-channel 128×128 needs 113 MiB of Adam state and YOLOv8n 51 MB, so neither fits the arena | A model that does not fit today running a measured step, or an allocator that fits 113 MiB of state |
| ggml backend | limit | `docs/GGML.md` §5: 21 VC4 nodes and 2 CPU nodes, 0 mismatch, max abs 8.01e-08, 28.47 ms best against 3.09 ms on four CPU threads; §8: no real LLM fits the 64 MiB client arena and the `N*K*4` weight panel | A real llama.cpp model running end to end, which needs ROPE and an arena the board does not have |
| PyTorch (`torch_vc4`) | open | `docs/FRAMEWORKS.md` §3 row 2, and the backend completeness table: `test_complete.py` passes every check on silicon (222 checks, peak 63.4 C); `floor`/`ceil`/`round`/`exp`/`log`/`erf` route to the element-verified engine bodies, worst abs 1.15e-02 on `exp` (rel 3.96e-06) and exact on the rounding bodies; `gelu('none')` is exact (6.56e-07) with an analytic autograd derivative; broadcast add and dimension reductions are verified against the CPU (4.77e-07 and 2.04e-04 abs).  Still raising: bilinear resize (the engine composes it from host gathers; no QPU kernel) and grouped conv beyond depthwise (the engine kernel exists, the extension has no binding) | A QPU bilinear kernel and the grouped-conv binding, then the backend table with every row measured |
| JAX (PJRT) | done | `docs/PJRT.md`: 25 VHLO operation names accepted (elementwise, compare/select, slice, reverse, pad, iota, gather, any-axis reduce with sum/max/min, batched and rank-3 `dot_general`, convolution) and every refusal names the operation.  `pjrt/tests/test_pjrt.py` on merged `main`: all cases within 5e-4 (conv 4.4e-05, batched dot 4.8e-07), named refusals refused, peak 64.5 C; 33 lowering checks run without a kernel | A case over 5e-4, a merge that drops a lowering, or a StableHLO producer whose encoding the reader does not carry |
| LiteRT (external delegate) | done | `litert/README.md`: the plugin ABI, the claimed ops (fp32 conv, depthwise conv, fully-connected, elementwise without broadcast, relu/logistic/tanh, softmax up to 16 classes) and what stays in LiteRT.  `litert/tests/test_litert.py` on silicon: 34 checks, 0 failures, max abs diff 1.71e-07 against the LiteRT CPU reference; 28 CPU-tier checks without the board | A claimed model differing from the LiteRT CPU reference, or a claim that falls back to host arithmetic |
| TensorFlow | open | `tf/README.md`: the StreamExecutor plugin registers `/device:VC:0` (TensorFlow 2.21 rejects a device type containing a digit), allocates from the engine arena, implements H2D/D2H and registers `Add` and `AddV2` for fp32.  On silicon: constant round-trip exact, `tf.add` at n=1024 and n=1000 with 0/1024 and 0/1000 over 1e-6 abs + 1e-6 rel (the FPU truncates, so agreement is not bit-exact), one launch per call, peak 59.1 C | Kernels for the ops a real graph needs (Mul/Sub/Div, the unary family, MatMul), then a Keras model placed on the device |
| Apache TVM | done | `tvm_byoc/README.md`: the `matmul + bias` composite is matched by `tvm_byoc/vc4/patterns.py`, lowered by `relax.ext.vc4` to a JSON descriptor and a module call, and executed by `tvm_byoc/vc4/runtime_vc4.cc`.  On silicon: one recorded call, 3 engine launches, max abs diff 2.861e-06 against Relax on llvm and the same against numpy, no NaNs, peak 60.1 C; 9/9 CPU checks for the pattern, descriptor and host path | A chain of more than one composite per module, a non-static shape, or a second pattern (`conv2d + relu`) |
| ONNX Runtime | open | `docs/FRAMEWORKS.md` §3 row 3: the Python provider in `infer/python/vc4infer/ort/` partitions a graph and runs the claimed nodes through the engine; a C execution provider needs the ORT plugin ABI header, which the 1.32.0 wheel does not ship | A C EP built against the ORT source headers, or the Python provider's claim table measured on a model and recorded here |
| Packaging (apt and pip) | done | `docs/INSTALL.md`: signed `vc4-apt` source, `vc4-toolkit` meta-package and pip index; `docs/VC4INFER.md` §Building and publishing: `build_deb.sh`, `build_wheel.sh`, `publish_apt.sh`; `infer/packaging/check_release.sh` fails on a tree/installed op-table mismatch; `docs/BENCHMARKS.md` §9: 0 run-time cost | A package that installs but fails `check_release.sh`, or a wheel that omits a module |
| Documentation | done | `docs/README.md` is the index; `docs/BENCHMARKS.md` is the number registry and states the conditions for every row; `docs/BUGS.md` records cause, fix and verification | A number that cannot be traced to the registry or a run log, or a broken index entry |
| Repository hygiene | done | `docs/BUGS.md` E8-E9: 239 tracked kernel blobs untracked and `/tmp` worktrees removed from tmpfs; `tools/check_kgen_sync.py` fails on generator drift; `infer/packaging/check_release.sh` fails on a stale package | A generated tree tracked again, or either drift check failing |
| Performance versus ncnn | open | `docs/NCNN.md`: ncnn 74.8 ms and 13.4 FPS against vc4infer 481.7 ms and 2.08 FPS at 160×160, a 6.44× gap with the same detections; of the ~480 ms, ~258 ms is QPU kernel time and ~222 ms is the launch path | A measured 160×160 run below 222 ms of launch time, or a graph that removes launches, then a re-run against ncnn |
| Arena capacity (640×640 and 320×320) | limit | `docs/BENCHMARKS.md` §3: 320×320 runs at 1323.1 ms with `--arena-mb 150`, 1/176400 over tolerance; 640×640 does not fit (working set + scratch ≈ 91 MiB and 15 convs exceed the im2col pitch) | A tiling or allocator change that fits 640×640, or a 320×320 run with 0/176400 over tolerance |
| Thermal envelope | limit | `docs/CLOCK-VARIANCE.md`: fanless, 56.4 C idle, 73.1 C after 112 s of 256³ GEMM, soft limit 80 C; `docs/BENCHMARKS.md` rule 4: numbers hold for runs under ~90 s from a ~56 C start | A fan or heatsink that keeps a long run under 80 C, or a workload whose sustained run fits the 90 s window |

## Completeness

Measured 2026-11-21 by `infer/tools/op_matrix.py`; the machine-readable record
is `docs/completeness.json`.  Two denominators, never interchanged:

- expressible denominator: 152/152 complete, 0 refused, 0 folded.
- whole opset 22: 152/193 complete, 41 structurally excluded.

The 41 exclusions are structural, not failures: control flow, sequences,
strings, metadata and non-fp32 tensors.  They are listed with their reason in
`docs/completeness.json` and in `docs/COMPLETENESS.md`.  Nothing is excluded
because the engine failed it.

## What this page does not claim

- Multi-node graphs: every completeness case is a single operator, so fusion,
  layout propagation and allocation are not covered by the 152/152.
- ONNX domains other than `ai.onnx`.
- int8 as a speed path: there is no exact packed byte MAC on this ISA, so
  int8 MAC/s equals fp32 (`docs/int8-RESULTS.md`).
- A framework outside its declared subset.
- A 640×640 YOLOv8n number: it does not fit.
- Runs longer than the thermal window without `tools/thermal-guard.sh`.

## Keeping it true

Run `python3 tools/status_check.py` from the repository root.  It checks the
allowed states, the evidence paths, the `done` evidence rule and the
completeness numbers against `docs/completeness.json`; it exits non-zero and
prints the row and the expected/actual value for each failure.

Evidence paths are relative to the repository root and written in backticks.
