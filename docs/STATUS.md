# Status — is the VC4 stack done?

No. This page is the single answer: one row per workstream, the state it is
in, the file or test behind that state, and the measurement that would move
it. `tools/status_check.py` fails when a row leaves the allowed states, when
a `done` row loses its evidence, when a cited path disappears, or when the
completeness numbers below stop matching `docs/completeness.json`.

States: 6 done, 4 limit, 4 open, 2 in progress.

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
| PyTorch (`torch_vc4`) | open | `docs/FRAMEWORKS.md` §3 row 2: declared-subset `PrivateUse1` backend, 46 gradient checks; `docs/BENCHMARKS.md` §4: 1.4× slower than the CPU at mm 256³.  The backend's own completeness table raises `floor`/`ceil` (the kernel does not generate), `exp`/`log`/`erf`/`round`, bilinear resize and grouped conv beyond depthwise; broadcast add and softmax wider than 16 classes are composite paths, not native | Fix the floor/ceil generator and the four wrong unary bodies, add bilinear resize and grouped conv, then re-run the backend completeness table |
| JAX (PJRT) | in progress | Branch `feat/pjrt-ops`, not on `main`; `docs/PJRT.md` on `main` records the 18-op subset, 16 cases, 0 over tolerance.  The branch widens it to slice, pad, iota, gather, any-axis reduce, batched `dot_general` and convolution; 33 lowering checks pass without a kernel, and the serialized QPU pass has not run | Run `pjrt/tests/test_pjrt.py` on an idle board, record per-case launches and values, then merge |
| LiteRT (external delegate) | in progress | Branch `feat/litert-delegate`, not on `main`.  Its claim table passes 37 checks and the CPU tiers 28 checks; no QPU timing exists.  The delegate refuses quantized, dynamic, broadcast and non-dense tensors by name | Run the `VC4_LITERT_GPU=1` tier on an idle board, record accuracy and timing, then merge |
| TensorFlow | open | `docs/FRAMEWORKS.md` §3 row 8: PluggableDevice via StreamExecutor, no plugin, runtime import unverified; the wheel does ship `stream_executor.h` and `pjrt_c_api.h` | A plugin that registers `/device:VC4:0` and runs one op on the QPU |
| Apache TVM | open | `docs/FRAMEWORKS.md` §3 row 6: Relax BYOC codegen, not started; the 0.27 wheel has no `tvm/relay`, so `relay.ext` is unavailable | A `relax.ext.vc4` pattern table and codegen that runs one fused graph on the QPU |
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
