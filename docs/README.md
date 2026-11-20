# vc4-stack documentation index

**Start with [TUTORIAL.md](TUTORIAL.md)** for the walkthrough, and
**[BENCHMARKS.md](BENCHMARKS.md)** for every measured number and comparison in
one table.

**Start with [TUTORIAL.md](TUTORIAL.md).**  It takes a stock Pi 3B to a running
YOLOv8n and a trained `torch_vc4` model, and every command in it was run on the
board or is explicitly marked as not re-run.  This directory is the single
source of documentation; code lives in the product directories it describes,
and every path below is relative to the repository root unless it starts with
`docs/`.

| document | contents |
|---|---|
| [`TUTORIAL.md`](TUTORIAL.md) | **start here**: prerequisites, install (apt/pip/source), verify, YOLO three ways, PyTorch backend, ONNX export, measured performance |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | every failure mode we hit: symptom, cause, fix (JIT status=2, policy errors, GPU wedge, flock, coherency, arena/CMA, FPU, `/tmp`) |
| [`API.md`](API.md) | reference for `vc4infer` (`Engine`/`Device`/`Detector`/`postprocess`/Ultralytics, errors, `host_policy`) and `torch_vc4` (op table, autograd, accuracy) |
| [`../README.md`](../README.md) | project index: what lives where, status, measured ceilings, scope |
| [`../infer/README.md`](../infer/README.md) | libvc4infer: API, build, kernel JIT, daemon backend, op coverage, limits |
| [`../training/README.md`](../training/README.md) | backward kernels and the MLP/conv trainer, how to run them |
| [`../vc4d/README.md`](../vc4d/README.md) | the privileged helper daemon: protocol, build, install |
| `NCNN.md` | **ncnn vs the VC4 engine**: same model, same image — ncnn 74.8 ms vs VC4 481.7 ms (6.4×), detections identical to 0.00 px, and why the gap is launch count rather than arithmetic |
| `ACCURACY.md` | **accuracy ledger**: VC4 vs the CPU on the same image (identical boxes, 0.00 px), per-op error bounds, training gradients, determinism, and where the error comes from (truncating FPU, no FMA, denormals) |
| `BUGS.md` | every bug, its root cause, the fix, the verification and the commit — plus the open ones with workarounds and the process failures that cost time |
| `BENCHMARKS.md` | **the registry**: hardware envelope, GEMM, YOLOv8n progression, 320/640, per-op torch_vc4 vs CPU, training, builds, copy bandwidth, ORT, backend overhead, clocks, completeness, framework effort |
| `BACKENDS.md` | the two backends: `managed` (vc4d, unprivileged, ~0.1 ms/launch) vs `direct` (root, no overhead) — measured 488 vs 410 ms on YOLOv8n, and when to use which |
| `ECOSYSTEM.md` | every VC4 repository, what each owns, how the two execution paths fit, and which repo to open for what |
| `MEDIUM-CLASSIFICATION.md` | lnstadrum (2021): a 225 K-param GLSL classifier on Pi Zero at 595 ms; the GLES 2.0 limits it documents, its 8-bit/channel-shuffle co-design, and the three ways it corroborates our launch-bound, CPU-supply-bound results |
| `VC4CV.md` | Seneral/VC4CV (MIT): the QPU stencil programs, the 16-bit VDW stride confirmation, the ZeroCompat camera-buffer erratum, and why it cannot run on this stack |
| `SOURCES-VERIFICATION.md` | link -> claim -> how we verified -> evidence -> status |
| `FINDINGS.md` | consolidated catalogue of everything measured, with what remains unproven |
| `REFERENCES.md` | every external source used, with URL, purpose, status; claims -> source table; Emma Anholt's work |
| `XCHECK.md` | independent decoder cross-validation (5,329 words, 0 unknown; 33/33 bit-exact rotates; corrections) |
| `NUMERICS.md` | the QPU FPU truncates (round toward zero), flushes denormals; 3-word dependent chains; register/field rules |
| `BUILD.md` | build parallelism on 1 GB: engine -j4 safe, VC4C -j2/-j1, ccache, `tools/build-fast.sh` |
| `CLOCK-VARIANCE.md` | core clock drooped 400->250 MHz under load *before* the fan; 89/89 samples at 400 MHz with it |
| [`OPS-UNARY.md`](OPS-UNARY.md) | the inverse/hyperbolic unary family (asin/acos/atan/atanh/asinh/acosh/sinh/cosh/tan): per-op method, enforced domain, measured accuracy, the refusals and their exact limits |
| `ONNX.md` | ONNX graph path: supported nodes, runner usage, measured modelA/B/C numbers, stride-2/depthwise decompositions, YOLOv8n coverage, the device byte invariant |
| `VC4INFER.md` | the packaged product: install, `run`/`bench`/`nodes`/`selftest`, guarantees, measured performance, package layout |
| `PYTHON-API.md` | the earlier `Engine`, `Device`, `Detector`, threading, error and Ultralytics reference (kept for history; `API.md` is current) |
| `FRAMEWORKS.md` | framework support survey: what runs today, what needs a relay, effort/evidence for each |
| `TRAINING.md` | training design, backward kernels, capacity limits |
| `TRAINING-RESULTS.md` | MLP 64-32-10 at 100% on SGD/Adam, gradchecks, throughput |
| `CONV-RESULTS.md` | col2im, im2col-row, maxpool fwd/bwd, conv dX/dW, conv-net at 100% |
| `ENGINE-RESULTS.md` | engine bring-up, 14 root-cause fixes, matmul/conv numbers |
| `VC4D.md` | privileged helper: protocol, security model, 82/82 hostile cases, 5-13% overhead |
| `DAEMON.md` | the daemon-backed engine: unprivileged client evidence, lock/arena sizing, measured launch overhead |
| `VC4MON.md` | read-only V3D sysfs module: 78 registers, no /dev/mem, exact parity with drm/vc4 debugfs, ioread32 vs memcpy_fromio |
| `V3DLIB.md` | V3DLib build/correctness, KMS and gpu_mem blockers, verdict, how to measure on a dedicated board |
| [`../infer/python/vc4infer/kgen/`](../infer/python/vc4infer/kgen) | QPU kernel generator (the Python package ships it; C clients use `infer/tools/kgen.py`) |
| `int8-RESULTS.md` | int8: no exact packed byte MAC, int8 MAC/s == fp32 |
| `STATUS.md` | current state of the work |
| `results/` | raw run records behind the numbers: `engine.md`, `daemon.md`, `build-fast.md`, `training.md` |
| `logs/` | the suite and benchmark logs: `suite-REPS5.log`, `suite-daemon.log`, `suite-train.log`, `bench-final.log` |

`COMPLETENESS.md` (the $1 work's authoritative per-op table for
`torch_vc4`) is linked from [API.md](API.md) when that document lands.  Raw
results are kept verbatim, so work-tree paths inside `results/` and `logs/`
refer to the machine the run was taken on, not to this layout.
