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
| [wiki](https://github.com/rughinnit/vc4-stack/wiki) | the same pages, generated from this directory by `tools/wiki_sync.py`; edit here, never there |
| [`TUTORIAL.md`](TUTORIAL.md) | **start here**: prerequisites, install (apt/pip/source), verify, YOLO three ways, PyTorch backend, ONNX export, measured performance |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | every failure mode we hit: symptom, cause, fix (JIT status=2, policy errors, GPU wedge, flock, coherency, arena/CMA, FPU, `/tmp`) |
| [`API.md`](API.md) | reference for `vc4infer` (`Engine`/`Device`/`Detector`/`postprocess`/Ultralytics, errors, `host_policy`) and `torch_vc4` (op table, autograd, accuracy) |
| [`../README.md`](../README.md) | project landing page: definition, headline numbers, install, hardware, frameworks, the document index, scope |
| [`../infer/README.md`](../infer/README.md) | libvc4infer: API, build, kernel JIT, daemon backend, op coverage, limits |
| [`../training/README.md`](../training/README.md) | backward kernels and the MLP/conv trainer, how to run them |
| [`../vc4d/README.md`](../vc4d/README.md) | the privileged helper daemon: protocol, build, install |
| [`../tf/README.md`](../tf/README.md) | the TensorFlow PluggableDevice: StreamExecutor API 0.0.1, the device type constraint, the implemented allocator/copy callbacks, the one op and what the next one costs |
| `NCNN.md` | **ncnn vs the VC4 engine**: same model, same image — ncnn 74.8 ms vs VC4 481.7 ms (6.4×), detections identical to 0.00 px, and why the gap is launch count rather than arithmetic |
| `ACCURACY.md` | **accuracy ledger**: VC4 vs the CPU on the same image (identical boxes, 0.00 px), per-op error bounds, training gradients, determinism, and where the error comes from (truncating FPU, no FMA, denormals) |
| [`ACCURACY-FPS.md`](ACCURACY-FPS.md) | **the latency-accuracy curve**: YOLOv8n at 96/128/160 with real COCO weights — launches, wall, FPS, QPU time and arena per point; the detections that survive each size; the 0.168 ms marginal launch and 3.3 GFLOP/s conv rate; and the workstation recipe for a trained 96/128 model and a 0.05 GFLOP, 8-conv student |
| [`COST-COMPARISON.md`](COST-COMPARISON.md) | **the cost of a delivered frame**: YOLOv8n at 96/128/160 on ncnn and vc4infer with CPU seconds per frame at equal FPS, the QPU share versus the CPU share, the fixed-instruction co-run burner, and the 8.7-17.1 frames per CPU-second break-even |
| [`CPU-BREAKDOWN.md`](CPU-BREAKDOWN.md) | **where a frame's CPU time goes**: the 0.313 s of a YOLOv8n-160 frame by phase (Python executor, host copies, C op setup, launch path, daemon), the two host-side changes it motivated, and the 24-29 % CPU and 16-21 % wall reduction they measured |
| `BUGS.md` | every bug, its root cause, the fix, the verification and the commit — plus the open ones with workarounds and the process failures that cost time |
| `BENCHMARKS.md` | **the registry**: hardware envelope, GEMM, YOLOv8n progression, 320/640, per-op torch_vc4 vs CPU, training, builds, copy bandwidth, ORT, backend overhead, clocks, completeness, framework effort |
| [`MODELS.md`](MODELS.md) | **model coverage beyond YOLO**: fifteen graphs measured against ONNX Runtime on the CPU, the three that run on the QPUs, the exact first refusal for every other one (arena, 7x7 stem, grouped conv, `ConstantOfShape`, `Upsample`, int8), and what each gap would need |
| `BACKENDS.md` | the two backends: `managed` (vc4d, unprivileged, ~0.1 ms/launch) vs `direct` (root, no overhead) — measured 488 vs 410 ms on YOLOv8n, and when to use which |
| `ECOSYSTEM.md` | every VC4 repository, what lives inside this tree, the old-path map, how the two execution paths fit, and which repo to open for what |
| `MEDIUM-CLASSIFICATION.md` | lnstadrum (2021): a 225 K-param GLSL classifier on Pi Zero at 595 ms; the GLES 2.0 limits it documents, its 8-bit/channel-shuffle co-design, and the three ways it corroborates our launch-bound, CPU-supply-bound results |
| `VC4CV.md` | Seneral/VC4CV (MIT): the QPU stencil programs, the 16-bit VDW stride confirmation, the ZeroCompat camera-buffer erratum, and why it cannot run on this stack |
| `SOURCES-VERIFICATION.md` | link -> claim -> how we verified -> evidence -> status |
| `FINDINGS.md` | consolidated catalogue of everything measured, with what remains unproven |
| `REFERENCES.md` | every external source used, with URL, purpose, status; claims -> source table; Emma Anholt's work |
| `XCHECK.md` | independent decoder cross-validation (5,329 words, 0 unknown; 33/33 bit-exact rotates; corrections) |
| `NUMERICS.md` | the QPU FPU truncates (round toward zero), flushes denormals; 3-word dependent chains; register/field rules |
| `BUILD.md` | build parallelism on 1 GB: engine -j4 safe, VC4C -j2/-j1, ccache, `tools/build-fast.sh`, and the from-checkout build, test and packaging commands |
| `CLOCK-VARIANCE.md` | core clock drooped 400->250 MHz under load *before* the fan; 89/89 samples at 400 MHz with it |
| [`OPS-UNARY.md`](OPS-UNARY.md) | the inverse/hyperbolic unary family (asin/acos/atan/atanh/asinh/acosh/sinh/cosh/tan): per-op method, enforced domain, measured accuracy, the refusals and their exact limits |
| [`OPS-NORM.md`](OPS-NORM.md) | the pooling / normalisation / softmax-variant / loss family (AveragePool, GlobalMaxPool, LpPool, GlobalLpPool, MaxUnpool, BatchNormalization, InstanceNormalization, GroupNormalization, LayerNormalization, MeanVarianceNormalization, LRN, Hardmax, LogSoftmax, Softmax, SoftmaxCrossEntropyLoss, NegativeLogLikelihoodLoss, Dropout): per-op method, the enforced epsilon placement and channel-axis layout, measured accuracy, every refusal with its exact reason, the five onnx.reference divergences, and two engine limits worked around |
| [`OPS-GEOMETRY.md`](OPS-GEOMETRY.md) | the geometry / detection block (ConvTranspose, Col2Im, AffineGrid, GridSample, RoiAlign, MaxRoiPool, NonMaxSuppression, Det, Einsum, DeformConv): col2im+GEMM versus a zero-insertion conv with both MAC counts, the four-tap gather that replaces `vc4_gather`, the NMS suppression closure, measured launches/accuracy per op, the daemon's float-uniform denial and two pool/view defects it exposed |
| `ONNX.md` | ONNX graph path: supported nodes, runner usage, measured modelA/B/C numbers, stride-2/depthwise decompositions, YOLOv8n coverage, the device byte invariant |
| [`VC4-MODEL.md`](VC4-MODEL.md) | a detector built to the measured launch and FLOP budget: the specification and its reasons, the measured per-conv cost at 96 and 128, 8 convs at 36.3 / 24.9 FPS, the 30 and 60 FPS gaps, and why no accuracy is claimed |
| `VC4INFER.md` | the packaged product: install, `run`/`bench`/`nodes`/`selftest`, guarantees, measured performance, package layout |
| `PYTHON-API.md` | the earlier `Engine`, `Device`, `Detector`, threading, error and Ultralytics reference (kept for history; `API.md` is current) |
| `FRAMEWORKS.md` | framework support survey: what runs today, what needs a relay, effort/evidence for each |
| [`../tvm_byoc/README.md`](../tvm_byoc/README.md) | TVM Relax BYOC vertical slice: the `matmul + bias` pattern table, the `relax.ext.vc4` codegen and its JSON descriptor, the runtime's `record`/`host`/`vc4` modes, CPU checks and the serialized silicon pass |
| `TRAINING.md` | training design, backward kernels, capacity limits, which frameworks train on the QPUs, and the workflow that works |
| [`POSITIONING.md`](POSITIONING.md) | how the stack maps layer by layer onto CUDA and ROCm, how far each layer is from the 28.8 GFLOP/s structural peak, the measured ceilings, and the ncnn gap that matters more |
| `TRAINING-RESULTS.md` | MLP 64-32-10 at 100% on SGD/Adam, gradchecks, throughput |
| `CONV-RESULTS.md` | col2im, im2col-row, maxpool fwd/bwd, conv dX/dW, conv-net at 100% |
| `ENGINE-RESULTS.md` | engine bring-up, 14 root-cause fixes, matmul/conv numbers |
| `VC4D.md` | privileged helper: protocol, security model, 90/90 hostile cases, 5-13% overhead |
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
