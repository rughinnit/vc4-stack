# vc4-stack documentation index

This directory is the single source of documentation.  Code lives in the product
directories it describes; every path below is relative to the repository root
unless it starts with `docs/`.

| document | contents |
|---|---|
| [`../README.md`](../README.md) | project index: what lives where, status, measured ceilings, scope |
| [`../infer/README.md`](../infer/README.md) | libvc4infer: API, build, kernel JIT, daemon backend, op coverage, limits |
| [`../training/README.md`](../training/README.md) | backward kernels and the MLP/conv trainer, how to run them |
| [`../vc4d/README.md`](../vc4d/README.md) | the privileged helper daemon: protocol, build, install |
| `SOURCES-VERIFICATION.md` | link -> claim -> how we verified -> evidence -> status |
| `FINDINGS.md` | consolidated catalogue of everything measured, with what remains unproven |
| `REFERENCES.md` | every external source used, with URL, purpose, status; claims -> source table; Emma Anholt's work |
| `XCHECK.md` | independent decoder cross-validation (5,329 words, 0 unknown; 33/33 bit-exact rotates; corrections) |
| `NUMERICS.md` | the QPU FPU truncates (round toward zero); 3-word dependent chains; register/field rules |
| `BUILD.md` | build parallelism on 1 GB: engine -j4 safe, VC4C -j2/-j1, ccache, `tools/build-fast.sh` |
| `CLOCK-VARIANCE.md` | core clock drooped 400->250 MHz under load *before* the fan; 89/89 samples at 400 MHz with it |
| `ONNX.md` | ONNX graph path: supported nodes, runner usage, measured modelA/B/C numbers, stride-2/depthwise decompositions, YOLOv8n coverage gap |
| `VC4INFER.md` | the packaged product: install, `run`/`bench`/`nodes`/`selftest`, guarantees, measured performance, package layout |
| `PYTHON-API.md` | `Engine`, `Device`, `Detector`, threading, error table, Ultralytics integration |
| `TRAINING.md` | training design, backward kernels, capacity limits |
| `TRAINING-RESULTS.md` | MLP 64-32-10 at 100% on SGD/Adam, gradchecks, throughput |
| `CONV-RESULTS.md` | col2im, im2col-row, maxpool fwd/bwd, conv dX/dW, conv-net at 100% |
| `ENGINE-RESULTS.md` | engine bring-up, 14 root-cause fixes, matmul/conv numbers |
| `VC4D.md` | privileged helper: protocol, security model, 82/82 hostile cases, 5-13% overhead |
| `VC4MON.md` | read-only V3D sysfs module: 78 registers, no /dev/mem, exact parity with drm/vc4 debugfs, ioread32 vs memcpy_fromio |
| `V3DLIB.md` | V3DLib build/correctness, KMS and gpu_mem blockers, verdict, how to measure on a dedicated board |
| [`../infer/python/vc4infer/kgen/`](../infer/python/vc4infer/kgen) | QPU kernel generator (the Python package ships it; C clients use `infer/tools/kgen.py`) |
| `int8-RESULTS.md` | int8: no exact packed byte MAC, int8 MAC/s == fp32 |
| `STATUS.md` | current state of the work |
| `results/` | raw run records behind the numbers: `engine.md`, `daemon.md`, `build-fast.md`, `training.md` |
| `logs/` | the suite and benchmark logs: `suite-REPS5.log`, `suite-daemon.log`, `suite-train.log`, `bench-final.log` |

Raw results are kept verbatim, so work-tree paths inside `results/` and `logs/`
refer to the machine the run was taken on, not to this layout.
