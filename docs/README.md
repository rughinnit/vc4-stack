# vc4-stack documentation index

| document | contents |
|---|---|
| `../README.md` | project overview, measured ceilings, scope |
| `SOURCES-VERIFICATION.md` | link -> claim -> how we verified -> evidence -> status |
| `FINDINGS.md` | consolidated catalogue of everything measured, with what remains unproven |
| `REFERENCES.md` | every external source used, with URL, purpose, status; claims -> source table; Emma Anholt's work |
| `XCHECK.md` | independent decoder cross-validation (5,329 words, 0 unknown; 33/33 bit-exact rotates; corrections) |
| `NUMERICS.md` | the QPU FPU truncates (round toward zero); 3-word dependent chains; register/field rules |
| `BUILD.md` | build parallelism on 1 GB: engine -j4 safe, VC4C -j2/-j1, ccache, `build-fast.sh` |
| `CLOCK-VARIANCE.md` | core clock drooped 400->250 MHz under load *before* the fan; 89/89 samples at 400 MHz with it |
| `TRAINING.md` | training design, backward kernels, capacity limits |
| `TRAINING-RESULTS.md` | MLP 64-32-10 at 100% on SGD/Adam, gradchecks, throughput |
| `CONV-RESULTS.md` | col2im, im2col-row, maxpool fwd/bwd, conv dX/dW, conv-net at 100% |
| `ENGINE-RESULTS.md` | engine bring-up, 14 root-cause fixes, matmul/conv numbers |
| `VC4D.md` | privileged helper: protocol, security model, 82/82 hostile cases, 5-13% overhead |
| `VC4MON.md` | read-only V3D sysfs module: 78 registers, no /dev/mem, exact parity with drm/vc4 debugfs, ioread32 vs memcpy_fromio |
| `V3DLIB.md` | V3DLib build/correctness, KMS and gpu_mem blockers, verdict, how to measure on a dedicated board |
| `REFERENCES.md` (QPULib, py-videocore) | QPULib is the dormant predecessor of V3DLib (examples + perf notes); py-videocore is a maintained MIT Python QPU DSL with a fourth independent encoder and an SGEMM example, but its VCSM path is dead on this kernel |
| `int8-RESULTS.md` | int8: no exact packed byte MAC, int8 MAC/s == fp32 |
| `STATUS.md` | current state of the work |
