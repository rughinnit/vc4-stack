# Accuracy

Speed has had most of the attention; this is the accuracy ledger. The question
that matters is not "is it bit-identical" (it is not, and no GPU is) but **"does
a model give the same answer here as it does on the CPU"**, and **where exactly
is the error and how big**.

Tolerances used throughout: `atol = rtol = 5e-4`, i.e. an element is "over" when
`|got − ref| > 5e-4 + 5e-4·max(|ref|, 1)`; the stricter `1e-4` form is also
reported.

## 1. End to end: YOLOv8n vs ONNX Runtime **on the CPU**, same model, same input

This is the comparison a user cares about — not versus a numpy reference
evaluator, but versus the thing they would otherwise run.

Real image (`bus.jpg`, letterboxed to 160×160), identical post-processing
(`vc4infer.postprocess.decode_yolo`) on both outputs:

| | |
|---|---:|
| raw tensor max abs | **1.37e-02** |
| raw tensor mean abs | 2.82e-05 |
| max relative error where `|ref| ≥ 1` | **3.19e-04** |
| over `5e-4` | **0 / 44,100** |
| over `1e-4` | 31 / 44,100 |
| **detections at conf 0.25** | **4 vs 4, same classes** |
| **max box difference** | **0.00 px** |
| max score difference | 8.5e-05 |

Side by side:

```
cpu  person 0.7784  [ 25.8  60.6  53.9 133.6]     vc4  person 0.7784  [ 25.8  60.6  53.9 133.6]
cpu  bus    0.7029  [ 20.3  33.6 138.6 112.4]     vc4  bus    0.7029  [ 20.3  33.6 138.6 112.4]
cpu  person 0.6193  [120.5  62.9 140.8 128.3]     vc4  person 0.6192  [120.5  62.9 140.8 128.3]
cpu  person 0.5971  [ 50.5  58.0  72.3 127.9]     vc4  person 0.5971  [ 50.5  58.0  72.3 127.9]
```

The same holds with random input (no detections, but the arithmetic still
compares): max abs **1.56e-02**, mean **4.7e-05**, **0/44,100** over 5e-4.

**Conclusion: the detection output is identical to the CPU's.** The raw tensor
differs by up to ~1e-2 in absolute terms, which is 3e-4 relative — below the
width of a box, and it does not move a single coordinate at 1-pixel precision.

## 2. End to end vs the independent reference (what `selftest` checks)

| case | over 5e-4 | max abs | max rel |
|---|---|---:|---:|
| modelA (conv/relu/pool/gemm/softmax) | **0/10** | 4.84e-06 | 4.84e-06 |
| modelB (stride-2, depthwise, 5×5 pool, concat, resize) | **0/16,384** | 4.41e-06 | 4.09e-06 |
| modelC (conv/GAP/gemm) | **0/4** | 4.06e-07 | 4.06e-07 |
| YOLOv8n 160 | **0/44,100** | 2.04e-02 | 2.23e-04 |
| YOLOv8n 320 | 1/176,400 | 4.40e-02 | 5.07e-04 |

The single 320 element is a DFL box coordinate at 1.003× tolerance
(`86.8006` vs `86.7566` on an 86-unit coordinate). It is identical in three
runs, so it is accumulation order, not a race.

## 3. Per operator

| op | error | note |
|---|---|---|
| `Exp` | rel **5.7e-6** | `2^(x·log2e)`, deg-5, exact scale |
| `Log` | rel **4.8e-7** | exponent/mantissa split + atanh series |
| `Sqrt` | rel **1.1e-7** | Newton `1/sqrt` |
| `Reciprocal` | rel **5.7e-8** | sign-split Newton |
| `Erf` | abs **4.1e-7** | A&S 7.1.26, exact ±1 for `|x|≥6` |
| `Tanh` | abs **1.2e-7** | `sign(1−t)/(1+t)` |
| `Softplus` / `Elu` / `Selu` | 1.3e-6 / 5.9e-7 / 8.7e-7 | |
| `Sin` / `Cos` | abs **4.8e-6** for `|x|≤1e5` | 3-part Cody-Waite; outside the range it refuses |
| `Gelu` (tanh, erf), `Mish` | within tolerance | composed from the above |
| **`Floor`, `Ceil`, `Round`** | **bit-exact** | `Round` is half-to-even |
| **`Abs`, `Neg`, `Sign`, `Clip`** | **bit-exact** | |
| **comparisons** (`Equal/Greater/Less/…`, `IsNaN`, `IsInf`) | **bit-exact** | integer key ordering + explicit NaN mask, no subtraction |
| elementwise (`Add`, `Mul`, `Relu`, `Silu`, `Sigmoid`) | ≤2.4e-7 | correctly-rounded multiply, see §5 |
| `Softmax` (≤16 classes) | 2.6e-7 … 3.4e-7 | |
| copies, concat, split, pad, transpose, resize-nearest | **byte-exact** | max abs error 0 on 14 shapes |
| conv / GEMM (fp32) | ≤2e-4 abs | accumulation order vs the reference |

Verified over **105,264 elements / 120 kernel runs** (unary) and **10,478
elements / 78 ONNX graphs**; 0 mismatches in both.

## 4. Training accuracy

| check | result |
|---|---|
| gradient checks (46) | **≤ 5.4e-6** norm-relative vs CPU autograd; per-element peak 6.4e-3 only where the denominator is noise (abs error 5e-5 on a gradient of 0.35) |
| backward ops, one per process (19) | all pass, `rel_L2 ≈ 1e-6 … 1e-7` |
| toy MLP, 300 steps | 100 % train / 100 % test |
| CNN demo, 40 steps | loss 2.64 → 5.8e-3 → **3.6e-4**, train and test |
| **determinism** | weights **bit-identical** across runs (`fnv` hashes), and equal to a run two sessions earlier |

Determinism is worth stating separately: the same input gives the *same bits*
every run, which is stronger than "accurate" and is what a training loop needs.

## 5. Where the error comes from

1. **The FPU truncates** (round toward zero) for `fadd` and `fmul`. A single
   `fmul` is therefore *not* IEEE. Where a correctly-rounded product matters,
   the engine builds one explicitly: split into 12-bit halves, compute the exact
   residual, and bump by one ulp past the half-way point (Veltkamp). That is the
   2.4e-7 elementwise row rather than ~1e-3.
2. **No FMA.** `a·b+c` is two roundings, not one. This is why a 64-layer network
   accumulates ~3e-4 relative and not ~1e-7.
3. **Denormals flush to zero.** Measured: `Abs(1.4e-45)` preserves the value,
   `LeakyRelu(1.4e-45, α=1)` returns 0, because it goes through `fmul`/`fadd`.
   Approximations keep their arguments away from that range.
4. **Accumulation order** differs from the reference, so the last bits of a
   long reduction differ by construction — that is the YOLO `2e-2` absolute
   figure on logits, not a defect.
5. **Approximations** for transcendentals (`§3`), each with its measured bound
   and its domain; outside the domain the op **refuses** rather than drifting.

## 6. What is *not* traded away

- **No quantisation**: the stack is fp32 throughout. int8 was rejected as a
  speed lever (no packed byte MAC), and no model is silently downcast. Any
  accuracy comparison here is fp32-vs-fp32.
- **No reduced-precision accumulation**: the VPM stores and the GEMM accumulate
  in fp32.
- **No silent approximations**: an op either meets its stated bound or raises.

## 7. How to check it yourself

```sh
vc4infer selftest                      # 4 models against stored references
vc4infer selftest --json               # per-case max abs / max rel / counts
python3 infer/tools/op_matrix.py --only Conv,Softmax --stage all   # per-op vs onnx.reference
cd ~/work/torch-vc4 && sh tests/run_all.sh   # forward and gradient checks vs CPU
```

`vc4infer selftest` prints `0 elements over tolerance` only when it is true; the
max-abs and max-rel columns are there so a change in error shows up even when it
is still inside the tolerance.
