# ONNX operator completeness (generated)

This file is **generated** by `infer/tools/op_matrix.py`; do not edit
it by hand.  It is the result of running the shipped runner against
synthesised single-operator graphs, not of reading a table.

| | |
|---|---|
| Generated | 2026-11-20T04:35:31Z |
| Engine | vc4infer 1.2.2 |
| Runner | `mode=hybrid`, `host_policy=device` |
| ONNX / opset | onnx 1.17.0, default domain opset **22** |
| Oracle | onnx.reference.ReferenceEvaluator 1.17.0 |
| Tolerance | `abs(got-ref) <= 0.0001 + 0.0001*max(|ref|,1)` |
| Hardware | Raspberry Pi 3 Model B Rev 1.2, kernel 6.18.50+rpt-rpi-v8, stock clocks |
| Scope | ONNX default domain (`ai.onnx`); other domains are not enumerated |
| Provenance | vc4infer 1.2.2 (in-tree infer/python + infer/build/libvc4infer.so, HEAD 87fcd46; the installed dist-packages deb is stale at 35 ops); stock clocks (core 400 MHz / V3D 300 MHz); clean kernel cache; quiet board (no GPU client, vc4-smi idle 0%, no throttle/undervoltage/ERRSTAT); run_tests.sh REPS=1 SUITE_EXIT=0 ALL TESTS PASSED |

## Headline

Two different denominators.  The first is the whole standard at
opset 22; the second removes only the operators a static fp32 device
cannot express *structurally* (shape/metadata, string, sequence or
optional values, control flow, and integer/bool-only tensor ops, including int8/uint8 quantized operators, because the runner accepts fp32 graph inputs only).  The
second number is never presented as the first.

| denominator | ops | **complete (vc4)** | vc4-limited | refused | mismatch | hang | folded | unmeasured | invalid | excluded | **pct_complete** | pct_expressible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ops in opset 22 | 193 | **40** | 4 | 104 | 1 | 0 | 3 | 0 | 0 | 41 | **20.73%** | 22.80% |
| ops not structurally host-only | 152 | **40** | 4 | 104 | 1 | 0 | 3 | 0 | 0 | 0 | **26.32%** | 28.95% |

`folded` = resolved by the engine at plan time with the host
reference evaluator (constant table generators).  It is reported,
but never counted as vc4.  `complete (vc4)` = every required form
tested here, including the attribute variants listed below, ran
within tolerance.  `pct_expressible`
counts a working general case (`vc4` + `vc4-limited`); it is not the
same number as `pct_complete` and is never presented as it.

Structural exclusions from the second denominator: **41** operators (control-flow 3, deprecated 2, io-decode 1, metadata 4, non-fp32 14, sequence/optional 12, string 5).  They are listed with their exact reason in `completeness.json` under `excluded`.

Sensitivity: if the 14 integer/bool-only operators (including the quantized ones) were counted as expressible, the denominator would be **166** and the complete figure **40/166 = 24.10%**.  They are excluded because `Runner.__init__` accepts fp32 graph inputs only, so no valid instance of those operators can reach the arena.

## MISMATCHES (bugs, highest priority)

| op | variant | max abs | max rel | worst ref -> got | explanation |
|---|---|---:|---:|---|---|
| `Resize` | nearest_half_pixel | 1.97 | 142 | 0.979 -> -0.992 | the engine computes floor((s+0.5)*scale-0.5) for every nearest mode; ONNX's default nearest_mode=round_prefer_floor rounds that coordinate, so pixels at .75 differ (docs/ONNX.md does not state a nearest_mode constraint). The floor variant of the same case passes. |
| `Resize` | sizes | 1.98 | 153 | 0.987 -> -0.991 | same half_pixel rounding bug: the sizes form also uses ONNX's default coordinate_transformation_mode=half_pixel. |

## Internal engine errors (bug reports)

These are not policy refusals: the node was dispatched, the
engine's own call failed, and the runner wrapped the library
error in `UnsupportedNode`.  No output was produced, so they are
scored as `refused`, but they are engine bugs, not constraints.

| op | variant | engine error |
|---|---|---|
| `Conv` | group2_3x3 | `engine error: copy2d_x: invalid argument ()` |

## Excluded operators (with the reason)

These are not part of the second denominator.  Nothing is excluded
because the engine fails it; the reason is structural.

| op | kind | reason |
|---|---|---|
| `And` | non-fp32 | bool tensors; the engine arena is fp32 activations only |
| `BitShift` | non-fp32 | integer tensors; the engine arena is fp32 activations only |
| `BitwiseAnd` | non-fp32 | integer tensors; the engine arena is fp32 activations only |
| `BitwiseNot` | non-fp32 | integer tensors; the engine arena is fp32 activations only |
| `BitwiseOr` | non-fp32 | integer tensors; the engine arena is fp32 activations only |
| `BitwiseXor` | non-fp32 | integer tensors; the engine arena is fp32 activations only |
| `ConcatFromSequence` | sequence/optional | sequence value |
| `Constant` | metadata | a literal, not a computation |
| `ConstantOfShape` | metadata | constant materialisation only |
| `ConvInteger` | non-fp32 | int8/uint8 activation (quantized); the engine arena is fp32 activations only |
| `DequantizeLinear` | non-fp32 | int8/uint8 activation (quantized); the engine arena is fp32 activations only |
| `If` | control-flow | subgraph branch chosen at run time |
| `ImageDecoder` | io-decode | decodes an encoded image; host I/O by nature |
| `Loop` | control-flow | subgraph executed a data-dependent number of times |
| `MatMulInteger` | non-fp32 | int8/uint8 activation (quantized); the engine arena is fp32 activations only |
| `Not` | non-fp32 | bool tensors; the engine arena is fp32 activations only |
| `Optional` | sequence/optional | optional value |
| `OptionalGetElement` | sequence/optional | optional value |
| `OptionalHasElement` | sequence/optional | optional value |
| `Or` | non-fp32 | bool tensors; the engine arena is fp32 activations only |
| `QLinearConv` | non-fp32 | int8/uint8 activation (quantized); the engine arena is fp32 activations only |
| `QLinearMatMul` | non-fp32 | int8/uint8 activation (quantized); the engine arena is fp32 activations only |
| `RegexFullMatch` | string | string tensors |
| `Scan` | control-flow | subgraph executed along a scanned axis |
| `Scatter` | deprecated | removed from the opset (last schema v11); an opset-22 model may not contain it |
| `SequenceAt` | sequence/optional | sequence value |
| `SequenceConstruct` | sequence/optional | sequence value |
| `SequenceEmpty` | sequence/optional | sequence value |
| `SequenceErase` | sequence/optional | sequence value |
| `SequenceInsert` | sequence/optional | sequence value |
| `SequenceLength` | sequence/optional | sequence value |
| `SequenceMap` | sequence/optional | sequence value and a graph body |
| `Shape` | metadata | shape metadata only (folded at plan time) |
| `Size` | metadata | element-count metadata only (folded at plan time) |
| `SplitToSequence` | sequence/optional | sequence value |
| `StringConcat` | string | string tensors |
| `StringNormalizer` | string | string tensors |
| `StringSplit` | string | string tensors |
| `TfIdfVectorizer` | string | string tensors |
| `Upsample` | deprecated | removed from the opset (last schema v10); an opset-22 model may not contain it |
| `Xor` | non-fp32 | bool tensors; the engine arena is fp32 activations only |

## Operators by domain

### `ai.onnx`

| ops | vc4 | vc4-limited | refused | mismatch | untested | host-only |
|---:|---:|---:|---:|---:|---:|---:|
| 193 | 40 | 4 | 104 | 1 | 3 | 41 |

#### Activation / elementwise

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Abs` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.107 |  |
| `Acos` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Acosh` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Add` | 14 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.13e-07 | 0.0791 |  |
| `Asin` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Asinh` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Atan` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Atanh` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Ceil` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0787 |  |
| `Celu` | 12 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Clip` | 13 | **vc4** | 3/3 | 0/0 | 0 | 0 | 0.084 |  |
| `Cos` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.09e-07 | 0.0776 |  |
| `Cosh` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Div` | 14 | refused | 0/1 | 0/0 | - | - | - | Div Div_generic: divisor is not a scalar constant |
| `Elu` | 22 | **vc4** | 1/1 | 0/0 | 1.85e-06 | 6.49e-06 | 0.0784 |  |
| `Erf` | 13 | **vc4** | 1/1 | 0/0 | 8.94e-07 | 1.55e-06 | 0.0778 |  |
| `Exp` | 13 | **vc4** | 1/1 | 0/0 | 1.79e-06 | 1.38e-06 | 0.0779 |  |
| `Floor` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0776 |  |
| `Gelu` | 20 | **vc4** | 1/1 | 0/0 | 5.36e-07 | 2.82e-06 | 0.38 |  |
| `HardSigmoid` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.02e-07 | 0.0822 |  |
| `HardSwish` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 2.34e-07 | 0.0784 |  |
| `Hardmax` | 13 | refused | 0/2 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `IsInf` | 20 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `IsNaN` | 20 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LeakyRelu` | 16 | **vc4** | 1/1 | 0/0 | 4.66e-10 | 1.14e-07 | 0.0776 |  |
| `Log` | 13 | **vc4** | 1/1 | 0/0 | 8.94e-08 | 3.01e-07 | 0.0786 |  |
| `LogSoftmax` | 13 | refused | 0/2 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Max` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Mean` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Min` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Mish` | 22 | **vc4** | 1/1 | 0/0 | 3.58e-07 | 1.35e-06 | 0.227 |  |
| `Mod` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Mul` | 14 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0807 |  |
| `Neg` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0768 |  |
| `PRelu` | 16 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Pow` | 15 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Reciprocal` | 13 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 9.93e-08 | 0.0782 |  |
| `Relu` | 14 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.079 |  |
| `Round` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0801 |  |
| `Selu` | 22 | **vc4** | 1/1 | 0/0 | 2.03e-06 | 8.34e-06 | 0.0766 |  |
| `Shrink` | 9 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Sigmoid` | 13 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 5.63e-07 | 0.0796 |  |
| `Sign` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0796 |  |
| `Sin` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.18e-07 | 0.0786 |  |
| `Sinh` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Softmax` | 13 | vc4-limited | 1/2 | 0/0 | 4.47e-07 | 2.72e-06 | 0.239 | Softmax Softmax_axis1_c4_4d: axis 1 of 4 classes needs the last axis or the NHWC layout (16 classes) |
| `Softplus` | 22 | **vc4** | 1/1 | 0/0 | 1.07e-06 | 3.20e-06 | 0.0892 |  |
| `Softsign` | 22 | **vc4** | 1/1 | 0/0 | 2.98e-08 | 8.68e-08 | 0.0801 |  |
| `Sqrt` | 13 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.28e-07 | 0.0786 |  |
| `Sub` | 14 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.18e-07 | 0.171 |  |
| `Sum` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Tan` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Tanh` | 13 | **vc4** | 1/1 | 0/0 | 7.75e-07 | 5.03e-06 | 0.0768 |  |
| `ThresholdedRelu` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

#### Signal / random / other

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `AffineGrid` | 20 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Bernoulli` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `BlackmanWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `DFT` | 20 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `HammingWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `HannWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `ImageDecoder` | 20 | excluded | 0/0 | 0/0 | - | - | - |  |
| `MelWeightMatrix` | 17 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Multinomial` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `NonMaxSuppression` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `NonZero` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RandomNormal` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RandomNormalLike` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RandomUniform` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RandomUniformLike` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `STFT` | 17 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

#### Logical / comparison

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `And` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitShift` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseAnd` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseNot` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseOr` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseXor` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Equal` | 19 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Greater` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GreaterOrEqual` | 16 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Less` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LessOrEqual` | 16 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Not` | 1 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Or` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Where` | 16 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Xor` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |

#### Matrix / reduction / loss

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `ArgMax` | 13 | refused | 0/2 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ArgMin` | 13 | refused | 0/2 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `CumSum` | 14 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Det` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Einsum` | 12 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Gemm` | 13 | **vc4** | 1/1 | 1/1 | 3.58e-07 | 3.26e-07 | 0.234 |  |
| `MatMul` | 13 | **vc4** | 1/1 | 0/0 | 4.77e-07 | 1.14e-06 | 0.153 |  |
| `NegativeLogLikelihoodLoss` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceL1` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceL2` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceLogSum` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceLogSumExp` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceMax` | 20 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceMean` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceMin` | 20 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceProd` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceSum` | 13 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ReduceSumSquare` | 18 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `SoftmaxCrossEntropyLoss` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `TopK` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Unique` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

#### NN / convolution / pooling

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `AveragePool` | 22 | refused | 0/3 | 0/1 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `BatchNormalization` | 15 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Col2Im` | 18 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Conv` | 22 | vc4-limited | 8/9 | 1/1 | 1.34e-04 | 0.232 | 0.74 | engine error: copy2d_x: invalid argument () |
| `ConvInteger` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ConvTranspose` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `DeformConv` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `DepthToSpace` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Dropout` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GlobalAveragePool` | 22 | **vc4** | 1/1 | 0/0 | 8.94e-08 | 1.49e-06 | 0.161 |  |
| `GlobalLpPool` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GlobalMaxPool` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GridSample` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GroupNormalization` | 21 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `InstanceNormalization` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LRN` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LayerNormalization` | 17 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LpNormalization` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LpPool` | 22 | refused | 0/3 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `MaxPool` | 22 | **vc4** | 3/3 | 0/1 | 0 | 0 | 0.0795 | numeric op with no VC4 kernel: ceil_mode |
| `MaxRoiPool` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `MaxUnpool` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `MeanVarianceNormalization` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RoiAlign` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `SpaceToDepth` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

#### Tensor / shape manipulation

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Cast` | 21 | refused | 0/1 | 0/1 | - | - | - | metadata node could not be resolved at plan time (metadata/control type conversion (Cast to INT32)) |
| `CastLike` | 21 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `CenterCropPad` | 18 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Compress` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Concat` | 13 | **vc4** | 1/1 | 0/2 | 0 | 0 | - | Concat Concat_rank2_axis1: axis 1 of rank 2 |
| `Constant` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ConstantOfShape` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Expand` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `EyeLike` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Flatten` | 21 | **vc4** | 1/1 | 1/1 | 0 | 0 | - |  |
| `Gather` | 13 | refused | 0/1 | 0/0 | - | - | - | Gather on activation data (only int64 shape tensors are implemented) |
| `GatherElements` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `GatherND` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Identity` | 21 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `OneHot` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Pad` | 21 | refused | 0/1 | 0/2 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Range` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Reshape` | 21 | **vc4** | 2/2 | 1/1 | 0 | 0 | - |  |
| `Resize` | 19 | **MISMATCH** | 3/4 | 0/3 | 1.98 | 153 | 0.528 | see MISMATCHES above |
| `ReverseSequence` | 10 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Scatter` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ScatterElements` | 18 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `ScatterND` | 18 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Shape` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Size` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Slice` | 13 | vc4-limited | 2/3 | 0/1 | 0 | 0 | 0.0825 | Slice Slice_two_axes: two sliced axes |
| `Split` | 18 | **vc4** | 2/2 | 0/0 | 0 | 0 | 0.156 |  |
| `Squeeze` | 21 | **vc4** | 2/2 | 0/0 | 0 | 0 | - |  |
| `Tile` | 13 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Transpose` | 21 | vc4-limited | 1/2 | 0/1 | 0 | 0 | 0.0722 | Transpose Transpose_perm_0132: perm [0, 1, 3, 2] needs a general permute kernel (the affine copy covers the head's [0,2,1,3]) |
| `Trilu` | 14 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `Unsqueeze` | 21 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `Upsample` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |

#### Control flow / sequence / string

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `ConcatFromSequence` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `If` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Loop` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Optional` | 15 | excluded | 0/0 | 0/0 | - | - | - |  |
| `OptionalGetElement` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `OptionalHasElement` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `RegexFullMatch` | 20 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Scan` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceAt` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceConstruct` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceEmpty` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceErase` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceInsert` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceLength` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SequenceMap` | 17 | excluded | 0/0 | 0/0 | - | - | - |  |
| `SplitToSequence` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `StringConcat` | 20 | excluded | 0/0 | 0/0 | - | - | - |  |
| `StringNormalizer` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `StringSplit` | 20 | excluded | 0/0 | 0/0 | - | - | - |  |
| `TfIdfVectorizer` | 9 | excluded | 0/0 | 0/0 | - | - | - |  |

#### Quantization

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `DequantizeLinear` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `DynamicQuantizeLinear` | 11 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `MatMulInteger` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QLinearConv` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QLinearMatMul` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QuantizeLinear` | 21 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

#### Recurrent

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `GRU` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `LSTM` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |
| `RNN` | 22 | refused | 0/1 | 0/0 | - | - | - | numeric op with no VC4 kernel: no implementation |

## Refusals, verbatim

Every row is the engine's own exception text for a graph that the
ReferenceEvaluator accepted.  `plan` means the policy refused before
any node executed; `run` means it refused during execution.

| op | variant | phase | exception | invariant |
|---|---|---|---|---|
| `Acos` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Acosh` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `AffineGrid` | theta_1x2x3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ArgMax` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ArgMax` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ArgMin` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ArgMin` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Asin` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Asinh` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Atan` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Atanh` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `AveragePool` | k2_s2_no_pad | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `AveragePool` | k3_s1_pad1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `AveragePool` | k3_s2_pad1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `AveragePool` | count_include_pad | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `BatchNormalization` | inference_c16 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Bernoulli` | p_uniform | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Cast` | f32_to_i32 | plan | `HostNotAllowed` | `metadata node could not be resolved at plan time (metadata/control type conversion (Cast to INT32))` |
| `Cast` | f32_to_f32 | plan | `HostNotAllowed` | `metadata node could not be resolved at plan time (metadata/control type conversion (Cast to FLOAT))` |
| `CastLike` | f32_to_f32 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Celu` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `CenterCropPad` | crop_last_axis | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Col2Im` | 4x4_blocks2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Compress` | axis0 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Concat` | rank2_axis1 | run | `UnsupportedNode` | `Concat Concat_rank2_axis1: axis 1 of rank 2` |
| `Concat` | axis0 | run | `UnsupportedNode` | `Concat Concat_axis0: axis 0 of rank 2` |
| `Conv` | group2_3x3 | run | `UnsupportedNode` | `engine error: copy2d_x: invalid argument ()` |
| `ConvTranspose` | 3x3_s2_p1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Cosh` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `CumSum` | axis1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `DFT` | real_onesided | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `DeformConv` | 3x3_offset_group1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `DepthToSpace` | blocksize2_dcr | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Det` | batched_3x3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Div` | generic | run | `UnsupportedNode` | `Div Div_generic: divisor is not a scalar constant` |
| `Dropout` | inference_identity | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `DynamicQuantizeLinear` | float_input | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Einsum` | ij_jk_ik | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Equal` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Expand` | broadcast | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `EyeLike` | 4x4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GRU` | t2_b2_h2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Gather` | axis1_indices_2x2 | plan | `HostNotAllowed` | `Gather on activation data (only int64 shape tensors are implemented)` |
| `GatherElements` | axis1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GatherND` | batch_dims0 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GlobalLpPool` | p2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GlobalMaxPool` | 1x16x4x4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Greater` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GreaterOrEqual` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GridSample` | bilinear | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `GroupNormalization` | groups2_c4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Hardmax` | last_axis_2d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Hardmax` | axis1_c16_4d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Hardmax` | axis1_c4_4d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `InstanceNormalization` | c4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `IsInf` | default | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `IsNaN` | default | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LRN` | size3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LSTM` | t2_b2_h2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LayerNormalization` | last_axis | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Less` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LessOrEqual` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LogSoftmax` | last_axis_2d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LogSoftmax` | axis1_c16_4d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LogSoftmax` | axis1_c4_4d | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LpNormalization` | axis_last_p2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LpPool` | k2_s2_no_pad | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LpPool` | k3_s1_pad1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `LpPool` | k3_s2_pad1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Max` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `MaxPool` | ceil_mode | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: ceil_mode` |
| `MaxRoiPool` | 2_rois_2x2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `MaxUnpool` | 2x2_s2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Mean` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `MeanVarianceNormalization` | axes_0_2_3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `MelWeightMatrix` | 4mel_dft8 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Min` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Mod` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Multinomial` | 2x4_sample3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `NegativeLogLikelihoodLoss` | mean | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `NonMaxSuppression` | 4_boxes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `NonZero` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `OneHot` | depth4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `PRelu` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Pad` | constant_pads | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Pad` | reflect | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Pad` | axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Pow` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `QuantizeLinear` | scale01_zp0 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RNN` | t2_b2_h2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RandomNormal` | shape_2x3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RandomNormalLike` | like | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RandomUniform` | shape_2x3 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RandomUniformLike` | like | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Range` | float_0_1_0p2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL1` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL1` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL1` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL2` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL2` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceL2` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSum` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSum` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSum` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSumExp` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSumExp` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceLogSumExp` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMax` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMax` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMax` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMean` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMean` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMean` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMin` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMin` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceMin` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceProd` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceProd` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceProd` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSum` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSum` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSum` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSumSquare` | all_axes | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSumSquare` | axis1_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ReduceSumSquare` | axis_last_no_keepdims | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Resize` | cubic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: mode=cubic (nearest, linear)` |
| `Resize` | align_corners | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: coordinate_transformation_mode=align_corners` |
| `ReverseSequence` | batch0_time1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `RoiAlign` | 2_rois | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `STFT` | frame8_step4 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ScatterElements` | axis1 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ScatterND` | rank2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Shrink` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Sinh` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Slice` | two_axes | run | `UnsupportedNode` | `Slice Slice_two_axes: two sliced axes` |
| `Slice` | step2 | run | `UnsupportedNode` | `Slice Slice_step2: step 2` |
| `Softmax` | axis1_c4_4d | run | `UnsupportedNode` | `Softmax Softmax_axis1_c4_4d: axis 1 of 4 classes needs the last axis or the NHWC layout (16 classes)` |
| `SoftmaxCrossEntropyLoss` | mean | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `SpaceToDepth` | blocksize2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Sum` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Tan` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `ThresholdedRelu` | generic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Tile` | 2x2 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `TopK` | k3_axis_last | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Transpose` | perm_0132 | run | `UnsupportedNode` | `Transpose Transpose_perm_0132: perm [0, 1, 3, 2] needs a general permute kernel (the affine copy covers the head's [0,2,1,3])` |
| `Transpose` | rank2 | run | `UnsupportedNode` | `Transpose Transpose_rank2: perm [1, 0] needs a general permute kernel (the affine copy covers the head's [0,2,1,3])` |
| `Trilu` | upper_k0 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Unique` | axis0 | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |
| `Where` | cond_init | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: no implementation` |

## Oracle gaps

These graphs are **valid** (the ONNX checker accepts them) but
`onnx.reference` cannot evaluate them, so only the engine's
refusal could be scored.  The exact reference error is in
`completeness.json` (`oracle_error`).

| op | variant | reference evaluator error |
|---|---|---|
| `Col2Im` | 4x4_blocks2 | `IndexError: index 2 is out of bounds for axis 0 with size 2` |
| `DFT` | real_onesided | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['inverse', 'onesided'] and linked attributes=[] (operator '` |
| `GlobalLpPool` | p2 | `RuntimeImplementationError: No registered implementation for operator 'GlobalLpPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. Yo` |
| `MaxRoiPool` | 2_rois_2x2 | `RuntimeImplementationError: No registered implementation for operator 'MaxRoiPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You ` |
| `MelWeightMatrix` | 4mel_dft8 | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['output_d` |
| `Multinomial` | 2x4_sample3 | `RuntimeImplementationError: No registered implementation for operator 'Multinomial' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You` |
| `STFT` | frame8_step4 | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['onesided'] and linked attributes=[] (operator 'STFT').` |

## Top refusals by real-model usage

Ranked over a local corpus of 13 distinct real ONNX models (modelA.onnx, modelB.onnx, modelC.onnx, yolov8n_160.onnx, light_bvlc_alexnet.onnx, light_densenet121.onnx, light_inception_v1.onnx, light_inception_v2.onnx, light_resnet50.onnx, light_shufflenet.onnx, light_squeezenet.onnx, light_vgg19.onnx, light_zfnet512.onnx): the number of models that contain the operator, then node count.

The corpus contains **7 refused operators** in total; all of them are listed (there is no tenth).

| # | op | status | models | nodes |
|---:|---|---|---:|---:|
| 1 | `AveragePool` | refused | 5 | 17 |
| 2 | `BatchNormalization` | refused | 4 | 292 |
| 3 | `Dropout` | refused | 4 | 6 |
| 4 | `LRN` | refused | 3 | 6 |
| 5 | `Sum` | refused | 2 | 29 |
| 6 | `Div` | refused | 1 | 2 |
| 7 | `Gather` | refused | 1 | 1 |

## Method

* Operator list: `onnx.defs.get_all_schemas_with_history()`, filtered
  to the default domain and to schemas in force at opset 22.
* Each graph is one node with seeded random inputs.  Conv weights and
  all structural parameters (axes, pads, shape, starts/ends) are
  initialisers; only fp32 activations are graph inputs.
* The oracle runs first.  If it rejects the graph, the case is
  `invalid-model` (a bug in this tool, never scored) and the run
  stops there.
* Then `Runner(...).check_policy()` runs.  A refusal at this point
  allocates no GPU work.  Only accepted graphs execute.
* Non-float outputs must match exactly; float outputs use the
  tolerance above, with NaN==NaN and matching infinities allowed.
* `gpu_ms` is the engine's own pack+kernel time for the node, not wall
  time, and is reported per case in `completeness.json`.

## What is not measured

* Other ONNX domains (`ai.onnx.ml`, `ai.onnx.preview.training`,
  vendor domains).
* Multi-node graphs: fusion, layout propagation and allocation are
  out of scope; every case is a single operator.
* Performance: `gpu_ms` is reported for information only.
* Operators excluded as structurally host-only are listed with their
  reason in `completeness.json` (`excluded`).

