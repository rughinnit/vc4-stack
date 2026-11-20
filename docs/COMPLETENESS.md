# ONNX operator completeness (generated)

This file is **generated** by `infer/tools/op_matrix.py`; do not edit
it by hand.  It is the result of running the shipped runner against
synthesised single-operator graphs, not of reading a table.

| | |
|---|---|
| Generated | 2026-11-20T16:17:00Z |
| Engine | vc4infer 1.2.8 |
| Python package | `/home/raspi/work/vc4-stack/infer/python/vc4infer/__init__.py` |
| Engine library | `/home/raspi/work/vc4-stack/infer/build/libvc4infer.so` |
| Runner | `mode=hybrid`, `host_policy=device` |
| ONNX / opset | onnx 1.17.0, default domain opset **22** |
| Oracle | onnx.reference.ReferenceEvaluator 1.17.0 |
| Tolerance | `abs(got-ref) <= 0.0001 + 0.0001*max(|ref|,1)` |
| Hardware | Raspberry Pi 3 Model B Rev 1.2, kernel 6.18.50+rpt-rpi-v8, stock clocks |
| Scope | ONNX default domain (`ai.onnx`); other domains are not enumerated |
| Provenance | vc4infer 1.2.8 repaired: rnn+geometry+daemon merges restored, stock 400 MHz, fanless |

## Headline

Two different denominators.  The first is the whole standard at
opset 22; the second removes only the operators a static fp32 device
cannot express *structurally* (shape/metadata, string, sequence or
optional values, control flow, and integer/bool-only tensor ops, including int8/uint8 quantized operators, because the runner accepts fp32 graph inputs only).  The
second number is never presented as the first.

| denominator | ops | **complete (vc4)** | vc4-limited | refused | mismatch | hang | folded | unmeasured | invalid | excluded | **pct_complete** | pct_expressible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ops in opset 22 | 193 | **143** | 2 | 0 | 0 | 0 | 4 | 0 | 0 | 41 | **74.09%** | 75.13% |
| ops not structurally host-only | 152 | **143** | 2 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | **94.08%** | 95.39% |

`folded` = resolved by the engine at plan time with the host
reference evaluator (constant table generators).  It is reported,
but never counted as vc4.  `complete (vc4)` = every required form
tested here, including the attribute variants listed below, ran
within tolerance.  `pct_expressible`
counts a working general case (`vc4` + `vc4-limited`); it is not the
same number as `pct_complete` and is never presented as it.

Structural exclusions from the second denominator: **41** operators (control-flow 3, deprecated 2, io-decode 1, metadata 4, non-fp32 14, sequence/optional 12, string 5).  They are listed with their exact reason in `completeness.json` under `excluded`.

Sensitivity: if the 14 integer/bool-only operators (including the quantized ones) were counted as expressible, the denominator would be **166** and the complete figure **143/166 = 86.14%**.  They are excluded because `Runner.__init__` accepts fp32 graph inputs only, so no valid instance of those operators can reach the arena.

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
| 193 | 143 | 2 | 0 | 0 | 7 | 41 |

#### Activation / elementwise

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Abs` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0767 |  |
| `Acos` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 1.98e-07 | 0.0794 |  |
| `Acosh` | 22 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 2.24e-07 | 0.076 |  |
| `Add` | 14 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.13e-07 | 0.0769 |  |
| `Asin` | 22 | **vc4** | 1/1 | 0/0 | 1.79e-07 | 3.07e-07 | 0.0786 |  |
| `Asinh` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 6.37e-06 | 0.0799 |  |
| `Atan` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 3.20e-07 | 0.0794 |  |
| `Atanh` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 5.93e-07 | 0.11 |  |
| `Ceil` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0842 |  |
| `Celu` | 12 | **vc4** | 1/1 | 0/0 | 2.98e-07 | 4.78e-07 | 0.747 |  |
| `Clip` | 13 | **vc4** | 3/3 | 0/0 | 0 | 0 | 0.0848 |  |
| `Cos` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.09e-07 | 0.0759 |  |
| `Cosh` | 22 | **vc4** | 1/1 | 0/0 | 9.54e-07 | 6.24e-07 | 0.0788 |  |
| `Div` | 14 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 1.19e-07 | 0.155 |  |
| `Elu` | 22 | **vc4** | 1/1 | 0/0 | 1.85e-06 | 6.49e-06 | 0.0778 |  |
| `Erf` | 13 | **vc4** | 1/1 | 0/0 | 8.94e-07 | 1.55e-06 | 0.0795 |  |
| `Exp` | 13 | **vc4** | 1/1 | 0/0 | 1.79e-06 | 1.38e-06 | 0.0784 |  |
| `Floor` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0845 |  |
| `Gelu` | 20 | **vc4** | 1/1 | 0/0 | 5.36e-07 | 2.82e-06 | 0.384 |  |
| `HardSigmoid` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.02e-07 | 0.0787 |  |
| `HardSwish` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 2.34e-07 | 0.0788 |  |
| `Hardmax` | 13 | **vc4** | 2/2 | 0/0 | 0 | 0 | 11.2 |  |
| `IsInf` | 20 | **vc4** | 1/1 | 0/0 | 0 | - | 0.0788 |  |
| `IsNaN` | 20 | **vc4** | 1/1 | 0/0 | 0 | - | 0.0758 |  |
| `LeakyRelu` | 16 | **vc4** | 1/1 | 0/0 | 4.66e-10 | 1.14e-07 | 0.0758 |  |
| `Log` | 13 | **vc4** | 1/1 | 0/0 | 8.94e-08 | 3.01e-07 | 0.0765 |  |
| `LogSoftmax` | 13 | **vc4** | 2/2 | 0/0 | 1.43e-06 | 1.13e-06 | 0.896 |  |
| `Max` | 13 | **vc4** | 2/2 | 0/0 | 0 | 0 | 1.77 |  |
| `Mean` | 13 | **vc4** | 2/2 | 0/0 | 1.19e-07 | 1.09e-05 | 0.232 |  |
| `Min` | 13 | **vc4** | 2/2 | 0/0 | 0 | 0 | 1.77 |  |
| `Mish` | 22 | **vc4** | 1/1 | 0/0 | 3.58e-07 | 1.35e-06 | 0.234 |  |
| `Mod` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.738 |  |
| `Mul` | 14 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0801 |  |
| `Neg` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.079 |  |
| `PRelu` | 16 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.379 |  |
| `Pow` | 15 | **vc4** | 1/1 | 0/0 | 4.29e-06 | 3.02e-06 | 0.248 |  |
| `Reciprocal` | 13 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 9.93e-08 | 0.08 |  |
| `Relu` | 14 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0837 |  |
| `Round` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0787 |  |
| `Selu` | 22 | **vc4** | 1/1 | 0/0 | 2.03e-06 | 8.34e-06 | 0.0795 |  |
| `Shrink` | 9 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.605 |  |
| `Sigmoid` | 13 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 5.63e-07 | 0.0858 |  |
| `Sign` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.085 |  |
| `Sin` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.18e-07 | 0.0802 |  |
| `Sinh` | 22 | **vc4** | 1/1 | 0/0 | 2.41e-06 | 6.45e-06 | 0.08 |  |
| `Softmax` | 13 | **vc4** | 2/2 | 0/0 | 4.47e-07 | 2.72e-06 | 0.543 |  |
| `Softplus` | 22 | **vc4** | 1/1 | 0/0 | 1.07e-06 | 3.20e-06 | 0.0992 |  |
| `Softsign` | 22 | **vc4** | 1/1 | 0/0 | 2.98e-08 | 8.68e-08 | 0.077 |  |
| `Sqrt` | 13 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.28e-07 | 0.0864 |  |
| `Sub` | 14 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.18e-07 | 0.155 |  |
| `Sum` | 13 | **vc4** | 2/2 | 0/0 | 2.38e-07 | 1.96e-07 | 0.746 |  |
| `Tan` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 2.03e-07 | 0.0798 |  |
| `Tanh` | 13 | **vc4** | 1/1 | 0/0 | 7.75e-07 | 5.03e-06 | 0.0807 |  |
| `ThresholdedRelu` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.279 |  |

#### Signal / random / other

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `AffineGrid` | 20 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 1.12e-07 | 1.31 |  |
| `Bernoulli` | 22 | **vc4** | 1/1 | 0/0 | - | - | 0.151 |  |
| `BlackmanWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `DFT` | 20 | **vc4** | 1/1 | 0/0 | 3.58e-07 | 3.92e-04 | 3.1 |  |
| `HammingWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `HannWindow` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `ImageDecoder` | 20 | excluded | 0/0 | 0/0 | - | - | - |  |
| `MelWeightMatrix` | 17 | folded (host) | 0/1 | 0/0 | 0 | 0 | - |  |
| `Multinomial` | 22 | **vc4** | 1/1 | 0/0 | - | - | 2.64 |  |
| `NonMaxSuppression` | 11 | **vc4** | 1/1 | 0/0 | 0 | - | 12.7 |  |
| `NonZero` | 13 | **vc4** | 1/1 | 0/0 | 0 | - | 0.155 |  |
| `RandomNormal` | 22 | **vc4** | 1/1 | 0/0 | - | - | 0.742 |  |
| `RandomNormalLike` | 22 | **vc4** | 1/1 | 0/0 | - | - | 0.739 |  |
| `RandomUniform` | 22 | **vc4** | 1/1 | 0/0 | - | - | 0.0817 |  |
| `RandomUniformLike` | 22 | **vc4** | 1/1 | 0/0 | - | - | 0.0811 |  |
| `STFT` | 17 | **vc4** | 1/1 | 0/0 | 9.54e-07 | 1.35e-05 | 1.68 |  |

#### Logical / comparison

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `And` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitShift` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseAnd` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseNot` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseOr` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `BitwiseXor` | 18 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Equal` | 19 | **vc4** | 1/1 | 0/0 | 0 | - | 0.0748 |  |
| `Greater` | 13 | **vc4** | 1/1 | 0/0 | 0 | - | 0.083 |  |
| `GreaterOrEqual` | 16 | **vc4** | 1/1 | 0/0 | 0 | - | 0.079 |  |
| `Less` | 13 | **vc4** | 1/1 | 0/0 | 0 | - | 0.0883 |  |
| `LessOrEqual` | 16 | **vc4** | 1/1 | 0/0 | 0 | - | 0.0827 |  |
| `Not` | 1 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Or` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Where` | 16 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.305 |  |
| `Xor` | 7 | excluded | 0/0 | 0/0 | - | - | - |  |

#### Matrix / reduction / loss

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `ArgMax` | 13 | **vc4** | 2/2 | 0/0 | 0 | - | 2.29 |  |
| `ArgMin` | 13 | **vc4** | 2/2 | 0/0 | 0 | - | 2.28 |  |
| `CumSum` | 14 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.13e-07 | 0.455 |  |
| `Det` | 22 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.60e-07 | 2.64 |  |
| `Einsum` | 12 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 4.54e-07 | 0.585 |  |
| `Gemm` | 13 | **vc4** | 1/1 | 1/1 | 3.58e-07 | 3.26e-07 | 0.228 |  |
| `MatMul` | 13 | **vc4** | 1/1 | 0/0 | 4.77e-07 | 1.14e-06 | 0.157 |  |
| `NegativeLogLikelihoodLoss` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-08 | 6.69e-08 | 0.893 |  |
| `ReduceL1` | 18 | **vc4** | 3/3 | 0/0 | 3.81e-06 | 2.38e-07 | 0.729 |  |
| `ReduceL2` | 18 | **vc4** | 3/3 | 0/0 | 9.54e-07 | 2.19e-07 | 0.811 |  |
| `ReduceLogSum` | 18 | **vc4** | 3/3 | 0/0 | 2.38e-07 | 1.68e-07 | 0.742 |  |
| `ReduceLogSumExp` | 18 | **vc4** | 3/3 | 0/0 | 1.19e-06 | 6.06e-07 | 0.818 |  |
| `ReduceMax` | 20 | **vc4** | 3/3 | 0/0 | 0 | 0 | 2.88 |  |
| `ReduceMean` | 18 | **vc4** | 3/3 | 0/0 | 1.19e-07 | 1.26e-07 | 0.749 |  |
| `ReduceMin` | 20 | **vc4** | 3/3 | 0/0 | 0 | 0 | 2.87 |  |
| `ReduceProd` | 18 | **vc4** | 3/3 | 0/0 | 7.45e-09 | 7.46e-08 | 1.11 |  |
| `ReduceSum` | 13 | **vc4** | 3/3 | 0/0 | 1.91e-06 | 1.59e-07 | 0.66 |  |
| `ReduceSumSquare` | 18 | **vc4** | 3/3 | 0/0 | 3.81e-06 | 1.80e-07 | 0.743 |  |
| `SoftmaxCrossEntropyLoss` | 13 | **vc4** | 1/1 | 0/0 | 9.54e-07 | 5.13e-07 | 1.63 |  |
| `TopK` | 11 | **vc4** | 1/1 | 0/0 | 0 | 0 | 14.4 |  |
| `Unique` | 11 | **vc4** | 1/1 | 0/0 | 0 | 0 | 4.63 |  |

#### NN / convolution / pooling

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `AveragePool` | 22 | **vc4** | 3/3 | 1/1 | 1.79e-07 | 3.33e-06 | 4.43 |  |
| `BatchNormalization` | 15 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 2.62e-06 | 1.1 |  |
| `Col2Im` | 18 | vc4-unverified | 1/1 | 0/0 | - | - | 1.24 |  |
| `Conv` | 22 | **vc4** | 9/9 | 1/1 | 1.34e-04 | 0.232 | 0.875 |  |
| `ConvInteger` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ConvTranspose` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 1.79e-05 | 3.02 |  |
| `DeformConv` | 22 | **vc4** | 1/1 | 0/0 | 6.56e-07 | 3.10e-05 | 231 |  |
| `DepthToSpace` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 2.3 |  |
| `Dropout` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `GlobalAveragePool` | 22 | **vc4** | 1/1 | 0/0 | 8.94e-08 | 1.49e-06 | 0.14 |  |
| `GlobalLpPool` | 22 | vc4-unverified | 1/1 | 0/0 | - | - | 0.379 |  |
| `GlobalMaxPool` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | 6.75 |  |
| `GridSample` | 22 | **vc4** | 1/1 | 0/0 | 5.36e-07 | 9.23e-06 | 17 |  |
| `GroupNormalization` | 21 | **vc4** | 1/1 | 0/0 | 9.54e-07 | 2.08e-06 | 2.36 |  |
| `InstanceNormalization` | 22 | **vc4** | 1/1 | 0/0 | 5.96e-07 | 3.88e-06 | 2.33 |  |
| `LRN` | 13 | **vc4** | 1/1 | 0/0 | 6.29e-05 | 6.82e-05 | 1.15 |  |
| `LayerNormalization` | 17 | **vc4** | 1/1 | 0/0 | 7.15e-07 | 1.87e-06 | 1.58 |  |
| `LpNormalization` | 22 | **vc4** | 1/1 | 0/0 | 1.19e-07 | 1.87e-07 | 0.757 |  |
| `LpPool` | 22 | **vc4** | 3/3 | 0/0 | 7.15e-07 | 3.36e-07 | 4.2 |  |
| `MaxPool` | 22 | **vc4** | 3/3 | 0/1 | 0 | 0 | 0.0756 | numeric op with no VC4 kernel: ceil_mode |
| `MaxRoiPool` | 22 | vc4-unverified | 1/1 | 0/0 | - | - | 17.5 |  |
| `MaxUnpool` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.154 |  |
| `MeanVarianceNormalization` | 13 | **vc4** | 1/1 | 0/0 | 3.58e-07 | 1.09e-06 | 3.66 |  |
| `RoiAlign` | 22 | **vc4** | 1/1 | 0/0 | 2.38e-07 | 7.27e-06 | 21.4 |  |
| `SpaceToDepth` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 4.59 |  |

#### Tensor / shape manipulation

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Cast` | 21 | **vc4** | 1/1 | 1/1 | 0 | 0 | 0.597 |  |
| `CastLike` | 21 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `CenterCropPad` | 18 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.518 |  |
| `Compress` | 11 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0756 |  |
| `Concat` | 13 | **vc4** | 1/1 | 0/2 | 0 | 0 | - | Concat Concat_rank2_axis1: axis 1 of rank 2 |
| `Constant` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ConstantOfShape` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Expand` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.367 |  |
| `EyeLike` | 22 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `Flatten` | 21 | **vc4** | 1/1 | 1/1 | 0 | 0 | - |  |
| `Gather` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.434 |  |
| `GatherElements` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0757 |  |
| `GatherND` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0789 |  |
| `Identity` | 21 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `OneHot` | 11 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.163 |  |
| `Pad` | 21 | **vc4** | 1/1 | 2/2 | 0 | 0 | 1.8 |  |
| `Range` | 11 | **vc4** | 1/1 | 0/0 | 0 | 0 | - |  |
| `Reshape` | 21 | **vc4** | 2/2 | 1/1 | 0 | 0 | - |  |
| `Resize` | 19 | **vc4** | 4/4 | 2/3 | 1.19e-07 | 3.08e-05 | 0.561 | numeric op with no VC4 kernel: mode=cubic (nearest, linear) |
| `ReverseSequence` | 10 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.796 |  |
| `Scatter` | 11 | excluded | 0/0 | 0/0 | - | - | - |  |
| `ScatterElements` | 18 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.157 |  |
| `ScatterND` | 18 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.225 |  |
| `Shape` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Size` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `Slice` | 13 | vc4-limited | 2/3 | 0/1 | 0 | 0 | 0.0894 | Slice Slice_two_axes: two sliced axes |
| `Split` | 18 | **vc4** | 2/2 | 0/0 | 0 | 0 | 0.162 |  |
| `Squeeze` | 21 | **vc4** | 2/2 | 0/0 | 0 | 0 | - |  |
| `Tile` | 13 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.451 |  |
| `Transpose` | 21 | vc4-limited | 1/2 | 0/1 | 0 | 0 | 0.078 | Transpose Transpose_perm_0132: perm [0, 1, 3, 2] needs a general permute kernel (the affine copy covers the head's [0,2,1,3]) |
| `Trilu` | 14 | **vc4** | 1/1 | 0/0 | 0 | 0 | 0.0806 |  |
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
| `DynamicQuantizeLinear` | 11 | **vc4** | 1/1 | 0/0 | 9.31e-10 | 6.36e-08 | 17.6 |  |
| `MatMulInteger` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QLinearConv` | 10 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QLinearMatMul` | 21 | excluded | 0/0 | 0/0 | - | - | - |  |
| `QuantizeLinear` | 21 | **vc4** | 1/1 | 0/0 | 0 | - | 0.252 |  |

#### Recurrent

| op | since | status | required forms | edge forms | max abs | max rel | gpu ms | refusal / constraint |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `GRU` | 22 | **vc4** | 1/1 | 0/0 | 4.02e-07 | 2.64e-06 | 5.3 |  |
| `LSTM` | 22 | **vc4** | 1/1 | 0/0 | 5.22e-07 | 2.65e-06 | 3.93 |  |
| `RNN` | 22 | **vc4** | 1/1 | 0/0 | 5.07e-07 | 2.01e-06 | 2.36 |  |

## Refusals, verbatim

Every row is the engine's own exception text for a graph that the
ReferenceEvaluator accepted.  `plan` means the policy refused before
any node executed; `run` means it refused during execution.

| op | variant | phase | exception | invariant |
|---|---|---|---|---|
| `Concat` | rank2_axis1 | run | `UnsupportedNode` | `Concat Concat_rank2_axis1: axis 1 of rank 2` |
| `Concat` | axis0 | run | `UnsupportedNode` | `Concat Concat_axis0: axis 0 of rank 2` |
| `MaxPool` | ceil_mode | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: ceil_mode` |
| `Resize` | cubic | plan | `HostNotAllowed` | `numeric op with no VC4 kernel: mode=cubic (nearest, linear)` |
| `Slice` | two_axes | run | `UnsupportedNode` | `Slice Slice_two_axes: two sliced axes` |
| `Slice` | step2 | run | `UnsupportedNode` | `Slice Slice_step2: step 2` |
| `Transpose` | perm_0132 | run | `UnsupportedNode` | `Transpose Transpose_perm_0132: perm [0, 1, 3, 2] needs a general permute kernel (the affine copy covers the head's [0,2,1,3])` |
| `Transpose` | rank2 | run | `UnsupportedNode` | `Transpose Transpose_rank2: perm [1, 0] needs a general permute kernel (the affine copy covers the head's [0,2,1,3])` |

## Oracle gaps

These graphs are **valid** (the ONNX checker accepts them) but
`onnx.reference` cannot evaluate them, so only the engine's
refusal could be scored.  The exact reference error is in
`completeness.json` (`oracle_error`).

| op | variant | reference evaluator error |
|---|---|---|
| `Col2Im` | 4x4_blocks2 | `IndexError: index 2 is out of bounds for axis 0 with size 2` |
| `GlobalLpPool` | p2 | `RuntimeImplementationError: No registered implementation for operator 'GlobalLpPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. Yo` |
| `MaxRoiPool` | 2_rois_2x2 | `RuntimeImplementationError: No registered implementation for operator 'MaxRoiPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You ` |
| `Multinomial` | 2x4_sample3 | `RuntimeImplementationError: No registered implementation for operator 'Multinomial' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You` |

## Top refusals by real-model usage

Ranked over a local corpus of 13 distinct real ONNX models (modelA.onnx, modelB.onnx, modelC.onnx, yolov8n_160.onnx, light_bvlc_alexnet.onnx, light_densenet121.onnx, light_inception_v1.onnx, light_inception_v2.onnx, light_resnet50.onnx, light_shufflenet.onnx, light_squeezenet.onnx, light_vgg19.onnx, light_zfnet512.onnx): the number of models that contain the operator, then node count.

The corpus contains **0 refused operators** in total; all of them are listed (there is no tenth).

| # | op | status | models | nodes |
|---:|---|---|---:|---:|

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

