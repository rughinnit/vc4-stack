# ONNX operator completeness: graph-validity pass (generated)

This file is **generated** by `infer/tools/op_matrix.py`; do not edit
it by hand.  This revision contains the operator enumeration and the
graph-validity pass only.  **No row is scored as supported:** the
vc4 sweep is deferred while the installed stack is being rebuilt.

| | |
|---|---|
| Generated | 2026-11-20T02:08:43Z |
| Engine | not imported (`ImportError: cannot import name '_runtime' from 'vc4infer' (unknown location)`) |
| Validity | `onnx.checker.check_model` + `onnx.reference.ReferenceEvaluator` 1.17.0 |
| ONNX / opset | onnx 1.17.0, default domain opset **22** |
| Tolerance (reserved for the sweep) | `abs(got-ref) <= 0.0001 + 0.0001*max(|ref|,1)` |
| Hardware | Raspberry Pi 3 Model B Rev 1.2, kernel 6.18.50+rpt-rpi-v8, stock clocks |
| Scope | ONNX default domain (`ai.onnx`); other domains are not enumerated |
| Provenance | graph-validity pass only; the vc4 sweep is deferred while the installed stack is rebuilt |

> An `unmeasured` row is **not** a supported row.  It stays outside
> the numerator until a sweep on a stable stack records `vc4`,
> `refused`, `hang` or `mismatch` for it.

## Denominators

| denominator | ops | graphs built | checker-valid | oracle-verified | with an oracle gap | invalid | measured on vc4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all ops in opset 22 | 193 | 152 | 152 | 145 | 7 | 0 | **0 (unmeasured)** |
| ops not structurally host-only | 152 | 152 | 152 | 145 | 7 | 0 | **0 (unmeasured)** |

| cases | built | checker-valid | oracle-verified | oracle gap | invalid | unmeasured |
|---:|---:|---:|---:|---:|---:|---:|
| 222 | 222 | 222 | 215 | 7 | 0 | 222 |

`checker-valid` is the graph-validity guarantee: the model passes
`onnx.checker` (attributes, types, ranks, output shapes).  `oracle-verified`
additionally means `onnx.reference` executed it and produced a
reference tensor.  The 7-case gap is recorded below.

Structural exclusions from the second denominator: **41** operators (control-flow 3, deprecated 2, io-decode 1, metadata 4, non-fp32 14, sequence/optional 12, string 5).

## Operators by domain

### `ai.onnx`

| ops | unmeasured | invalid | host-only |
|---:|---:|---:|---:|
| 193 | 152 | 0 | 41 |

#### Activation / elementwise

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `Abs` | 13 | built | ok | verified | unmeasured |  |
| `Acos` | 22 | built | ok | verified | unmeasured |  |
| `Acosh` | 22 | built | ok | verified | unmeasured |  |
| `Add` | 14 | built | ok | verified | unmeasured |  |
| `Asin` | 22 | built | ok | verified | unmeasured |  |
| `Asinh` | 22 | built | ok | verified | unmeasured |  |
| `Atan` | 22 | built | ok | verified | unmeasured |  |
| `Atanh` | 22 | built | ok | verified | unmeasured |  |
| `Ceil` | 13 | built | ok | verified | unmeasured |  |
| `Celu` | 12 | built | ok | verified | unmeasured |  |
| `Clip` | 13 | built | ok | verified | unmeasured |  |
| `Cos` | 22 | built | ok | verified | unmeasured |  |
| `Cosh` | 22 | built | ok | verified | unmeasured |  |
| `Div` | 14 | built | ok | verified | unmeasured |  |
| `Elu` | 22 | built | ok | verified | unmeasured |  |
| `Erf` | 13 | built | ok | verified | unmeasured |  |
| `Exp` | 13 | built | ok | verified | unmeasured |  |
| `Floor` | 13 | built | ok | verified | unmeasured |  |
| `Gelu` | 20 | built | ok | verified | unmeasured |  |
| `HardSigmoid` | 22 | built | ok | verified | unmeasured |  |
| `HardSwish` | 22 | built | ok | verified | unmeasured |  |
| `Hardmax` | 13 | built | ok | verified | unmeasured |  |
| `IsInf` | 20 | built | ok | verified | unmeasured |  |
| `IsNaN` | 20 | built | ok | verified | unmeasured |  |
| `LeakyRelu` | 16 | built | ok | verified | unmeasured |  |
| `Log` | 13 | built | ok | verified | unmeasured |  |
| `LogSoftmax` | 13 | built | ok | verified | unmeasured |  |
| `Max` | 13 | built | ok | verified | unmeasured |  |
| `Mean` | 13 | built | ok | verified | unmeasured |  |
| `Min` | 13 | built | ok | verified | unmeasured |  |
| `Mish` | 22 | built | ok | verified | unmeasured |  |
| `Mod` | 13 | built | ok | verified | unmeasured |  |
| `Mul` | 14 | built | ok | verified | unmeasured |  |
| `Neg` | 13 | built | ok | verified | unmeasured |  |
| `PRelu` | 16 | built | ok | verified | unmeasured |  |
| `Pow` | 15 | built | ok | verified | unmeasured |  |
| `Reciprocal` | 13 | built | ok | verified | unmeasured |  |
| `Relu` | 14 | built | ok | verified | unmeasured |  |
| `Round` | 22 | built | ok | verified | unmeasured |  |
| `Selu` | 22 | built | ok | verified | unmeasured |  |
| `Shrink` | 9 | built | ok | verified | unmeasured |  |
| `Sigmoid` | 13 | built | ok | verified | unmeasured |  |
| `Sign` | 13 | built | ok | verified | unmeasured |  |
| `Sin` | 22 | built | ok | verified | unmeasured |  |
| `Sinh` | 22 | built | ok | verified | unmeasured |  |
| `Softmax` | 13 | built | ok | verified | unmeasured |  |
| `Softplus` | 22 | built | ok | verified | unmeasured |  |
| `Softsign` | 22 | built | ok | verified | unmeasured |  |
| `Sqrt` | 13 | built | ok | verified | unmeasured |  |
| `Sub` | 14 | built | ok | verified | unmeasured |  |
| `Sum` | 13 | built | ok | verified | unmeasured |  |
| `Tan` | 22 | built | ok | verified | unmeasured |  |
| `Tanh` | 13 | built | ok | verified | unmeasured |  |
| `ThresholdedRelu` | 22 | built | ok | verified | unmeasured |  |

#### Signal / random / other

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `AffineGrid` | 20 | built | ok | verified | unmeasured |  |
| `Bernoulli` | 22 | built | ok | verified | unmeasured |  |
| `BlackmanWindow` | 17 | built | ok | verified | unmeasured |  |
| `DFT` | 20 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (TypeError) |
| `HammingWindow` | 17 | built | ok | verified | unmeasured |  |
| `HannWindow` | 17 | built | ok | verified | unmeasured |  |
| `ImageDecoder` | 20 | - | - | - | excluded |  |
| `MelWeightMatrix` | 17 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (TypeError) |
| `Multinomial` | 22 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (RuntimeImplementationError) |
| `NonMaxSuppression` | 11 | built | ok | verified | unmeasured |  |
| `NonZero` | 13 | built | ok | verified | unmeasured |  |
| `RandomNormal` | 22 | built | ok | verified | unmeasured |  |
| `RandomNormalLike` | 22 | built | ok | verified | unmeasured |  |
| `RandomUniform` | 22 | built | ok | verified | unmeasured |  |
| `RandomUniformLike` | 22 | built | ok | verified | unmeasured |  |
| `STFT` | 17 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (TypeError) |

#### Logical / comparison

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `And` | 7 | - | - | - | excluded |  |
| `BitShift` | 11 | - | - | - | excluded |  |
| `BitwiseAnd` | 18 | - | - | - | excluded |  |
| `BitwiseNot` | 18 | - | - | - | excluded |  |
| `BitwiseOr` | 18 | - | - | - | excluded |  |
| `BitwiseXor` | 18 | - | - | - | excluded |  |
| `Equal` | 19 | built | ok | verified | unmeasured |  |
| `Greater` | 13 | built | ok | verified | unmeasured |  |
| `GreaterOrEqual` | 16 | built | ok | verified | unmeasured |  |
| `Less` | 13 | built | ok | verified | unmeasured |  |
| `LessOrEqual` | 16 | built | ok | verified | unmeasured |  |
| `Not` | 1 | - | - | - | excluded |  |
| `Or` | 7 | - | - | - | excluded |  |
| `Where` | 16 | built | ok | verified | unmeasured |  |
| `Xor` | 7 | - | - | - | excluded |  |

#### Matrix / reduction / loss

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `ArgMax` | 13 | built | ok | verified | unmeasured |  |
| `ArgMin` | 13 | built | ok | verified | unmeasured |  |
| `CumSum` | 14 | built | ok | verified | unmeasured |  |
| `Det` | 22 | built | ok | verified | unmeasured |  |
| `Einsum` | 12 | built | ok | verified | unmeasured |  |
| `Gemm` | 13 | built | ok | verified | unmeasured |  |
| `MatMul` | 13 | built | ok | verified | unmeasured |  |
| `NegativeLogLikelihoodLoss` | 22 | built | ok | verified | unmeasured |  |
| `ReduceL1` | 18 | built | ok | verified | unmeasured |  |
| `ReduceL2` | 18 | built | ok | verified | unmeasured |  |
| `ReduceLogSum` | 18 | built | ok | verified | unmeasured |  |
| `ReduceLogSumExp` | 18 | built | ok | verified | unmeasured |  |
| `ReduceMax` | 20 | built | ok | verified | unmeasured |  |
| `ReduceMean` | 18 | built | ok | verified | unmeasured |  |
| `ReduceMin` | 20 | built | ok | verified | unmeasured |  |
| `ReduceProd` | 18 | built | ok | verified | unmeasured |  |
| `ReduceSum` | 13 | built | ok | verified | unmeasured |  |
| `ReduceSumSquare` | 18 | built | ok | verified | unmeasured |  |
| `SoftmaxCrossEntropyLoss` | 13 | built | ok | verified | unmeasured |  |
| `TopK` | 11 | built | ok | verified | unmeasured |  |
| `Unique` | 11 | built | ok | verified | unmeasured |  |

#### NN / convolution / pooling

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `AveragePool` | 22 | built | ok | verified | unmeasured |  |
| `BatchNormalization` | 15 | built | ok | verified | unmeasured |  |
| `Col2Im` | 18 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (IndexError) |
| `Conv` | 22 | built | ok | verified | unmeasured |  |
| `ConvInteger` | 10 | - | - | - | excluded |  |
| `ConvTranspose` | 22 | built | ok | verified | unmeasured |  |
| `DeformConv` | 22 | built | ok | verified | unmeasured |  |
| `DepthToSpace` | 13 | built | ok | verified | unmeasured |  |
| `Dropout` | 22 | built | ok | verified | unmeasured |  |
| `GlobalAveragePool` | 22 | built | ok | verified | unmeasured |  |
| `GlobalLpPool` | 22 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (RuntimeImplementationError) |
| `GlobalMaxPool` | 22 | built | ok | verified | unmeasured |  |
| `GridSample` | 22 | built | ok | verified | unmeasured |  |
| `GroupNormalization` | 21 | built | ok | verified | unmeasured |  |
| `InstanceNormalization` | 22 | built | ok | verified | unmeasured |  |
| `LRN` | 13 | built | ok | verified | unmeasured |  |
| `LayerNormalization` | 17 | built | ok | verified | unmeasured |  |
| `LpNormalization` | 22 | built | ok | verified | unmeasured |  |
| `LpPool` | 22 | built | ok | verified | unmeasured |  |
| `MaxPool` | 22 | built | ok | verified | unmeasured |  |
| `MaxRoiPool` | 22 | built | ok | gap | unmeasured | onnx.reference 1.17.0 cannot evaluate this checker-valid graph (RuntimeImplementationError) |
| `MaxUnpool` | 22 | built | ok | verified | unmeasured |  |
| `MeanVarianceNormalization` | 13 | built | ok | verified | unmeasured |  |
| `RoiAlign` | 22 | built | ok | verified | unmeasured |  |
| `SpaceToDepth` | 13 | built | ok | verified | unmeasured |  |

#### Tensor / shape manipulation

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `Cast` | 21 | built | ok | verified | unmeasured |  |
| `CastLike` | 21 | built | ok | verified | unmeasured |  |
| `CenterCropPad` | 18 | built | ok | verified | unmeasured |  |
| `Compress` | 11 | built | ok | verified | unmeasured |  |
| `Concat` | 13 | built | ok | verified | unmeasured |  |
| `Constant` | 21 | - | - | - | excluded |  |
| `ConstantOfShape` | 21 | - | - | - | excluded |  |
| `Expand` | 13 | built | ok | verified | unmeasured |  |
| `EyeLike` | 22 | built | ok | verified | unmeasured |  |
| `Flatten` | 21 | built | ok | verified | unmeasured |  |
| `Gather` | 13 | built | ok | verified | unmeasured |  |
| `GatherElements` | 13 | built | ok | verified | unmeasured |  |
| `GatherND` | 13 | built | ok | verified | unmeasured |  |
| `Identity` | 21 | built | ok | verified | unmeasured |  |
| `OneHot` | 11 | built | ok | verified | unmeasured |  |
| `Pad` | 21 | built | ok | verified | unmeasured |  |
| `Range` | 11 | built | ok | verified | unmeasured |  |
| `Reshape` | 21 | built | ok | verified | unmeasured |  |
| `Resize` | 19 | built | ok | verified | unmeasured |  |
| `ReverseSequence` | 10 | built | ok | verified | unmeasured |  |
| `Scatter` | 11 | - | - | - | excluded |  |
| `ScatterElements` | 18 | built | ok | verified | unmeasured |  |
| `ScatterND` | 18 | built | ok | verified | unmeasured |  |
| `Shape` | 21 | - | - | - | excluded |  |
| `Size` | 21 | - | - | - | excluded |  |
| `Slice` | 13 | built | ok | verified | unmeasured |  |
| `Split` | 18 | built | ok | verified | unmeasured |  |
| `Squeeze` | 21 | built | ok | verified | unmeasured |  |
| `Tile` | 13 | built | ok | verified | unmeasured |  |
| `Transpose` | 21 | built | ok | verified | unmeasured |  |
| `Trilu` | 14 | built | ok | verified | unmeasured |  |
| `Unsqueeze` | 21 | built | ok | verified | unmeasured |  |
| `Upsample` | 10 | - | - | - | excluded |  |

#### Control flow / sequence / string

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `ConcatFromSequence` | 11 | - | - | - | excluded |  |
| `If` | 21 | - | - | - | excluded |  |
| `Loop` | 21 | - | - | - | excluded |  |
| `Optional` | 15 | - | - | - | excluded |  |
| `OptionalGetElement` | 18 | - | - | - | excluded |  |
| `OptionalHasElement` | 18 | - | - | - | excluded |  |
| `RegexFullMatch` | 20 | - | - | - | excluded |  |
| `Scan` | 21 | - | - | - | excluded |  |
| `SequenceAt` | 11 | - | - | - | excluded |  |
| `SequenceConstruct` | 11 | - | - | - | excluded |  |
| `SequenceEmpty` | 11 | - | - | - | excluded |  |
| `SequenceErase` | 11 | - | - | - | excluded |  |
| `SequenceInsert` | 11 | - | - | - | excluded |  |
| `SequenceLength` | 11 | - | - | - | excluded |  |
| `SequenceMap` | 17 | - | - | - | excluded |  |
| `SplitToSequence` | 11 | - | - | - | excluded |  |
| `StringConcat` | 20 | - | - | - | excluded |  |
| `StringNormalizer` | 10 | - | - | - | excluded |  |
| `StringSplit` | 20 | - | - | - | excluded |  |
| `TfIdfVectorizer` | 9 | - | - | - | excluded |  |

#### Quantization

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `DequantizeLinear` | 21 | - | - | - | excluded |  |
| `DynamicQuantizeLinear` | 11 | built | ok | verified | unmeasured |  |
| `MatMulInteger` | 10 | - | - | - | excluded |  |
| `QLinearConv` | 10 | - | - | - | excluded |  |
| `QLinearMatMul` | 21 | - | - | - | excluded |  |
| `QuantizeLinear` | 21 | built | ok | verified | unmeasured |  |

#### Recurrent

| op | since | graph | checker | oracle | vc4 | note |
|---|---:|---|---|---|---|---|
| `GRU` | 22 | built | ok | verified | unmeasured |  |
| `LSTM` | 22 | built | ok | verified | unmeasured |  |
| `RNN` | 22 | built | ok | verified | unmeasured |  |

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

## Oracle gaps

These graphs are checker-valid but `onnx.reference` cannot
evaluate them in this onnx release, so a future sweep can score
a refusal but cannot verify a computed result.  The exact error
is in `completeness.json`.

| op | variant | reference evaluator error |
|---|---|---|
| `Col2Im` | 4x4_blocks2 | `IndexError: index 2 is out of bounds for axis 0 with size 2` |
| `DFT` | real_onesided | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['inverse', 'onesided'] and linked attributes=[] (operator '` |
| `GlobalLpPool` | p2 | `RuntimeImplementationError: No registered implementation for operator 'GlobalLpPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. Yo` |
| `MaxRoiPool` | 2_rois_2x2 | `RuntimeImplementationError: No registered implementation for operator 'MaxRoiPool' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You ` |
| `MelWeightMatrix` | 4mel_dft8 | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['output_d` |
| `Multinomial` | 2x4_sample3 | `RuntimeImplementationError: No registered implementation for operator 'Multinomial' and domain '', schema.has_function is False, schema.has_context_dependent_function is False. You` |
| `STFT` | frame8_step4 | `TypeError: Issues with types [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>] and attributes ['onesided'] and linked attributes=[] (operator 'STFT').` |

## Unmeasured operators that real models use

Ranked over a local corpus of 13 distinct ONNX models (modelA.onnx, modelB.onnx, modelC.onnx, yolov8n_160.onnx, light_bvlc_alexnet.onnx, light_densenet121.onnx, light_inception_v1.onnx, light_inception_v2.onnx, light_resnet50.onnx, light_shufflenet.onnx, light_squeezenet.onnx, light_vgg19.onnx, light_zfnet512.onnx).  This is the order in which the deferred sweep matters, not a result.

| # | op | models | nodes |
|---:|---|---:|---:|
| 1 | `Conv` | 13 | 471 |
| 2 | `Relu` | 12 | 392 |
| 3 | `MaxPool` | 12 | 40 |
| 4 | `Softmax` | 10 | 10 |
| 5 | `Gemm` | 9 | 15 |
| 6 | `Reshape` | 8 | 48 |
| 7 | `Concat` | 7 | 106 |
| 8 | `AveragePool` | 5 | 17 |
| 9 | `BatchNormalization` | 4 | 292 |
| 10 | `Dropout` | 4 | 6 |

## Method

* Operator list: `onnx.defs.get_all_schemas_with_history()`, filtered to the default domain and to schemas in force at opset 22.
* One node per graph, seeded inputs.  Structural parameters (axes,
  pads, shape, starts/ends) are initialisers; only fp32 activations
  are graph inputs.
* Every graph is checked with `onnx.checker.check_model` first.  If
  the checker rejects it, the case is `invalid-model` and is never
  scored.
* The graph then runs through `onnx.reference.ReferenceEvaluator` to
  produce the oracle tensor.  A reference-implementation gap is
  recorded, not treated as an engine result.
* The vc4 sweep is the same set of graphs through
  `Runner(mode='hybrid', host_policy='device')`, compared with the
  tolerance above.  It was **not run** for this revision.

## Command for the deferred sweep

Run only when the installed package is stable and no a concurrent job is
using the GPU (one process, one arena, one operator at a time):

```sh
cd ~/work/vc4-stack
TMPDIR=$HOME/scratch/tmp python3 infer/tools/op_matrix.py \
    --note "vc4infer $(python3 -c 'import vc4infer;print(vc4infer.__version__)'), clean kernel cache"
```

A subset is `--only Conv,Resize,Softmax`; the no-GPU pass is
`--skip-gpu`.  Every accepted graph runs sequentially; a case that
does not return in `--case-timeout` seconds is recorded as `hang`,
and a lost daemon stops the sweep instead of failing the report.

