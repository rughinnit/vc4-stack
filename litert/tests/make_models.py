#!/usr/bin/env python3
"""Generate the small .tflite models used by the delegate tests.

The wheel ships the FlatBuffers schema but no TensorFlow converter, so the
models are built directly from ai_edge_litert.schema_py_generated.  That keeps
the test self-contained (no TensorFlow, no network) and makes every weight a
deterministic function of the seed.

Models:
  chain.tflite     conv3x3+relu -> conv1x1 -> softmax       (all claimable)
  mixed.tflite     conv3x3+relu -> avgpool2x2 -> conv1x1 -> softmax
                   (the average pool is left to LiteRT)
  hostonly.tflite  avgpool2x2 -> sin                        (nothing claimable)
"""
import os

import flatbuffers
import numpy as np

from ai_edge_litert import schema_py_generated as s

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


class Builder:
    def __init__(self):
        self.tensors = []
        self.buffers = [s.BufferT()]
        self.ops = []
        self.codes = []
        self.code_index = {}

    def tensor(self, name, dims, typ=s.TensorType.FLOAT32, buf=0):
        t = s.TensorT()
        t.shape = np.array(dims, dtype=np.int32)
        t.type = typ
        t.buffer = buf
        t.name = name.encode()
        t.quantization = s.QuantizationParametersT()
        self.tensors.append(t)
        return len(self.tensors) - 1

    def const(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        b = s.BufferT()
        b.data = np.frombuffer(arr.tobytes(), dtype=np.uint8)
        self.buffers.append(b)
        return self.tensor(name, arr.shape, buf=len(self.buffers) - 1)

    def op(self, builtin, ins, outs, opts_type=s.BuiltinOptions.NONE, opts=None,
           version=1):
        if builtin not in self.code_index:
            self.code_index[builtin] = len(self.codes)
            code = s.OperatorCodeT()
            code.builtinCode = builtin
            code.deprecatedBuiltinCode = builtin
            code.version = version
            self.codes.append(code)
        t = s.OperatorT()
        t.opcodeIndex = self.code_index[builtin]
        t.inputs = np.array(ins, dtype=np.int32)
        t.outputs = np.array(outs, dtype=np.int32)
        t.builtinOptionsType = opts_type
        t.builtinOptions = opts
        self.ops.append(t)

    def save(self, path, ins, outs, name=b"main"):
        sg = s.SubGraphT()
        sg.tensors = self.tensors
        sg.operators = self.ops
        sg.inputs = np.array(ins, dtype=np.int32)
        sg.outputs = np.array(outs, dtype=np.int32)
        sg.name = name
        m = s.ModelT()
        m.version = 3
        m.subgraphs = [sg]
        m.operatorCodes = self.codes
        m.buffers = self.buffers
        m.description = b"vc4 litert delegate test"
        fb = flatbuffers.Builder(4096)
        fb.Finish(m.Pack(fb), file_identifier=b"TFL3")
        with open(path, "wb") as f:
            f.write(bytes(fb.Output()))


def conv_opts(padding, stride=1, activation=s.ActivationFunctionType.NONE):
    o = s.Conv2DOptionsT()
    o.padding = padding
    o.strideW = stride
    o.strideH = stride
    o.dilationWFactor = 1
    o.dilationHFactor = 1
    o.fusedActivationFunction = activation
    return o


def pool_opts(padding, stride=2, k=2):
    o = s.Pool2DOptionsT()
    o.padding = padding
    o.strideW = stride
    o.strideH = stride
    o.filterWidth = k
    o.filterHeight = k
    o.fusedActivationFunction = s.ActivationFunctionType.NONE
    return o


def softmax_opts(beta=1.0):
    o = s.SoftmaxOptionsT()
    o.beta = beta
    return o


def build_chain(path, seed=1):
    rng = np.random.default_rng(seed)
    C = 16
    b = Builder()
    x = b.tensor("input", [1, 8, 8, C])
    w1 = b.const("w1", rng.standard_normal((C, 3, 3, C)) * 0.1)
    b1 = b.const("b1", rng.standard_normal(C).astype(np.float32) * 0.01)
    y1 = b.tensor("conv1", [1, 8, 8, C])
    w2 = b.const("w2", rng.standard_normal((C, 1, 1, C)) * 0.1)
    b2 = b.const("b2", rng.standard_normal(C).astype(np.float32) * 0.01)
    y2 = b.tensor("conv2", [1, 8, 8, C])
    y3 = b.tensor("out", [1, 8, 8, C])
    b.op(s.BuiltinOperator.CONV_2D, [x, w1, b1], [y1],
         s.BuiltinOptions.Conv2DOptions,
         conv_opts(s.Padding.SAME, activation=s.ActivationFunctionType.RELU))
    b.op(s.BuiltinOperator.CONV_2D, [y1, w2, b2], [y2],
         s.BuiltinOptions.Conv2DOptions, conv_opts(s.Padding.VALID))
    b.op(s.BuiltinOperator.SOFTMAX, [y2], [y3],
         s.BuiltinOptions.SoftmaxOptions, softmax_opts())
    b.save(path, [x], [y3])
    return rng.standard_normal((1, 8, 8, C)).astype(np.float32)


def build_mixed(path, seed=2):
    rng = np.random.default_rng(seed)
    C = 16
    b = Builder()
    x = b.tensor("input", [1, 8, 8, C])
    w1 = b.const("w1", rng.standard_normal((C, 3, 3, C)) * 0.1)
    b1 = b.const("b1", rng.standard_normal(C).astype(np.float32) * 0.01)
    y1 = b.tensor("conv1", [1, 8, 8, C])
    y2 = b.tensor("pool", [1, 4, 4, C])
    w2 = b.const("w2", rng.standard_normal((C, 1, 1, C)) * 0.1)
    b2 = b.const("b2", rng.standard_normal(C).astype(np.float32) * 0.01)
    y3 = b.tensor("conv2", [1, 4, 4, C])
    y4 = b.tensor("out", [1, 4, 4, C])
    b.op(s.BuiltinOperator.CONV_2D, [x, w1, b1], [y1],
         s.BuiltinOptions.Conv2DOptions,
         conv_opts(s.Padding.SAME, activation=s.ActivationFunctionType.RELU))
    b.op(s.BuiltinOperator.AVERAGE_POOL_2D, [y1], [y2],
         s.BuiltinOptions.Pool2DOptions, pool_opts(s.Padding.SAME))
    b.op(s.BuiltinOperator.CONV_2D, [y2, w2, b2], [y3],
         s.BuiltinOptions.Conv2DOptions, conv_opts(s.Padding.VALID))
    b.op(s.BuiltinOperator.SOFTMAX, [y3], [y4],
         s.BuiltinOptions.SoftmaxOptions, softmax_opts())
    b.save(path, [x], [y4])
    return rng.standard_normal((1, 8, 8, C)).astype(np.float32)


def build_hostonly(path, seed=3):
    rng = np.random.default_rng(seed)
    C = 16
    b = Builder()
    x = b.tensor("input", [1, 8, 8, C])
    y1 = b.tensor("pool", [1, 4, 4, C])
    y2 = b.tensor("out", [1, 4, 4, C])
    b.op(s.BuiltinOperator.AVERAGE_POOL_2D, [x], [y1],
         s.BuiltinOptions.Pool2DOptions, pool_opts(s.Padding.SAME))
    b.op(s.BuiltinOperator.SIN, [y1], [y2])
    b.save(path, [x], [y2])
    return rng.standard_normal((1, 8, 8, C)).astype(np.float32)


def main():
    os.makedirs(MODELS, exist_ok=True)
    for name, fn in (("chain", build_chain), ("mixed", build_mixed),
                     ("hostonly", build_hostonly)):
        path = os.path.join(MODELS, name + ".tflite")
        x = fn(path)
        np.save(path + ".input.npy", x)
        print("wrote %s (%d bytes, input %s)" %
              (path, os.path.getsize(path), list(x.shape)))


if __name__ == "__main__":
    main()
