#!/usr/bin/env python3
"""Materialise ConstantOfShape weights in an ONNX graph.

The ONNX backend test data under ``.../test/data/light`` keeps the topology of
AlexNet, DenseNet, Inception, ResNet, ShuffleNet, SqueezeNet, VGG and ZFNet but
stores every weight as a ``ConstantOfShape`` node whose shape is an int64
initialiser and whose value is the default, zero.  The engine's static plan has
no implementation for ``ConstantOfShape`` (it is metadata, not a numeric op),
so the graph is refused before a single node is dispatched.

This pass replaces each such node with a zero-filled float32 initialiser of the
same shape and drops the now-unused shape initialisers and inputs.  The
replacement is exact: zero is the value the node would have produced, so both
ONNX Runtime and the engine see the same tensors.  It is a test-corpus
workaround, not something a normal exporter produces.
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def materialise(model_path, out_path):
    m = onnx.load(model_path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    replaced = 0
    keep = []
    for n in g.node:
        if n.op_type != "ConstantOfShape":
            keep.append(n)
            continue
        shape = None
        if n.input and n.input[0] in inits:
            shape = [int(v) for v in numpy_helper.to_array(inits[n.input[0]])]
        if not shape:
            keep.append(n)
            continue
        value = 0.0
        for a in n.attribute:
            if a.name == "value":
                value = float(numpy_helper.to_array(a.t).ravel()[0])
        arr = np.full(shape, value, dtype=np.float32)
        inits[n.output[0]] = numpy_helper.from_array(arr, n.output[0])
        replaced += 1
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input if i}
    used |= {o.name for o in g.output}
    new_init = [inits[name] for name in inits if name in used]
    del g.initializer[:]
    g.initializer.extend(new_init)
    # Drop parameters that existed only to feed a ConstantOfShape shape.
    live_inputs = [i for i in g.input
                   if i.name not in inits or i.name in used]
    del g.input[:]
    g.input.extend(live_inputs)
    # These test graphs are IR v3, where every initialiser must also appear in
    # graph.input.  IR v4 allows initialisers that are not inputs, which is
    # what a materialised weight is; ONNX Runtime still treats an initialiser
    # listed as an input as overridable and does not require a feed for it.
    m.ir_version = max(int(m.ir_version), 4)
    onnx.checker.check_model(m)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(m, out_path)
    return replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    a = ap.parse_args()
    n = materialise(a.model, a.out)
    print("%s -> %s (%d ConstantOfShape nodes materialised)"
          % (a.model, a.out, n))


if __name__ == "__main__":
    sys.exit(main())
