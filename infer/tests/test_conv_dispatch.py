#!/usr/bin/env python3
"""test_conv_dispatch.py -- which conv path a shape selects, without a device.

Three refusals on the model corpus were decided by the dispatch classifier
before any kernel ran, and each is cheap to pin here:

  * a dense conv with ``Cin == 1`` is ``group == 1``, which is also
    ``group == Cin``.  The equality was read as depthwise, so
    ``super-resolution-10``'s 1->64 5x5 stem was refused as a "channel
    multiplier".  A dense conv is dense; only ``group > 1`` can be depthwise.
  * a true depthwise conv with a channel multiplier (``group == Cin > 1``,
    ``Cout != Cin``) is still refused, and the message names that constraint.
  * a grouped conv whose per-group blocks are not 16-aligned still refuses for
    the alignment (ShuffleNet's group-4 3x3).

``coverage.conv_kind`` is the single classifier the static report and the
executor both call, so this file also checks that the plan's NHWC layout
decision agrees for the dense single-channel form.

No GPU work, no kernel cache, no ctypes call into libvc4infer.

usage: TMPDIR=$HOME/scratch/tmp python3 tests/test_conv_dispatch.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import coverage                              # noqa: E402
from vc4infer import optimizer as O                        # noqa: E402
from vc4infer.optimizer.plan import conv_nhwc              # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def conv_model(cin, cout, k, group, hw=8, pad=None):
    pad = (k - 1) // 2 if pad is None else pad
    rng = np.random.default_rng(7)
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                      [1, cin, hw, hw])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                      [1, cout, hw, hw])
    w = numpy_helper.from_array(
        rng.uniform(-0.4, 0.4, (cout, cin // group, k, k)).astype(np.float32),
        "W")
    b = numpy_helper.from_array(np.zeros(cout, np.float32), "B")
    node = helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                            kernel_shape=[k, k], pads=[pad] * 4,
                            strides=[1, 1], dilations=[1, 1], group=group)
    g = helper.make_graph([node], "conv", [x], [y], [w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m


def conv_node(m):
    return O.Graph.from_model(m).nodes[0]


def report(m):
    rep = coverage.analyse_model(m)
    return rep["nodes"][0]


def main():
    print("conv dispatch classification (no device)")

    check(coverage.conv_kind(1, 1) == "dense",
          "group 1 with one input channel is dense")
    check(coverage.conv_kind(16, 16) == "depthwise",
          "group == Cin > 1 is depthwise")
    check(coverage.conv_kind(4, 32) == "grouped",
          "a proper group count is grouped")
    check(coverage.conv_kind(1, 16) == "dense",
          "group 1 with many channels is dense")

    # the super-resolution stem: 1 -> 64, 5x5, dense
    m = conv_model(1, 64, 5, 1)
    n = report(m)
    check(n["status"] == "supported",
          "dense 1->64 5x5 is supported (%s)" % n["reason"])
    check("dense" in n["reason"], "the reason says dense (%s)" % n["reason"])
    node = conv_node(m)
    check(conv_nhwc(node, [1, 1, 8, 8], [64, 1, 5, 5], 1),
          "the plan takes the NHWC path for it")

    # a real channel multiplier stays refused, with that constraint named
    m = conv_model(16, 32, 3, 16)
    n = report(m)
    check(n["status"] == "unsupported" and
          "channel multiplier" in n["reason"] and
          "Cout 32 != Cin 16" in n["reason"],
          "depthwise multiplier is refused by name (%s)" % n["reason"])

    # ShuffleNet's group-4 3x3: per-group blocks are 8 channels
    m = conv_model(32, 32, 3, 4)
    n = report(m)
    check(n["status"] == "unsupported" and
          "16-aligned per-group blocks" in n["reason"],
          "group 4 still refuses for the alignment (%s)" % n["reason"])

    # a dense conv with Cin=1 and Cout=1 is the degenerate case of the same
    # classifier and must not become depthwise either
    m = conv_model(1, 1, 3, 1)
    n = report(m)
    check(n["status"] == "supported", "dense 1->1 3x3 is supported")
    check(coverage.conv_kind(1, 1) == "dense",
          "1->1 stays on the dense path")

    print("")
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  - " + f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
