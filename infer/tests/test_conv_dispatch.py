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

The third defect is one layer below the classifier: a 1x1 conv to 1000 output
channels pads Cout to 1008, computes, and slices the real 1000 channels back
with ``copy2d_x``.  A copy2d store is at most 16 lanes wide and 16 words deep,
and ``_slice_ch`` passed the whole remainder (1000) as both.  The tests below
drive ``_slice_ch`` with a fake engine that enforces the C library's limits and
check the block/tail arithmetic, so the trigger is pinned to the unaligned
channel count rather than to the 13x13 extent.

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

from vc4infer import _binding as vc4infer                  # noqa: E402
from vc4infer import coverage                              # noqa: E402
from vc4infer import optimizer as O                        # noqa: E402
from vc4infer._onnxrun import Runner, Tensor               # noqa: E402
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


class FakePool:
    def __init__(self):
        self.off = 4096

    def alloc(self, n):
        b = vc4infer.Buf(self.off, int(n))
        self.off += (int(n) + 63) // 64 * 64
        return b

    def release(self, buf):
        pass


class RecordingCopy2d:
    """The C library's copy2d_x contract, with every call recorded."""

    def __init__(self):
        self.calls = []

    def copy2d_x(self, src, dst, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane, nblk, sbstep, dbstep, units, depth):
        if depth < 1 or depth > 16 or nlane > 16 or not slane:
            raise ValueError("copy2d_x: invalid argument")
        call = dict(rows=rows, vpr=vpr, srow=srow, drow=drow, svec=svec,
                    dvec=dvec, slane=slane, nlane=nlane, nblk=nblk,
                    sbstep=sbstep, dbstep=dbstep, units=units, depth=depth,
                    src_off=src.off, src_bytes=src.bytes,
                    dst_off=dst.off, dst_bytes=dst.bytes)
        # the extent checks from src/vc4_gen.c copy2d_run
        s_ext = ((rows - 1) * srow + (nblk - 1) * sbstep + (vpr - 1) * svec
                 + (nlane - 1) * slane + 4)
        d_ext = ((rows - 1) * drow + (nblk - 1) * dbstep + (vpr - 1) * dvec
                 + depth * 4)
        if s_ext > src.bytes:
            raise ValueError("source extent %d > buffer %d"
                             % (s_ext, src.bytes))
        if d_ext > dst.bytes:
            raise ValueError("dest extent %d > buffer %d"
                             % (d_ext, dst.bytes))
        self.calls.append(call)


def slice_harness():
    r = object.__new__(Runner)
    eng = RecordingCopy2d()
    r.eng = eng
    r.pool = FakePool()
    r._garbage = []
    r._tmp = lambda t: t
    return r, eng


def slice_of(cn, c=1008, h=13, w=13):
    r, eng = slice_harness()
    pitch = 16
    x = Tensor(vc4infer.Buf(0, h * pitch * c * 4 + 64), (1, c, h, w),
               "nhwc", pitch)
    out = r._slice_ch(x, 0, cn)
    return eng.calls, out


def test_channel_slice():
    print("channel slice arithmetic (16 lanes x 16 words per copy)")

    # the trigger: the old single call passed cn as both nlane and depth
    _, eng = slice_harness()
    try:
        eng.copy2d_x(vc4infer.Buf(0, 1 << 20), vc4infer.Buf(0, 1 << 20),
                     13, 13, 0, 0, 0, 0, 4, 1000, 1, 64, 64, 1, 1000)
        check(False, "a 1000-lane, 1000-word store is rejected")
    except ValueError as e:
        check("invalid argument" in str(e),
              "1000 lanes x 1000 words is invalid argument (%s)" % e)

    calls, out = slice_of(1000)
    check(len(calls) == 2, "1000 channels is a block call and a tail (%d)"
          % len(calls))
    head, tail = calls
    check(head["nlane"] == 16 and head["depth"] == 16 and
          head["nblk"] == 62 and head["sbstep"] == 64,
          "62 aligned 16-channel blocks (%d x %d lanes, %d blocks)"
          % (head["depth"], head["nlane"], head["nblk"]))
    check(tail["nlane"] == 8 and tail["depth"] == 8 and tail["nblk"] == 1,
          "the 8-channel remainder is one 8-lane call (%d x %d)"
          % (tail["depth"], tail["nlane"]))
    check(tail["src_off"] == head["src_off"] + 62 * 16 * 4 and
          tail["dst_off"] == head["dst_off"] + 62 * 16 * 4,
          "the tail starts 62 blocks into the channel range")
    check(out.shape == (1, 1000, 13, 13) and out.pitch == 13,
          "the slice keeps 1000 channels at pitch 13 (%s, %d)"
          % (out.shape, out.pitch))

    # the 13x13 extent is not the trigger: an 8x8 and a 3x1 slice split the
    # same way, and Cout=1008 needs no slice at all
    for h, w in ((8, 8), (3, 1)):
        calls, _ = slice_of(1000, h=h, w=w)
        check([c["nblk"] for c in calls] == [62, 1] and
              [c["depth"] for c in calls] == [16, 8],
              "%dx%d: same 62 blocks + 8-word tail" % (h, w))
    calls, _ = slice_of(1008)
    check(len(calls) == 1 and calls[0]["nblk"] == 63 and
          calls[0]["depth"] == 16,
          "Cout=1008 is one aligned call, no slice in the first place")

    # the DFL head's 1-channel conv is the old single-call form, still valid
    calls, _ = slice_of(8, c=16, w=4, h=4)
    check(len(calls) == 1 and calls[0]["nlane"] == 8 and
          calls[0]["depth"] == 8,
          "an 8-channel slice is one 8-lane call")

    # a remainder of one channel still needs a one-word store
    calls, _ = slice_of(17, c=32, w=4, h=4)
    check(len(calls) == 2 and calls[1]["nlane"] == 1 and
          calls[1]["depth"] == 1,
          "a 1-channel remainder is a 1-lane, 1-word store")


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

    test_channel_slice()

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
