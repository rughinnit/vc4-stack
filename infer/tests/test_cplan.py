#!/usr/bin/env python3
"""test_cplan.py -- the frozen plan, its boundary and its equivalence, no GPU.

A plan is a recording of one frame's device calls, so the property that
matters on the CPU side is: replaying it issues the same calls, in the same
order, with the same buffers and arguments, as the Python executor did, and
the values that come out are equal.  Both halves are checked here against a
stub engine that keeps one bytearray as the arena and computes the small set
of ops the synthetic graphs use, so no ctypes call reaches libvc4infer and no
GPU is touched.

The stub mirrors the C walker's decoding (tests/test_plan.c proves the C side
of the same contract, including the 152-byte record layout).

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_cplan.py
"""
import ctypes
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "infer", "python"))

import vc4infer                                            # noqa: E402
from vc4infer import _binding, _plan                       # noqa: E402
from vc4infer._errors import UnsupportedNode               # noqa: E402
from vc4infer._onnxrun import Runner, Tensor               # noqa: E402

FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def raises(fn, kind, needle=None):
    try:
        fn()
    except kind as e:
        return (needle is None or needle in str(e)), str(e)
    except Exception as e:                                 # noqa: BLE001
        return False, "wrong exception %s: %s" % (type(e).__name__, e)
    return False, "no exception"


# ------------------------------------------------------------ stub engine
class FakeStats:
    def __init__(self, launches=0):
        self.launches = launches
        self.pack_ns = 0
        self.kernel_ns = 0
        self.flops = 0


class FakeLib:
    @staticmethod
    def vc4_plan_op_size():
        return ctypes.sizeof(_binding.PlanOp)


def _fbits(v):
    import struct
    return struct.unpack("<f", struct.pack("<I", int(v) & 0xFFFFFFFF))[0]


class FakeEngine:
    """The slice of the engine API the runner uses, over a host bytearray.

    Elementwise and unary are computed with numpy so the final tensors are
    real values and their equality is a check, not a tautology.  The copies
    are a deterministic model of the layout call: the same arguments move the
    same words, which is all an equivalence test needs (the physical oracle
    is tests/test_copyops.c on silicon).  ``plan_run`` decodes a PlanOp array
    exactly as src/vc4_plan.c does and funnels it through the same methods,
    so a record and its Python call produce the same entry in ``calls``.
    """

    plan_supported = True
    lib = FakeLib()

    def __init__(self, arena=8 << 20):
        self._arena = bytearray(arena)
        self._next = 64
        self.calls = []
        self._stats = FakeStats()

    # -- arena -------------------------------------------------------------
    @property
    def arena_bytes(self):
        return len(self._arena)

    @property
    def stats(self):
        # a fresh snapshot, the way vc4_get_stats fills a struct: the runner
        # subtracts one reading from the next
        return FakeStats(self._stats.launches)

    def alloc(self, nbytes):
        n = (int(nbytes) + 63) // 64 * 64
        if self._next + n > len(self._arena):
            raise vc4infer.Error("fake arena exhausted")
        b = vc4infer.Buf(self._next, int(nbytes))
        # every buffer starts zeroed, so a frame cannot depend on what the
        # last frame left in a recycled block: with a stub engine whose conv
        # writes nothing, stale bytes would otherwise make build and replay
        # take different data-dependent branches
        self._arena[self._next:self._next + n] = bytes(n)
        self._next += n
        return b

    def upload(self, arr, buf=None):
        if isinstance(arr, np.ndarray):
            data = np.ascontiguousarray(arr, dtype=np.float32).ravel().tobytes()
        else:
            data = bytes(arr)
        if buf is None:
            buf = self.alloc(len(data))
        self._arena[int(buf.off):int(buf.off) + len(data)] = data
        return buf

    def download(self, buf, nbytes):
        raw = bytes(self._arena[int(buf.off):int(buf.off) + int(nbytes)])
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _floats(self, off, n):
        return np.frombuffer(self._arena, dtype=np.float32,
                             count=int(n), offset=int(off))

    def _zero(self, buf):
        if buf is not None:
            n = int(buf.bytes)
            self._arena[int(buf.off):int(buf.off) + n] = bytes(n)

    def _call(self, kind, *args):
        self.calls.append((kind,) + args)
        self._stats.launches += 1

    # -- compute -----------------------------------------------------------
    def elementwise(self, op, a, out, n, b=None):
        x = self._floats(a.off, n)
        if op == _binding.OP_ADD:
            y = x + self._floats(b.off, n)
        elif op == _binding.OP_MUL:
            y = x * self._floats(b.off, n)
        elif op == _binding.OP_RELU:
            y = np.maximum(x, 0.0)
        elif op == _binding.OP_SIGMOID:
            y = 1.0 / (1.0 + np.exp(-x))
        elif op == _binding.OP_SILU:
            y = x / (1.0 + np.exp(-x))
        else:
            y = x
        self._floats(out.off, n)[:] = np.asarray(y, np.float32)
        self._call("elementwise", int(op))

    def unary(self, op, a, out, n, p0=0.0, p1=0.0, p2=0.0):
        x = self._floats(a.off, n)
        if op == _binding.UOP_ABS:
            y = np.abs(x)
        elif op == _binding.UOP_NEG:
            y = -x
        elif op == _binding.UOP_AFFINE:
            y = x * np.float32(p0) + np.float32(p1)
        elif op == _binding.UOP_HARDSWISH:
            y = x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)
        else:
            y = x
        self._floats(out.off, n)[:] = np.asarray(y, np.float32)
        self._call("unary", int(op))

    def _copy(self, src, dst, count):
        src, dst = int(src), int(dst)
        if src < 0 or dst < 0:
            return
        lo = max(0, min(int(count), (len(self._arena) - src) // 4,
                        (len(self._arena) - dst) // 4))
        if lo > 0:
            self._arena[dst:dst + lo * 4] = self._arena[src:src + lo * 4]

    def copy2d_x(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane=16, nblk=1, sbstep=64, dbstep=64, units=4, depth=16):
        n = min(int(nlane), int(depth), 16)
        for r in range(int(rows)):
            for v in range(int(vpr)):
                for b in range(int(nblk)):
                    self._copy(int(a.off) + r * int(srow) + v * int(svec)
                               + b * int(sbstep),
                               int(out.off) + r * int(drow) + v * int(dvec)
                               + b * int(dbstep), n)
        self._call("copy2d_x", int(rows), int(vpr), int(nlane), int(nblk))

    def copy2d(self, a, out, rows, vpr, srow, drow, svec, dvec, slane):
        self.copy2d_x(a, out, rows, vpr, srow, drow, svec, dvec, slane,
                      16, 1, 64, 64, 4, 16)

    def copy2d_n(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane, units, depth):
        self.copy2d_x(a, out, rows, vpr, srow, drow, svec, dvec, slane,
                      nlane, 1, 64, 64, units, depth)

    def fill2d(self, out, rows, vpr, drow, dvec, value=0.0):
        for r in range(int(rows)):
            for v in range(int(vpr)):
                self._floats(int(out.off) + r * int(drow) + v * int(dvec),
                             16)[:] = value
        self._call("fill2d", float(value))

    def blit(self, a, out, nbytes):
        self._copy(int(a.off), int(out.off), int(nbytes) // 4)
        self._call("blit", int(nbytes))

    def gather(self, tab, out, n, fill=0.0):
        idx = np.frombuffer(self._arena, dtype=np.int32, count=int(n),
                            offset=int(tab.off))
        for i, off in enumerate(idx):
            if off < 0:
                self._floats(int(out.off) + i * 4, 1)[:] = fill
            else:
                # the table holds absolute arena byte offsets
                self._floats(int(out.off) + i * 4, 1)[:] = \
                    self._floats(int(off), 1)
        self._call("gather", int(n))

    def conv_nhwc_wpanel_bytes(self, Cin, Cout, KH, KW, bias=True):
        return 4096

    def conv_nhwc_pack_w(self, wt, Cin, Cout, KH, KW, bias, wpanel):
        self._call("pack_w", int(Cin), int(Cout))

    def conv_nhwc_pads(self, x, nimg, Cin, H, W, in_pitch, wpanel, Cout, KH,
                       KW, stride, pt, pl, pb, pr, dil, out, act=0):
        ho = (H + pt + pb - dil * (KH - 1) - 1) // stride + 1
        wo = (W + pl + pr - dil * (KW - 1) - 1) // stride + 1
        self._zero(out)
        self._call("conv_nhwc_pads", int(Cin), int(Cout))
        return ho, wo, (wo + 15) // 16 * 16

    def conv_nhwc_dil(self, x, nimg, Cin, H, W, in_pitch, wpanel, Cout, KH,
                      KW, stride, pad, dil, out, act=0):
        ho = (H + 2 * pad - dil * (KH - 1) - 1) // stride + 1
        wo = (W + 2 * pad - dil * (KW - 1) - 1) // stride + 1
        self._zero(out)
        self._call("conv_nhwc_dil", int(Cin), int(Cout))
        return ho, wo, (wo + 15) // 16 * 16

    def conv_nhwc_group(self, x, nimg, Cin, H, W, in_pitch, wpanels, Cout,
                        groups, KH, KW, stride, pad, dil, out, act=0):
        ho = (H + 2 * pad - dil * (KH - 1) - 1) // stride + 1
        wo = (W + 2 * pad - dil * (KW - 1) - 1) // stride + 1
        self._zero(out)
        self._call("conv_nhwc_group", int(Cin), int(Cout))
        return ho, wo, (wo + 15) // 16 * 16

    def conv2d(self, x, w, out, Cin, H, W, Cout, KH, KW, bias=None,
               stride=1, pad=1):
        ho = (H + 2 * pad - KH) // stride + 1
        wo = (W + 2 * pad - KW) // stride + 1
        self._zero(out)
        self._call("conv2d", int(Cin), int(Cout))
        return ho, wo

    def maxpool_nhwc(self, x, nimg, C, H, W, in_pitch, k, pad, out):
        ho = H + 2 * pad - k + 1
        wo = W + 2 * pad - k + 1
        self._zero(out)
        self._call("maxpool_nhwc", int(C))
        return ho, wo, (wo + 15) // 16 * 16

    def maxpooln(self, x, out, C, H, W, n, stride, pad):
        ho = (H + 2 * pad - n) // stride + 1
        wo = (W + 2 * pad - n) // stride + 1
        self._zero(out)
        self._call("maxpooln", int(C))
        return ho, wo

    def upsample2x_nhwc(self, x, nimg, C, H, W, in_pitch, out):
        self._zero(out)
        self._call("upsample2x_nhwc", int(C))
        return H * 2, W * 2, (W * 2 + 15) // 16 * 16

    def softmax_fwd(self, x, out, rows, cols):
        self._zero(out)
        self._call("softmax_fwd", int(rows))

    def softmax(self, x, out, rows, cols):
        self._zero(out)
        self._call("softmax", int(rows))

    def matmul(self, a, b, c, M, N, K, layout=0, act=0):
        x = self._floats(a.off, M * K).reshape(M, K)
        w = self._floats(b.off, K * N).reshape(K, N)
        y = x @ w
        if act == _binding.ACT_RELU:
            y = np.maximum(y, 0.0)
        self._floats(c.off, M * N)[:] = y.astype(np.float32).ravel()
        self._call("matmul", int(M), int(N), int(K))

    # -- the C walker's decoding -------------------------------------------
    def plan_run(self, ops):
        for op in ops:
            k = op.kind
            if k == _binding.PLAN_ALIAS:
                continue
            a = vc4infer.Buf(op.a_off, op.a_bytes)
            out = vc4infer.Buf(op.c_off, op.c_bytes)
            b = (vc4infer.Buf(op.b_off, op.b_bytes)
                 if k == _binding.PLAN_EW and op.p[1] else None)
            p = [int(v) for v in op.p]
            if k == _binding.PLAN_EW:
                self.elementwise(p[0], a, out, op.n, b=b)
            elif k == _binding.PLAN_UNARY:
                self.unary(p[0], a, out, op.n, _fbits(p[1]), _fbits(p[2]),
                           _fbits(p[3]))
            elif k == _binding.PLAN_COPY2D_X:
                self.copy2d_x(a, out, *p[:13])
            elif k == _binding.PLAN_COPY2D_N:
                self.copy2d_n(a, out, *p[:10])
            elif k == _binding.PLAN_COPY2D:
                self.copy2d(a, out, *p[:7])
            elif k == _binding.PLAN_FILL2D:
                self.fill2d(out, p[0], p[1], p[2], p[3], _fbits(p[4]))
            elif k == _binding.PLAN_GATHER:
                self.gather(a, out, op.n, _fbits(p[0]))
            elif k == _binding.PLAN_BLIT:
                self.blit(a, out, op.n)
            else:
                raise AssertionError("stub walker: kind %d" % k)


# ------------------------------------------------------------------ models
def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _model(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def save(m, name):
    d = tempfile.mkdtemp(prefix="cplan.")
    p = os.path.join(d, name)
    onnx.save(m, p)
    return p


def mk_elementwise():
    c = numpy_helper.from_array(np.asarray(2.0, np.float32), "c")
    n = [helper.make_node("Mul", ["X", "c"], ["m"]),
         helper.make_node("Add", ["m", "c"], ["a"]),
         helper.make_node("Relu", ["a"], ["r"]),
         helper.make_node("Abs", ["r"], ["Y"])]
    return _model(n, [_vi("X", [1, 4, 8, 8])], [_vi("Y", [1, 4, 8, 8])], [c])


def mk_fallback():
    """MatMul is not representable; the Relu after it is."""
    w = numpy_helper.from_array(
        np.random.default_rng(3).standard_normal((8, 6)).astype(np.float32),
        "W")
    n = [helper.make_node("MatMul", ["X", "W"], ["mm"]),
         helper.make_node("Relu", ["mm"], ["Y"])]
    return _model(n, [_vi("X", [4, 8])], [_vi("Y", [4, 6])], [w])


def mk_alias():
    """Reshape on an NCHW tensor is a view: zero calls, an ALIAS step."""
    shp = numpy_helper.from_array(np.asarray([1, 64], np.int64), "shp")
    n = [helper.make_node("Reshape", ["X", "shp"], ["r"]),
         helper.make_node("Relu", ["r"], ["Y"])]
    return _model(n, [_vi("X", [1, 16, 2, 2])], [_vi("Y", [1, 64])], [shp])


def mk_copies():
    """A conv output (NHWC) through a channel concat and a split.

    This is the shape of the copy work that dominates the launch count: the
    concat and the split become COPY2D_X records, the conv stays Python, and
    the two activations after the split are C steps again.
    """
    w = numpy_helper.from_array(np.zeros((16, 16, 3, 3), np.float32), "W")
    sp = numpy_helper.from_array(np.asarray([16, 16], np.int64), "sp")
    n = [helper.make_node("Conv", ["X", "W"], ["c"], pads=[1, 1, 1, 1],
                          strides=[1, 1], kernel_shape=[3, 3]),
         helper.make_node("Concat", ["c", "c"], ["cat"], axis=1),
         helper.make_node("Split", ["cat", "sp"], ["s0", "s1"], axis=1),
         helper.make_node("Relu", ["s0"], ["Y0"]),
         helper.make_node("Abs", ["s1"], ["Y1"])]
    return _model(n, [_vi("X", [1, 16, 8, 8])],
                  [_vi("Y0", [1, 16, 8, 8]), _vi("Y1", [1, 16, 8, 8])],
                  [w, sp])


def mk_host_transfer():
    """Transpose goes through host numpy, so it cannot be a C step."""
    n = [helper.make_node("Transpose", ["X"], ["t"], perm=[0, 1, 3, 2]),
         helper.make_node("Relu", ["t"], ["Y"])]
    return _model(n, [_vi("X", [1, 2, 4, 4])], [_vi("Y", [1, 2, 4, 4])])


# ------------------------------------------------------------------- tests
def run_pair(model, feeds, passes="none"):
    """Steady-state Python frames vs the plan's replay frame.

    The comparison is frame 2, not frame 1, on both sides: the first frame of
    any run also packs weight panels, and those are plan-time data that a
    later frame does not repeat.  Taking steady-state frames on both sides is
    the like-for-like comparison.
    """
    path = save(model, "m.onnx")
    old = os.environ.get("VC4_NO_CPLAN")
    os.environ["VC4_NO_CPLAN"] = "1"
    try:
        eng = FakeEngine()
        r = Runner(path, eng, passes=passes)
        r.run(feeds)
        mark = len(eng.calls)
        out_py = r.run(feeds)                 # steady-state Python frame
        calls_py = list(eng.calls[mark:])
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old
    eng2 = FakeEngine()
    r2 = Runner(path, eng2, passes=passes)
    out_build = r2.run(feeds)                 # the build frame
    calls_build = list(eng2.calls)
    out_plan = r2.run(feeds)                  # the replay frame
    calls_plan = list(eng2.calls[len(calls_build):])
    return (calls_py, out_py, calls_build, calls_plan, out_plan, r2)


def test_elementwise():
    print("elementwise chain")
    x = np.random.default_rng(1).standard_normal((1, 4, 8, 8)).astype(np.float32)
    calls_py, out_py, calls_build, calls_plan, out_plan, r = run_pair(
        mk_elementwise(), {"X": x}, passes="ew")
    check(r.cplan is not None, "a plan was built")
    check(calls_build == calls_py, "the build frame issues the same calls")
    check(calls_plan == calls_py,
          "replay issues the same calls (%d)" % len(calls_py))
    check(np.array_equal(out_py[0], out_plan[0]),
          "replay output is byte-identical")
    check(r.cplan.summary()["py_nodes"] == 0,
          "every node is a C step: %s" % r.cplan.summary())
    # the affine pair folds to one AFFINE unary under the ew pass
    check(any(c[0] == "unary" for c in calls_py),
          "the ew pass produced a unary launch")


def test_fallback():
    print("fallback boundary")
    x = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float32)
    calls_py, out_py, calls_build, calls_plan, out_plan, r = run_pair(
        mk_fallback(), {"X": x})
    sm = r.cplan.summary()
    check(sm["py_nodes"] == 1, "one Python step (the MatMul): %s" % sm)
    check(sm["c_nodes"] == 1, "one C node (the Relu): %s" % sm)
    check(calls_plan == calls_py, "calls match across the boundary")
    check(np.array_equal(out_py[0], out_plan[0]), "output matches")
    kinds = [k for k, _ in r.cplan.steps]
    check(kinds == ["p", "c"], "the plan yields before the Python node: %s"
          % kinds)


def test_alias():
    print("zero-call node becomes an alias step")
    x = np.random.default_rng(4).standard_normal((1, 16, 2, 2))
    calls_py, out_py, calls_build, calls_plan, out_plan, r = run_pair(
        mk_alias(), {"X": x.astype(np.float32)})
    sm = r.cplan.summary()
    check(sm["c_ops"] == 1, "one device call for two nodes: %s" % sm)
    check(calls_plan == calls_py, "calls match")
    check(np.array_equal(out_py[0], out_plan[0]), "output matches")
    check("r" in r.cplan.desc and
          r.cplan.steps[0][1][0].a_off == r.cplan.desc["r"][0],
          "the Relu record reads the buffer the Reshape named")


def test_copies():
    print("concat and split are copy records")
    x = np.random.default_rng(10).standard_normal((1, 16, 8, 8))
    calls_py, out_py, calls_build, calls_plan, out_plan, r = run_pair(
        mk_copies(), {"X": x.astype(np.float32)})
    sm = r.cplan.summary()
    check(sm["py_nodes"] == 1, "only the conv is Python: %s" % sm)
    check(sm["c_nodes"] == 4, "concat, split and two activations are C: %s"
          % sm)
    recs = [op for kind, step in r.cplan.steps if kind == "c"
            for op in step if op.kind == _binding.PLAN_COPY2D_X]
    check(len(recs) == 4,
          "the concat and split are four COPY2D_X records (%d)" % len(recs))
    check(calls_plan == calls_py, "calls match")
    check(all(np.array_equal(a, b) for a, b in zip(out_py, out_plan)),
          "both outputs match")


def test_host_transfer():
    print("host-transfer node stays in Python")
    x = np.random.default_rng(5).standard_normal((1, 2, 4, 4))
    calls_py, out_py, calls_build, calls_plan, out_plan, r = run_pair(
        mk_host_transfer(), {"X": x.astype(np.float32)})
    sm = r.cplan.summary()
    check(sm["py_nodes"] == 1, "Transpose is the Python step: %s" % sm)
    check(calls_plan == calls_py, "calls match")
    check(np.array_equal(out_py[0], out_plan[0]), "output matches")


def test_profile_after_build():
    print("profile run after a plan exists")
    x = np.random.default_rng(6).standard_normal((1, 4, 8, 8)).astype(np.float32)
    path = save(mk_elementwise(), "m.onnx")
    eng = FakeEngine()
    r = Runner(path, eng, passes="none")
    r.run({"X": x})
    check(r.cplan is not None, "built")
    before = len(eng.calls)
    out = r.run({"X": x}, profile=True)
    check(len(eng.calls) > before, "the profile run executes the graph again")
    check(len(r.profile) == len(r.graph.node),
          "one profile entry per node (%d)" % len(r.profile))
    check(r.profile[-1]["launches"] >= 1, "launches are attributed")
    out2 = r.run({"X": x})
    check(np.array_equal(out[0], out2[0]),
          "a plan run after a profile run still matches")


def test_feed_guard():
    print("feed guard")
    x = np.random.default_rng(7).standard_normal((1, 4, 8, 8)).astype(np.float32)
    path = save(mk_elementwise(), "m.onnx")
    eng = FakeEngine()
    r = Runner(path, eng, passes="none")
    r.run({"X": x})
    ok, msg = raises(lambda: r.run({"X": np.zeros((1, 4, 4, 8), np.float32)}),
                     UnsupportedNode, "frozen plan")
    check(ok, "a different feed shape is refused with a plan message")


def test_disable():
    print("VC4_NO_CPLAN")
    x = np.random.default_rng(8).standard_normal((1, 4, 8, 8)).astype(np.float32)
    path = save(mk_elementwise(), "m.onnx")
    old = os.environ.get("VC4_NO_CPLAN")
    os.environ["VC4_NO_CPLAN"] = "1"
    try:
        eng = FakeEngine()
        r = Runner(path, eng, passes="none")
        r.run({"X": x})
        r.run({"X": x})
        check(r.cplan is None, "no plan is built")
        check(getattr(r.pool, "pin", None) is None, "the pool stays unpinned")
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old


def test_convert_copies():
    print("copy calls convert to records")
    a = _binding.Buf(0x1000, 0x4000)
    out = _binding.Buf(0x8000, 0x8000)
    tr = _plan.Trace(FakeEngine())
    tr.copy2d_x(a, out, 8, 4, 64, 128, 64, 128, 4, 16, 2, 64, 64, 4, 16)
    conv = _plan._convert(tr.calls[0])
    check(conv is not None, "copy2d_x is representable")
    check(conv[0] == _binding.PLAN_COPY2D_X, "it becomes a COPY2D_X record")
    rec = _plan._record(conv)
    check(rec.kind == _binding.PLAN_COPY2D_X and rec.p[8] == 2 and
          rec.p[12] == 16 and rec.c_off == 0x8000 and rec.a_off == 0x1000,
          "record fields survive the conversion")
    # a call that is not on the list is not representable
    tr2 = _plan.Trace(FakeEngine())
    listing = [("other", "conv_nhwc_pads")]
    check(_plan._convert(listing[0]) is None,
          "an unlisted entry point is refused")


def test_launch_count():
    print("launch count is unchanged")
    x = np.random.default_rng(9).standard_normal((1, 4, 8, 8)).astype(np.float32)
    path = save(mk_elementwise(), "m.onnx")
    eng = FakeEngine()
    r = Runner(path, eng, passes="ew")
    r.run({"X": x})
    n0 = eng.stats.launches
    r.run({"X": x})
    n1 = eng.stats.launches - n0
    check(n0 == n1, "build and replay launch the same kernels (%d)" % n0)


def test_fallback_midprocess():
    """A pin mismatch on a later frame: drop once, continue in Python.

    The failure a handler with a data-dependent allocation would cause is
    simulated by handing the pinned pool a recorded size that no longer
    matches.  What is checked is the contract: the frame is re-run from the
    feeds and is correct, the plan is not rebuilt, and later frames keep
    working from the same arena.
    """
    print("fallback after the plan is in use")
    x = np.random.default_rng(11).standard_normal((4, 8)).astype(np.float32)
    path = save(mk_fallback(), "m.onnx")
    eng = FakeEngine()
    r = Runner(path, eng, passes="none")
    first = r.run({"X": x})                   # build
    replay = r.run({"X": x})                  # replay
    check(r.cplan is not None, "the plan is in use")
    check(np.array_equal(first[0], replay[0]), "the replay frame is correct")
    key = sorted(k for k in r.pool.pin if k.startswith("n0:"))[0]
    seg, off, nbytes = r.pool.pin[key]
    r.pool.pin[key] = (seg, off, nbytes + 64)  # the tripwire
    free_before = list(r.pool.free)
    segs_before = len(r.pool.segs)
    out = r.run({"X": x})                     # trips, drops, re-runs
    check(r.plan_error is not None and "n0:" in r.plan_error,
          "the mismatch is recorded: %s" % r.plan_error)
    check(r.cplan is None, "the plan is gone")
    check(r._cplan_enabled is False, "no rebuild is attempted")
    check(np.array_equal(first[0], out[0]),
          "the tripping frame returns the Python answer")
    check(r.pool.free and
          sum(sz for _o, sz in r.pool.free) ==
          sum(sz for _o, sz in free_before),
          "the allocator keeps its free bytes (%d of %d)"
          % (sum(sz for _o, sz in r.pool.free),
             sum(sz for _o, sz in free_before)))
    for _ in range(20):
        out = r.run({"X": x})
    check(np.array_equal(first[0], out[0]), "later frames stay correct")
    check(len(r.pool.segs) == segs_before,
          "no extra pool segment over 20 frames (%d)" % len(r.pool.segs))


def test_yolo_structure():
    """The real 160 graph: the plan builds, replays and issues the same calls.

    Byte equality on the real graph needs the QPUs (the contract is recorded
    in docs/CPU-BREAKDOWN.md); with a stub engine what is checked is the part
    the stub can answer: every node is classified, every Python step is a
    node the stub really cannot express, and the steady-state calls of the
    Python path and the replay are the same list.
    """
    print("yolov8n_160 structure")
    path = os.path.join(REPO, "infer", "packaging", "vc4infer", "data",
                        "models", "yolov8n_160.onnx")
    if not os.path.isfile(path):
        print("  SKIP: %s is not present" % path)
        return
    feed = {"images": np.zeros((1, 3, 160, 160), np.float32)}
    old = os.environ.get("VC4_NO_CPLAN")
    os.environ["VC4_NO_CPLAN"] = "1"
    try:
        eng = FakeEngine(arena=96 << 20)
        r = Runner(path, eng, passes="all")
        r.run(feed)
        mark = len(eng.calls)
        r.run(feed)
        calls_py = list(eng.calls[mark:])
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old
    eng2 = FakeEngine(arena=96 << 20)
    r2 = Runner(path, eng2, passes="all")
    r2.run(feed)
    mark = len(eng2.calls)
    r2.run(feed)
    calls_plan = list(eng2.calls[mark:])
    check(r2.cplan is not None, "the plan survives replay (no pin mismatch)")
    if r2.cplan is None:
        return
    sm = r2.cplan.summary()
    py_ops = {r2.graph.node[i].op_type for i in sm["py"]}
    check(py_ops <= {"Conv", "Constant", "MaxPool", "Resize", "Sub",
                     "Softmax", "Gemm"},
          "the Python steps are the op classes the plan cannot express: %s"
          % sorted(py_ops))
    check(sm["c_nodes"] >= 40,
          "the copy and elementwise nodes are records (%d)" % sm["c_nodes"])
    kinds = {r2.graph.node[i].op_type for i, n in enumerate(r2.graph.node)
             if i not in r2.cplan.py_nodes and i not in r2._skip
             for _ in [0]}
    check({"Concat", "Split"} <= kinds,
          "concat and split are C steps on the real graph")
    check(calls_plan == calls_py,
          "steady-state Python and replay issue the same %d calls"
          % len(calls_py))


def test_abi():
    print("record ABI")
    lib = _binding.plan_op_size()
    if lib is None:
        print("  SKIP: the loaded libvc4infer has no vc4_plan_op_size")
        return
    check(lib == ctypes.sizeof(_binding.PlanOp),
          "libvc4infer and _binding agree on vc4_plan_op (%d B)" % lib)


def main():
    test_convert_copies()
    test_elementwise()
    test_fallback()
    test_alias()
    test_copies()
    test_host_transfer()
    test_profile_after_build()
    test_feed_guard()
    test_disable()
    test_launch_count()
    test_fallback_midprocess()
    test_yolo_structure()
    test_abi()
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("OK: 13 groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
