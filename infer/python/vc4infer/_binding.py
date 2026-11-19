#!/usr/bin/env python3
"""ctypes binding for libvc4infer.

Same API surface as the C library; the arena is addressed with integer
offsets, so a caller can either keep raw bytes or hand a numpy array through
frombuffer() on the uncached host pointer.

    import vc4infer, numpy as np
    eng = vc4infer.Engine()
    a = eng.upload(np.random.rand(128,128).astype(np.float32))
    b = eng.upload(np.random.rand(128,128).astype(np.float32))
    c = eng.alloc(128*128*4)
    eng.matmul(a, b, c, 128, 128, 128)
    out = eng.download(c, 128*128*4).reshape(128,128)

Strict dispatch is preserved: an unsupported op raises Unsupported (the C
library returns VC4_ERR_UNSUPPORTED and never falls back to the CPU).
"""
import ctypes
import os
import sys

try:                       # numpy is convenient, not required
    import numpy as np
except ImportError:        # pragma: no cover
    np = None

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_lib():
    """Locate libvc4infer: $VC4INFER_LIB, the in-tree build, then the system.

    The package installs the library as /usr/lib/<triplet>/libvc4infer.so.1;
    an in-tree checkout keeps it in infer/build.  The first existing candidate
    wins, so a developer build is never shadowed by an installed one.
    """
    env = os.environ.get("VC4INFER_LIB")
    if env:
        return env
    cands = [os.path.join(_HERE, "libvc4infer.so"),
             os.path.join(_HERE, "..", "..", "build", "libvc4infer.so"),
             "/usr/lib/aarch64-linux-gnu/libvc4infer.so.1",
             "/usr/lib/libvc4infer.so.1",
             "/usr/local/lib/libvc4infer.so"]
    for c in cands:
        if os.path.isfile(c):
            return os.path.realpath(c)
    try:
        import ctypes.util
        found = ctypes.util.find_library("vc4infer")
        if found:
            return found
    except Exception:
        pass
    raise ImportError("libvc4infer.so not found (tried %s); set VC4INFER_LIB"
                      % ", ".join(cands))


_LIB = _find_lib()

VC4_OK = 0
ERR_INVAL, ERR_NOMEM, ERR_IO, ERR_TIMEOUT = -1, -2, -3, -4
ERR_UNSUPPORTED = -5
ERR_GPU, ERR_KERNEL = -6, -7

_STATUS = {
    ERR_INVAL: "invalid argument",
    ERR_NOMEM: "out of memory: no arena (another process holds it, or CMA "
               "is exhausted)",
    ERR_IO: "I/O error: vc4d socket or /dev/mem",
    ERR_TIMEOUT: "QPU timeout; the context is dead",
    ERR_UNSUPPORTED: "unsupported operation",
    ERR_GPU: "V3D/queue error after a launch",
    ERR_KERNEL: "QPU kernel blob missing and generation failed",
}

OP_MATMUL, OP_CONV2D, OP_ADD, OP_MUL, OP_RELU, OP_SIGMOID, OP_SILU = range(7)
LAYOUT_ROW_MAJOR = 0

# vc4_uop (mirrors infer/include/vc4infer.h)
(UOP_ABS, UOP_NEG, UOP_IDENTITY, UOP_CLIP, UOP_LEAKYRELU, UOP_FLOOR,
 UOP_CEIL, UOP_ROUND, UOP_SIGN, UOP_EXP, UOP_LOG, UOP_SQRT, UOP_RECIPROCAL,
 UOP_TANH, UOP_ERF, UOP_ELU, UOP_SELU, UOP_SOFTPLUS, UOP_SOFTSIGN, UOP_GELU,
 UOP_GELU_ERF, UOP_HARDSIGMOID, UOP_HARDSWISH, UOP_SIN, UOP_COS) = range(25)
SEL_WHERE, SEL_CLIP3, SEL_PRELU = range(3)
RED_SUM, RED_MEAN, RED_MAX, RED_MIN, RED_ARGMAX, RED_ARGMIN = range(6)


class Buf(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint32), ("bytes", ctypes.c_size_t)]


class OpInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("backend", ctypes.c_char_p),
                ("supported", ctypes.c_int), ("note", ctypes.c_char_p)]


class Stats(ctypes.Structure):
    _fields_ = [("launches", ctypes.c_uint64), ("pack_ns", ctypes.c_uint64),
                ("kernel_ns", ctypes.c_uint64), ("gpu_cycles", ctypes.c_uint64),
                ("flops", ctypes.c_uint64)]


class Error(RuntimeError):
    """An engine call failed; ``code`` is the vc4_status value when known."""

    def __init__(self, msg, code=None):
        self.code = code
        super().__init__(msg)


class Unsupported(Error):
    pass


class Engine:
    def __init__(self, cache_dir=None, arena_bytes=0):
        self.lib = ctypes.CDLL(_LIB)
        L = self.lib
        L.vc4_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                               ctypes.c_size_t]
        L.vc4_init.restype = ctypes.c_int
        L.vc4_shutdown.argtypes = [ctypes.c_void_p]
        L.vc4_strerror.restype = ctypes.c_char_p
        L.vc4_last_error.argtypes = [ctypes.c_void_p]
        L.vc4_last_error.restype = ctypes.c_char_p
        L.vc4_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(Buf)]
        L.vc4_alloc.restype = ctypes.c_int
        L.vc4_upload.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf), ctypes.c_void_p,
                                 ctypes.c_size_t]
        L.vc4_upload.restype = ctypes.c_int
        L.vc4_download.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                   ctypes.c_void_p, ctypes.c_size_t]
        L.vc4_download.restype = ctypes.c_int
        L.vc4_host_ptr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)]
        L.vc4_host_ptr.restype = ctypes.c_void_p
        L.vc4_matmul.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                 ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                 ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                 ctypes.c_int]
        L.vc4_matmul.restype = ctypes.c_int
        # ctx, in, Cin, H, W, wt, Cout, KH, KW, bias, stride, pad, out, HO, WO
        L.vc4_conv2d.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(Buf),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(Buf),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(Buf),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(Buf),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.vc4_conv2d.restype = ctypes.c_int
        L.vc4_elementwise.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                      ctypes.POINTER(Buf), ctypes.c_size_t]
        L.vc4_elementwise.restype = ctypes.c_int
        L.vc4_maxpooln.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + [
            ctypes.c_int] * 6 + [ctypes.POINTER(Buf),
                                 ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int)]
        L.vc4_maxpooln.restype = ctypes.c_int
        L.vc4_softmax_fwd.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                      ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.POINTER(Buf)]
        L.vc4_softmax_fwd.restype = ctypes.c_int
        L.vc4_conv_nhwc_wpanel_bytes.argtypes = [ctypes.c_int] * 5
        L.vc4_conv_nhwc_wpanel_bytes.restype = ctypes.c_size_t
        L.vc4_conv_nhwc_pack_w.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 4 + [ctypes.POINTER(Buf), ctypes.POINTER(Buf)]
        L.vc4_conv_nhwc_pack_w.restype = ctypes.c_int
        L.vc4_conv_nhwc.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 5 + [ctypes.POINTER(Buf)] + [ctypes.c_int] * 5 + \
            [ctypes.POINTER(Buf), ctypes.POINTER(ctypes.c_int),
             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.vc4_conv_nhwc.restype = ctypes.c_int
        L.vc4_conv_nhwc_dil.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 5 + [ctypes.POINTER(Buf)] + [ctypes.c_int] * 6 + \
            [ctypes.POINTER(Buf), ctypes.POINTER(ctypes.c_int),
             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.vc4_conv_nhwc_dil.restype = ctypes.c_int
        L.vc4_conv_nhwc_group.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.POINTER(Buf))] + \
            [ctypes.c_int] * 7 + [ctypes.POINTER(Buf),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
        L.vc4_conv_nhwc_group.restype = ctypes.c_int
        L.vc4_maxpool_nhwc.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 7 + [ctypes.POINTER(Buf),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
        L.vc4_maxpool_nhwc.restype = ctypes.c_int
        L.vc4_upsample2x_nhwc.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_int] * 5 + [ctypes.POINTER(Buf),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
        L.vc4_upsample2x_nhwc.restype = ctypes.c_int
        L.vc4_unary.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                ctypes.c_size_t] + [ctypes.c_float] * 3
        L.vc4_unary.restype = ctypes.c_int
        L.vc4_select3.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                  ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                  ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                  ctypes.c_size_t]
        L.vc4_select3.restype = ctypes.c_int
        L.vc4_gather.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                 ctypes.POINTER(Buf), ctypes.c_size_t,
                                 ctypes.c_float]
        L.vc4_gather.restype = ctypes.c_int
        L.vc4_blit.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                               ctypes.POINTER(Buf), ctypes.c_size_t]
        L.vc4_blit.restype = ctypes.c_int
        L.vc4_copy2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                 ctypes.POINTER(Buf)] + \
            [ctypes.c_uint32] * 7
        L.vc4_copy2d.restype = ctypes.c_int
        L.vc4_copy2d_n.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                   ctypes.POINTER(Buf)] + \
            [ctypes.c_uint32] * 10
        L.vc4_copy2d_n.restype = ctypes.c_int
        L.vc4_copy2d_x.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                   ctypes.POINTER(Buf)] + \
            [ctypes.c_uint32] * 13
        L.vc4_copy2d_x.restype = ctypes.c_int
        L.vc4_fill2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf)] + \
            [ctypes.c_uint32] * 4 + [ctypes.c_float]
        L.vc4_fill2d.restype = ctypes.c_int
        L.vc4_reduce.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                 ctypes.POINTER(Buf), ctypes.c_uint32,
                                 ctypes.c_uint32, ctypes.c_int]
        L.vc4_reduce.restype = ctypes.c_int
        L.vc4_cumsum.argtypes = [ctypes.c_void_p, ctypes.POINTER(Buf),
                                 ctypes.POINTER(Buf), ctypes.c_uint32,
                                 ctypes.c_uint32]
        L.vc4_cumsum.restype = ctypes.c_int
        L.vc4_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(Stats)]
        L.vc4_get_stats.restype = None
        L.vc4_supported_ops.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        L.vc4_supported_ops.restype = ctypes.POINTER(OpInfo)
        L.vc4_print_supported_ops.argtypes = [ctypes.c_void_p]
        L.vc4_arena_bytes.argtypes = [ctypes.c_void_p]
        L.vc4_arena_bytes.restype = ctypes.c_size_t

        self.ctx = ctypes.c_void_p()
        if not arena_bytes:
            # A 320x320 YOLO needs ~91 MiB (activations + weights + the
            # per-conv im2col scratch); the daemon's default 72 MiB is too
            # small.  The client region is what the engine can see, so make
            # it settable without touching the daemon launch line.
            mb = int(os.environ.get("VC4INFER_ARENA_MB", "0"))
            if mb > 0:
                arena_bytes = mb << 20
        if cache_dir:
            # kgen.py writes the blob into the cache directory; it does not
            # create it, so the caller's cache path must exist.
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
        st = L.vc4_init(ctypes.byref(self.ctx),
                        cache_dir.encode() if cache_dir else None, arena_bytes)
        if st != VC4_OK:
            raise Error("vc4_init failed: %s (%d)"
                        % (_STATUS.get(st, "unknown error"), st), st)

    # ---- lifecycle
    def close(self):
        if self.ctx:
            self.lib.vc4_shutdown(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---- helpers
    def _check(self, st, what):
        if st == VC4_OK:
            return
        msg = self.lib.vc4_last_error(self.ctx).decode()
        if st == ERR_UNSUPPORTED:
            raise Unsupported("%s: %s" % (what, msg))
        raise Error("%s: %s (%s)"
                    % (what, self.lib.vc4_strerror(st).decode(), msg), st)

    @property
    def arena_bytes(self):
        return self.lib.vc4_arena_bytes(self.ctx)

    def alloc(self, nbytes):
        b = Buf()
        self._check(self.lib.vc4_alloc(self.ctx, nbytes, ctypes.byref(b)), "alloc")
        return b

    def upload(self, arr, buf=None):
        """arr: numpy array or anything exposing the buffer protocol (fp32 bytes)."""
        if np is not None and isinstance(arr, np.ndarray):
            raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        else:
            raw = bytes(memoryview(arr).cast("B"))
        if buf is None:
            buf = self.alloc(len(raw))
        tmp = ctypes.create_string_buffer(raw, len(raw))
        self._check(self.lib.vc4_upload(self.ctx, ctypes.byref(buf),
                                        ctypes.cast(tmp, ctypes.c_void_p),
                                        len(raw)), "upload")
        return buf

    def download(self, buf, nbytes):
        raw = ctypes.create_string_buffer(nbytes)
        self._check(self.lib.vc4_download(self.ctx, ctypes.byref(buf), raw, nbytes),
                    "download")
        return np.frombuffer(raw.raw, dtype=np.float32) if np is not None else raw.raw

    def view(self, buf, count):
        """Zero-copy view of the uncached arena mapping (numpy required)."""
        if np is None:
            raise RuntimeError("numpy is not installed")
        p = self.lib.vc4_host_ptr(self.ctx, ctypes.byref(buf))
        return np.ctypeslib.as_array((ctypes.c_float * count).from_address(p))

    # ---- ops
    def matmul(self, a, b, c, M, N, K, layout=LAYOUT_ROW_MAJOR):
        self._check(self.lib.vc4_matmul(self.ctx, ctypes.byref(a), ctypes.byref(b),
                                        ctypes.byref(c), M, N, K, layout), "matmul")

    def conv2d(self, x, w, out, Cin, H, W, Cout, KH, KW, bias=None, stride=1, pad=1):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        self._check(self.lib.vc4_conv2d(
            self.ctx, ctypes.byref(x), Cin, H, W, ctypes.byref(w), Cout, KH, KW,
            ctypes.byref(bias) if bias is not None else None, stride, pad,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo)), "conv2d")
        return ho.value, wo.value

    def elementwise(self, op, a, out, n, b=None):
        self._check(self.lib.vc4_elementwise(
            self.ctx, op, ctypes.byref(a),
            ctypes.byref(b) if b is not None else None, ctypes.byref(out), n),
            "elementwise")

    def maxpooln(self, x, out, C, H, W, N, stride, pad):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        self._check(self.lib.vc4_maxpooln(
            self.ctx, ctypes.byref(x), C, H, W, N, stride, pad,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo)), "maxpooln")
        return ho.value, wo.value

    def softmax_fwd(self, x, out, rows, cols):
        self._check(self.lib.vc4_softmax_fwd(
            self.ctx, ctypes.byref(x), rows, cols, ctypes.byref(out)),
            "softmax_fwd")

    # ---- generic kernels (vc4_gen.c)
    def unary(self, op, a, out, n, p0=0.0, p1=0.0, p2=0.0):
        self._check(self.lib.vc4_unary(self.ctx, int(op), ctypes.byref(a),
                                       ctypes.byref(out), int(n),
                                       float(p0), float(p1), float(p2)),
                    "unary")

    def select3(self, op, a, b, c, out, n):
        self._check(self.lib.vc4_select3(self.ctx, int(op), ctypes.byref(a),
                                         ctypes.byref(b), ctypes.byref(c),
                                         ctypes.byref(out), int(n)), "select3")

    def gather(self, tab, out, n, fill=0.0):
        self._check(self.lib.vc4_gather(self.ctx, ctypes.byref(tab),
                                        ctypes.byref(out), int(n), float(fill)),
                    "gather")

    def blit(self, a, out, nbytes):
        self._check(self.lib.vc4_blit(self.ctx, ctypes.byref(a),
                                      ctypes.byref(out), int(nbytes)), "blit")

    def copy2d(self, a, out, rows, vpr, srow, drow, svec, dvec, slane):
        """16-lane vector copy: dst[rows][vpr][16] <- src at lane*slane."""
        self._check(self.lib.vc4_copy2d(
            self.ctx, ctypes.byref(a), ctypes.byref(out), int(rows), int(vpr),
            int(srow), int(drow), int(svec), int(dvec), int(slane)),
            "copy2d")

    def copy2d_n(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane, units, depth):
        """copy2d with an explicit store shape and lane count."""
        self._check(self.lib.vc4_copy2d_n(
            self.ctx, ctypes.byref(a), ctypes.byref(out), int(rows), int(vpr),
            int(srow), int(drow), int(svec), int(dvec), int(slane),
            int(nlane), int(units), int(depth)), "copy2d_n")

    def copy2d_x(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane=16, nblk=1, sbstep=64, dbstep=64, units=4, depth=16):
        """The general layout copy (see vc4infer.h)."""
        self._check(self.lib.vc4_copy2d_x(
            self.ctx, ctypes.byref(a), ctypes.byref(out), int(rows), int(vpr),
            int(srow), int(drow), int(svec), int(dvec), int(slane),
            int(nlane), int(nblk), int(sbstep), int(dbstep), int(units),
            int(depth)), "copy2d_x")

    def fill2d(self, out, rows, vpr, drow, dvec, value=0.0):
        self._check(self.lib.vc4_fill2d(
            self.ctx, ctypes.byref(out), int(rows), int(vpr), int(drow),
            int(dvec), float(value)), "fill2d")

    def reduce(self, a, out, outer, red, op):
        self._check(self.lib.vc4_reduce(self.ctx, ctypes.byref(a),
                                        ctypes.byref(out), int(outer), int(red),
                                        int(op)), "reduce")

    def cumsum(self, a, out, rows, cols):
        self._check(self.lib.vc4_cumsum(self.ctx, ctypes.byref(a),
                                        ctypes.byref(out), int(rows),
                                        int(cols)), "cumsum")

    # ---- NHWC (the YOLO plane)
    def conv_nhwc_wpanel_bytes(self, Cin, Cout, KH, KW, bias=True):
        return int(self.lib.vc4_conv_nhwc_wpanel_bytes(Cin, Cout, KH, KW,
                                                      1 if bias else 0))

    def conv_nhwc_pack_w(self, wt, Cin, Cout, KH, KW, bias, wpanel):
        self._check(self.lib.vc4_conv_nhwc_pack_w(
            self.ctx, ctypes.byref(wt), Cin, Cout, KH, KW,
            ctypes.byref(bias) if bias is not None else None,
            ctypes.byref(wpanel)), "conv_nhwc_pack_w")

    def conv_nhwc(self, x, nimg, Cin, H, W, in_pitch, wpanel, Cout, KH, KW,
                  stride, pad, out):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        op = ctypes.c_int(0)
        self._check(self.lib.vc4_conv_nhwc(
            self.ctx, ctypes.byref(x), nimg, Cin, H, W, in_pitch,
            ctypes.byref(wpanel), Cout, KH, KW, stride, pad,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo),
            ctypes.byref(op)), "conv_nhwc")
        return ho.value, wo.value, op.value

    def conv_nhwc_dil(self, x, nimg, Cin, H, W, in_pitch, wpanel, Cout, KH, KW,
                      stride, pad, dil, out):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        op = ctypes.c_int(0)
        self._check(self.lib.vc4_conv_nhwc_dil(
            self.ctx, ctypes.byref(x), nimg, Cin, H, W, in_pitch,
            ctypes.byref(wpanel), Cout, KH, KW, stride, pad, dil,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo),
            ctypes.byref(op)), "conv_nhwc_dil")
        return ho.value, wo.value, op.value

    def conv_nhwc_group(self, x, nimg, Cin, H, W, in_pitch, wpanels, Cout,
                        groups, KH, KW, stride, pad, dil, out):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        op = ctypes.c_int(0)
        arr = (ctypes.POINTER(Buf) * len(wpanels))()
        for i, w in enumerate(wpanels):
            arr[i] = ctypes.pointer(w)
        self._check(self.lib.vc4_conv_nhwc_group(
            self.ctx, ctypes.byref(x), nimg, Cin, H, W, in_pitch, arr, Cout,
            groups, KH, KW, stride, pad, dil, ctypes.byref(out),
            ctypes.byref(ho), ctypes.byref(wo), ctypes.byref(op)),
            "conv_nhwc_group")
        return ho.value, wo.value, op.value

    def maxpool_nhwc(self, x, nimg, C, H, W, in_pitch, k, pad, out):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        op = ctypes.c_int(0)
        self._check(self.lib.vc4_maxpool_nhwc(
            self.ctx, ctypes.byref(x), nimg, C, H, W, in_pitch, k, pad,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo),
            ctypes.byref(op)), "maxpool_nhwc")
        return ho.value, wo.value, op.value

    def upsample2x_nhwc(self, x, nimg, C, H, W, in_pitch, out):
        ho = ctypes.c_int(0)
        wo = ctypes.c_int(0)
        op = ctypes.c_int(0)
        self._check(self.lib.vc4_upsample2x_nhwc(
            self.ctx, ctypes.byref(x), nimg, C, H, W, in_pitch,
            ctypes.byref(out), ctypes.byref(ho), ctypes.byref(wo),
            ctypes.byref(op)), "upsample2x_nhwc")
        return ho.value, wo.value, op.value

    @property
    def stats(self):
        s = Stats()
        self.lib.vc4_get_stats(self.ctx, ctypes.byref(s))
        return s

    # ---- registry
    def supported_ops(self):
        n = ctypes.c_size_t(0)
        t = self.lib.vc4_supported_ops(ctypes.byref(n))
        return [dict(name=t[i].name.decode(), backend=t[i].backend.decode(),
                     supported=bool(t[i].supported), note=t[i].note.decode())
                for i in range(n.value)]

    def print_supported_ops(self):
        self.lib.vc4_print_supported_ops(None if False else ctypes.c_void_p(
            ctypes.cast(sys.stdout, ctypes.c_void_p).value))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", action="store_true", help="print the op table")
    ap.add_argument("--bench", action="store_true", help="run a 256^3 matmul")
    args = ap.parse_args()
    eng = Engine()
    print("arena %.1f MiB" % (eng.arena_bytes / (1 << 20)))
    if args.ops or not args.bench:
        for o in eng.supported_ops():
            print("%-20s %-6s %-24s %s" % (o["name"],
                                           "yes" if o["supported"] else "NO",
                                           o["backend"], o["note"]))
    if args.bench:
        import time
        for n in (128, 256):
            a = eng.upload(np.random.rand(n, n).astype(np.float32) * 2 - 1)
            b = eng.upload(np.random.rand(n, n).astype(np.float32) * 2 - 1)
            c = eng.alloc(n * n * 4)
            eng.matmul(a, b, c, n, n, n)          # warm
            t0 = time.monotonic()
            reps = 5
            for _ in range(reps):
                eng.matmul(a, b, c, n, n, n)
            dt = (time.monotonic() - t0) / reps
            print("matmul %d^3: %.3f ms  %.2f GFLOP/s" %
                  (n, dt * 1e3, 2.0 * n ** 3 / dt / 1e9))
    eng.close()
