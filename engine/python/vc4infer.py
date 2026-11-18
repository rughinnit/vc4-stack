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
_LIB = os.environ.get("VC4INFER_LIB",
                      os.path.join(_HERE, "..", "build", "libvc4infer.so"))

VC4_OK = 0
ERR_UNSUPPORTED = -5

OP_MATMUL, OP_CONV2D, OP_ADD, OP_MUL, OP_RELU, OP_SIGMOID, OP_SILU = range(7)
LAYOUT_ROW_MAJOR = 0


class Buf(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint32), ("bytes", ctypes.c_size_t)]


class OpInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("backend", ctypes.c_char_p),
                ("supported", ctypes.c_int), ("note", ctypes.c_char_p)]


class Error(RuntimeError):
    pass


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
        L.vc4_conv2d.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(Buf)] * 3 + [
            ctypes.c_int] * 9 + [ctypes.POINTER(Buf), ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int)]
        L.vc4_conv2d.restype = ctypes.c_int
        L.vc4_elementwise.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.POINTER(Buf), ctypes.POINTER(Buf),
                                      ctypes.POINTER(Buf), ctypes.c_size_t]
        L.vc4_elementwise.restype = ctypes.c_int
        L.vc4_supported_ops.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        L.vc4_supported_ops.restype = ctypes.POINTER(OpInfo)
        L.vc4_print_supported_ops.argtypes = [ctypes.c_void_p]
        L.vc4_arena_bytes.argtypes = [ctypes.c_void_p]
        L.vc4_arena_bytes.restype = ctypes.c_size_t

        self.ctx = ctypes.c_void_p()
        st = L.vc4_init(ctypes.byref(self.ctx),
                        cache_dir.encode() if cache_dir else None, arena_bytes)
        if st != VC4_OK:
            raise Error("vc4_init failed: %d" % st)

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
        raise Error("%s: %s (%s)" % (what, self.lib.vc4_strerror(st).decode(), msg))

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
