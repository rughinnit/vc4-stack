#!/usr/bin/env python3
"""ctypes harness for the statically registered VC4 ggml backend.

The backend is compiled into libvc4ggml.so and registered by calling
ggml_vc4_register() - no dlopen of a DL plugin, no llama.cpp.

    VC4INFER_DAEMON=1 python3 ctypes_test.py
"""
import ctypes
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("VC4GGML_LIB", os.path.join(HERE, "..", "build", "libvc4ggml.so"))


def main():
    lib = ctypes.CDLL(LIB)
    lib.vc4ggml_test_tiny.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.vc4ggml_test_tiny.restype = ctypes.c_int
    lib.vc4ggml_test_matmul.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.vc4ggml_test_matmul.restype = ctypes.c_int
    lib.vc4ggml_test_llama.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.vc4ggml_test_llama.restype = ctypes.c_int

    results = {}
    ok = True
    for name in ("tiny", "matmul", "llama"):
        buf = ctypes.create_string_buffer(8192)
        rc = getattr(lib, "vc4ggml_test_" + name)(buf, len(buf))
        text = buf.value.decode("utf-8", "replace")
        print(text, end="")
        results[name] = {"rc": rc, "report": text}
        ok = ok and rc == 0

    print(json.dumps({"ok": ok, "tests": results}, indent=1)[:2000])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
