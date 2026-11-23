"""checkout_guard.py -- make a tool test the checkout it belongs to.

A tool started as ``python3 infer/tools/op_matrix.py`` puts ``infer/tools`` on
``sys.path``, not the checkout's ``infer/python``, so ``import vc4infer``
resolves to whatever is installed instead.  That is silent: the tool runs,
prints plausible numbers, and tests an engine nobody asked about.  It
happened once - a wide-matmul mismatch from an older installed wheel was
reported as a failure of the tree.

Tools that do not prepend the checkout's package call ``require_checkout``
after parsing their arguments and before opening an engine.  The guard
resolves the loaded package, its shared library and its kernel generator,
compares them with the checkout the tool lives in, and stops with one line
naming what was loaded and what was expected.  ``--allow-installed`` is the
explicit way to test an installed wheel on purpose.

Tools that already prepend ``<checkout>/infer/python`` (``plan_diff.py``,
``breakdown.py``, ``verify.py`` through ``onnxrun``) cannot be wrong this
way and do not need it.
"""
import importlib
import os
import sys

EXIT_MISMATCH = 2


def _real(path):
    return os.path.realpath(path) if path else None


def checkout_root(tool_file):
    """The repository root above tool_file, or None outside a checkout."""
    d = os.path.dirname(os.path.abspath(tool_file))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "infer", "python", "vc4infer")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return None


def resolved():
    """Where the loaded engine actually came from."""
    out = {}
    try:
        v = importlib.import_module("vc4infer")
        out["vc4infer"] = _real(getattr(v, "__file__", None))
    except Exception:                                     # pragma: no cover
        out["vc4infer"] = None
    try:
        b = importlib.import_module("vc4infer._binding")
        out["libvc4infer"] = _real(getattr(b, "_LIB", None))
    except Exception:                                     # pragma: no cover
        out["libvc4infer"] = None
    for name in ("vc4infer.kgen", "vc4infer.kgen.kgen"):
        try:
            m = importlib.import_module(name)
            out[name] = _real(getattr(m, "__file__", None))
        except Exception:                                 # pragma: no cover
            out[name] = None
    return out


def _under(path, roots):
    if not path:
        return True                     # absent: nothing to contradict
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def require_checkout(tool_file, allow_installed=False, stream=None):
    """Refuse to run against an engine outside the checkout.

    Returns True when every resolved path is inside the checkout, False when
    ``allow_installed`` let a mismatch through, and exits with
    EXIT_MISMATCH otherwise.
    """
    stream = stream or sys.stderr
    name = os.path.basename(tool_file)
    root = checkout_root(tool_file)
    paths = resolved()
    if root is None:
        print("%s: cannot find a vc4-stack checkout above %s; refusing to "
              "guess which engine is under test"
              % (name, os.path.abspath(tool_file)), file=stream)
        raise SystemExit(EXIT_MISMATCH)
    pkg = _real(os.path.join(root, "infer", "python", "vc4infer"))
    lib_roots = [_real(os.path.join(root, "infer", "build")), pkg]
    wrong = []
    if not _under(paths["vc4infer"], [pkg]):
        wrong.append("vc4infer from %s (checkout: %s)"
                     % (paths["vc4infer"], pkg))
    if not _under(paths["libvc4infer"], lib_roots):
        wrong.append("libvc4infer from %s (checkout: %s)"
                     % (paths["libvc4infer"],
                        os.path.join(root, "infer", "build")))
    for key in ("vc4infer.kgen", "vc4infer.kgen.kgen"):
        if not _under(paths[key], [pkg]):
            wrong.append("%s from %s (checkout: %s)"
                         % (key, paths[key], pkg))
    if not wrong:
        return True
    if allow_installed:
        print("%s: --allow-installed: testing the engine at %s"
              % (name, paths["vc4infer"]), file=stream)
        return False
    print("%s: refusing to test another engine: %s; set PYTHONPATH=%s to use "
          "this checkout, or pass --allow-installed to test that package on "
          "purpose" % (name, "; ".join(wrong),
                       os.path.join(root, "infer", "python")), file=stream)
    raise SystemExit(EXIT_MISMATCH)
