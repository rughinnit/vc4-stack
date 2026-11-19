"""Paths, defaults and the environment the engine runs with.

Kept free of public-API imports so ``session``/``cli``/``__init__`` can all use
it without an import cycle.
"""
import os
import time

from . import _binding, _onnxrun

Runner = _onnxrun.Runner
UnsupportedNode = _onnxrun.UnsupportedNode
Engine = _binding.Engine
Error = _binding.Error
Unsupported = _binding.Unsupported


def available_devices():
    """Devices this build can run on.  There is exactly one: the VC4 QPU."""
    return ["vc4"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_KGEN = os.path.join(_HERE, "kgen", "kgen.py")
INSTALL_KGEN = "/usr/lib/vc4infer/kgen/kgen.py"
INSTALL_MODELS = "/usr/share/vc4infer/models"
SOCKET = "/run/vc4d.sock"


def default_cache_dir():
    """Per-user kernel blob cache ($VC4INFER_CACHE, else ~/.cache/vc4infer)."""
    env = os.environ.get("VC4INFER_CACHE")
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "vc4infer", "kcache")


def default_kgen():
    """The kernel generator shipped inside this package."""
    for p in (os.environ.get("VC4INFER_KGEN"), _PKG_KGEN, INSTALL_KGEN):
        if p and os.path.isfile(p):
            return p
    return None


def models_dir():
    """Directory holding the selftest models (shipped in the .deb)."""
    for d in (os.environ.get("VC4INFER_MODELS"), INSTALL_MODELS,
              os.path.join(_HERE, "..", "packaging", "vc4infer",
                           "data", "models")):
        if d and os.path.isdir(d):
            return d
    return INSTALL_MODELS


def prepare_env():
    """The defaults the library and CLI run with.

    ``VC4INFER_DAEMON=1`` (vc4d, unprivileged) unless the caller set it;
    ``VC4INFER_KGEN`` points at the packaged kernel generator, so a wheel in a
    venv does not depend on the repository layout.
    """
    os.environ.setdefault("VC4INFER_DAEMON", "1")
    kg = default_kgen()
    if kg and not os.environ.get("VC4INFER_KGEN"):
        os.environ["VC4INFER_KGEN"] = kg
    return {
        "daemon": os.environ.get("VC4INFER_DAEMON", "0") not in ("", "0"),
        "socket": SOCKET,
        "kgen": kg,
        "cache": default_cache_dir(),
    }


def daemon_reachable():
    return os.path.exists(SOCKET)


def open_engine(cache_dir=None, arena_mb=None, retry_s=3.0):
    """Create a binding Engine, with actionable errors for a missing daemon.

    vc4d serves one 64 MiB client arena, so a concurrent user (another shell,
    the test suite) makes the allocation fail with NOMEM.  Retry that case for
    a few seconds before reporting it, so back-to-back commands are robust.
    """
    prepare_env()
    if arena_mb:
        os.environ["VC4INFER_ARENA_MB"] = str(int(arena_mb))
    daemon = os.environ.get("VC4INFER_DAEMON", "0") not in ("", "0")
    if daemon and not daemon_reachable():
        raise Error(
            "vc4d is not running (no %s). Start it with `sudo systemctl start "
            "vc4d`, or run as root with VC4INFER_DAEMON=0." % SOCKET)
    deadline = time.monotonic() + max(0.0, float(retry_s))
    while True:
        try:
            return Engine(cache_dir=cache_dir or default_cache_dir())
        except Error as e:
            if getattr(e, "code", None) == _binding.ERR_NOMEM and \
                    time.monotonic() < deadline:
                time.sleep(0.25)
                continue
            raise Error(
                "%s\n(vc4d must be running and the GPU free; it serves one "
                "64 MiB client arena, so a concurrent session makes this fail)"
                % e, getattr(e, "code", None))
