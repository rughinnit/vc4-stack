"""Paths, defaults and the environment the engine runs with.

Kept free of public-API imports so ``session``/``cli``/``__init__`` can all use
it without an import cycle.
"""
import os
import sys
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


def socket_path():
    """The socket the client will connect to: VC4D_SOCKET overrides the
    default, exactly as libvc4client's remote_socket() does.  The precheck
    below has to look at the same path, or a test daemon on another socket is
    reported as "vc4d is not running"."""
    return os.environ.get("VC4D_SOCKET") or SOCKET


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


BACKENDS = ("managed", "direct")

BACKEND_HELP = {
    "managed": "vc4d daemon: unprivileged, validated, ~0.1 ms per launch",
    "direct": "root + /dev/mem: no daemon, no per-launch overhead, no "
              "validation",
}


def backend():
    """The backend in effect: ``managed`` (vc4d) or ``direct`` (/dev/mem)."""
    explicit = os.environ.get("VC4INFER_BACKEND")
    if explicit in BACKENDS:
        return explicit
    daemon = os.environ.get("VC4INFER_DAEMON", "1") not in ("", "0")
    return "managed" if daemon else "direct"


def set_backend(name):
    """Select the backend for this process.

    ``managed`` is the default and the only supported one for normal use: the
    QPUs are reached through vc4d, which validates every register write and
    address, so a buggy or hostile client cannot take the machine down.

    ``direct`` talks to /dev/mem itself.  It is what the benchmarks use to
    show the cost of the daemon (7 % on a 256^3 GEMM, 15 % on YOLOv8n at 160)
    and it needs root.  It performs no validation: a wrong offset corrupts
    memory, and the GPU cannot be recovered without a reboot.
    """
    name = (name or "").strip().lower()
    if name not in BACKENDS:
        raise ValueError("backend must be one of: %s" % ", ".join(BACKENDS))
    if name == "direct":
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            raise PermissionError(
                "the direct backend needs root (it opens /dev/mem): "
                "run with sudo, or use --backend managed")
        if not os.path.exists("/dev/mem"):
            raise FileNotFoundError(
                "the direct backend needs /dev/mem, which is not present")
        os.environ["VC4INFER_DAEMON"] = "0"
        if not os.environ.get("VC4INFER_BACKEND"):
            sys.stderr.write(
                "vc4infer: direct backend - /dev/mem, no validation, root "
                "only; run with --backend managed for the safe path\n")
    else:
        os.environ["VC4INFER_DAEMON"] = "1"
    os.environ["VC4INFER_BACKEND"] = name
    return name


def backend_summary():
    b = backend()
    return "%s (%s)" % (b, BACKEND_HELP[b])


def prepare_env():
    """The defaults the library and CLI run with.

    ``VC4INFER_DAEMON=1`` (vc4d, unprivileged) unless the caller set it;
    ``VC4INFER_KGEN`` points at the packaged kernel generator, so a wheel in a
    venv does not depend on the repository layout.
    """
    explicit = os.environ.get("VC4INFER_BACKEND")
    if explicit in BACKENDS:
        os.environ["VC4INFER_DAEMON"] = "1" if explicit == "managed" else "0"
    else:
        os.environ.setdefault("VC4INFER_DAEMON", "1")
    kg = default_kgen()
    if kg and not os.environ.get("VC4INFER_KGEN"):
        os.environ["VC4INFER_KGEN"] = kg
    return {
        "backend": backend(),
        "daemon": os.environ.get("VC4INFER_DAEMON", "0") not in ("", "0"),
        "socket": socket_path(),
        "kgen": kg,
        "cache": default_cache_dir(),
    }


def daemon_reachable():
    return os.path.exists(socket_path())


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
    # The daemon reads VC4INFER_ARENA_MB itself; in direct mode the client owns
    # the arena, so the size has to be passed to the library explicitly.
    arena_bytes = 0
    if not daemon:
        try:
            mb = int(os.environ.get("VC4INFER_ARENA_MB", "64"))
        except ValueError:
            mb = 64
        arena_bytes = max(8, mb) << 20
    if daemon and not daemon_reachable():
        raise Error(
            "vc4d is not running (no %s). Start it with `sudo systemctl start "
            "vc4d`, set VC4D_SOCKET to a private daemon's socket, or run "
            "as root with VC4INFER_DAEMON=0." % socket_path())
    deadline = time.monotonic() + max(0.0, float(retry_s))
    while True:
        try:
            return Engine(cache_dir=cache_dir or default_cache_dir(),
                          arena_bytes=arena_bytes)
        except Error as e:
            if getattr(e, "code", None) == _binding.ERR_NOMEM and \
                    time.monotonic() < deadline:
                time.sleep(0.25)
                continue
            raise Error(
                "%s\n(vc4d must be running and the GPU free; it serves one "
                "64 MiB client arena, so a concurrent session makes this fail)"
                % e, getattr(e, "code", None))
