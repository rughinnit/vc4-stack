#!/usr/bin/env python3
"""test_runtime_env.py -- the daemon socket override, without a device.

The C client reads ``VC4D_SOCKET`` for every connection; the Python side used
to check and name only the system path, so a private daemon was reachable from
the library but the engine refused before it got there.  These checks pin the
override in both places.

    python3 infer/tests/test_runtime_env.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _runtime                         # noqa: E402

FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def main():
    saved = os.environ.get("VC4D_SOCKET")
    try:
        os.environ.pop("VC4D_SOCKET", None)
        check(_runtime.socket_path() == "/run/vc4d.sock",
              "the default socket is the system daemon")
        check(_runtime.daemon_reachable() ==
              os.path.exists("/run/vc4d.sock"),
              "the default reachability check uses that path")

        tmp = tempfile.mkdtemp(prefix="vc4-sock-",
                               dir=os.environ.get("TMPDIR") or None)
        sock = os.path.join(tmp, "private.sock")
        os.environ["VC4D_SOCKET"] = sock
        check(_runtime.socket_path() == sock,
              "VC4D_SOCKET overrides the path")
        check(not _runtime.daemon_reachable(),
              "a missing private socket is reported as unreachable")
        open(sock, "w").close()
        check(_runtime.daemon_reachable(),
              "an existing private socket is reachable")
        check(_runtime.prepare_env()["socket"] == sock,
              "prepare_env reports the private socket")
    finally:
        if saved is None:
            os.environ.pop("VC4D_SOCKET", None)
        else:
            os.environ["VC4D_SOCKET"] = saved
    if FAILS:
        print("\n%d FAILED" % len(FAILS))
        return 1
    print("\nall runtime-env checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
