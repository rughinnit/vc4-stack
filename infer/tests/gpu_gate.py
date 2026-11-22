"""gpu_gate.py -- opt-in gate for any test that touches the QPUs.

On this board a second GPU client is not a slow neighbour, it is a corrupt
one: the QPUs are shared with no preemption, and a run that starts while
another owns them can wedge the queue or reboot the board.  A test file that
constructs an ``Engine`` or calls libvc4infer is a client, so it asks for the
QPUs explicitly before it opens anything:

    from gpu_gate import require_gpu
    require_gpu(__file__)          # or a label

Without ``VC4_REQUIRE_GPU=1`` the test prints one line saying what it would
have run and exits 77, the conventional "skipped" status.  ``run_tests.sh``
reports 77 as SKIP and lists the names at the end, so a CPU-only run is honest
about what it did not cover instead of looking complete.

The gate is deliberately env-only and not a config file: whoever starts the
test already has the shell, and the person who owns the board is the person
who exports the variable.
"""
import os
import sys

EXIT_SKIP = 77
VAR = "VC4_REQUIRE_GPU"

# Anything that is not one of these means "yes": the opt-in is a deliberate
# act, and misspelling it is not one.
_OFF = ("", "0", "no", "false", "off")


def opted_in():
    """True when the caller has said this process may use the QPUs."""
    return os.environ.get(VAR, "").strip().lower() not in _OFF


def require_gpu(name):
    """Continue if the QPUs are ours, otherwise print why and exit 77."""
    if opted_in():
        return True
    label = os.path.basename(name) if os.sep in str(name) else str(name)
    print("SKIP %s: needs the QPUs; set %s=1 to run it" % (label, VAR))
    sys.stdout.flush()
    sys.exit(EXIT_SKIP)
