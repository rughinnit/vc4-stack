#!/usr/bin/env python3
"""test_kgen_slot.py -- kernel generation refuses a blob the code slot cannot hold.

CPU only: no engine, no board.  Two checks against the generator the engine
JITs from (the package copy, which tools/check_kgen_sync.py keeps identical to
infer/tools/kgen.py):

  1. up2 at C=512 is the kernel that made v8s160 refuse: the unrolled channel
     loop was 21440 B against the 16384 B client code slot.  The runtime loop
     must generate well under the slot, at every C the engine can pass.
  2. a kernel that really is too large is refused *at generation* with a
     sentence naming the size and the slot, not later when a launch fails.
     im2col unrolls KH*KW taps, so 8x16 taps at Cin=16 is a real over-size
     generation.

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_kgen_slot.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KGEN = os.path.join(HERE, "..", "python", "vc4infer", "kgen", "kgen.py")
SLOT = 16384
FAILS = []


def run(*args):
    return subprocess.run([sys.executable, KGEN] + list(args),
                          capture_output=True, text=True)


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)


def main():
    with tempfile.TemporaryDirectory() as td:
        for c in (16, 64, 128, 256, 512):
            out = os.path.join(td, "up2_%d.bin" % c)
            r = run("up2", out, "--cin", str(c))
            check(r.returncode == 0, "up2 C=%d generates" % c, r.stderr)
            if r.returncode == 0:
                n = os.path.getsize(out)
                check(n <= SLOT,
                      "up2 C=%d is %d B, inside the %d B code slot"
                      % (c, n, SLOT))

        out = os.path.join(td, "too_big.bin")
        r = run("im2col", out, "--cin", "16", "--k", "2048",
                "--kh", "8", "--kw", "16")
        check(r.returncode == 4,
              "an over-slot kernel is refused with exit 4", r.returncode)
        check("exceed the %d B client code slot" % SLOT in r.stderr,
              "the refusal names the slot and the size", r.stderr)
        check(not os.path.exists(out),
              "no over-slot blob is written", out)

    print("%s %d failure(s)" % ("FAIL" if FAILS else "PASS", len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
