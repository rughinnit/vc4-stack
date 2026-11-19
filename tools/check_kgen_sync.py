#!/usr/bin/env python3
"""Fail if the two copies of the QPU kernel generator have drifted apart.

`infer/tools/` holds the generator the engine JITs from (both in a source tree
and, as /usr/lib/vc4infer/kgen/, in the installed package).  `infer/python/
vc4infer/kgen/` holds the copy the wheel ships.  They are byte-identical by
construction but were *tracked twice*, and once diverged in opposite
directions: the tools copy had cstride/dilation, the wheel copy had nvec and
the generic-op body, so the committed state could not generate the kernels it
claimed and a pip install silently produced wrong grouped/dilated convs.

Run from the repo root or from CI; exit 1 on drift.
"""
import filecmp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "infer", "tools")
DST = os.path.join(ROOT, "infer", "python", "vc4infer", "kgen")

# kgen.py is generated into the package from tools/; the rest live beside it.
FILES = ("kgen.py", "kops.py", "qenc.py", "gen.py", "vpmwords.py")


def main():
    bad = []
    for name in FILES:
        a = os.path.join(SRC, name)
        b = os.path.join(DST, name)
        for p in (a, b):
            if not os.path.exists(p):
                bad.append("%s: missing %s" % (name, p))
                break
        else:
            if not filecmp.cmp(a, b, shallow=False):
                bad.append("%s: %s differs from %s" % (name, a, b))
    if bad:
        for line in bad:
            print("DRIFT: " + line, file=sys.stderr)
        print("fix: cp infer/tools/*.py infer/python/vc4infer/kgen/  "
              "(or run tools/sync_kgen.sh)", file=sys.stderr)
        return 1
    print("kgen copies in sync: %s" % ", ".join(FILES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
