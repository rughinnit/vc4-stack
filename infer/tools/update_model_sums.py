#!/usr/bin/env python3
"""update_model_sums.py -- write the SHA256SUMS that ships beside the models.

The manifest at ``infer/packaging/vc4infer/data/models/SHA256SUMS`` is the
contract between the packaged model data and ``vc4infer selftest``: it lists
every file of that directory in ``sha256sum`` format, and selftest compares the
files it is about to use against it before it runs a case.

    python3 infer/tools/update_model_sums.py            # rewrite in place
    python3 infer/tools/update_model_sums.py --check    # exit 1 when stale
    python3 infer/tools/update_model_sums.py DIR -o OUT

Entries are sorted by file name, so the manifest is stable between runs and a
diff shows exactly which model changed.  The file is generated, never edited by
hand: a hand-written digest is how a wrong one ships.
"""
import argparse
import hashlib
import os
import sys

SUMS = "SHA256SUMS"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.normpath(
    os.path.join(HERE, "..", "packaging", "vc4infer", "data", "models"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_files(models):
    """Every regular file in the directory except the manifest itself."""
    return [n for n in sorted(os.listdir(models))
            if n != SUMS and os.path.isfile(os.path.join(models, n))]


def render(models):
    """The manifest text: one ``<sha256>  <name>`` line per file."""
    return "".join("%s  %s\n" % (sha256_file(os.path.join(models, n)), n)
                   for n in model_files(models))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="regenerate the selftest model manifest")
    ap.add_argument("models", nargs="?", default=DEFAULT_DIR,
                    help="model directory (default: the packaged one)")
    ap.add_argument("-o", "--out", default=None,
                    help="write here instead of DIR/%s" % SUMS)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when the manifest is stale")
    args = ap.parse_args(argv)

    models = os.path.abspath(args.models)
    if not os.path.isdir(models):
        sys.stderr.write("no such directory: %s\n" % models)
        return 2
    files = model_files(models)
    if not files:
        sys.stderr.write("refusing to write an empty manifest: %s has no "
                         "files\n" % models)
        return 2
    text = render(models)
    path = args.out or os.path.join(models, SUMS)

    if args.check:
        try:
            with open(path) as fh:
                have = fh.read()
        except OSError:
            have = None
        if have != text:
            sys.stderr.write("%s is stale or missing; regenerate it with %s\n"
                             % (path, os.path.basename(__file__)))
            return 1
        print("%s: %d files, up to date" % (path, len(files)))
        return 0

    with open(path, "w") as fh:
        fh.write(text)
    print("wrote %s (%d files)" % (path, len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
