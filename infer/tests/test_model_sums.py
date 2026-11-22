#!/usr/bin/env python3
"""test_model_sums.py -- the shipped model manifest, without a GPU.

The package data carries SHA256SUMS beside the selftest models, and selftest
refuses to run a case when a file there does not match it.  This test keeps
both halves honest:

  * every file of the packaged model directory is listed, and the recorded
    digests are recomputed here rather than taken from the tool that wrote
    them, so a stale manifest fails;
  * infer/tools/update_model_sums.py reproduces the checked-in file byte for
    byte, so the manifest really is generated and its order is stable;
  * a same-size corruption in a copy of the directory, and a missing file,
    are reported as MODEL DATA MISMATCH naming the file and both digests,
    with the model-data exit status rather than the engine's;
  * cmd_selftest stops with that status before it opens an engine (the engine
    is stubbed here, so this file needs no GPU and no daemon);
  * a directory without a manifest still runs (a user's own models).

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_model_sums.py

Exit status is 0 only when every check passed.
"""
import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "packaging", "vc4infer", "data", "models")
TOOL = os.path.join(ROOT, "tools", "update_model_sums.py")
SUMS = os.path.join(DATA, "SHA256SUMS")
sys.path.insert(0, os.path.join(ROOT, "python"))

from vc4infer import cli                                     # noqa: E402

FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_files():
    return sorted(n for n in os.listdir(DATA)
                  if n != cli.MODEL_SUMS and
                  os.path.isfile(os.path.join(DATA, n)))


def recomputed():
    """The manifest text built from the files alone (sha256sum format)."""
    return "".join("%s  %s\n" % (sha256(os.path.join(DATA, n)), n)
                   for n in data_files())


def listed():
    """(name, digest) pairs as the checked-in manifest states them."""
    out = []
    with open(SUMS) as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(None, 1)
                out.append((name.strip(), digest))
    return out


def capture(fn, *a):
    """Run fn, return (result, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a)
    return res, buf.getvalue()


# --------------------------------------------------------------------- tests
def test_manifest_is_current():
    check(os.path.isfile(SUMS), "manifest: %s exists" % SUMS)
    if not os.path.isfile(SUMS):
        return
    with open(SUMS) as fh:
        have = fh.read()
    want = recomputed()
    diff = ""
    if have != want:
        hl, wl = have.splitlines(), want.splitlines()
        for i in range(max(len(hl), len(wl))):
            h = hl[i] if i < len(hl) else "(absent)"
            w = wl[i] if i < len(wl) else "(absent)"
            if h != w:
                diff = "line %d: manifest %r, data %r" % (i + 1, h, w)
                break
    check(have == want, "manifest: every digest is the file's digest", diff)

    names = [n for n, _ in listed()]
    check(names == data_files(), "manifest: every file listed, in sorted order",
          "manifest %s, directory %s" % (names, data_files()))
    check(len(names) == len(set(names)), "manifest: no file listed twice")
    bad = [n for n, d in listed()
           if len(d) != 64 or any(c not in "0123456789abcdef" for c in d)]
    check(not bad, "manifest: digests are 64 lowercase hex characters", bad)
    crlf = b"\r" in open(SUMS, "rb").read()
    check(not crlf, "manifest: LF line endings")


def test_tool_regenerates_manifest(tmp):
    out = os.path.join(tmp, "SHA256SUMS.tool")
    run = subprocess.run([sys.executable, TOOL, DATA, "-o", out],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check(run.returncode == 0, "tool: exits 0",
          run.stderr.decode().strip())
    if run.returncode != 0:
        return
    with open(out) as fh:
        got = fh.read()
    with open(SUMS) as fh:
        want = fh.read()
    check(got == want, "tool: reproduces the checked-in manifest byte for byte")
    run = subprocess.run([sys.executable, TOOL, DATA, "--check"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check(run.returncode == 0, "tool: --check passes on the tree",
          run.stderr.decode().strip())


def test_check_accepts_packaged_data():
    problems = cli._verify_models(DATA)
    check(problems == [], "check: the packaged directory has no problems",
          problems)
    status, text = capture(cli._report_model_data, DATA)
    check(status is None, "report: the packaged directory is not a mismatch",
          status)
    check("model data:" in text and "match" in text,
          "report: says the files match the manifest", text.strip())


def test_corrupted_copy_names_the_file(tmp):
    bad = os.path.join(tmp, "corrupt")
    shutil.copytree(DATA, bad)
    victim = "modelA.onnx"
    path = os.path.join(bad, victim)
    with open(path, "r+b") as fh:            # one byte, same file size
        fh.seek(4096)
        b = fh.read(1)
        fh.seek(4096)
        fh.write(bytes([b[0] ^ 0xFF]))
    want = dict(listed())[victim]
    problems = cli._verify_models(bad)
    check(problems is not None and len(problems) == 1,
          "corrupt: exactly one file reported", problems)
    got = dict((n, (e, a)) for n, e, a in (problems or []))
    check(victim in got and got[victim][0] == want and
          got[victim][1] == sha256(path),
          "corrupt: names %s, manifest digest and file digest both reported"
          % victim, got.get(victim))
    check(got.get(victim, (None, None))[0] != got.get(victim, (None, None))[1],
          "corrupt: the two digests differ")

    status, text = capture(cli._report_model_data, bad)
    lines = [l for l in text.splitlines() if victim in l]
    check(status == cli.MODEL_DATA_MISMATCH,
          "corrupt: reported as MODEL DATA MISMATCH (exit %d)"
          % cli.MODEL_DATA_MISMATCH, status)
    check("MODEL DATA MISMATCH" in text,
          "corrupt: the outcome is named", text.strip())
    check(len(lines) == 1 and want in lines[0] and sha256(path) in lines[0],
          "corrupt: one line naming the file and both digests", lines)
    check("ERROR" not in text and "Softmax" not in text,
          "corrupt: not reported as an engine error", text.strip())
    check(cli.MODEL_DATA_MISMATCH not in (0, 1, 2, 3, 4, 5, 6, 7),
          "corrupt: the status is distinct from the engine's and usage's")

    os.remove(os.path.join(bad, "modelC.onnx"))
    problems = cli._verify_models(bad)
    missing = [p for p in (problems or []) if p[0] == "modelC.onnx"]
    check(len(missing) == 1 and missing[0][2] is None,
          "missing: a listed file that is gone is reported as such", problems)
    _, text = capture(cli._report_model_data, bad)
    check(any("modelC.onnx" in l and "(missing)" in l
              for l in text.splitlines()),
          "missing: the line says (missing)", text.strip())


def test_selftest_stops_before_the_engine(tmp):
    bad = os.path.join(tmp, "selftest-corrupt")
    shutil.copytree(DATA, bad)
    victim = "yolov8n_160.onnx"
    with open(os.path.join(bad, victim), "r+b") as fh:   # the case that started this
        fh.seek(1000000)
        b = fh.read(1)
        fh.seek(1000000)
        fh.write(bytes([b[0] ^ 0xFF]))

    opened = []

    def fake_engine(*a, **k):
        opened.append(a)
        raise AssertionError("engine opened on data that failed the check")

    real_backend, real_engine = cli._backend_str, cli.Engine
    cli._backend_str = lambda: "vc4d (/run/vc4d.sock, stub)"
    cli.Engine = fake_engine
    try:
        code, text = capture(cli.cmd_selftest,
                             types.SimpleNamespace(models=bad, cache=None))
    finally:
        cli._backend_str, cli.Engine = real_backend, real_engine
    check(code == cli.MODEL_DATA_MISMATCH and not opened,
          "selftest: exits %d before it opens an engine"
          % cli.MODEL_DATA_MISMATCH, (code, opened))
    check("MODEL DATA MISMATCH" in text and "SELFTEST NOT RUN" in text,
          "selftest: names the mismatch and says no case ran", text.strip())
    check(any(victim in l and "(missing)" not in l for l in text.splitlines()),
          "selftest: names the damaged file", text.strip())


def test_malformed_manifest_is_a_usage_error(tmp):
    bad = os.path.join(tmp, "malformed")
    shutil.copytree(DATA, bad)
    with open(os.path.join(bad, cli.MODEL_SUMS), "w") as fh:
        fh.write("not-a-digest  modelA.onnx\n")
    err = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stderr(err):
            cli._verify_models(bad)
    except SystemExit as e:
        code = e.code
    check(code == 2 and "not a sha256sum line" in err.getvalue(),
          "manifest: a malformed line is refused with its line number",
          err.getvalue().strip())


def test_no_manifest_is_not_an_error(tmp):
    own = os.path.join(tmp, "own-models")
    os.makedirs(own)
    shutil.copy(os.path.join(DATA, "modelA.onnx"), own)
    check(cli._verify_models(own) is None,
          "absent: no manifest means no check, not a failure")
    status, text = capture(cli._report_model_data, own)
    check(status is None, "absent: the run may continue", status)
    check(text.count("integrity check skipped") == 1,
          "absent: said once", text.strip())


def main():
    tmp = tempfile.mkdtemp(prefix="model-sums-", dir=os.environ.get("TMPDIR"))
    try:
        for fn in (test_manifest_is_current, test_tool_regenerates_manifest,
                   test_check_accepts_packaged_data,
                   test_corrupted_copy_names_the_file,
                   test_selftest_stops_before_the_engine,
                   test_malformed_manifest_is_a_usage_error,
                   test_no_manifest_is_not_an_error):
            print("-- %s" % fn.__name__)
            if fn.__code__.co_argcount:
                fn(tmp)
            else:
                fn()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print("\nFAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("\nall model manifest checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
