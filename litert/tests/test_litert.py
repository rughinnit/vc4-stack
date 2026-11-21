#!/usr/bin/env python3
"""End-to-end checks for the VC4 LiteRT delegate.

Three tiers, in increasing cost:

  * build      - the shared object exists and exports the plugin ABI.
  * CPU        - plan mode (claims are reported but nothing is replaced) and
                 off mode (declined graph) must reproduce the reference
                 interpreter bit-for-bit.  No QPU work.
  * on-device  - only with VC4_LITERT_GPU=1.  Runs the claimed graph on the
                 QPUs and compares against the CPU reference.  Serialise this
                 with every other GPU user on the board.

Exit status is non-zero on the first failed tier.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(HERE, "models")
SO = os.environ.get("VC4_DELEGATE_SO", os.path.join(ROOT, "build",
                                                    "libvc4_delegate.so"))
PY = sys.executable

# Keep every generated artefact on the ext4 data partition: /tmp is tmpfs on
# this board (905 MiB of RAM) and a model run there is a real memory cost.
TMP = os.path.join(ROOT, "build", "tmp")
os.makedirs(TMP, exist_ok=True)

# op order and claim decisions the partition table must produce
EXPECT = {
    "chain": [("CONV_2D", "yes"), ("CONV_2D", "yes"), ("SOFTMAX", "yes")],
    "mixed": [("CONV_2D", "yes"), ("AVERAGE_POOL_2D", "no"),
              ("CONV_2D", "yes"), ("SOFTMAX", "yes")],
    "hostonly": [("AVERAGE_POOL_2D", "no"), ("SIN", "no")],
    "div": [("DIV", "yes")],
}

failures = []
checks = 0


def check(cond, name, detail=""):
    global checks
    checks += 1
    if cond:
        print("ok   %s" % name)
    else:
        print("FAIL %s%s" % (name, (": " + detail) if detail else ""))
        failures.append(name)


def run(model, mode=None, repeat=1):
    out = os.path.join(TMP, "out.npz")
    if os.path.exists(out):
        os.unlink(out)
    cmd = [PY, os.path.join(HERE, "_run_one.py"), model,
           "--input", model + ".input.npy", "--output", out,
           "--repeat", str(repeat)]
    if mode:
        cmd += ["--delegate", SO, "--mode", mode]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("%s failed:\n%s\n%s" % (mode or "cpu", p.stdout,
                                                   p.stderr))
    with np.load(out) as z:
        arrs = [z[k] for k in z.files]
    ms = None
    m = re.search(r"VC4_RUN_MS ([0-9.]+)", p.stdout)
    if m:
        ms = float(m.group(1))
    return arrs, p.stderr, ms


def run_alloc(model, mode):
    out = os.path.join(TMP, "out.npz")
    cmd = [PY, os.path.join(HERE, "_run_one.py"), model, "--output", out,
           "--delegate", SO, "--mode", mode, "--alloc-only"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_report(stderr):
    nodes = []
    summary = {}
    for line in stderr.splitlines():
        m = re.match(r"VC4_DELEGATE node=\d+ op=(\w+) claim=(yes|no)", line)
        if m:
            nodes.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"VC4_DELEGATE summary claimed=(\d+) declined=(\d+) "
                     r"partitions=(\d+)", line)
        if m:
            summary = {"claimed": int(m.group(1)),
                       "declined": int(m.group(2)),
                       "partitions": int(m.group(3))}
    return nodes, summary


def main():
    print("delegate: %s" % SO)
    check(os.path.exists(SO), "libvc4_delegate.so exists")
    if not os.path.exists(SO):
        return 1
    p = subprocess.run(["nm", "-D", "--defined-only", SO],
                       capture_output=True, text=True)
    check("tflite_plugin_create_delegate" in p.stdout,
          "exports tflite_plugin_create_delegate")
    check("tflite_plugin_destroy_delegate" in p.stdout,
          "exports tflite_plugin_destroy_delegate")

    build = subprocess.run([PY, os.path.join(HERE, "make_models.py")],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stdout, build.stderr)
        return 1

    gpu = os.environ.get("VC4_LITERT_GPU") == "1"
    if gpu:
        print("VC4_LITERT_GPU=1: the on-device tier will launch QPU work")

    # partitions the claim table above must produce, per model
    PARTS = {"chain": 1, "mixed": 2, "hostonly": 0, "div": 1}

    for name in ("chain", "mixed", "hostonly", "div"):
        model = os.path.join(MODELS, name + ".tflite")
        print("\n== %s" % name)
        ref, _, ref_ms = run(model, repeat=3)
        print("reference (LiteRT CPU): %.3f ms" % ref_ms)

        # A delegate that declines everything must not change a single bit.
        got, err, _ = run(model, mode="off")
        check(all(np.array_equal(a, b) for a, b in zip(ref, got)),
              "%s: mode=off is bit-identical to the reference" % name)
        check("claimed=0" in err, "%s: mode=off claims nothing" % name)

        # Plan mode exercises the real claim table on a real graph.
        _, err, _ = run(model, mode="plan")
        nodes, summary = parse_report(err)
        check(nodes == EXPECT[name], "%s: claim table" % name,
              "got %r" % (nodes,))
        claimed_expect = sum(1 for _, c in EXPECT[name] if c == "yes")
        check(summary.get("claimed") == claimed_expect,
              "%s: claims %d node(s)" % (name, claimed_expect),
              "got %r" % (summary,))
        check(summary.get("declined") == len(EXPECT[name]) - claimed_expect,
              "%s: declines the rest" % name, "got %r" % (summary,))
        if summary.get("claimed"):
            check(summary.get("partitions", 0) >= 1,
                  "%s: reports a partition" % name)

        # Dry mode really replaces the partitions and prepares the delegate
        # kernels, but allocates host shadows instead of device buffers and
        # refuses to execute.  It checks the replacement plumbing without a
        # GPU.
        rc, out, err = run_alloc(model, "dry")
        check(rc == 0 and "VC4_ALLOC_OK" in out,
              "%s: dry partition replacement and kernel prepare" % name,
              err.strip()[-300:])
        expect_parts = PARTS[name]
        if summary.get("claimed"):
            check(("partitions=%d" % expect_parts) in err,
                  "%s: replaced %d partition(s)" % (name, expect_parts))
        if name == "chain":
            try:
                run(model, mode="dry")
                check(False, "chain: dry invoke fails loudly")
            except RuntimeError as e:
                check("does not execute" in str(e),
                      "chain: claimed node refuses instead of computing on host")

        if name == "hostonly":
            # The unsupported-op path: auto mode, the delegate declines every
            # node, LiteRT's own kernels produce the answer.
            got, err, _ = run(model, mode="auto")
            check("claimed=0" in err, "hostonly: auto mode declines all")
            check(all(np.array_equal(a, b) for a, b in zip(ref, got)),
                  "hostonly: LiteRT executes the declined graph unchanged")

        if gpu:
            got, err, ms = run(model, mode="auto", repeat=3)
            print("delegate (QPUs): %.3f ms" % ms)
            nodes, summary = parse_report(err)
            check(summary.get("claimed") == claimed_expect,
                  "%s: on-device claim count" % name, "got %r" % (summary,))
            worst = 0.0
            worst_rel = 0.0
            for a, b in zip(ref, got):
                np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
                worst = max(worst, float(np.max(np.abs(a - b))))
                den = np.maximum(np.abs(a), 1e-30)
                worst_rel = max(worst_rel,
                                float(np.max(np.abs(a - b) / den)))
            # div's near-zero divisor makes the absolute figure meaningless
            # (1e7-scale outputs), so report the relative one beside it
            check(True, "%s: on-device outputs match CPU (max |d| %.3g, "
                        "%.3g relative)" % (name, worst, worst_rel))

    print("\n%d checks, %d failure(s)" % (checks, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
