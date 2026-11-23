#!/usr/bin/env python3
"""test_artifact.py -- the .vc4x container and its loader checks, without a GPU.

Everything here runs on a machine with no VideoCore IV: the container format,
the key checks, the section hashes, the pinned arena allocator and the
"initializer is only a panel" analysis.  Building a real artifact and running
it is the on-board path (``vc4infer build`` and the artifact measurement
suite), not this file.

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_artifact.py

Exit status is 0 only when every check passed.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding                              # noqa: E402
from vc4infer import artifact as art                        # noqa: E402
from vc4infer._errors import Vc4InferError                  # noqa: E402
from vc4infer._onnxrun import Pool, Tensor, _PanelWeight    # noqa: E402

FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def raises(fn, kind, needle=None):
    try:
        fn()
    except kind as e:
        text = str(e)
        return (needle is None or needle in text), text
    except Exception as e:                                 # noqa: BLE001
        return False, "raised %s instead: %s" % (type(e).__name__, e)
    return False, "did not raise"


# ------------------------------------------------------------------ fixtures
def make_container(path, *, extra_keys=None, sections=None, plan=None):
    sections = sections or [("graph", b"onnx-bytes"), ("weights", b"w" * 64)]
    manifest = {
        "format": art.FORMAT,
        "format_version": art.LAYOUT_VERSION,
        "layout_version": art.LAYOUT_VERSION,
        "created": art._now(),
        "keys": {
            "generator_digest": art.generator_digest(),
            "abi_tag": art.kernel_abi_tag(),
            "engine_digest": art.engine_digest(),
            "board": art.board_model(),
            "arch": "test",
            "engine_version": "test",
        },
        "model": {"name": "t.onnx", "graph_inline": True,
                  "graph_section": "graph",
                  "graph_sha256": hashlib.sha256(b"onnx-bytes").hexdigest(),
                  "nodes": 1, "opt": "none", "passes": [], "opt_keys": []},
        "shapes": {},
        "plan": plan or {
            "arena_bytes": 1 << 26, "pool_bytes": 4096,
            "segment_bytes": [4096], "scratch_bytes": 1024,
            "allocations": [{"key": "a#0", "seg": 0, "offset": 0,
                             "bytes": 128, "requested": 128}],
        },
        "weights": {"section": "weights", "bytes": 64, "entries": [],
                    "dropped": []},
        "kernels": {"embedded": True, "entries": [], "bytes": 0},
        "tuning": {"autotune": "off", "records": []},
    }
    if extra_keys:
        manifest["keys"].update(extra_keys)
    art.write(path, manifest, sections)
    return manifest


def tamper(path, fn):
    """Rewrite the manifest after load/mutate via the private container."""
    with open(path, "rb") as f:
        head = f.read(art.HEADER.size)
    magic, version, mlen, plen, digest = art.HEADER.unpack(head)
    with open(path, "rb") as f:
        f.seek(art.HEADER.size)
        manifest = json.loads(f.read(mlen))
        payload = f.read(plen)
    fn(manifest)
    hdr = json.dumps(manifest, sort_keys=True).encode("ascii")
    with open(path, "wb") as f:
        f.write(art.HEADER.pack(art.MAGIC, version, len(hdr), len(payload),
                                hashlib.sha256(hdr).digest()))
        f.write(hdr)
        f.write(payload)


# ------------------------------------------------------------------- format
def test_roundtrip(tmp):
    p = os.path.join(tmp, "ok.vc4x")
    make_container(p)
    a = art.Artifact.load(p, verify=True)
    check(a.graph_bytes() == b"onnx-bytes", "roundtrip: graph section read back")
    check(a.keys["board"] == art.board_model(), "roundtrip: keys preserved")
    check(len(a.verify()) == 2, "roundtrip: verify() walks every section")
    check(a.pool_bytes == 4096 and a.scratch_bytes == 1024,
          "roundtrip: plan sizes preserved")
    check(a.allocations["a#0"] == (0, 0, 128),
          "roundtrip: allocation map read back")
    check(len(str(a)) > 0 and "Artifact" in repr(a),
          "roundtrip: repr is informative")
    check(art.is_artifact(p), "roundtrip: magic detection")
    check(not art.is_artifact(os.path.join(HERE, "test_artifact.py")),
          "roundtrip: a Python file is not an artifact")


def test_container_damage(tmp):
    good = os.path.join(tmp, "good.vc4x")
    make_container(good)
    with open(good, "rb") as f:
        blob = f.read()

    p = os.path.join(tmp, "short.vc4x")
    with open(p, "wb") as f:
        f.write(blob[:20])
    ok, text = raises(lambda: art.Artifact.load(p), art.ArtifactError)
    check(ok and "shorter than" in text, "damage: a truncated header is refused",
          text)

    p = os.path.join(tmp, "magic.vc4x")
    with open(p, "wb") as f:
        f.write(b"NOPE" + blob[4:])
    ok, text = raises(lambda: art.Artifact.load(p), art.ArtifactError)
    check(ok and "not a vc4x" in text, "damage: a bad magic is refused", text)

    p = os.path.join(tmp, "hdr.vc4x")
    data = bytearray(blob)
    data[art.HEADER.size + 2] ^= 0x40           # inside the manifest
    with open(p, "wb") as f:
        f.write(data)
    ok, text = raises(lambda: art.Artifact.load(p), art.ArtifactError)
    check(ok and "manifest" in text.lower(),
          "damage: a manifest checksum mismatch is refused", text)

    p = os.path.join(tmp, "len.vc4x")
    with open(p, "wb") as f:
        f.write(blob[:-3])
    ok, text = raises(lambda: art.Artifact.load(p), art.ArtifactError)
    check(ok and "payload" in text.lower(),
          "damage: a short payload is refused", text)


def test_section_hash(tmp):
    p = os.path.join(tmp, "corrupt.vc4x")
    make_container(p)
    with open(p, "rb") as f:
        blob = bytearray(f.read())
    head = art.HEADER.unpack(bytes(blob[:art.HEADER.size]))
    blob[art.HEADER.size + head[2]] ^= 0xff     # first byte of the graph
    with open(p, "wb") as f:
        f.write(blob)
    a = art.Artifact.load(p)
    ok, text = raises(a.verify, art.ArtifactError)
    check(ok and "sha256" in text, "sections: verify() catches corruption",
          text)
    a = art.Artifact.load(p)
    ok, text = raises(a.graph_bytes, art.ArtifactError)
    check(ok and "sha256" in text,
          "sections: graph_bytes() hashes the graph it returns", text)


def test_keys(tmp):
    base = os.path.join(tmp, "key.vc4x")
    make_container(base)
    a = art.Artifact.load(base)
    check(a.validate(arena=1 << 26), "keys: an intact artifact validates")

    cases = {
        "generator_digest": "0" * 64,
        "engine_digest": "f" * 64,
        "abi_tag": "some-other-abi",
        "board": "Raspberry Pi 5 Model B",
    }
    for key, value in cases.items():
        p = os.path.join(tmp, "key-%s.vc4x" % key)
        make_container(p, extra_keys={key: value})
        a = art.Artifact.load(p)
        ok, text = raises(lambda a=a: a.validate(arena=1 << 26),
                          art.ArtifactMismatch)
        check(ok and key in text,
              "keys: %s mismatch is refused by name" % key, text)
        try:
            a.validate(arena=1 << 26)
        except art.ArtifactMismatch as e:
            check(e.key == key, "keys: %s mismatch reports its key" % key)

    p = os.path.join(tmp, "layout.vc4x")
    make_container(p)

    def bump(m):
        m["layout_version"] = art.LAYOUT_VERSION + 1

    tamper(p, bump)
    ok, text = raises(lambda: art.Artifact.load(p), art.ArtifactMismatch)
    check(ok and "layout_version" in text,
          "keys: a newer layout version is refused at load", text)

    p = os.path.join(tmp, "arena.vc4x")
    make_container(p)
    a = art.Artifact.load(p)
    ok, text = raises(lambda: a.validate(arena=1024), art.ArtifactMismatch)
    check(ok and "arena" in text,
          "keys: a plan that does not fit the arena is refused", text)


def test_referenced_graph(tmp):
    p = os.path.join(tmp, "ref.vc4x")
    onnx_path = os.path.join(tmp, "ref.onnx")
    with open(onnx_path, "wb") as f:
        f.write(b"graph-v1")

    def manifest_ref(m):
        m["sections"] = [s for s in m["sections"] if s["name"] != "graph"]
        m["model"]["graph_inline"] = False
        m["model"]["graph_path"] = onnx_path
        m["model"]["graph_sha256"] = hashlib.sha256(b"graph-v1").hexdigest()

    make_container(p, sections=[("weights", b"w" * 64)])
    tamper(p, manifest_ref)
    a = art.Artifact.load(p)
    check(a.graph_bytes() == b"graph-v1",
          "reference: the graph is read from the referenced file")
    with open(onnx_path, "wb") as f:
        f.write(b"graph-v2")
    ok, text = raises(a.graph_bytes, art.ArtifactError)
    check(ok and "changed" in text,
          "reference: a changed referenced graph is refused", text)
    os.unlink(onnx_path)
    a = art.Artifact.load(p)
    ok, text = raises(a.graph_bytes, art.ArtifactError)
    check(ok and "not readable" in text,
          "reference: a missing referenced graph is refused", text)


# --------------------------------------------------------------- pinned pool
class FakeEngine:
    """Just enough of the binding Engine for the pool allocator."""

    def __init__(self, base=0x100000, arena=1 << 26):
        self.base = base
        self.top = base
        self.arena_bytes = arena

    def alloc(self, n):
        n = (int(n) + 63) & ~63
        b = _binding.Buf(self.top, n)
        self.top += n
        return b


def test_pinned_pool(tmp):
    pin = {"a#0": (0, 0, 128), "b#0": (0, 128, 128),
           "c#0": (0, 0, 256)}
    keys = []

    def label():
        return keys.pop(0)

    eng = FakeEngine()
    pool = Pool(eng, 4096, pin=pin, segment_bytes=[4096], labeler=label)
    keys[:] = ["a#0"]
    a = pool.alloc(100)
    check(a.off == eng.base and a.bytes == 100,
          "plan: a pinned allocation lands at its recorded offset")
    keys[:] = ["b#0"]
    b = pool.alloc(120)
    check(b.off == eng.base + 128,
          "plan: the next pinned allocation follows it")
    pool.release(a)
    keys[:] = ["c#0"]
    ok, text = raises(lambda: pool.alloc(200), art.ArtifactPlanError)
    check(ok and "overlaps" in text,
          "plan: a plan that puts two live tensors together is refused", text)
    keys[:] = ["d#0"]
    ok, text = raises(lambda: pool.alloc(64), art.ArtifactPlanError)
    check(ok and "not in the artifact" in text,
          "plan: an unplanned allocation is refused by key", text)

    keys[:] = ["a#0"]
    ok, text = raises(lambda: pool.alloc(200), art.ArtifactPlanError)
    check(ok and "in the plan" in text,
          "plan: a size that changed since the build is refused", text)


# ----------------------------------------------------- dropped initializers
def conv_model(shared=False):
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 16, 8, 8])
    z = helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1, 16, 8, 8])
    w = numpy_helper.from_array(np.zeros((16, 16, 3, 3), np.float32), "w")
    b = numpy_helper.from_array(np.zeros(16, np.float32), "b")
    nodes = [helper.make_node("Conv", ["x", "w", "b"], ["y"], name="conv")]
    outs = [y]
    if shared:
        # the same weight feeds an Add, so its bytes are needed as a tensor
        nodes.append(helper.make_node("Add", ["w", "y"], ["z"], name="add"))
        outs = [y, z]
    g = helper.make_graph(nodes, "g", [x], outs, [w, b])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])


def test_dropped(tmp):
    model = conv_model()
    host = {"w": np.zeros((16, 16, 3, 3), np.float32),
            "b": np.zeros(16, np.float32)}
    panels = {("panel", "conv", 16, 16, 3, 3, 1, None): object()}
    got = {e["name"] for e in art._dropped_initializers(model, panels, host)}
    check(got == {"w", "b"},
          "drop: a panel-backed conv's weight and bias are not uploaded", got)

    got = {e["name"] for e in
           art._dropped_initializers(conv_model(shared=True), panels, host)}
    check(got == {"b"},
          "drop: an initializer with a second consumer is kept", got)

    got = {e["name"] for e in art._dropped_initializers(model, {}, host)}
    check(got == set(),
          "drop: without a panel entry nothing is dropped", got)

    # the stub refuses arena access instead of reading an unrelated buffer
    stub = _PanelWeight((4, 8))
    check(isinstance(stub, Tensor) and stub.shape == (4, 8),
          "drop: the stub is a Tensor with the weight's shape")
    ok, text = raises(lambda: stub.buf, art.ArtifactPlanError)
    check(ok and "packed panel" in text,
          "drop: reading the dropped weight's arena bytes is refused", text)


# -------------------------------------------------------------- exec state
def test_seed_cache(tmp):
    """The embedded blobs must land with the stamp the cache compares."""
    p = os.path.join(tmp, "kern.vc4x")
    stamp = "0123456789abcdef" * 4
    blob = b"\x00\x01\x02\x03" * 4
    make_container(p, sections=[("graph", b"onnx-bytes"),
                                ("kernels", blob)])
    tamper(p, lambda m: m.update({"kernels": {
        "embedded": True, "section": "kernels", "bytes": len(blob),
        "entries": [{"name": "gemm_test", "offset": 0, "bytes": len(blob),
                     "sha256": hashlib.sha256(blob).hexdigest(),
                     "stamp": stamp}]}}))
    a = art.Artifact.load(p)
    cache = os.path.join(tmp, "kc")
    written, present = art.seed_cache(a, cache)
    check((written, present) == (1, 0), "seed: the first call writes the blob",
          (written, present))
    check(os.path.isfile(os.path.join(cache, "gemm_test.bin")),
          "seed: the blob file is there")
    with open(os.path.join(cache, "gemm_test.bin.stamp"), encoding="ascii") as f:
        check(f.read().strip() == stamp,
              "seed: the stamp is the blob's own cache digest, not the "
              "generator digest")
    written, present = art.seed_cache(a, cache)
    check((written, present) == (0, 1),
          "seed: the second call leaves a valid entry alone", (written, present))
    stub = os.path.join(tmp, "nostamp.vc4x")
    make_container(stub, sections=[("graph", b"onnx-bytes"),
                                   ("kernels", blob)])
    tamper(stub, lambda m: m.update({"kernels": {
        "embedded": True, "section": "kernels", "bytes": len(blob),
        "entries": [{"name": "gemm_nostamp", "offset": 0, "bytes": len(blob),
                     "sha256": hashlib.sha256(blob).hexdigest(),
                     "stamp": None}]}}))
    a2 = art.Artifact.load(stub)
    art.seed_cache(a2, cache)
    check(not os.path.exists(os.path.join(cache, "gemm_nostamp.bin.stamp")),
          "seed: a blob with no recorded stamp gets no false stamp")


def test_exec_state(tmp):
    raw = {
        "silu_skip": [3], "silu_src": {"7": "x"},
        "skip": [3], "skip_reason": {"3": "fused"},
        "fused": {"3": {"fused": {"by": "silu"}}},
        "fused_skip": {"3": "silu"},
        "alias": {"4": ["out", "src"]},
    }
    p = os.path.join(tmp, "state.vc4x")
    make_container(p)
    tamper(p, lambda m: m.update({"exec": raw}))
    a = art.Artifact.load(p)
    st = a.exec_state
    check(st["skip"] == {3} and st["silu_src"] == {7: "x"} and
          st["alias"] == {4: ("out", "src")},
          "exec: ints and tuples survive the manifest", st)
    check(art.Artifact.load(os.path.join(tmp, "ok.vc4x")).exec_state is None,
          "exec: an artifact with no rewrite tables says so")


def test_frozen_plan(tmp=None):
    """The plan records survive the container and refuse what they cannot use.

    A build records the plan in the build process's pool segments; a load
    receives different ones and every offset is rebased.  The refusals are the
    ABI, the record count, the segment geometry and the kernel table - each
    names what disagrees instead of replaying into another buffer.
    """
    from vc4infer import _binding, _plan
    import ctypes

    op = _binding.PlanOp()
    op.kind = _binding.PLAN_EW
    op.a_off, op.a_bytes = 1050, 40
    op.c_off, op.c_bytes = 2040, 40
    op.n = 16
    op.p[0] = 3
    plan = _plan.Plan([("c", (_binding.PlanOp * 1)(op)), ("p", 7), ("s", 8)],
                      [(3,), (7,), (8,)],
                      {"a": (1050, 40, (1, 4), "nchw", 0),
                       "y": (2040, 40, (1, 4), "nchw", 0)},
                      {"X": ((1, 4), 4)}, 1, 1, [7])
    block, blob = _plan.frozen_plan(plan, [(1000, 100), (2000, 50)], 152,
                                    "kernels", 9)
    loaded = _plan.plan_from_frozen(block, blob, 152,
                                    [(3000, 100), (5000, 50)], "kernels")
    rec = loaded.steps[0][1][0]
    check(rec.a_off == 3050 and rec.c_off == 5040,
          "the operation offsets are rebased (%d, %d)"
          % (rec.a_off, rec.c_off))
    check(loaded.desc["a"][0] == 3050 and loaded.desc["y"][0] == 5040,
          "the output descriptors are rebased")
    check(loaded.from_artifact and
          [k for k, _ in loaded.steps] == ["c", "p", "s"] and
          loaded.nodes == [(3,), (7,), (8,)],
          "the step and node tables survive")

    def refused(label, new_block, new_blob, new_segs, new_abi, new_kernels):
        try:
            _plan.plan_from_frozen(new_block, new_blob, new_abi, new_segs,
                                   new_kernels)
        except _plan.FrozenPlanError as e:
            check(True, "%s refused: %s" % (label, str(e)[:60]))
            return
        check(False, "%s was not refused" % label)

    b = dict(block)
    b["version"] = _plan.FROZEN_VERSION + 1
    refused("version", b, blob, [(3000, 100), (5000, 50)], 152, "kernels")
    b = dict(block)
    b["abi_bytes"] = 160
    refused("record size", b, blob, [(3000, 100), (5000, 50)], 152,
            "kernels")
    refused("section length", block, blob[:-8],
            [(3000, 100), (5000, 50)], 152, "kernels")
    refused("segment count", block, blob, [(3000, 150)], 152, "kernels")
    refused("segment sizes", block, blob, [(3000, 100), (5000, 60)], 152,
            "kernels")
    refused("kernel table", block, blob, [(3000, 100), (5000, 50)], 152,
            "other")


def main():
    tmp = tempfile.mkdtemp(prefix="vc4x-test-", dir=os.environ.get("TMPDIR"))
    try:
        for fn in (test_roundtrip, test_container_damage, test_section_hash,
                   test_keys, test_referenced_graph, test_pinned_pool,
                   test_dropped, test_seed_cache, test_exec_state,
                   test_frozen_plan):
            print("-- %s" % fn.__name__)
            fn(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print("\nFAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("\nall artifact checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
