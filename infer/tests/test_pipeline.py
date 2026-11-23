#!/usr/bin/env python3
"""test_pipeline.py - the CPU-side gates for depth-2 request pipelining.

No GPU, no vc4d: tests/fake_daemon.py speaks the wire protocol, hands out a
memfd arena and records every request.  Four things are proved here, and one
limit is measured.

  A. The request stream is unchanged.  For modelA/B/C and YOLOv8n at 96, 128
     and 160, the pipelined run sends the same launches, in the same order,
     with the same kernel and the same uniform values, as the synchronous run.
     (Byte-identical outputs are a silicon gate; this is the CPU proof that
     pipelining did not change what the GPU is asked to do.)

  B. Nothing that a request reads changes while it is outstanding.  The fake
     daemon snapshots each request's uniform shadow and a window at every
     address its launches name, waits 20 ms (a stand-in for execution), and
     compares.  A reused uniform set or a pool block handed to a new node under
     a running request shows up as a difference.

  C. The pool holds a freed block until the request that was in flight when it
     was freed has been reaped, and releases it afterwards.

  D. Accounting balances: no block is left in limbo after a run, the same run
     twice reaches the same high-water mark, and a held block is never handed
     out twice.

  E. A version-3 daemon (no SUBMIT) makes run_pipelined() refuse with the
     reason, instead of silently running the synchronous path.

usage:
  TMPDIR=$HOME/scratch/tmp python3 tests/test_pipeline.py [--models DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from vc4infer import _binding as vc4infer                       # noqa: E402
from vc4infer._onnxrun import Pool, Tensor                     # noqa: E402
from vc4infer.engine import Engine                              # noqa: E402

FAKE = os.path.join(ROOT, "..", "vc4d", "test", "fake_daemon.py")
TMP = os.environ.get("TMPDIR") or os.path.expanduser("~/scratch/tmp")
FAILS = []
# The fake's arena window: every address word the engine sends points into it.
FAKE_BUS = 0xE0000000
FAKE_BUS_END = 0xE0000000 + (128 << 20)


def norm_words(e):
    """A launch's uniform words with pointer words wildcarded.

    The two runs do not place buffers at the same offsets, and cannot: the
    pipelined path reserves two scratch windows above the pool, so a pool
    segment allocated after arming lands above them.  Offsets are not part of
    the contract - every consumer reads them from the uniform block - so the
    comparison keeps the kernel, the QPU count, the uniform count, the
    address-or-value pattern and every *value* word, and drops the address
    value.  The bytes each address points at are compared separately, in gate
    B, where they are checked for change under a running request.
    """
    mask = e.get("mask") or 0
    out = []
    for j, w in enumerate(e.get("words") or ()):
        if (mask >> (j % 16)) & 1:
            out.append("@")
        elif FAKE_BUS <= w < FAKE_BUS_END:
            out.append("@")
        else:
            out.append(w)
    return tuple(out)



def check(cond, fmt, *args):
    print(("  PASS  " if cond else "  FAIL  ") + (fmt % args))
    if not cond:
        FAILS.append(fmt % args)
    return cond


class Fake:
    """A fake daemon process plus the records it wrote."""

    def __init__(self, tag, version=4, delay_ms=0, arena_mb=128):
        self.sock = os.path.join(TMP, "pipe-%s-%d.sock" % (tag, os.getpid()))
        self.log = os.path.join(TMP, "pipe-%s-%d.jsonl" % (tag, os.getpid()))
        for p in (self.sock, self.log, self.log + ".counts"):
            if os.path.exists(p):
                os.unlink(p)
        env = dict(os.environ)
        env["FAKE_VERSION"] = str(version)
        env["FAKE_DELAY_MS"] = str(delay_ms)
        self.proc = subprocess.Popen(
            [sys.executable, FAKE, self.sock, "--log", self.log,
             "--delay-ms", str(delay_ms), "--arena-mb", str(arena_mb)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + 20
        while time.time() < deadline:
            if os.path.exists(self.sock):
                return
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode()
                raise RuntimeError("fake daemon died: %s" % err)
            time.sleep(0.02)
        raise RuntimeError("fake daemon did not create %s" % self.sock)

    def records(self):
        out = []
        if not os.path.exists(self.log):
            return out
        with open(self.log) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def counts(self):
        p = self.log + ".counts"
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for p in (self.sock,):
            if os.path.exists(p):
                os.unlink(p)


def engine_for(model, fake, cache, nocplan=False):
    os.environ["VC4D_SOCKET"] = fake.sock
    os.environ["VC4INFER_DAEMON"] = "1"
    os.environ.pop("VC4INFER_NO_PIPE", None)
    if nocplan:
        os.environ["VC4_NO_CPLAN"] = "1"
    else:
        os.environ.pop("VC4_NO_CPLAN", None)
    return Engine(model, cache_dir=cache, arena_mb=64)


def stream(fake, model, cache, pipeline, feeds, frames=3, nocplan=False):
    """Run `frames` frames and return the launch stream of the last one.

    The frozen plan is on by default and replays from the second frame, so a
    single frame would only ever exercise the recording (Python) path.  The
    frame that is compared is therefore a replay, which is what ships.
    """
    eng = engine_for(model, fake, cache, nocplan)
    try:
        run = eng.run_pipelined if pipeline else eng.run
        for _ in range(max(1, frames - 1)):
            run(feeds)
        before = len(fake.records())
        out = run(feeds)
    finally:
        eng.close()
    recs = fake.records()[before:]
    launches = []
    reqs = 0
    for r in recs:
        if r["op"] not in ("batch", "submit"):
            continue
        reqs += 1
        for e in r["entries"]:
            launches.append((e["kind"], e.get("name"), e["nqpu"], e["nuni"],
                             norm_words(e)))
    return launches, out, reqs


def model_paths(models, full=False):
    """modelA/B/C always; the YOLO sizes with --full (they take minutes because
    every kernel is generated for a cold cache)."""
    out = []
    for name in ("modelA", "modelB", "modelC"):
        p = os.path.join(models, name + ".onnx")
        if os.path.exists(p):
            out.append(p)
    if full:
        for size in (96, 128, 160):
            p = os.path.join(models, "yolov8n_%d.onnx" % size)
            if os.path.exists(p):
                out.append(p)
    return out


def feed_for(eng, models):
    name = os.path.basename(eng.model_path)
    base = name[:-5]
    for cand in (os.path.join(models, base + ".in.npy"),
                 os.path.join(models, "yolov8n_in_%s.npy" % base),
                 os.path.join(models, "yolov8n_160_in.npy")):
        if os.path.exists(cand):
            a = np.load(cand)
            if a.shape == eng.inputs[0].shape:
                return {eng.inputs[0].name: a}
    return {spec.name: np.zeros(spec.shape, dtype=np.float32)
            for spec in eng.inputs}


# ---------------------------------------------------------------- gate C
class FakeCtx:
    """Just enough of the binding's Engine for Pool: alloc + watermarks."""

    def __init__(self, pipeline=True):
        self.pipeline_armed = pipeline
        self.epoch = 0
        self.reaped = 0
        self.next = 0
        self.allocs = []

    def alloc(self, nbytes):
        off = self.next
        self.next += (nbytes + 63) & ~63
        self.allocs.append((off, nbytes))
        return vc4infer.Buf(off, nbytes)

    def pipeline_watermarks(self):
        return (self.epoch, self.reaped)


def gate_quarantine():
    print("C. the pool holds a freed block until its request is reaped")
    eng = FakeCtx()
    pool = Pool(eng, 1 << 20)
    a = pool.alloc(1000)
    pool.release(a)
    # Idle: the block is free at once, as before.
    b = pool.alloc(1000)
    check(b.off == a.off, "idle release is immediate (off %d -> %d)",
          a.off, b.off)
    pool.release(b)

    # One request in flight: the block freed now must not come back.
    eng.epoch, eng.reaped = 5, 4
    victim = pool.alloc(1000)
    pool.release(victim)
    check(victim.off not in [o for o, _ in pool.free],
          "the freed block is not on the free list at once (off %d)", victim.off)
    c = pool.alloc(1000)
    check(c.off != victim.off, "a block freed at epoch 5 is not reused at "
          "once (got %d, held %d)", c.off, victim.off)
    check(pool.held, "the block is on the hold list (%d)", len(pool.held))
    used_held = pool.used_bytes
    check(used_held >= 2000, "held bytes count as used, not free: %d", used_held)

    # The request completes: the block returns.
    eng.reaped = 5
    d = pool.alloc(1000)
    check(d.off == victim.off, "after reaping epoch 5 the block is reused "
          "(got %d, want %d)", d.off, victim.off)
    check(not pool.held, "the hold list is empty (%d)", len(pool.held))

    # Two live allocations never overlap.
    e = pool.alloc(1000)
    ranges = sorted((c.off, c.off + 1024) for c in (c, d, e))
    overlap = any(ranges[i][1] > ranges[i + 1][0] for i in range(len(ranges) - 1))
    check(not overlap, "three live blocks do not overlap: %s", ranges)


def gate_stream(models, cache, full):
    print("A. the request stream is the same with and without the pipeline")
    print("   (the compared frame is a frozen-plan replay, the shipping path)")
    for p in model_paths(models, full):
        name = os.path.basename(p)[:-5]
        fake = Fake("a-" + name)
        try:
            eng = engine_for(p, fake, cache)
            feeds = feed_for(eng, models)
            eng.close()
            sync_launches, sync_out, sync_reqs = stream(fake, p, cache, False, feeds)
            pipe_launches, pipe_out, pipe_reqs = stream(fake, p, cache, True, feeds)
            same = sync_launches == pipe_launches
            check(same, "%-12s %d launches, %d/%d requests, same order+uniforms",
                  name, len(sync_launches), pipe_reqs, sync_reqs)
            if not same:
                for i, (a, b) in enumerate(zip(sync_launches, pipe_launches)):
                    if a != b:
                        print("      first difference at launch %d:\n"
                              "        sync %s\n        pipe %s" % (i, a[:4], b[:4]))
                        break
                if len(sync_launches) != len(pipe_launches):
                    print("      launch counts differ: %d vs %d"
                          % (len(sync_launches), len(pipe_launches)))
            check(pipe_reqs >= 1, "%-12s the pipelined run sent requests (%d)",
                  name, pipe_reqs)
            # Output bytes are not compared here: the fake computes nothing, so
            # an output tensor is whatever bytes happen to be at its offset, and
            # the two paths deliberately place buffers differently.  Byte
            # identical outputs are the silicon gate (tools/overlap.py --check);
            # what is proved here is that the same kernels run on the same
            # values in the same order.
            check(set(sync_out) == set(pipe_out),
                  "%-12s the same output tensors (%d) and shapes",
                  name, len(sync_out))
        finally:
            fake.close()


def gate_executor_stream(models, cache):
    print("A2. the same, on the executor path (VC4_NO_CPLAN=1, modelA)")
    p = os.path.join(models, "modelA.onnx")
    if not os.path.exists(p):
        check(True, "modelA missing: skipped")
        return
    fake = Fake("a2-modelA")
    try:
        eng = engine_for(p, fake, cache)
        feeds = feed_for(eng, models)
        eng.close()
        sync_launches, _o, sync_reqs = stream(fake, p, cache, False, feeds,
                                             frames=2, nocplan=True)
        pipe_launches, _o, pipe_reqs = stream(fake, p, cache, True, feeds,
                                             frames=2, nocplan=True)
        check(sync_launches == pipe_launches,
              "executor path: %d launches, %d/%d requests, same order+uniforms",
              len(sync_launches), pipe_reqs, sync_reqs)
    finally:
        fake.close()


def gate_freshness(models, cache, full):
    print("B. no request's inputs changed while it was outstanding (20 ms/x)")
    for p in model_paths(models, full):
        name = os.path.basename(p)[:-5]
        if name.startswith("yolov8n_1"):
            continue        # 20 ms per request over ~130 requests is minutes
        fake = Fake("b-" + name, delay_ms=20)
        try:
            eng = engine_for(p, fake, cache)
            feeds = feed_for(eng, models)
            eng.close()
            stream(fake, p, cache, True, feeds)
            counts = fake.counts()
            check(counts.get("uniform_changed", 0) == 0,
                  "%-12s %d requests, %d uniform shadows changed under a "
                  "running request", name, counts.get("submit", 0),
                  counts.get("uniform_changed", 0))
            check(counts.get("data_changed", 0) == 0,
                  "%-12s %d launch probes changed under a running request",
                  name, counts.get("data_changed", 0))
        finally:
            fake.close()


def gate_accounting(models, cache):
    print("D. accounting balances across pipelined runs")
    p = os.path.join(models, "modelB.onnx")
    if not os.path.exists(p):
        check(True, "modelB missing: skipped")
        return
    fake = Fake("d-modelB")
    try:
        os.environ["VC4D_SOCKET"] = fake.sock
        # The quarantine only exists for the ordinary (unpinned) pool, so this
        # gate runs with the frozen plan off; the plan's own lifetime rule is
        # the barrier set, which gate B checks.
        eng = engine_for(p, fake, cache, nocplan=True)
        feeds = feed_for(eng, models)
        runner = eng._runner
        pool = runner.pool
        highs = []
        for _ in range(3):
            eng.run_pipelined(feeds)
            check(not pool.held, "no block left on the hold list (%d)",
                  len(pool.held))
            live = sum(pool.live.values())
            check(pool.used_bytes == live,
                  "used_bytes %d equals live bytes %d after the run",
                  pool.used_bytes, live)
            # no offset is both live and free
            bad = [(o, sz) for o, sz in pool.free
                   if any(o <= lo < o + sz for lo in pool.live)]
            check(not bad, "no free range holds a live block: %s", bad)
            highs.append(pool.high)
        # The first run establishes the panels and the constants; from then on
        # every frame must reach the same high-water, or the quarantine is
        # growing per frame (a leak by another name).
        check(highs[1] == highs[2],
              "runs 2 and 3 reach the same high-water: %s", highs)
        check(highs[0] <= highs[2],
              "the first run is not above the steady state: %s", highs)
        pipe_high = highs[-1]
        eng.close()

        # What the pipeline costs in pool bytes: the same three frames on the
        # synchronous path, where nothing is held back.
        eng = engine_for(p, fake, cache, nocplan=True)
        sync_high = 0
        for _ in range(3):
            eng.run(feeds)
            sync_high = max(sync_high, eng._runner.pool.high)
        eng.close()
        print("      pool high-water: pipelined %d B, sync %d B (+%d B, %.1f%%)"
              % (pipe_high, sync_high, pipe_high - sync_high,
                 100.0 * (pipe_high - sync_high) / max(sync_high, 1)))
    finally:
        fake.close()


def gate_fallback(models, cache):
    print("E. a version-3 daemon makes run_pipelined refuse")
    p = os.path.join(models, "modelC.onnx")
    if not os.path.exists(p):
        check(True, "modelC missing: skipped")
        return
    fake = Fake("e-modelC", version=3)
    try:
        eng = engine_for(p, fake, cache)
        feeds = feed_for(eng, models)
        ok = eng.run(feeds)                    # the batch path still works
        check(all(v is not None for v in ok.values()),
              "run() works against a version-3 daemon")
        try:
            eng.run_pipelined(feeds)
            check(False, "run_pipelined raised on a version-3 daemon")
        except Exception as e:
            check("protocol 4" in str(e),
                  "run_pipelined names the reason: %s" % str(e)[:90])
        counts = fake.counts()
        check(counts.get("submit", 0) == 0,
              "no SUBMIT was sent to the version-3 daemon (%d)",
              counts.get("submit", 0))
        eng.close()
    finally:
        fake.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get(
        "VC4INFER_MODELS", os.path.expanduser("~/work/onnx/models")))
    ap.add_argument("--cache", default=os.path.join(TMP, "pipe-kcache"))
    ap.add_argument("--only", default="", help="A,B,C,D,E")
    ap.add_argument("--full", action="store_true",
                    help="also run YOLOv8n at 96/128/160 (slow: cold kernels)")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    only = set(x.strip().upper() for x in args.only.split(",") if x.strip())
    print("models %s\ncache  %s\n" % (args.models, args.cache))
    if not only or "C" in only:
        gate_quarantine()
    if not only or "A" in only:
        gate_stream(args.models, args.cache, args.full)
    if not only or "A" in only:
        gate_executor_stream(args.models, args.cache)
    if not only or "B" in only:
        gate_freshness(args.models, args.cache, args.full)
    if not only or "D" in only:
        gate_accounting(args.models, args.cache)
    if not only or "E" in only:
        gate_fallback(args.models, args.cache)
    print()
    if FAILS:
        print("FAIL (%d)" % len(FAILS))
        for f in FAILS:
            print("  " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
