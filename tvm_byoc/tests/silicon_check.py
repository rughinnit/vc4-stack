"""The serialized silicon pass for the VC4 Relax BYOC backend.

This is the only check here that wants the QPU.  It builds the same
``matmul + bias`` composite as the CPU checks, compiles it with the module in
``vc4`` mode, and compares the result against the same graph executed by Relax
on the CPU and against numpy.  It also prints the engine's own counters from
the runtime log.

The board runs one GPU client at a time, so the script refuses to start unless
``--vc4`` is passed explicitly.

    TMPDIR=$HOME/scratch/tmp \
    VC4_TVM_LOG=$HOME/scratch/logs/tvm-byoc-silicon.log \
    /path/to/tvm/bin/python tvm_byoc/tests/silicon_check.py --vc4

Exit status: 0 when both comparisons pass, 1 otherwise, 2 when --vc4 is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import tvm  # noqa: E402
from tvm import relax  # noqa: E402

from tvm_byoc.vc4 import build_for_vc4  # noqa: E402

WORK = pathlib.Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
LOG = pathlib.Path(os.environ.get("VC4_TVM_LOG") or WORK / "tvm-byoc-silicon.log")


def build(m, k, n):
    bias = (np.arange(n, dtype="float32") * 0.5).astype("float32")
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType((m, k), "float32"))
    w = relax.Var("w", relax.TensorType((k, n), "float32"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            out = bb.emit(relax.op.add(relax.op.matmul(x, w), relax.const(bias, "float32")))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get(), bias


def execute(mod, args, mode):
    offloaded = build_for_vc4(mod, mode=mode)
    ex = tvm.compile(offloaded, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](*[tvm.runtime.tensor(a) for a in args]).numpy()


def events():
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vc4", action="store_true", help="allow dispatch to the VC4 device")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.vc4:
        print("refusing to dispatch to the VC4 device without --vc4")
        print("the board runs one GPU client at a time; see tvm_byoc/README.md")
        return 2
    if os.environ.get("VC4_TVM_MODE") not in (None, "vc4"):
        print(f"VC4_TVM_MODE={os.environ['VC4_TVM_MODE']} overrides the descriptor; unset it")
        return 2

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")

    mod, bias = build(args.m, args.k, args.n)
    x = np.random.RandomState(args.seed).rand(args.m, args.k).astype("float32")
    w = np.random.RandomState(args.seed + 1).rand(args.k, args.n).astype("float32")

    cpu = execute(mod, (x, w), "host")
    device = execute(mod, (x, w), "vc4")
    numpy_ref = x @ w + bias

    d_cpu = float(np.max(np.abs(device - cpu)))
    d_np = float(np.max(np.abs(device - numpy_ref)))
    nan = bool(np.isnan(device).any())

    calls = [e for e in events() if e["event"] == "call"]
    engine = [e for e in events() if e["event"] == "engine"]
    print(f"shape M={args.m} K={args.k} N={args.n}, bias {args.n} row floats")
    print(f"log {LOG}")
    print(f"calls recorded: {len(calls)}  engine reports: {len(engine)}")
    for e in calls:
        print(f"  call mode={e['mode']} M={e['M']} N={e['N']} K={e['K']} "
              f"x_fnv1a={e['x_fnv1a']} w_fnv1a={e['w_fnv1a']}")
    for e in engine:
        print(f"  engine launches={e['launches']} flops={e['flops']}")
    print(f"max|vc4 - relax cpu| = {d_cpu:.3e}")
    print(f"max|vc4 - numpy|     = {d_np:.3e}")
    print(f"NaN in output: {nan}")

    ok = not nan and d_cpu < 1e-4 and d_np < 1e-4
    if not ok:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
