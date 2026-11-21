#!/usr/bin/env python3
"""Lowering checks for the vc4 PJRT plugin that need no QPU.

Every case is exported to a StableHLO portable artifact with ``jax.export``
(the serialization the PJRT compile path receives), lowered by
``pjrt/tools/pjrt_decode``, and checked against expectations derived from the
shapes and the operands:

* the instruction list, so an op that starts or stops lowering is visible;
* the GEMM, convolution and reduction geometry the executor will use;
* for the index-table ops (slice, reverse, pad, gather) the exact source
  element of every output element, computed in numpy by running the same
  operation on an ``arange`` tensor -- the table is the whole lowering, so a
  wrong table is a wrong result on silicon.

Nothing here launches a kernel: ``pjrt_decode`` only runs the bytecode reader
and the lowering.  A separate pass, on an idle board, runs the same programs
through ``tests/test_pjrt.py``.

    make -C pjrt decode
    python3 pjrt/tests/test_lower.py
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
DECODE = os.environ.get(
    "PJRT_DECODE", os.path.join(HERE, "..", "build", "pjrt_decode"))

FAILED = []
CASES = 0


def check(name, cond, detail=""):
    global CASES
    CASES += 1
    if cond:
        print(f"  {name:26s} ok")
    else:
        print(f"  {name:26s} FAIL {detail}")
        FAILED.append(name)


# ------------------------------------------------------------------ decode

def parse(text):
    prog = {"tensors": [], "insns": [], "tables": [], "outputs": []}
    for line in text.splitlines():
        m = re.match(r"tensors=(\d+) insns=(\d+) params=(\d+) outputs=(\d+)",
                     line)
        if m:
            prog["ninsns"] = int(m.group(2))
            continue
        m = re.match(r"  t(\d+)\s+(const\s+)?dtype=(\d+) shape=\[([^\]]*)\]",
                     line)
        if m:
            dims = [int(x) for x in m.group(4).split(",") if x != ""]
            prog["tensors"].append(
                {"slot": int(m.group(1)), "const": bool(m.group(2)),
                 "dtype": int(m.group(3)), "dims": dims})
            continue
        m = re.match(r"  (\w+)\s+a=(-?\d+) b=(-?\d+) c=(-?\d+) -> t(\d+)(.*)",
                     line)
        if m:
            insn = {"op": m.group(1), "a": int(m.group(2)), "b": int(m.group(3)),
                    "c": int(m.group(4)), "dst": int(m.group(5))}
            for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(6)):
                insn[k] = int(v)
            prog["insns"].append(insn)
            continue
        m = re.match(r"  table(\d+) n=(\d+):(.*)", line)
        if m:
            prog["tables"].append([int(x) for x in m.group(3).split()])
            continue
        m = re.match(r"outputs:(.*)", line)
        if m:
            prog["outputs"] = [int(x) for x in m.group(1).split()]
    return prog


def lower(fn, *args):
    """Export `fn` to a VHLO portable artifact and lower it."""
    exported = jax.export.export(jax.jit(fn))(*args)
    blob = exported.mlir_module_serialized
    fd, path = tempfile.mkstemp(suffix=".mlir", dir=os.environ.get("TMPDIR"))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        out = subprocess.run([DECODE, path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    text = out.stdout + out.stderr
    if out.returncode != 0 or text.startswith("ERROR"):
        return None, text.strip()
    return parse(text), None


def ops(prog):
    return [i["op"] for i in prog["insns"]]


def table(prog, which=0):
    return prog["tables"][which]


# ------------------------------------------------------------------ helpers

def interior(arr, axis, amount):
    """Insert `amount` fill slots between consecutive elements of `axis`."""
    if amount == 0:
        return arr
    shape = list(arr.shape)
    shape[axis] = (shape[axis] - 1) * (amount + 1) + 1
    out = np.full(shape, -1, dtype=np.int64)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(None, None, amount + 1)
    out[tuple(sl)] = arr
    return out


# ------------------------------------------------------------------- cases

def cases():
    x = (np.arange(6, dtype=np.float32).reshape(2, 3) / 10).astype(np.float32)
    w = (np.arange(12, dtype=np.float32).reshape(3, 4) / 10).astype(np.float32)
    c = (np.arange(12, dtype=np.float32).reshape(4, 3) / 10).astype(np.float32)
    x34 = np.arange(12, dtype=np.float32).reshape(3, 4)
    y3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    idx = np.array([2, 0], dtype=np.int32)
    rows = np.array([2, 0], dtype=np.int32)
    cols = np.array([1, 3], dtype=np.int32)
    k16 = np.ones((16, 16, 3, 3), np.float32)
    x16 = np.ones((1, 16, 5, 5), np.float32)

    def check_table(name, fn, args, want):
        prog, err = lower(fn, *args)
        if prog is None:
            check(name, False, err)
            return
        got = table(prog)
        check(name, got == list(want), f"{got[:8]} != {list(want)[:8]}")

    print("\nindex-table ops (table compared with the numpy source index):")
    src2 = np.arange(12).reshape(3, 4)
    check_table("slice", lambda a: a[1:3, 0:2], (x34,), src2[1:3, 0:2].ravel())
    check_table("slice_step", lambda a: a[::2, ::3], (x34,), src2[::2, ::3].ravel())
    check_table("slice_rank3", lambda a: a[:, 1:3, ::2], (y3,),
                np.arange(24).reshape(2, 3, 4)[:, 1:3, ::2].ravel())
    check_table("reverse", lambda a: a[::-1, ::-1], (x34,),
                src2[::-1, ::-1].ravel())
    check_table("gather_axis0", lambda a: a[idx], (x34,), src2[idx].ravel())
    check_table("gather_2d",
                lambda a: a[rows, cols], (x34,), src2[rows, cols].ravel())
    def gather3_multi(a):
        return a[jnp.array([1, 0]), jnp.array([2, 0])]

    a3 = np.arange(24).reshape(2, 3, 4)
    check_table("gather_rank3_multi", gather3_multi, (y3,),
                a3[[1, 0], [2, 0]].ravel())
    check_table("gather_rank3_axis0", lambda a: a[jnp.array([1, 0])], (y3,),
                a3[[1, 0]].ravel())

    # pad: -1 marks the fill, everything else is the source element
    pad_cases = [
        ("pad_low_high", ((1, 2), (0, 3)), (0, 0)),
        ("pad_interior", ((1, 2), (0, 0)), (1, 2)),
        ("pad_both_edges", ((0, 0), (2, 1)), (0, 0)),
    ]
    for name, edges, inter in pad_cases:
        want = src2.copy()
        for ax, amt in enumerate(inter):
            want = interior(want, ax, amt)
        want = np.pad(want, edges, constant_values=-1)
        check_table(name,
                    lambda a, e=tuple(edges), i=tuple(inter):
                    jax.lax.pad(a, 0.0, [(e[0][0], e[0][1], i[0]),
                                         (e[1][0], e[1][1], i[1])]),
                    (x34,), want.ravel())

    print("\niota:")
    prog, err = lower(lambda a: jnp.arange(6, dtype=jnp.float32), x)
    check("iota_no_kernel", prog is not None and ops(prog) == [],
          err or str(ops(prog) if prog else ""))
    if prog:
        consts = [t for t in prog["tensors"] if t["const"]]
        check("iota_constant", len(consts) == 1 and consts[0]["dims"] == [6],
              str(consts))
    prog, err = lower(lambda a: jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
                      (x,))
    check("iota_rank3", prog is not None and ops(prog) == [], err or "")
    prog, err = lower(lambda a: jnp.take(a, jnp.arange(2), axis=0), x)
    check("iota_feeds_gather", prog is not None and "gather" in ops(prog),
          err or str(ops(prog) if prog else ""))

    prog, err = lower(lambda a, b: jnp.minimum(a, b), x, w[:2, :3])
    check("minimum", prog is not None and ops(prog) == ["compare", "select"]
          and prog["insns"][0].get("cmp") == 3, err or "")

    print("\nreductions:")
    prog, err = lower(lambda a: jnp.sum(a, axis=None), x)
    check("sum_all", prog is not None and prog["insns"][0].get("outer") == 1
          and prog["insns"][0].get("red") == 6
          and prog["insns"][0].get("redop") == 0, err or "")
    prog, err = lower(lambda a: jnp.sum(a, axis=1), x)
    check("sum_trailing", prog is not None and ops(prog) == ["reduce"]
          and prog["insns"][0].get("outer") == 2
          and prog["insns"][0].get("red") == 3, err or "")
    prog, err = lower(lambda a: jnp.max(a, axis=0), x)
    check("max_leading", prog is not None and ops(prog) == ["gather", "reduce"]
          and prog["insns"][1].get("redop") == 2, err or "")
    prog, err = lower(lambda a: jnp.min(a, axis=1), x)
    check("min_trailing", prog is not None and ops(prog) == ["reduce"]
          and prog["insns"][0].get("redop") == 3, err or "")
    prog, err = lower(lambda a: jnp.sum(a, axis=(0, 2)), y3)
    check("sum_nontrailing", prog is not None and ops(prog) == ["gather", "reduce"]
          and prog["insns"][1].get("outer") == 3
          and prog["insns"][1].get("red") == 8, err or "")

    print("\ndot_general:")
    prog, err = lower(lambda a, b: a @ b, x, w)
    check("dot_rank2", prog is not None and ops(prog) == ["dot"]
          and prog["insns"][0].get("M") == 2
          and prog["insns"][0].get("N") == 4
          and prog["insns"][0].get("K") == 3, err or "")
    prog, err = lower(lambda p, q: jnp.einsum("mk,nk->mn", p, q), x, c)
    check("dot_rhs_transposed", prog is not None
          and ops(prog) == ["gather", "dot"], err or "")
    prog, err = lower(lambda p, q: jnp.einsum("ijk,kl->ijl", p, q), y3,
                      np.ones((4, 4), np.float32))
    check("dot_rank3", prog is not None and ops(prog) == ["dot"]
          and prog["insns"][0].get("M") == 6
          and prog["insns"][0].get("N") == 4
          and prog["insns"][0].get("K") == 4, err or "")
    prog, err = lower(lambda a: jnp.einsum("bij,bkj->bik", a, a), y3)
    check("dot_batched", prog is not None and "dot_n" in ops(prog)
          and prog["insns"][-1].get("batches") == 2
          and prog["insns"][-1].get("M") == 3
          and prog["insns"][-1].get("N") == 3
          and prog["insns"][-1].get("K") == 4, err or "")

    print("\nconvolution:")
    def conv(a, k):
        return jax.lax.conv_general_dilated(
            a, k, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))

    def conv_valid(a, k):
        return jax.lax.conv_general_dilated(
            a, k, (1, 1), [(0, 0), (0, 0)],
            dimension_numbers=("NCHW", "OIHW", "NCHW"))

    prog, err = lower(conv, x16, k16)
    check("conv_same", prog is not None and "conv" in ops(prog)
          and prog["insns"][-1].get("Cin") == 16
          and prog["insns"][-1].get("Cout") == 16
          and prog["insns"][-1].get("k") == 3
          and prog["insns"][-1].get("pad") == 1
          and prog["insns"][-1].get("images") == 1
          and prog["insns"][-1].get("out") == 5, err or "")
    prog, err = lower(conv, np.ones((2, 16, 5, 5), np.float32), k16)
    check("conv_two_images", prog is not None and "conv" in ops(prog)
          and prog["insns"][-1].get("images") == 2, err or "")
    prog, err = lower(conv_valid, x16, np.ones((16, 16, 1, 1), np.float32))
    check("conv_1x1", prog is not None and "conv" in ops(prog)
          and prog["insns"][-1].get("k") == 1, err or "")

    print("\nrefusals name the op and the unsupported attribute:")
    want = [
        ("conv_channels", conv,
         (np.ones((1, 3, 5, 5), np.float32),
          np.ones((16, 3, 3, 3), np.float32)), "convolution_v1",
         "input channels"),
        ("conv_stride",
         lambda a, k: jax.lax.conv_general_dilated(
             a, k, (2, 2), "SAME",
             dimension_numbers=("NCHW", "OIHW", "NCHW")),
         (x16, k16), "convolution_v1", "window_strides"),
        ("conv_dilation",
         lambda a, k: jax.lax.conv_general_dilated(
             a, k, (1, 1), "SAME", rhs_dilation=(2, 2),
             dimension_numbers=("NCHW", "OIHW", "NCHW")),
         (x16, k16), "convolution_v1", "rhs_dilation"),
        ("reduce_product", lambda a: jnp.prod(a), (x,), "reduce_v1",
         "multiply"),
        ("divide", lambda a: jnp.mean(a), (x,), "vhlo.divide_v1", ""),
    ]
    for name, fn, args, opname, word in want:
        prog, err = lower(fn, *args)
        ok = prog is None and opname in err and word in err
        check(name, ok, (err or "lowered")[:90])


def main():
    if not os.path.exists(DECODE):
        print(f"missing {DECODE}: run make -C pjrt decode")
        return 1
    print(f"decoder: {DECODE}")
    cases()
    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} of {CASES}: {', '.join(FAILED)}")
        return 1
    print(f"all {CASES} lowering checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
