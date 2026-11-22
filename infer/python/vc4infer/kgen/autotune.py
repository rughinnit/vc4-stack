"""The autotuner's search space, as data.  CPU only, no device.

`vc4_autotune` (infer/src/vc4_autotune.c) is the implementation that runs:
this module mirrors its tables and its legality filter so the space can be
inspected, unit-tested and generated for every candidate without a board.
`infer/tests/test_autotune_space.py` pins the two together by comparing this
enumeration with `build/test_autotune --space` and by generating each
candidate's blob through kgen.

The searched axes are the QPU counts, which change loop bounds and uniform
strides only:

  matmul   pack_q   QPUs for the A-panel packer       1..4, divides M/8
           gemm_q   QPUs for the asm10 GEMM           1..12, <= M/8
  conv2d   im2col_q QPUs for the im2col gather        1..4, divides M/16
           gemm_q   QPUs for the asm3 GEMM            1..12, <= M/16

`M` is the padded output row count (Ho * align16(Wo)), so `M/8` and `M/16` are
the row-block counts the two GEMMs iterate.  The tile shape and the vectors
per group are not free: they are part of each kernel's hand-verified schedule.

Shape dictionaries use the field names of `vc4_tune_shape`; parameter
dictionaries use those of `vc4_tune_params`.
"""
MAX_QPU = 12
GEMM_LADDER = (12, 8, 6, 4, 3, 2, 1)
AUX_LADDER = (4, 2, 1)

BIAS = 1

MATMUL = "matmul"
CONV2D = "conv2d"

SHAPE_KEYS = ("m", "n", "k", "cin", "cout", "h", "w", "kh", "kw", "pad",
              "stride", "flags")


def align8(v):
    return (v + 7) & ~7


def align16(v):
    return (v + 15) & ~15


def align64(v):
    return (v + 63) & ~63


def matmul_shape(m, n, k):
    """A[M][K] * B[K][N], padded to the generator's 8/64/16 tiles."""
    shape = dict.fromkeys(SHAPE_KEYS, 0)
    shape.update(m=align8(m), n=align64(n), k=align16(k))
    return shape


def conv_shape(cin, cout, h, w, kh, kw, pad, stride=1, bias=True):
    """NCHW conv2d layer geometry, in the terms the engine dispatches it."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    ho = (h + 2 * pad - kh) // stride + 1
    wp = align16(w + 2 * pad)
    shape = dict.fromkeys(SHAPE_KEYS, 0)
    shape.update(cin=cin, cout=cout, h=h, w=w, kh=kh, kw=kw, pad=pad,
                 stride=stride, flags=BIAS if bias else 0)
    shape["m"] = ho * wp
    shape["n"] = cout
    shape["k"] = align16(cin * kh * kw) + (16 if bias else 0)
    return shape


def row_blocks(op, shape):
    """Row blocks one GEMM launch has work for: 8 rows (asm10) or 16 (asm3)."""
    return shape["m"] // (16 if op == CONV2D else 8)


def aux_of(op, params):
    return params["im2col_q"] if op == CONV2D else params["pack_q"]


def set_aux(op, params, q):
    out = dict(params)
    if op == CONV2D:
        out["im2col_q"] = q
    else:
        out["pack_q"] = q
    return out


def defaults(op, shape):
    """The executor's own launch: the baseline and the fallback."""
    tr = row_blocks(op, shape)
    p = {"pack_q": 0, "gemm_q": min(tr, MAX_QPU), "im2col_q": 0}
    if op == CONV2D:
        p["im2col_q"] = 4 if tr % 4 == 0 else 2 if tr % 2 == 0 else 1
    else:
        p["pack_q"] = min(tr, 4)
    return p


def legal(op, shape, params):
    """1 when the kernels can run this launch for this shape."""
    tr = row_blocks(op, shape)
    aux = aux_of(op, params)
    if not 1 <= params["gemm_q"] <= MAX_QPU or params["gemm_q"] > tr:
        return 0
    if not 1 <= aux <= 4 or aux > tr or tr % aux:
        return 0
    if op == CONV2D and params["pack_q"] != 0:
        return 0
    if op != CONV2D and params["im2col_q"] != 0:
        return 0
    return 1


def space(op, shape):
    """The coarse pass in measurement order: the default first, then each
    legal GEMM count, then each legal auxiliary count."""
    d = defaults(op, shape)
    out = [d]
    tr = row_blocks(op, shape)
    for g in GEMM_LADDER:
        if g > MAX_QPU or g > tr or g == d["gemm_q"]:
            continue
        c = dict(d)
        c["gemm_q"] = g
        out.append(c)
    for a in AUX_LADDER:
        if a > 4 or a > tr or tr % a or a == aux_of(op, d):
            continue
        out.append(set_aux(op, d, a))
    return out


def refine(op, shape, best, seen):
    """The fine pass around a coarse winner, deduplicated against `seen`."""
    out = []
    tr = row_blocks(op, shape)
    for dist in (1, 2):
        for sign in (-1, 1):
            g = best["gemm_q"] + sign * dist
            if not 1 <= g <= MAX_QPU or g > tr:
                continue
            c = dict(best)
            c["gemm_q"] = g
            if c not in seen and c not in out:
                out.append(c)
    for a in AUX_LADDER:
        if a > 4 or a > tr or tr % a or a == aux_of(op, best):
            continue
        c = set_aux(op, best, a)
        if c not in seen and c not in out:
            out.append(c)
    return out


def describe(op, shape):
    """One line naming the space: useful in a report or a log."""
    tr = row_blocks(op, shape)
    d = defaults(op, shape)
    return ("%s %ux%ux%u: %d row blocks, default %s, %d coarse candidates"
            % (op, shape["m"], shape["n"], shape["k"], tr,
               "/".join(str(d[k]) for k in ("pack_q", "gemm_q", "im2col_q")),
               len(space(op, shape))))
