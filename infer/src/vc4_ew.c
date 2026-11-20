/*
 * vc4_ew.c - elementwise ops on hand-written QPU kernels.
 *
 * One QPU iteration covers 16 vectors = 256 floats: 16 TMU gathers per operand
 * (16 lanes = 16 consecutive floats = one 64 B line), the arithmetic in the
 * ALU, and one VPM->DMA store (the verified 4-chunk sequence with stride 0,
 * which lands 256 contiguous floats).
 *
 * add / mul / relu are exact IEEE fp32.  sigmoid and silu are computed as
 *     1/(1+exp(-x))  with  exp(-x) = 2^(-x*log2 e)
 * using a degree-5 minimax polynomial for 2^f on [-0.5,0.5] and a
 * bit-trick + 3x Newton reciprocal.  Measured worst-case error is reported by
 * tests/test_ew.c (typically < 2e-7 absolute).
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EW_GROUP 256u
#define EW_KMAX  VC4_MAX_QPU

/* vc4_op numbering (public API) is not kgen.py's OP_*: the public enum starts
 * with MATMUL/CONV2D, so ADD=2 while kgen's ADD=0.  Translate explicitly. */
static int ew_kgen_op(vc4_op op)
{
    switch (op) {
    case VC4_OP_ADD:     return 0;
    case VC4_OP_MUL:     return 1;
    case VC4_OP_RELU:    return 2;
    case VC4_OP_SIGMOID: return 3;
    case VC4_OP_SILU:    return 4;
    case VC4_OP_RELU_BWD:return 5;
    case VC4_OP_RSQRT:   return 6;
    /* the comparison/predicate family: fixed bodies in kgen._ew_cmp */
    case VC4_OP_CMP_EQ:  return 10;
    case VC4_OP_CMP_GT:  return 11;
    case VC4_OP_CMP_GE:  return 12;
    case VC4_OP_CMP_LT:  return 13;
    case VC4_OP_CMP_LE:  return 14;
    case VC4_OP_ISNAN:   return 15;
    case VC4_OP_ISINF:   return 16;
    default:             return -1;
    }
}

static int ew_is_binary(vc4_op op)
{
    return op == VC4_OP_ADD || op == VC4_OP_MUL || op == VC4_OP_RELU_BWD ||
           op == VC4_OP_CMP_EQ || op == VC4_OP_CMP_GT ||
           op == VC4_OP_CMP_GE || op == VC4_OP_CMP_LT ||
           op == VC4_OP_CMP_LE;
}

static int ew_op_ok(vc4_op op)
{
    switch (op) {
    case VC4_OP_ADD:
    case VC4_OP_MUL:
    case VC4_OP_RELU:
    case VC4_OP_SIGMOID:
    case VC4_OP_SILU:
    case VC4_OP_RELU_BWD:
    case VC4_OP_RSQRT:
    case VC4_OP_CMP_EQ:
    case VC4_OP_CMP_GT:
    case VC4_OP_CMP_GE:
    case VC4_OP_CMP_LT:
    case VC4_OP_CMP_LE:
    case VC4_OP_ISNAN:
    case VC4_OP_ISINF:
        return 1;
    default:
        return 0;
    }
}

vc4_status vc4__ew_run(vc4_ctx *ctx, vc4_op op, uint64_t n,
                       uint32_t a_bus, uint32_t b_bus, uint32_t o_bus)
{
    if (n % EW_GROUP)
        return VC4_ERR_INVAL;
    uint64_t ngroups = n / EW_GROUP;
    if (!ngroups)
        return VC4_ERR_INVAL;

    char name[48], nop[8];
    char *argv[] = { (char *)"ew", (char *)"--op", nop, NULL };
    snprintf(nop, sizeof(nop), "%d", ew_kgen_op(op));

    /* One group per QPU per launch.  A QPU that runs a *second* group in the
       same program (the round-robin loop's second iteration) reliably hangs
       with the long sigmoid/silu bodies at 12 QPUs, and intermittently loses
       a group with the short ones.  Batching keeps every launch to exactly
       one iteration; the extra launch costs a few microseconds. */
    vc4_status st = VC4_OK;
    for (uint64_t base = 0; base < ngroups && st == VC4_OK; base += EW_KMAX) {
        uint64_t left = ngroups - base;
        int nq = left < (uint64_t)EW_KMAX ? (int)left : EW_KMAX;
        snprintf(name, sizeof(name), "ew_%d_q%d", ew_kgen_op(op), nq);

        vc4_kernel *k = NULL;
        st = vc4__kernel(ctx, name, argv, nq, &k);
        if (st != VC4_OK)
            break;
        for (int i = 0; i < nq; ++i) {
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = a_bus;
            u[1] = b_bus;
            u[2] = o_bus;
            u[3] = 1;                       /* one group for this QPU      */
            u[4] = (uint32_t)(base + (uint64_t)i);  /* global group index  */
            u[5] = (uint32_t)nq;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
    }
    return st;
}

vc4_status vc4_elementwise(vc4_ctx *ctx, vc4_op op,
                           const vc4_buf *a, const vc4_buf *b, vc4_buf *out,
                           size_t n)
{
    if (!ctx || !a || !out || !n)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (!ew_op_ok(op))
        return VC4_ERR_UNSUPPORTED;
    if (ew_is_binary(op) && !b)
        return VC4_ERR_INVAL;
    if (a->bytes < n * 4 || out->bytes < n * 4)
        return VC4_ERR_INVAL;
    if (ew_is_binary(op) && b->bytes < n * 4)
        return VC4_ERR_INVAL;

    const size_t np = ((n + EW_GROUP - 1) / EW_GROUP) * EW_GROUP;
    const int need_pad = (np != n);

    vc4__scratch_reset(ctx);

    uint32_t a_bus = vc4_bus_addr(ctx, a);
    /* The generated kernel issues one TMU gather per operand per vector even
       for the unary ops.  A NULL b must not become bus address 0 (a TMU load
       from 0 wedges the QPU): point the unused operand at a instead. */
    uint32_t b_bus = b ? vc4_bus_addr(ctx, b) : a_bus;
    uint32_t o_bus = vc4_bus_addr(ctx, out);
    float *o_tmp = NULL;

    if (need_pad) {
        /* tail handling: compute the rounded-up group in scratch and copy the
           real values back (byte work, no arithmetic on the CPU) */
        float *ap = vc4__scratch_alloc(ctx, np * 4, NULL);
        if (!ap)
            return VC4_ERR_NOMEM;
        vc4__invalidate_off(ctx, a->off, n * 4);
        vc4__write(ctx, (uint32_t)((uint8_t *)ap - ctx->arena),
                   ctx->arena + a->off, n * 4);
        memset((uint8_t *)ap + n * 4, 0, (np - n) * 4);
        vc4__mark_dirty(ctx, (uint32_t)((uint8_t *)ap - ctx->arena), np * 4);
        a_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)ap - ctx->arena);
        if (ew_is_binary(op)) {
            float *bp = vc4__scratch_alloc(ctx, np * 4, NULL);
            if (!bp)
                return VC4_ERR_NOMEM;
            vc4__invalidate_off(ctx, b->off, n * 4);
            vc4__write(ctx, (uint32_t)((uint8_t *)bp - ctx->arena),
                       ctx->arena + b->off, n * 4);
            memset((uint8_t *)bp + n * 4, 0, (np - n) * 4);
            vc4__mark_dirty(ctx, (uint32_t)((uint8_t *)bp - ctx->arena),
                            np * 4);
            b_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)bp - ctx->arena);
        }
        o_tmp = vc4__scratch_alloc(ctx, np * 4, NULL);
        if (!o_tmp)
            return VC4_ERR_NOMEM;
        o_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)o_tmp - ctx->arena);
    }

    vc4_status st = vc4__ew_run(ctx, op, np, a_bus, b_bus, o_bus);
    if (st == VC4_OK) {
        ctx->st.flops += (uint64_t)n * (ew_is_binary(op) ? 1 : 4);
        if (o_tmp) {
            vc4__l2_flush(ctx);
            vc4__write(ctx, out->off, o_tmp, n * 4);
        }
    }
    vc4__scratch_reset(ctx);
    return st;
}

vc4_status vc4_relu(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n)
{
    return vc4_elementwise(ctx, VC4_OP_RELU, a, NULL, out, n);
}

vc4_status vc4_sigmoid(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n)
{
    return vc4_elementwise(ctx, VC4_OP_SIGMOID, a, NULL, out, n);
}

vc4_status vc4_silu(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n)
{
    return vc4_elementwise(ctx, VC4_OP_SILU, a, NULL, out, n);
}

vc4_status vc4_rsqrt(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n)
{
    return vc4_elementwise(ctx, VC4_OP_RSQRT, a, NULL, out, n);
}

vc4_status vc4_relu_backward(vc4_ctx *ctx, const vc4_buf *a,
                             const vc4_buf *dy, vc4_buf *out, size_t n)
{
    return vc4_elementwise(ctx, VC4_OP_RELU_BWD, a, dy, out, n);
}

/* ==================================================================
 * the binary / backward elementwise family (kops.py EWB)
 *
 * Same harness as build_ew (one group = 16 vectors = 256 floats, or 4
 * vectors = 64 floats for the long bodies).  Operand a is the forward input
 * (or the left operand), b is the incoming gradient (or the right operand).
 * p0/p1 are the op attributes, read from uniforms 6/7.
 * ================================================================== */
typedef struct {
    const char *name;
    int         heavy;      /* generated with 4 vectors per group */
} ewb_desc;

/* `name` is the kops.EWB key; `heavy` must match kops.EWB_HEAVY exactly (the
 * host's group size and the generated nvec are the same decision).  Only
 * gelu_erf_bwd is heavy: its body embeds the erf and exp cores, 1979 words at
 * 4 vectors per group. */
static const ewb_desc EWB_DESC[VC4_BW_COUNT] = {
    [VC4_BW_SIGMOID]     = { "sigmoid_bwd", 0 },
    [VC4_BW_SILU]        = { "silu_bwd", 0 },
    [VC4_BW_TANH]        = { "tanh_bwd", 0 },
    [VC4_BW_EXP]         = { "exp_bwd", 0 },
    [VC4_BW_LOG]         = { "log_bwd", 0 },
    [VC4_BW_SQRT]        = { "sqrt_bwd", 0 },
    [VC4_BW_ABS]         = { "abs_bwd", 0 },
    [VC4_BW_CLIP]        = { "clip_bwd", 0 },
    [VC4_BW_LEAKYRELU]   = { "leakyrelu_bwd", 0 },
    [VC4_BW_HARDSWISH]   = { "hardswish_bwd", 0 },
    [VC4_BW_HARDSIGMOID] = { "hardsigmoid_bwd", 0 },
    [VC4_BW_GELU]        = { "gelu_bwd", 0 },
    [VC4_BW_GELU_ERF]    = { "gelu_erf_bwd", 1 },
    [VC4_BW_ERF]         = { "erf_bwd", 0 },
};

vc4_status vc4__ewb(vc4_ctx *ctx, const char *opname, int heavy,
                    const vc4_buf *a, const vc4_buf *b, vc4_buf *out,
                    size_t n, float p0, float p1)
{
    if (!ctx || !a || !b || !out || !n || !opname)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (a->bytes < n * 4 || b->bytes < n * 4 || out->bytes < n * 4)
        return VC4_ERR_INVAL;
    const size_t group = heavy ? 64u : 256u;
    const size_t np = ((n + group - 1) / group) * group;
    const int need_pad = np != n;

    vc4__scratch_reset(ctx);
    uint32_t a_bus = vc4_bus_addr(ctx, a);
    uint32_t b_bus = vc4_bus_addr(ctx, b);
    uint32_t o_bus = vc4_bus_addr(ctx, out);
    float *o_tmp = NULL;

    if (need_pad) {
        float *ap = vc4__scratch_alloc(ctx, np * 4, NULL);
        float *bp = vc4__scratch_alloc(ctx, np * 4, NULL);
        if (!ap || !bp) {
            vc4__scratch_reset(ctx);
            return VC4_ERR_NOMEM;
        }
        vc4__invalidate_off(ctx, a->off, n * 4);
        vc4__write(ctx, (uint32_t)((uint8_t *)ap - ctx->arena),
                   ctx->arena + a->off, n * 4);
        memset((uint8_t *)ap + n * 4, 0, (np - n) * 4);
        vc4__mark_dirty(ctx, (uint32_t)((uint8_t *)ap - ctx->arena), np * 4);
        vc4__invalidate_off(ctx, b->off, n * 4);
        vc4__write(ctx, (uint32_t)((uint8_t *)bp - ctx->arena),
                   ctx->arena + b->off, n * 4);
        memset((uint8_t *)bp + n * 4, 0, (np - n) * 4);
        vc4__mark_dirty(ctx, (uint32_t)((uint8_t *)bp - ctx->arena), np * 4);
        a_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)ap - ctx->arena);
        b_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)bp - ctx->arena);
        o_tmp = vc4__scratch_alloc(ctx, np * 4, NULL);
        if (!o_tmp) {
            vc4__scratch_reset(ctx);
            return VC4_ERR_NOMEM;
        }
        o_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)o_tmp - ctx->arena);
    }

    char name[80];
    char *argv[] = { (char *)"ewb", (char *)"--op", (char *)opname, NULL };
    uint32_t p0b, p1b;
    memcpy(&p0b, &p0, 4);
    memcpy(&p1b, &p1, 4);
    uint64_t ngroups = np / group;
    vc4_status st = VC4_OK;
    for (uint64_t base = 0; base < ngroups && st == VC4_OK;
         base += VC4_MAX_QPU) {
        uint64_t left = ngroups - base;
        int nq = left < (uint64_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        snprintf(name, sizeof(name), "ewb_%s_q%d", opname, nq);
        vc4_kernel *k = NULL;
        st = vc4__kernel(ctx, name, argv, nq, &k);
        if (st != VC4_OK)
            break;
        for (int i = 0; i < nq; ++i) {
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = a_bus;
            u[1] = b_bus;
            u[2] = o_bus;
            u[3] = 1;
            u[4] = (uint32_t)(base + (uint64_t)i);
            u[5] = (uint32_t)nq;
            u[6] = p0b;
            u[7] = p1b;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
    }
    if (st == VC4_OK) {
        ctx->st.flops += (uint64_t)n * 4;
        if (o_tmp) {
            vc4__l2_flush(ctx);
            vc4__write(ctx, out->off, o_tmp, n * 4);
        }
    }
    vc4__scratch_reset(ctx);
    return st;
}

/*
 * The derivative of each op.  Most are short compositions of the atoms that
 * are verified on this board (ew add/mul/sigmoid/rsqrt, ewb sub/div/exp/abs),
 * because the whole-body kops form of those ops at 16 vectors does not fit the
 * 2048-instruction code slot.  The exact erf-form GELU derivative is special:
 * it has no short composition, so it runs as one fused kernel generated at
 * 4 vectors per group (kops.EWB_HEAVY; 1979 words, so it fits).  Each
 * composition recomputes the forward value with the *same* primitives the
 * forward op uses, so the gradient is consistent with it.
 */
static vc4_status bw_sigmoid(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                             vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf y, u, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &y);
    float *p2 = vc4__ws_alloc(c, n * 4, &u);
    float *p3 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2 && p3) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_sigmoid(c, x, &y, n);                 /* s */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &y, &u, n, -1.0f, 1.0f, 0);  /* 1-s */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &y, &u, &t, n);            /* s(1-s) */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

static vc4_status bw_tanh(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf x2, s, u, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &x2);
    float *p2 = vc4__ws_alloc(c, n * 4, &s);
    float *p3 = vc4__ws_alloc(c, n * 4, &u);
    float *p4 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2 && p3 && p4) ? VC4_OK : VC4_ERR_NOMEM;
    /* tanh(x) = 2*sigmoid(2x) - 1  =>  d/dx = 4 s (1-s), s = sigmoid(2x) */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &x2, n, 2.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_sigmoid(c, &x2, &s, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &s, &u, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &s, &u, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, 4.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

static vc4_status bw_silu(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf s, y, u, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &s);
    float *p2 = vc4__ws_alloc(c, n * 4, &y);
    float *p3 = vc4__ws_alloc(c, n * 4, &u);
    float *p4 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2 && p3 && p4) ? VC4_OK : VC4_ERR_NOMEM;
    /* d/dx [x s] = s + y (1-s),  y = x*s */
    if (st == VC4_OK)
        st = vc4_sigmoid(c, x, &s, n);
    if (st == VC4_OK)
        st = vc4_silu(c, x, &y, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &s, &u, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &y, &u, &t, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, &s, &t, &t, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

static vc4_status bw_sqrt(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf r, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &r);
    float *p2 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2) ? VC4_OK : VC4_ERR_NOMEM;
    /* 1/(2 sqrt x) = 0.5 * rsqrt(x) */
    if (st == VC4_OK)
        st = vc4_rsqrt(c, x, &r, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &r, &t, n, 0.5f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

static vc4_status bw_exp(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                         vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf y;
    float *p = vc4__ws_alloc(c, n * 4, &y);
    vc4_status st = p ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4__ewb(c, "exp", 0, x, x, &y, n, 0, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &y, o, n);
    vc4__ws_end(c, top);
    return st;
}

static vc4_status bw_log(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                         vc4_buf *o, size_t n)
{
    return vc4__ewb(c, "div", 0, g, x, o, n, 0, 0);   /* g / x */
}

/* d/dx erf = 2/sqrt(pi) * exp(-x^2) */
static vc4_status bw_erf(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                         vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf t;
    float *p = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = p ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, x, x, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, -1.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4__ewb(c, "exp", 0, &t, &t, &t, n, 0, 0);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n,
                       1.1283791670955126f, 0.0f, 0);      /* 2/sqrt(pi) */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

/*
 * tanh-form GELU derivative:
 *   0.5(1+t) + 0.5*x*(1-t^2)*GELU_S*(1+3*GELU_C*x^2),
 *   t = tanh(GELU_S*(x + GELU_C*x^3))
 * composed from the same primitives (the fused body does not fit the code
 * slot at 16 vectors per group).
 */
static vc4_status bw_gelu(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n)
{
    const float GC = 0.044715f, GS = 0.7978845608028654f;
    size_t top = vc4__ws_begin(c);
    vc4_buf x2, a, b, u, t, one_m_t2, d;
    float *p1 = vc4__ws_alloc(c, n * 4, &x2);
    float *p2 = vc4__ws_alloc(c, n * 4, &a);
    float *p3 = vc4__ws_alloc(c, n * 4, &b);
    float *p4 = vc4__ws_alloc(c, n * 4, &u);
    float *p5 = vc4__ws_alloc(c, n * 4, &t);
    float *p6 = vc4__ws_alloc(c, n * 4, &one_m_t2);
    float *p7 = vc4__ws_alloc(c, n * 4, &d);
    vc4_status st = (p1 && p2 && p3 && p4 && p5 && p6 && p7) ? VC4_OK
                                                            : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, x, x, &x2, n);          /* x^2 */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &x2, &a, n, GC, 1.0f, 0); /* 1+c x^2 */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, x, &a, &b, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &b, &u, n, GS, 0.0f, 0);
    /* t = tanh(u) = 2*sigmoid(2u) - 1 */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &u, &b, n, 2.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_sigmoid(c, &b, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, 2.0f, -1.0f, 0);
    /* d = 0.5*(1+t) */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &d, n, 0.5f, 0.5f, 0);
    /* one_m_t2 = 1 - t^2 */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &t, &t, &one_m_t2, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &one_m_t2, &one_m_t2, n, -1.0f, 1.0f,
                       0);
    /* du/dx = GS*(1 + 3c x^2) */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &x2, &a, n, 3.0f * GC, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &a, &a, n, GS, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &one_m_t2, &a, &one_m_t2, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &one_m_t2, x, &one_m_t2, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &one_m_t2, &one_m_t2, n, 0.5f, 0.0f,
                       0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, &d, &one_m_t2, &d, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &d, o, n);
    vc4__ws_end(c, top);
    return st;
}

/* Exact erf-form GELU derivative, one fused kernel:
 *   g * ( 0.5*(1 + erf(x/sqrt 2)) + x*exp(-x^2/2)/sqrt(2 pi) )
 * The body is written from the same erf/exp primitives the forward
 * VC4_UOP_GELU_ERF uses, so the gradient is the derivative of the kernel that
 * ran.  It is generated at 4 vectors per group (kops.EWB_HEAVY) because the
 * 16-vector body is 4222 words against a 2048-instruction slot; the host's
 * group size here must be 64 floats to match. */
static vc4_status bw_gelu_erf(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                              vc4_buf *o, size_t n)
{
    return vc4__ewb(c, "gelu_erf_bwd", 1, x, g, o, n, 0.0f, 0.0f);
}

/* ---- helpers for the relu_bwd-based masks --------------------------------
 * The ewb kernels that read their second stream (sub/div/eqmask/abs_bwd/...)
 * do not see that stream correctly on this board, while the ones that only
 * read `a` (exp, rcp) and the verified ew kernels (add/mul/relu/
 * relu_bwd/sigmoid/silu/rsqrt) do.  So every derivative below is built from
 * the verified set: a mask is `relu_bwd(z, ones)`, i.e. 1 where z > 0.
 */
static vc4_status ws_ones(vc4_ctx *c, vc4_buf *b, size_t n)
{
    *b = (vc4_buf){ 0, 0 };
    if (!vc4__ws_alloc(c, n * 4, b))
        return VC4_ERR_NOMEM;
    float *o = malloc(n * 4);
    if (!o)
        return VC4_ERR_NOMEM;
    for (size_t i = 0; i < n; ++i) o[i] = 1.0f;
    memcpy(c->arena + b->off, o, n * 4);
    vc4__mark_dirty(c, b->off, n * 4);
    free(o);
    return VC4_OK;
}

/* out = g * (a > 0 ? 1 : 0) */
static vc4_status bw_relu(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n)
{
    return vc4_relu_backward(c, x, g, o, n);
}

/* out = g / z  (g * rcp(z)) */
static vc4_status bw_div_by(vc4_ctx *c, const vc4_buf *g, const vc4_buf *z,
                            vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf r;
    float *p = vc4__ws_alloc(c, n * 4, &r);
    vc4_status st = p ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4__ewb(c, "rcp", 0, z, z, &r, n, 0, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &r, o, n);
    vc4__ws_end(c, top);
    return st;
}

/* out = g * 1[x in (lo, hi]] : relu_bwd(x-lo,g) - relu_bwd(x-hi,g) */
static vc4_status bw_clip(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                          vc4_buf *o, size_t n, float lo, float hi)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf a, b, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &a);
    float *p2 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &t, n, 1.0f, -lo, 0);
    if (st == VC4_OK)
        st = vc4_relu_backward(c, &t, g, &a, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &t, n, 1.0f, -hi, 0);
    if (st == VC4_OK) {
        b = t;
        st = vc4_relu_backward(c, &b, g, &t, n);
    }
    if (st == VC4_OK) {
        /* a - t */
        vc4_buf nm;
        float *p3 = vc4__ws_alloc(c, n * 4, &nm);
        if (!p3)
            st = VC4_ERR_NOMEM;
        else {
            st = vc4_unary(c, VC4_UOP_AFFINE, &t, &nm, n, -1.0f, 0.0f, 0);
            if (st == VC4_OK)
                st = vc4_elementwise(c, VC4_OP_ADD, &a, &nm, o, n);
        }
    }
    vc4__ws_end(c, top);
    return st;
}

/* leakyrelu: g*(x>0 ? 1 : alpha) */
static vc4_status bw_leaky(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                           vc4_buf *o, size_t n, float alpha)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf m, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &m);
    float *p2 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_relu_backward(c, x, g, &m, n);          /* g where x>0 */
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, g, &m, &t, n);
    if (st == VC4_OK) {
        /* t = g + (-1)*m = g - m */
        vc4_buf nm;
        float *p3 = vc4__ws_alloc(c, n * 4, &nm);
        if (!p3)
            st = VC4_ERR_NOMEM;
        else {
            st = vc4_unary(c, VC4_UOP_AFFINE, &m, &nm, n, -1.0f, 0.0f, 0);
            if (st == VC4_OK)
                st = vc4_elementwise(c, VC4_OP_ADD, g, &nm, &t, n);
        }
    }
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, alpha, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, &m, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

/* hardsigmoid: alpha * 1[0 < alpha*x+beta < 1] * g */
static vc4_status bw_hardsig(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                             vc4_buf *o, size_t n, float alpha, float beta)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf s, ones, m1, m2, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &s);
    float *p2 = vc4__ws_alloc(c, n * 4, &m1);
    float *p3 = vc4__ws_alloc(c, n * 4, &m2);
    float *p4 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2 && p3 && p4) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = ws_ones(c, &ones, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &s, n, alpha, beta, 0);
    if (st == VC4_OK)
        st = vc4_relu_backward(c, &s, &ones, &m1, n);          /* s > 0 */
    if (st == VC4_OK) {
        vc4_buf u;
        float *p5 = vc4__ws_alloc(c, n * 4, &u);
        if (!p5)
            st = VC4_ERR_NOMEM;
        else {
            st = vc4_unary(c, VC4_UOP_AFFINE, &s, &u, n, 1.0f, -1.0f, 0);
            if (st == VC4_OK)
                st = vc4_relu_backward(c, &u, &ones, &m2, n);  /* s > 1 */
        }
    }
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &m2, &m2, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &m1, &m2, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, alpha, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &t, o, n);
    vc4__ws_end(c, top);
    return st;
}

/* hardswish: g * (s + x*alpha*1[0<s<1]), s = clip(alpha*x+beta,0,1) */
static vc4_status bw_hardswish(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                               vc4_buf *o, size_t n, float alpha, float beta)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf s, ones, m1, m2, m, t, d;
    float *p1 = vc4__ws_alloc(c, n * 4, &s);
    float *p2 = vc4__ws_alloc(c, n * 4, &m1);
    float *p3 = vc4__ws_alloc(c, n * 4, &m2);
    float *p4 = vc4__ws_alloc(c, n * 4, &m);
    float *p5 = vc4__ws_alloc(c, n * 4, &t);
    float *p6 = vc4__ws_alloc(c, n * 4, &d);
    vc4_status st = (p1 && p2 && p3 && p4 && p5 && p6) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = ws_ones(c, &ones, n);
    /* s = min(max(t,0),1) = 1 - relu(1 - relu(t)) */
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &t, n, alpha, beta, 0);
    if (st == VC4_OK)
        st = vc4_relu(c, &t, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &t, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_relu(c, &t, &t, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &t, &s, n, -1.0f, 1.0f, 0);
    /* mask = 1[0<s<1] */
    if (st == VC4_OK)
        st = vc4_relu_backward(c, &s, &ones, &m1, n);
    if (st == VC4_OK) {
        vc4_buf u;
        float *p7 = vc4__ws_alloc(c, n * 4, &u);
        if (!p7)
            st = VC4_ERR_NOMEM;
        else {
            st = vc4_unary(c, VC4_UOP_AFFINE, &s, &u, n, 1.0f, -1.0f, 0);
            if (st == VC4_OK)
                st = vc4_relu_backward(c, &u, &ones, &m2, n);
        }
    }
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &m2, &m2, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, &m1, &m2, &m, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &m, &m, n, alpha, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, x, &m, &m, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, &s, &m, &d, n);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_MUL, g, &d, o, n);
    vc4__ws_end(c, top);
    return st;
}

/* |x| derivative: g*sign(x) = relu_bwd(x,g) - relu_bwd(-x,g) */
static vc4_status bw_abs(vc4_ctx *c, const vc4_buf *x, const vc4_buf *g,
                         vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(c);
    vc4_buf pos, neg, t;
    float *p1 = vc4__ws_alloc(c, n * 4, &pos);
    float *p2 = vc4__ws_alloc(c, n * 4, &neg);
    float *p3 = vc4__ws_alloc(c, n * 4, &t);
    vc4_status st = (p1 && p2 && p3) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_relu_backward(c, x, g, &pos, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, x, &t, n, -1.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_relu_backward(c, &t, g, &neg, n);
    if (st == VC4_OK)
        st = vc4_unary(c, VC4_UOP_AFFINE, &neg, &neg, n, -1.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(c, VC4_OP_ADD, &pos, &neg, o, n);
    vc4__ws_end(c, top);
    return st;
}

vc4_status vc4_bwop_run(vc4_ctx *ctx, vc4_bwop op, const vc4_buf *x,
                        const vc4_buf *g, vc4_buf *out, size_t n, float p0,
                        float p1)
{
    if (!ctx || !x || !g || !out || !n)
        return VC4_ERR_INVAL;
    if (op < 0 || op >= VC4_BW_COUNT)
        return VC4_ERR_UNSUPPORTED;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    switch (op) {
    case VC4_BW_SIGMOID:     return bw_sigmoid(ctx, x, g, out, n);
    case VC4_BW_SILU:        return bw_silu(ctx, x, g, out, n);
    case VC4_BW_TANH:        return bw_tanh(ctx, x, g, out, n);
    case VC4_BW_EXP:         return bw_exp(ctx, x, g, out, n);
    case VC4_BW_LOG:         return bw_div_by(ctx, g, x, out, n);
    case VC4_BW_SQRT:        return bw_sqrt(ctx, x, g, out, n);
    case VC4_BW_ERF:         return bw_erf(ctx, x, g, out, n);
    case VC4_BW_GELU:        return bw_gelu(ctx, x, g, out, n);
    case VC4_BW_GELU_ERF:    return bw_gelu_erf(ctx, x, g, out, n);
    case VC4_BW_ABS:         return bw_abs(ctx, x, g, out, n);
    case VC4_BW_CLIP:        return bw_clip(ctx, x, g, out, n, p0, p1);
    case VC4_BW_LEAKYRELU:   return bw_leaky(ctx, x, g, out, n, p0);
    case VC4_BW_HARDSWISH:   return bw_hardswish(ctx, x, g, out, n, p0, p1);
    case VC4_BW_HARDSIGMOID: return bw_hardsig(ctx, x, g, out, n, p0, p1);
    default:                 return VC4_ERR_UNSUPPORTED;
    }
}
