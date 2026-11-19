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
    default:             return -1;
    }
}

static int ew_is_binary(vc4_op op)
{
    return op == VC4_OP_ADD || op == VC4_OP_MUL || op == VC4_OP_RELU_BWD;
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
