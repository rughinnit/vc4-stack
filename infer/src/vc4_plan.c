/*
 * vc4_plan.c - replay a frozen execution plan on the ARM.
 *
 * The ONNX executor used to spend its per-node time in Python: dispatch
 * lookup, attribute reads, Tensor objects, pool bookkeeping.  A plan records
 * the device calls one frame makes and replays them as a straight line over
 * arena offsets, so the per-node work happens once (when the plan is built)
 * instead of every frame.
 *
 * Nothing here does arithmetic, allocates or moves host data: every record is
 * one of the byte-moving or elementwise entry points the executor already
 * called on that node.  The bytes moved and the launch count are therefore
 * the recording's, not this file's decision.
 *
 * The hook indirection exists for testing without a VideoCore: the walker's
 * decoding and its stop-on-error behaviour are covered by tests/test_plan.c,
 * which supplies a recorder instead of the real entry points.
 */
#include "vc4_int.h"

#include <string.h>

static float plan_fbits(int64_t v)
{
    uint32_t u = (uint32_t)v;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static vc4_buf plan_buf(uint32_t off, uint32_t bytes)
{
    vc4_buf b;
    b.off = off;
    b.bytes = bytes;
    return b;
}

static vc4_status plan_call(vc4_ctx *ctx, const vc4_plan_op *op,
                            const vc4_buf *a, const vc4_buf *b,
                            vc4_buf *out)
{
    switch ((vc4_plan_kind)op->kind) {
    case VC4_PLAN_EW:
        return vc4_elementwise(ctx, (vc4_op)op->p[0], a, b, out, (size_t)op->n);
    case VC4_PLAN_UNARY:
        return vc4_unary(ctx, (vc4_uop)op->p[0], a, out, (size_t)op->n,
                         plan_fbits(op->p[1]), plan_fbits(op->p[2]),
                         plan_fbits(op->p[3]));
    case VC4_PLAN_COPY2D:
        return vc4_copy2d(ctx, a, out, (uint32_t)op->p[0], (uint32_t)op->p[1],
                          (uint32_t)op->p[2], (uint32_t)op->p[3],
                          (uint32_t)op->p[4], (uint32_t)op->p[5],
                          (uint32_t)op->p[6]);
    case VC4_PLAN_COPY2D_N:
        return vc4_copy2d_n(ctx, a, out, (uint32_t)op->p[0],
                            (uint32_t)op->p[1], (uint32_t)op->p[2],
                            (uint32_t)op->p[3], (uint32_t)op->p[4],
                            (uint32_t)op->p[5], (uint32_t)op->p[6],
                            (uint32_t)op->p[7], (uint32_t)op->p[8],
                            (uint32_t)op->p[9]);
    case VC4_PLAN_COPY2D_X:
        return vc4_copy2d_x(ctx, a, out, (uint32_t)op->p[0],
                            (uint32_t)op->p[1], (uint32_t)op->p[2],
                            (uint32_t)op->p[3], (uint32_t)op->p[4],
                            (uint32_t)op->p[5], (uint32_t)op->p[6],
                            (uint32_t)op->p[7], (uint32_t)op->p[8],
                            (uint32_t)op->p[9], (uint32_t)op->p[10],
                            (uint32_t)op->p[11], (uint32_t)op->p[12]);
    case VC4_PLAN_FILL2D:
        return vc4_fill2d(ctx, out, (uint32_t)op->p[0], (uint32_t)op->p[1],
                          (uint32_t)op->p[2], (uint32_t)op->p[3],
                          plan_fbits(op->p[4]));
    case VC4_PLAN_GATHER:
        return vc4_gather(ctx, a, out, (size_t)op->n, plan_fbits(op->p[0]));
    case VC4_PLAN_BLIT:
        return vc4_blit(ctx, a, out, (size_t)op->n);
    case VC4_PLAN_REDUCE:
        return vc4_reduce(ctx, a, out, (uint32_t)op->p[0],
                          (uint32_t)op->p[1], (vc4_redop)op->p[2]);
    case VC4_PLAN_CUMSUM:
        return vc4_cumsum(ctx, a, out, (uint32_t)op->p[0],
                          (uint32_t)op->p[1]);
    case VC4_PLAN_ALIAS:
        return VC4_OK;
    default:
        return VC4_ERR_INVAL;
    }
}

size_t vc4_plan_op_size(void)
{
    return sizeof(vc4_plan_op);
}

vc4_status vc4_plan_run_h(vc4_ctx *ctx, const vc4_plan_op *ops, size_t n,
                          const vc4_plan_hooks *hooks)
{
    size_t i;

    if (!ops || !n)
        return VC4_ERR_INVAL;

    for (i = 0; i < n; i++) {
        const vc4_plan_op *op = &ops[i];
        vc4_buf a, b, out;
        vc4_status st;
        int has_b;

        if (op->kind >= VC4_PLAN_KINDS)
            return VC4_ERR_INVAL;
        if (op->kind == VC4_PLAN_ALIAS)
            continue;
        /* vc4_elementwise takes b as an optional second operand; every other
         * record has at most one input. */
        has_b = (op->kind == VC4_PLAN_EW) && (op->p[1] != 0);
        a = plan_buf(op->a_off, op->a_bytes);
        b = plan_buf(op->b_off, op->b_bytes);
        out = plan_buf(op->c_off, op->c_bytes);
        if (hooks && hooks->call)
            st = hooks->call(hooks->user, op, &a, has_b ? &b : NULL, &out);
        else
            st = plan_call(ctx, op, &a, has_b ? &b : NULL, &out);
        if (st != VC4_OK)
            return st;
    }
    return VC4_OK;
}

vc4_status vc4_plan_run(vc4_ctx *ctx, const vc4_plan_op *ops, size_t n)
{
    return vc4_plan_run_h(ctx, ops, n, NULL);
}
