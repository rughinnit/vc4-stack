/* test_plan.c - the frozen-plan walker, with no VideoCore in the loop.
 *
 * The walker's job is decoding, not arithmetic: every record maps to exactly
 * one entry point with exactly the arguments that were recorded.  A hook
 * stands in for those entry points, so this test can prove the mapping and
 * the stop-on-error rule on a machine with no QPU.
 *
 * What it does not prove: that the recorded arguments reproduce the frame.
 * That is the plan/executor equivalence test (tests/test_cplan.py, synthetic
 * graphs) and, on silicon, the byte-for-byte model comparison.
 *
 * usage: test_plan
 */
#include "vc4infer.h"
#include "util.h"

static int errors;

struct call {
    int kind;
    uint32_t a_off, b_off, c_off;
    uint64_t n;
    int64_t p[14];
    int has_b;
};

struct recorder {
    struct call calls[16];
    int n;
    int fail_at;          /* 1-based index of a call that fails */
    vc4_status fail_with;
};

static vc4_status record(void *user, const vc4_plan_op *op, const vc4_buf *a,
                         const vc4_buf *b, const vc4_buf *out)
{
    struct recorder *r = user;

    if (r->n < (int)(sizeof r->calls / sizeof r->calls[0])) {
        struct call *c = &r->calls[r->n];
        memcpy(c->p, op->p, sizeof c->p);
        c->kind = op->kind;
        c->a_off = a->off;
        c->b_off = b ? b->off : 0xFFFFFFFFu;
        c->c_off = out->off;
        c->n = (uint64_t)op->n;
        c->has_b = b != NULL;
    }
    ++r->n;
    if (r->fail_at && r->n == r->fail_at)
        return r->fail_with;
    return VC4_OK;
}

static void expect(int cond, const char *what)
{
    printf("%-4s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++errors;
}

static vc4_plan_op op_of(int kind)
{
    vc4_plan_op op;
    memset(&op, 0, sizeof op);
    op.kind = (uint16_t)kind;
    return op;
}

int main(void)
{
    struct recorder rec;
    vc4_plan_hooks hooks;
    vc4_plan_op ops[11];
    vc4_status st;
    size_t i;
    int k;

    memset(&rec, 0, sizeof rec);
    hooks.call = record;
    hooks.user = &rec;

    /* the ABI the Python writer mirrors */
    expect(vc4_plan_op_size() == sizeof(vc4_plan_op),
           "vc4_plan_op_size() is sizeof(vc4_plan_op)");
    printf("     record size: %zu B\n", vc4_plan_op_size());

    /* one record of every kind, with distinguishable arguments */
    for (i = 0; i < sizeof ops / sizeof ops[0]; i++)
        ops[i] = op_of(VC4_PLAN_ALIAS);
    ops[0].kind = VC4_PLAN_EW;
    ops[0].a_off = 0x100; ops[0].a_bytes = 0x1000;
    ops[0].b_off = 0x200; ops[0].b_bytes = 0x1000;
    ops[0].c_off = 0x300; ops[0].c_bytes = 0x1000;
    ops[0].n = 256;
    ops[0].p[0] = 3;                 /* vc4_op: ADD */
    ops[0].p[1] = 1;                 /* b present */

    ops[1].kind = VC4_PLAN_UNARY;
    ops[1].a_off = 0x400; ops[1].a_bytes = 0x800;
    ops[1].c_off = 0x500; ops[1].c_bytes = 0x800;
    ops[1].n = 64;
    ops[1].p[0] = 1;                 /* vc4_uop: NEG */
    {
        /* -0.0 and 1.5, carried as bit patterns and recovered exactly */
        float f = -0.0f, g = 1.5f;
        uint32_t u;
        memcpy(&u, &f, 4); ops[1].p[1] = (int64_t)u;
        memcpy(&u, &g, 4); ops[1].p[2] = (int64_t)u;
    }

    ops[2].kind = VC4_PLAN_COPY2D;
    ops[2].a_off = 0x600; ops[2].a_bytes = 0x400;
    ops[2].c_off = 0x700; ops[2].c_bytes = 0x400;
    for (k = 0; k < 7; k++)
        ops[2].p[k] = 10 + k;

    ops[3].kind = VC4_PLAN_COPY2D_N;
    ops[3].a_off = 0x800; ops[3].a_bytes = 0x400;
    ops[3].c_off = 0x900; ops[3].c_bytes = 0x400;
    for (k = 0; k < 10; k++)
        ops[3].p[k] = 20 + k;

    ops[4].kind = VC4_PLAN_COPY2D_X;
    ops[4].a_off = 0xA00; ops[4].a_bytes = 0x400;
    ops[4].c_off = 0xB00; ops[4].c_bytes = 0x400;
    for (k = 0; k < 13; k++)
        ops[4].p[k] = 30 + k;

    ops[5].kind = VC4_PLAN_FILL2D;
    ops[5].c_off = 0xC00; ops[5].c_bytes = 0x400;
    for (k = 0; k < 4; k++)
        ops[5].p[k] = 40 + k;
    {
        float v = -INFINITY;
        uint32_t u;
        memcpy(&u, &v, 4);
        ops[5].p[4] = (int64_t)u;
    }

    ops[6].kind = VC4_PLAN_GATHER;
    ops[6].a_off = 0xD00; ops[6].a_bytes = 0x400;
    ops[6].c_off = 0xE00; ops[6].c_bytes = 0x400;
    ops[6].n = 96;
    ops[6].p[0] = 0;                 /* fill bits of 0.0 */

    ops[7].kind = VC4_PLAN_BLIT;
    ops[7].a_off = 0xF00; ops[7].a_bytes = 0x400;
    ops[7].c_off = 0x1000; ops[7].c_bytes = 0x400;
    ops[7].n = 4096;

    ops[8].kind = VC4_PLAN_REDUCE;
    ops[8].a_off = 0x1100; ops[8].a_bytes = 0x400;
    ops[8].c_off = 0x1200; ops[8].c_bytes = 0x100;
    ops[8].p[0] = 4; ops[8].p[1] = 64; ops[8].p[2] = 2;

    ops[9].kind = VC4_PLAN_CUMSUM;
    ops[9].a_off = 0x1300; ops[9].a_bytes = 0x400;
    ops[9].c_off = 0x1400; ops[9].c_bytes = 0x400;
    ops[9].p[0] = 2; ops[9].p[1] = 256;

    ops[10].kind = VC4_PLAN_ALIAS;

    st = vc4_plan_run_h(NULL, ops, sizeof ops / sizeof ops[0], &hooks);
    expect(st == VC4_OK, "a full plan runs to completion");
    expect(rec.n == 10, "the alias record issues no call (10 of 11)");
    if (rec.n == 10) {
        expect(rec.calls[0].kind == VC4_PLAN_EW &&
               rec.calls[0].a_off == 0x100 && rec.calls[0].b_off == 0x200 &&
               rec.calls[0].c_off == 0x300 && rec.calls[0].n == 256 &&
               rec.calls[0].p[0] == 3 && rec.calls[0].has_b,
               "EW decodes op, three buffers, n and b presence");
        expect(rec.calls[1].kind == VC4_PLAN_UNARY &&
               rec.calls[1].p[0] == 1 && rec.calls[1].p[1] == 0x80000000u &&
               rec.calls[1].p[2] == 0x3FC00000u && !rec.calls[1].has_b,
               "UNARY carries the float parameters bit-exactly and no b");
        expect(rec.calls[2].kind == VC4_PLAN_COPY2D &&
               rec.calls[2].p[0] == 10 && rec.calls[2].p[6] == 16,
               "COPY2D decodes its seven fields");
        expect(rec.calls[3].kind == VC4_PLAN_COPY2D_N &&
               rec.calls[3].p[0] == 20 && rec.calls[3].p[9] == 29,
               "COPY2D_N decodes its ten fields");
        expect(rec.calls[4].kind == VC4_PLAN_COPY2D_X &&
               rec.calls[4].p[0] == 30 && rec.calls[4].p[12] == 42,
               "COPY2D_X decodes its thirteen fields");
        expect(rec.calls[5].kind == VC4_PLAN_FILL2D &&
               rec.calls[5].p[4] == (int64_t)0xFF800000u,
               "FILL2D carries -inf as bits");
        expect(rec.calls[6].kind == VC4_PLAN_GATHER && rec.calls[6].n == 96,
               "GATHER carries n and the table buffer");
        expect(rec.calls[7].kind == VC4_PLAN_BLIT && rec.calls[7].n == 4096,
               "BLIT carries the byte count");
        expect(rec.calls[8].kind == VC4_PLAN_REDUCE &&
               rec.calls[8].p[0] == 4 && rec.calls[8].p[1] == 64 &&
               rec.calls[8].p[2] == 2,
               "REDUCE carries outer, red and the op");
        expect(rec.calls[9].kind == VC4_PLAN_CUMSUM &&
               rec.calls[9].p[0] == 2 && rec.calls[9].p[1] == 256,
               "CUMSUM carries rows and cols");
    }

    /* a failing call stops the walk and is reported unchanged */
    memset(&rec, 0, sizeof rec);
    rec.fail_at = 3;
    rec.fail_with = VC4_ERR_GPU;
    st = vc4_plan_run_h(NULL, ops, sizeof ops / sizeof ops[0], &hooks);
    expect(st == VC4_ERR_GPU, "the first failing call's status is returned");
    expect(rec.n == 3, "the walk stops at the failing record");

    /* argument checks */
    expect(vc4_plan_run_h(NULL, NULL, 1, &hooks) == VC4_ERR_INVAL,
           "a NULL plan is refused");
    expect(vc4_plan_run_h(NULL, ops, 0, &hooks) == VC4_ERR_INVAL,
           "an empty plan is refused");
    {
        vc4_plan_op bad = op_of(VC4_PLAN_KINDS);
        st = vc4_plan_run_h(NULL, &bad, 1, &hooks);
        expect(st == VC4_ERR_INVAL, "an out-of-range kind is refused");
    }

    printf("\n%s (%d failure%s)\n", errors ? "FAIL" : "OK", errors,
           errors == 1 ? "" : "s");
    return errors ? 1 : 0;
}
