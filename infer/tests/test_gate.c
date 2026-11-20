/* test_gate.c - the entry points that used to be refused are open, and their
 * argument checks still run before anything is launched.
 *
 * vc4_gather / vc4_reduce / vc4_cumsum / vc4_select3 are element-verified on
 * silicon now (tests/test_gated.c); vc4_blit has ridden copy2d for longer.
 * Nothing here launches anything: each call passes an argument the function
 * rejects cheaply, and the check is that the refusal is the normal argument
 * error, not the old "not verified on this board" gate.
 *
 * usage: test_gate
 */
#include "vc4infer.h"
#include "util.h"

static int errors;

static void expect_arg_refusal(vc4_ctx *ctx, const char *what, vc4_status st,
                               const char *frag)
{
    const char *msg = vc4_last_error(ctx);
    if (st != VC4_ERR_UNSUPPORTED && st != VC4_ERR_INVAL) {
        printf("FAIL %s: status %d (%s), expected an argument refusal\n", what,
               (int)st, vc4_strerror(st));
        ++errors;
        return;
    }
    if (frag && (!msg || !*msg)) {
        printf("FAIL %s: refused with no message\n", what);
        ++errors;
        return;
    }
    if (msg && strstr(msg, "not verified on this board")) {
        printf("FAIL %s: still gated: %s\n", what, msg);
        ++errors;
        return;
    }
    if (frag && !strstr(msg, frag)) {
        printf("FAIL %s: message does not mention %s: %s\n", what, frag, msg);
        ++errors;
        return;
    }
    printf("  %-12s open, refuses: %s\n", what, msg);
}

int main(void)
{
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        return 1;
    }
    printf("libvc4infer %s entry-point check\n", VC4INFER_VERSION_STRING);

    vc4_buf a = { 4096, 1024 }, b = { 8192, 1024 }, c = { 12288, 1024 };
    vc4_buf o = { 16384, 1024 };

    /* table / output too small for the requested element count */
    expect_arg_refusal(ctx, "vc4_gather",
                       vc4_gather(ctx, &a, &o, 100000, 0.0f), NULL);
    /* red = 0 is not reducible */
    expect_arg_refusal(ctx, "vc4_reduce",
                       vc4_reduce(ctx, &a, &o, 16, 0, VC4_RED_SUM),
                       "not reducible");
    /* the scan axis must be a multiple of 256 */
    expect_arg_refusal(ctx, "vc4_cumsum", vc4_cumsum(ctx, &a, &o, 1, 100),
                       "multiple of 256");
    /* op outside the enum */
    expect_arg_refusal(ctx, "vc4_select3",
                       vc4_select3(ctx, (vc4_selop)99, &a, &b, &c, &o, 16),
                       "not implemented");

    /* blit keeps its 64 B granularity refusal */
    expect_arg_refusal(ctx, "vc4_blit", vc4_blit(ctx, &a, &o, 100),
                       "multiple of 64");

    if (errors) {
        printf("ENTRY-POINT CHECK FAILED\n");
        vc4_shutdown(ctx);
        return 1;
    }
    printf("FORMERLY GATED KERNELS ARE OPEN (argument refusals still fire)\n");
    vc4_shutdown(ctx);
    return 0;
}
