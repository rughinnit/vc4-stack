/* test_gate.c - the unverified-kernel gate, and a value word that is not an
 * address.
 *
 * Four kernels are still refused by vc4__refuse_unverified(): gather,
 * reduce, cumsum and select3 are element-exact in the ALU/TMU model but on
 * silicon their group load/store path still returns another launch's bytes
 * (docs/KERNELS.md).  vc4_blit is *not* gated any more: it rides the verified
 * copy2d kernel and is byte-exact (tests/test_gated.c).
 *
 * Nothing in this file launches anything.
 *
 * usage: test_gate
 */
#include "vc4infer.h"
#include "util.h"

static int errors;

static void expect_refusal(vc4_ctx *ctx, const char *what, vc4_status st)
{
    const char *msg;
    if (st != VC4_ERR_UNSUPPORTED) {
        printf("FAIL %s: status %d (%s), expected VC4_ERR_UNSUPPORTED\n", what,
               (int)st, vc4_strerror(st));
        ++errors;
        return;
    }
    msg = vc4_last_error(ctx);
    if (!msg || !*msg) {
        printf("FAIL %s: refused with no message\n", what);
        ++errors;
        return;
    }
    if (!strstr(msg, what)) {
        printf("FAIL %s: message does not name the kernel: %s\n", what, msg);
        ++errors;
        return;
    }
    printf("  %-12s refused: %s\n", what, msg);
}

int main(void)
{
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        return 1;
    }
    printf("libvc4infer %s unverified-kernel gate\n", VC4INFER_VERSION_STRING);

    /* The gate runs before any buffer is read, so these are never touched */
    vc4_buf a = { 4096, 1024 }, b = { 8192, 1024 }, c = { 12288, 1024 };
    vc4_buf o = { 16384, 1024 };

    expect_refusal(ctx, "vc4_gather", vc4_gather(ctx, &a, &o, 16, 0.0f));
    expect_refusal(ctx, "vc4_reduce",
                   vc4_reduce(ctx, &a, &o, 256, 16, VC4_RED_SUM));
    expect_refusal(ctx, "vc4_cumsum", vc4_cumsum(ctx, &a, &o, 1, 256));
    expect_refusal(ctx, "vc4_select3",
                   vc4_select3(ctx, VC4_SEL_WHERE, &a, &b, &c, &o, 16));

    /* blit is fixed: a length that is not a multiple of 64 B must be refused
     * by blit's *own* check, not by the gate (which would name the kernel as
     * unverified).  The granularity check runs before any buffer is touched. */
    {
        vc4_status st = vc4_blit(ctx, &a, &o, 100);
        const char *msg = vc4_last_error(ctx);
        if (st != VC4_ERR_UNSUPPORTED || !msg ||
            !strstr(msg, "multiple of 64") ||
            strstr(msg, "not verified on this board")) {
            printf("FAIL vc4_blit: 100 B returned %d (%s), expected the "
                   "64 B granularity refusal\n", (int)st,
                   msg ? msg : "(no message)");
            ++errors;
        } else {
            printf("  %-12s open, refuses a non-64 B length\n", "vc4_blit");
        }
    }

    /* The uniform rule has a second corner, fixed in 1.2.7 (G4): vc4d used to
     * refuse any uniform word >= 0xC0000000 as a possible bus address, and
     * every float <= -2.0 is one (-2.0f == 0xC0000000).  The engine refused
     * such a parameter itself in daemon mode.  LAUNCH now carries a per-slot
     * address mask, so the value must be *accepted* and computed, in both
     * modes, with the same result. */
    {
        vc4_buf ua = { 0, 0 }, uo = { 0, 0 };
        if (vc4_alloc(ctx, 64, &ua) != VC4_OK ||
            vc4_alloc(ctx, 64, &uo) != VC4_OK) {
            printf("FAIL: vc4_alloc for the uniform check\n");
            ++errors;
        } else {
            vc4_status st = vc4_unary(ctx, VC4_UOP_AFFINE, &ua, &uo, 16, -2.0f,
                                      0.0f, 0.0f);
            const char *msg = vc4_last_error(ctx);
            if (st != VC4_OK) {
                printf("FAIL uniform value -2.0: st=%d (%s), expected the "
                       "launch to run (G4)\n", (int)st,
                       msg ? msg : "(no message)");
                ++errors;
            } else {
                printf("  %-12s launched: a parameter of -2.0 is legal\n",
                       "uniform/-2.0");
            }
        }
    }

    vc4_shutdown(ctx);
    printf("%s\n", errors ? "GATE TEST FAILED"
                          : "FOUR KERNELS STILL REFUSE, BLIT IS OPEN");
    return errors ? 1 : 0;
}
