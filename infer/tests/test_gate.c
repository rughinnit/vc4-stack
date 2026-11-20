/* test_gate.c - the refusal contract for the unverified kernels.
 *
 * Nothing in this file launches anything.  Every entry point below must refuse
 * BEFORE it builds a kernel, with VC4_ERR_UNSUPPORTED and a message that names
 * the kernel.  That is the product requirement: a caller of the public API
 * must not be able to wedge the board, and these kernels are confirmed to
 * either hang the QPU (killing the vc4d context for every later client) or
 * return wrong results:
 *
 *   vc4_gather, vc4_cumsum        hang every shape (TMU setup)
 *   vc4_reduce                    hangs every shape (256-output group, r4 seed)
 *   vc4_select3(WHERE)            returns success with every element wrong
 *   vc4_blit                      returns success and copies nothing
 *
 * The test deliberately does not set VC4_ALLOW_UNVERIFIED_KERNELS: with the
 * override set, the calls would launch and one of them would take the GPU down
 * (that is exactly what the gate exists to prevent).
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

    /* the gate runs before any buffer is read, so these are never touched */
    vc4_buf a = { 4096, 1024 }, b = { 8192, 1024 }, c = { 12288, 1024 };
    vc4_buf o = { 16384, 1024 };

    expect_refusal(ctx, "vc4_gather", vc4_gather(ctx, &a, &o, 16, 0.0f));
    expect_refusal(ctx, "vc4_blit", vc4_blit(ctx, &a, &o, 64));
    expect_refusal(ctx, "vc4_reduce",
                   vc4_reduce(ctx, &a, &o, 256, 16, VC4_RED_SUM));
    expect_refusal(ctx, "vc4_cumsum", vc4_cumsum(ctx, &a, &o, 1, 256));
    expect_refusal(ctx, "vc4_select3",
                   vc4_select3(ctx, VC4_SEL_WHERE, &a, &b, &c, &o, 16));

    /* The uniform rule has a second corner: vc4d refuses any uniform word
     * >= 0xC0000000 as a possible bus address, and every float <= -2.0 is one
     * (-2.0f == 0xC0000000).  In daemon mode the engine must refuse such a
     * parameter itself, naming the value, instead of letting the daemon report
     * "uniform is a bus/arena/MMIO address".  In direct mode the value is legal
     * and this check does not exist. */
    if (getenv("VC4INFER_DAEMON") && *getenv("VC4INFER_DAEMON") &&
        strcmp(getenv("VC4INFER_DAEMON"), "0") != 0) {
        vc4_status st = vc4_unary(ctx, VC4_UOP_AFFINE, &a, &o, 16, -2.0f,
                                  0.0f, 0.0f);
        const char *msg = vc4_last_error(ctx);
        if (st != VC4_ERR_UNSUPPORTED || !msg || !strstr(msg, "0x")) {
            printf("FAIL uniform value: st=%d (%s), expected a refusal naming "
                   "the bit pattern\n", (int)st, msg ? msg : "(no message)");
            ++errors;
        } else {
            printf("  %-12s refused: %s\n", "uniform/-2.0", msg);
        }
    } else {
        printf("  uniform/-2.0  skipped (direct mode: the value is legal)\n");
    }

    vc4_shutdown(ctx);
    printf("%s\n", errors ? "GATE TEST FAILED" : "ALL UNVERIFIED KERNELS REFUSE");
    return errors ? 1 : 0;
}
