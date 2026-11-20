/*
 * test_daemon_uni.c - G4: scalar parameters <= -2.0 through the daemon.
 *
 * The daemon used to reject every uniform word >= 0xC0000000 as a possible GPU
 * bus address, and the engine then refused the call itself with a message
 * naming the bit pattern (vc4_guard.c), because the protocol carried no way to
 * say "this word is a value".  Every float <= -2.0f is in that window:
 * -2.0f = 0xC0000000, -3.0f = 0xC0400000, -inf = 0xFF800000.  So
 * `LeakyRelu(-2)`, `Clip(-3, 3)` and any op whose attribute is at or below
 * -2.0 worked in direct mode and failed in managed mode.
 *
 * With VC4D_LAUNCH_UNI_MASK the launcher declares which uniform slots are
 * addresses and the daemon checks exactly those.  This test runs the real
 * kernels through the daemon and compares every element with a CPU reference;
 * the values below land in the uniform block as 0xC0000000/0xC0400000 and are
 * what used to be refused.
 *
 * usage: test_daemon_uni            (needs VC4INFER_DAEMON=1)
 */
#include "vc4infer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define N 1024

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

typedef float (*ref_fn)(float, float, float);

static float ref_leaky(float x, float a, float unused)
{
    (void)unused;
    return x > 0.0f ? x : a * x;
}
static float ref_clip(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
static float ref_affine(float x, float s, float b)
{
    return s * x + b;
}

static void one(vc4_ctx *ctx, vc4_uop op, const char *label,
                float p0, float p1, ref_fn ref)
{
    vc4_buf a, o;
    if (vc4_alloc(ctx, N * 4, &a) != VC4_OK ||
        vc4_alloc(ctx, N * 4, &o) != VC4_OK) {
        printf("  FAIL  %s: alloc\n", label);
        fails++;
        return;
    }
    float *x = malloc(N * 4), *got = malloc(N * 4);
    for (int i = 0; i < N; ++i)
        x[i] = ((float)(i % 29) - 14.0f) * 0.5f;      /* -7.0 .. 7.0 */
    vc4_upload(ctx, &a, x, N * 4);
    vc4_status st = vc4_unary(ctx, op, &a, &o, N, p0, p1, 0.0f);
    if (st != VC4_OK) {
        CHECK(0, "%s: %s (%s)", label, vc4_strerror(st), vc4_last_error(ctx));
        free(x);
        free(got);
        return;
    }
    vc4_download(ctx, &o, got, N * 4);
    long bad = 0;
    double worst = 0;
    for (int i = 0; i < N; ++i) {
        double want = ref(x[i], p0, p1);
        double err = fabs((double)got[i] - want);
        if (err > worst)
            worst = err;
        if (err > 1e-6)
            bad++;
    }
    CHECK(bad == 0, "%s (p0=%g p1=%g): %ld/%d elements wrong, worst |err|=%.3g",
          label, (double)p0, (double)p1, bad, N, worst);
    free(x);
    free(got);
}

int main(void)
{
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s\n", vc4_strerror(st));
        return 1;
    }
    if (!getenv("VC4INFER_DAEMON")) {
        printf("note: VC4INFER_DAEMON is unset, running the direct path "
               "(the daemon case is the point of the test)\n");
    }
    printf("arena %.1f MiB, backend %s\n", vc4_arena_bytes(ctx) / 1048576.0,
           getenv("VC4INFER_DAEMON") ? "vc4d" : "direct");

    /* the three shapes the bug named; -2.0f and -3.0f are the offending words */
    one(ctx, VC4_UOP_LEAKYRELU, "LeakyRelu(-2)", -2.0f, 0.0f, ref_leaky);
    one(ctx, VC4_UOP_CLIP, "Clip(-3,3)", -3.0f, 3.0f, ref_clip);
    one(ctx, VC4_UOP_AFFINE, "Affine(-2 x + 1)", -2.0f, 1.0f, ref_affine);

    vc4_shutdown(ctx);
    printf(fails ? "G4 ENGINE FAIL (%d)\n" : "G4 ENGINE PASS\n", fails);
    return fails ? 1 : 0;
}
