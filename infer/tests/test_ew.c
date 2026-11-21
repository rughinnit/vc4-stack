/* test_ew.c - elementwise ops vs CPU scalar references.
 *
 * add/mul/relu must be bit-exact (they are single IEEE fp32 operations).
 * sigmoid/silu are approximations: the measured worst case is printed and
 * checked against a documented tolerance.
 *
 * div is the kops.EWB body (reciprocal of |b|, sign restored, then multiply).
 * It is checked relatively, with a signed divisor and one near-zero entry per
 * 64 elements: the body used to clamp the divisor itself to [TINY, BIG], which
 * turned every negative divisor into +TINY and the result into a * 1e38.
 *
 * usage: test_ew [reps]
 */
#include "vc4infer.h"
#include "util.h"

static int run(vc4_ctx *ctx, vc4_op op, const char *name, size_t n, int reps,
               int binary, double tol, int exact, int rel)
{
    float *a = malloc(n * 4), *b = malloc(n * 4), *o = malloc(n * 4);
    if (!a || !b || !o)
        return 1;
    for (size_t i = 0; i < n; ++i) {
        a[i] = frand() * 8.0f;
        b[i] = frand() * 8.0f;
    }
    if (op == VC4_OP_DIV) {
        /* both signs in every vector, ordinary magnitudes, and one divisor
           near zero per 64 elements: 1/|b| is 1e7 there, so the result is only
           bounded by the relative check */
        for (size_t i = 0; i < n; ++i) {
            b[i] = (i & 1 ? -1.0f : 1.0f) *
                   (0.5f + (frand() + 1.0f) * 0.75f);
            if (i % 64 == 17)
                b[i] = -1e-7f;
        }
    }
    vc4_buf ba, bb, bo;
    if (vc4_alloc(ctx, n * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, n * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, n * 4, &bo) != VC4_OK) {
        printf("FAIL: arena alloc for %s n=%zu\n", name, n);
        return 1;
    }
    vc4_upload(ctx, &ba, a, n * 4);
    vc4_upload(ctx, &bb, b, n * 4);

    int clean = 0;
    double worst = 0;
    for (int r = 0; r < reps; ++r) {
        memset(o, 0xCD, n * 4);
        vc4_upload(ctx, &bo, o, n * 4);
        vc4_status st = vc4_elementwise(ctx, op, &ba, binary ? &bb : NULL, &bo, n);
        if (st != VC4_OK) {
            printf("FAIL: %s -> %s (%s)\n", name, vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        vc4_download(ctx, &bo, o, n * 4);

        long mm = 0;
        double maxabs = 0;
        for (size_t i = 0; i < n; ++i) {
            double ref, got = o[i];
            switch (op) {
            case VC4_OP_ADD:     ref = (double)a[i] + (double)b[i]; break;
            case VC4_OP_MUL:     ref = (double)a[i] * (double)b[i]; break;
            case VC4_OP_RELU:    ref = a[i] > 0 ? (double)a[i] : 0.0; break;
            case VC4_OP_SIGMOID: ref = 1.0 / (1.0 + exp(-(double)a[i])); break;
            case VC4_OP_SILU:    ref = (double)a[i] / (1.0 + exp(-(double)a[i])); break;
            case VC4_OP_DIV:     ref = (double)a[i] / (double)b[i]; break;
            default: ref = 0; break;
            }
            double err = fabs(got - ref);
            if (rel && fabs(ref) > 1e-30)
                err /= fabs(ref);
            if (err > maxabs)
                maxabs = err;
            if (exact) {
                float fref = (op == VC4_OP_ADD) ? (a[i] + b[i])
                           : (op == VC4_OP_MUL) ? (a[i] * b[i])
                           : (a[i] > 0 ? a[i] : 0.0f);
                float fgot = o[i];   /* compare 32-bit floats, not a double */
                if (memcmp(&fgot, &fref, 4) != 0)
                    mm++;
            } else if (err > tol) {
                mm++;
            }
        }
        if (maxabs > worst)
            worst = maxabs;
        if (mm == 0)
            clean++;
        printf("  %-10s n=%zu rep%d: mismatches=%ld/%zu max %s=%.3g%s\n",
               name, n, r, mm, n, rel ? "rel" : "abs", maxabs,
               exact ? " (bit-exact)" : "");
    }
    printf("%s %-10s n=%zu: %d/%d clean, worst %s = %.3g%s\n",
           clean == reps ? "PASS" : "FAIL", name, n, clean, reps,
           rel ? "|err|/|ref|" : "|err|", worst,
           exact ? " (must be 0 for bit-exact ops)" : "");
    free(a); free(b); free(o);
    return clean == reps ? 0 : 1;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    int lock = gpu_lock();
    if (lock < 0)
        return 2;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s\n", vc4_strerror(st));
        gpu_unlock(lock);
        return 2;
    }
    printf("libvc4infer %s elementwise  arena %.1f MiB  reps=%d\n",
           VC4INFER_VERSION_STRING, (double)vc4_arena_bytes(ctx) / (1 << 20), reps);

    int rc = 0;
    rc |= run(ctx, VC4_OP_ADD, "add", 4096, reps, 1, 0, 1, 0);
    rc |= run(ctx, VC4_OP_MUL, "mul", 4096, reps, 1, 0, 1, 0);
    rc |= run(ctx, VC4_OP_RELU, "relu", 4096, reps, 0, 0, 1, 0);
    rc |= run(ctx, VC4_OP_ADD, "add+tail", 1000, reps, 1, 0, 1, 0);  /* not 256 */
    rc |= run(ctx, VC4_OP_DIV, "div", 4096, reps, 1, 5e-7, 0, 1);
    rc |= run(ctx, VC4_OP_DIV, "div+tail", 1000, reps, 1, 5e-7, 0, 1);
    rc |= run(ctx, VC4_OP_SIGMOID, "sigmoid", 4096, reps, 0, 1e-5, 0, 0);
    rc |= run(ctx, VC4_OP_SILU, "silu", 4096, reps, 0, 1e-5, 0, 0);

    vc4_print_supported_ops(stdout);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return rc;
}
