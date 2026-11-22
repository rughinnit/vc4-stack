/* test_conv.c - vc4_conv2d vs a direct CPU convolution oracle.
 *
 * Every output element of every run is compared.  The CPU reference is the
 * verifier only.
 *
 * usage: test_conv [reps]
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

#define TOL 1e-4

static int one_conv(vc4_ctx *ctx, int Cin, int H, int W, int Cout,
                    int KH, int KW, int pad, int use_bias, int reps)
{
    const int ho = (H + 2 * pad - KH) + 1;
    const int wo = (W + 2 * pad - KW) + 1;
    const size_t nin = (size_t)Cin * H * W;
    const size_t nwt = (size_t)Cout * Cin * KH * KW;
    const size_t nout = (size_t)Cout * ho * wo;

    float *in = malloc(nin * 4), *wt = malloc(nwt * 4), *bi = malloc((size_t)Cout * 4);
    float *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    if (!in || !wt || !bi || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand() * 0.25f;
    for (int i = 0; i < Cout; ++i) bi[i] = frand();
    memset(out, 0xCD, nout * 4);

    vc4_buf bin, bwt, bbi, bout;
    if (vc4_alloc(ctx, nin * 4, &bin) != VC4_OK ||
        vc4_alloc(ctx, nwt * 4, &bwt) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Cout * 4, &bbi) != VC4_OK ||
        vc4_alloc(ctx, nout * 4, &bout) != VC4_OK) {
        printf("FAIL: arena alloc for conv %dx%dx%d\n", Cin, H, W);
        return 1;
    }
    vc4_upload(ctx, &bin, in, nin * 4);
    vc4_upload(ctx, &bwt, wt, nwt * 4);
    if (use_bias)
        vc4_upload(ctx, &bbi, bi, (size_t)Cout * 4);
    vc4_upload(ctx, &bout, out, nout * 4);

    ref_conv(in, Cin, H, W, wt, Cout, KH, KW, use_bias ? bi : NULL, pad, ref, ho, wo);

    int clean = 0;
    double best_ms = 0, best_gf = 0;
    for (int rep = 0; rep < reps; ++rep) {
        memset(out, 0, nout * 4);
        vc4_upload(ctx, &bout, out, nout * 4);

        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        int gho = 0, gwo = 0;
        uint64_t t0 = now_ns();
        vc4_status st = vc4_conv2d(ctx, &bin, Cin, H, W, &bwt, Cout, KH, KW,
                                   use_bias ? &bbi : NULL, 1, pad, &bout,
                                   &gho, &gwo);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: vc4_conv2d(%d,%d,%d / %dx%dx%d) -> %s (%s)\n",
                   Cin, H, W, Cout, KH, KW, vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        if (gho != ho || gwo != wo) {
            printf("FAIL: conv2d reported %dx%d, expected %dx%d\n", gho, gwo, ho, wo);
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);

        check_t ck = { 0, 0, 0, 1 };
        for (size_t i = 0; i < nout; ++i)
            check_elem(&ck, out[i], ref[i], TOL, i < 8);

        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) + (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall_ms = (double)(t1 - t0) / 1e6;
        double flops = 2.0 * Cout * ho * wo * Cin * KH * KW;
        double gf = flops / (gpu_ms * 1e6);
        if (gf > best_gf) { best_gf = gf; best_ms = gpu_ms; }
        if (ck.mismatches == 0)
            clean++;
        printf("  conv %dx%dx%d -> %dx%dx%d k%dx%d pad%d %s rep%d: mismatches=%ld/%zu "
               "maxabs=%.3g | gpu %.3f ms %.2f GFLOP/s (wall %.3f ms)\n",
               Cin, H, W, Cout, ho, wo, KH, KW, pad, use_bias ? "+bias" : "no-bias",
               rep, ck.mismatches, nout, ck.max_abs, gpu_ms, gf, wall_ms);
    }
    printf("%s conv %dx%dx%d->%dx%dx%d k%d pad%d%s: %d/%d clean, best %.3f ms %.2f GFLOP/s\n",
           clean == reps ? "PASS" : "FAIL", Cin, H, W, Cout, ho, wo, KH, pad,
           use_bias ? "+bias" : "", clean, reps, best_ms, best_gf);
    free(in); free(wt); free(bi); free(out); free(ref);
    return clean == reps ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

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
    printf("libvc4infer %s conv2d  arena %.1f MiB  reps=%d\n",
           VC4INFER_VERSION_STRING, (double)vc4_arena_bytes(ctx) / (1 << 20), reps);

    int rc = 0;
    rc |= one_conv(ctx, 16, 16, 16, 16, 3, 3, 1, 1, reps);   /* Cin=16 */
    rc |= one_conv(ctx, 32, 16, 32, 32, 3, 3, 1, 1, reps);
    rc |= one_conv(ctx, 16, 8, 16, 16, 1, 1, 0, 0, reps);    /* 1x1, no bias */
    rc |= one_conv(ctx, 32, 8, 24, 16, 1, 1, 0, 1, reps);    /* 1x1 + bias   */
    rc |= one_conv(ctx, 16, 12, 12, 16, 3, 3, 0, 1, reps);   /* no padding   */
    rc |= one_conv(ctx, 64, 8, 16, 32, 3, 3, 1, 1, reps);    /* Cin=64, K=576 */

    /* strict dispatch: shapes outside the table must be refused, not computed */
    {
        vc4_buf dummy;
        int ho = 0, wo = 0;
        vc4_alloc(ctx, 64 * 64 * 4, &dummy);
        vc4_status s2 = vc4_conv2d(ctx, &dummy, 16, 8, 8, &dummy, 16, 5, 5,
                                   NULL, 1, 2, &dummy, &ho, &wo);
        printf("%s unsupported 5x5 -> %s\n", s2 == VC4_ERR_UNSUPPORTED ? "PASS" : "FAIL",
               vc4_strerror(s2));
        rc |= (s2 == VC4_ERR_UNSUPPORTED) ? 0 : 1;
        s2 = vc4_conv2d(ctx, &dummy, 16, 8, 8, &dummy, 16, 3, 3, NULL, 2, 1,
                        &dummy, &ho, &wo);
        printf("%s unsupported stride 2 -> %s\n", s2 == VC4_ERR_UNSUPPORTED ? "PASS" : "FAIL",
               vc4_strerror(s2));
        rc |= (s2 == VC4_ERR_UNSUPPORTED) ? 0 : 1;
    }

    vc4_print_supported_ops(stdout);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return rc;
}
