/* diag_conv.c - real conv2d against a CPU oracle, with the panel path
 * (im2col -> pack) validated implicitly by the result.
 * usage: diag_conv Cin H W Cout KH KW pad [bias]
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int Cin = argc > 1 ? atoi(argv[1]) : 16;
    int H = argc > 2 ? atoi(argv[2]) : 8;
    int W = argc > 3 ? atoi(argv[3]) : 8;
    int Cout = argc > 4 ? atoi(argv[4]) : 16;
    int KH = argc > 5 ? atoi(argv[5]) : 1;
    int KW = argc > 6 ? atoi(argv[6]) : 1;
    int pad = argc > 7 ? atoi(argv[7]) : 0;
    int use_bias = argc > 8 ? atoi(argv[8]) : 0;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    const int ho = (H + 2 * pad - KH) + 1, wo = (W + 2 * pad - KW) + 1;
    float *in = malloc((size_t)Cin * H * W * 4);
    float *wt = malloc((size_t)Cout * Cin * KH * KW * 4);
    float *bs = calloc((size_t)Cout, 4);
    float *out = malloc((size_t)Cout * ho * wo * 4);
    for (size_t i = 0; i < (size_t)Cin * H * W; ++i) in[i] = frand();
    for (size_t i = 0; i < (size_t)Cout * Cin * KH * KW; ++i) wt[i] = frand();
    for (int i = 0; i < Cout; ++i) bs[i] = frand();

    vc4_buf bi, bw, bb, bo;
    vc4_alloc(ctx, (size_t)Cin * H * W * 4, &bi);
    vc4_alloc(ctx, (size_t)Cout * Cin * KH * KW * 4, &bw);
    if (use_bias) vc4_alloc(ctx, (size_t)Cout * 4, &bb);
    vc4_alloc(ctx, (size_t)Cout * ho * wo * 4, &bo);
    vc4_upload(ctx, &bi, in, (size_t)Cin * H * W * 4);
    vc4_upload(ctx, &bw, wt, (size_t)Cout * Cin * KH * KW * 4);
    if (use_bias) vc4_upload(ctx, &bb, bs, (size_t)Cout * 4);
    memset(out, 0xCD, (size_t)Cout * ho * wo * 4);
    vc4_upload(ctx, &bo, out, (size_t)Cout * ho * wo * 4);

    int ho2 = 0, wo2 = 0;
    vc4_status st = vc4_conv2d(ctx, &bi, Cin, H, W, &bw, Cout, KH, KW,
                               use_bias ? &bb : NULL, 1, pad, &bo, &ho2, &wo2);
    printf("conv %dx%dx%d -> %dx%dx%d k%dx%d pad%d%s -> %s\n", Cin, H, W, Cout,
           H, W, KH, KW, pad, use_bias ? " +bias" : "", vc4_strerror(st));
    if (st != VC4_OK) { printf("  %s\n", vc4_last_error(ctx)); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bo, out, (size_t)Cout * ho * wo * 4);

    double worst = 0;
    long bad = 0;
    for (int co = 0; co < Cout; ++co)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < wo; ++x) {
                double acc = use_bias ? bs[co] : 0.0;
                for (int ci = 0; ci < Cin; ++ci)
                    for (int ky = 0; ky < KH; ++ky)
                        for (int kx = 0; kx < KW; ++kx) {
                            int sy = y + ky - pad, sx = x + kx - pad;
                            if (sy < 0 || sy >= H || sx < 0 || sx >= W) continue;
                            acc += (double)in[((size_t)ci * H + sy) * W + sx] *
                                   wt[(((size_t)co * Cin + ci) * KH + ky) * KW + kx];
                        }
                double got = out[((size_t)co * ho + y) * wo + x];
                double err = fabs(got - acc);
                double den = fabs(acc) > 1.0 ? fabs(acc) : 1.0;
                if (err > 1e-4 * den) {
                    if (bad < 6)
                        printf("  co%d y%d x%d got %.6g ref %.6g\n", co, y, x, got, acc);
                    bad++;
                }
                if (err / den > worst) worst = err / den;
            }
    printf("conv: %ld/%d mismatches (worst rel %.3g) -> %s\n", bad, Cout * ho * wo,
           worst, bad ? "FAIL" : "PASS");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return bad ? 1 : 0;
}
