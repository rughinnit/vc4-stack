/* diag_col2im.c - bring-up checks for the new backward kernels (CPU oracle) */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include <math.h>

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); \
    printf("\n"); g_fail++; } } while (0)

static int t_col2im(vc4_ctx *ctx)
{
    const int Cin = 16, KH = 3, KW = 3, HO = 4, WO = 4, pad = 0;
    const int K = Cin * KH * KW;
    const int Hp = HO + KH - 1;
    const int Wp = (int)((WO + KW - 1 + 15) & ~15);
    const int M = HO * Wp;
    const int N = Cin * Hp * Wp;
    printf("-- col2im Cin=%d K=%d HO=%d WO=%d Hp=%d Wp=%d M=%d\n",
           Cin, K, HO, WO, Hp, Wp, M);
    float *dA = calloc((size_t)M * K, 4);
    for (int y = 0; y < HO; ++y)
        for (int x = 0; x < WO; ++x)
            for (int k = 0; k < K; ++k)
                dA[((size_t)y * Wp + x) * K + k] = frand();
    double *ref = calloc((size_t)N, 8);
    for (int c = 0; c < Cin; ++c)
        for (int t = 0; t < KH * KW; ++t) {
            int ky = t / KW, kx = t % KW;
            for (int y = 0; y < HO; ++y)
                for (int x = 0; x < WO; ++x) {
                    float v = dA[((size_t)y * Wp + x) * K + t * Cin + c];
                    ref[((size_t)c * Hp + y + ky) * Wp + x + kx] += v;
                }
        }
    vc4_buf bA, bX;
    CHECK(vc4_alloc(ctx, (size_t)M * K * 4, &bA) == VC4_OK, "alloc A");
    CHECK(vc4_alloc(ctx, (size_t)N * 4 + 64, &bX) == VC4_OK, "alloc X");
    vc4_upload(ctx, &bA, dA, (size_t)M * K * 4);
    int hp, wp;
    vc4_status st = vc4_col2im(ctx, &bA, Cin, K, KH, KW, HO, WO, &bX, &hp, &wp);
    CHECK(st == VC4_OK, "col2im: %s (%s)", vc4_strerror(st), vc4_last_error(ctx));
    if (st != VC4_OK) return 1;
    CHECK(hp == Hp && wp == Wp, "dims %d %d", hp, wp);
    float *got = malloc((size_t)N * 4);
    vc4_download(ctx, &bX, got, (size_t)N * 4);
    check_t ck = {0};
    for (int i = 0; i < N; ++i)
        check_elem(&ck, got[i], ref[i], 2e-6, 1);
    printf("%s col2im vs oracle: max_abs=%.3g max_rel=%.3g mism=%ld\n",
           ck.mismatches ? "FAIL" : "PASS", ck.max_abs, ck.max_rel, ck.mismatches);
    if (ck.mismatches) g_fail++;
    free(dA); free(ref); free(got);
    return 0;
}

static int t_pool(vc4_ctx *ctx)
{
    const char *e = getenv("POOL_C");
    const int C = e ? atoi(e) : 16;
    const int H = 16, W = 16, op = 16;
    const int Ho = H / 2, Wo = W / 2;
    printf("-- pool C=%d H=%d W=%d -> %dx%d pitch %d\n", C, H, W, Ho, Wo, op);
    float *x = malloc((size_t)C * H * W * 4 + 64);
    for (size_t i = 0; i < (size_t)C * H * W; ++i) x[i] = frand();
    memset(x + (size_t)C * H * W, 0, 64);
    vc4_buf bX, bO, bD, bDX;
    vc4_alloc(ctx, (size_t)C * H * W * 4 + 64, &bX);
    vc4_alloc(ctx, (size_t)C * Ho * op * 4, &bO);
    vc4_alloc(ctx, (size_t)C * Ho * op * 4, &bD);
    vc4_alloc(ctx, (size_t)C * H * W * 4, &bDX);
    vc4_upload(ctx, &bX, x, (size_t)C * H * W * 4 + 64);
    int ho, wo;
    vc4_status st = vc4_maxpool2x2(ctx, &bX, C, H, W, op, &bO, &ho, &wo);
    CHECK(st == VC4_OK, "pool fwd: %s", vc4_strerror(st));
    if (st != VC4_OK) return 1;
    CHECK(ho == Ho && wo == Wo, "pool dims");
    float *o = malloc((size_t)C * Ho * op * 4);
    vc4_download(ctx, &bO, o, (size_t)C * Ho * op * 4);
    long bad = 0;
    float maxerr = 0;
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < Ho; ++y)
            for (int px = 0; px < op; ++px) {
                float ref = 0;
                if (px < Wo) {
                    ref = x[((size_t)c * H + 2 * y) * W + 2 * px];
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            float v = x[((size_t)c * H + 2 * y + dy) * W + 2 * px + dx];
                            if (v > ref) ref = v;
                        }
                }
                float g = o[((size_t)c * Ho + y) * op + px];
                float e = fabsf(g - ref);
                if (e > maxerr) maxerr = e;
                if (e > 0) bad++;
            }
    printf("%s pool fwd vs oracle: bad=%ld max_abs=%.3g\n",
           bad ? "FAIL" : "PASS", bad, maxerr);
    if (bad) g_fail++;

    /* backward: dy in the padded layout */
    float *dy = malloc((size_t)C * Ho * op * 4);
    for (size_t i = 0; i < (size_t)C * Ho * op; ++i) dy[i] = frand();
    vc4_upload(ctx, &bD, dy, (size_t)C * Ho * op * 4);
    st = vc4_maxpool2x2_backward(ctx, &bX, &bD, C, H, W, op, &bDX);
    CHECK(st == VC4_OK, "pool bwd: %s", vc4_strerror(st));
    if (st != VC4_OK) return 1;
    float *dx = malloc((size_t)C * H * W * 4);
    vc4_download(ctx, &bDX, dx, (size_t)C * H * W * 4);
    bad = 0; maxerr = 0;
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int px = 0; px < W; ++px) {
                int gy = y & ~1, gx = px & ~1;
                float mx = x[((size_t)c * H + gy) * W + gx];
                for (int dy2 = 0; dy2 < 2; ++dy2)
                    for (int dx2 = 0; dx2 < 2; ++dx2) {
                        float v = x[((size_t)c * H + gy + dy2) * W + gx + dx2];
                        if (v > mx) mx = v;
                    }
                float ref = (x[((size_t)c * H + y) * W + px] == mx)
                            ? dy[((size_t)c * Ho + (y / 2)) * op + (px / 2)] : 0.0f;
                float e = fabsf(dx[((size_t)c * H + y) * W + px] - ref);
                if (e > maxerr) maxerr = e;
                if (e > 0) bad++;
            }
    printf("%s pool bwd vs oracle: bad=%ld max_abs=%.3g\n",
           bad ? "FAIL" : "PASS", bad, maxerr);
    if (bad) g_fail++;
    free(x); free(o); free(dy); free(dx);
    return 0;
}

static int t_im2col(vc4_ctx *ctx)
{
    const int Cin = 16, H = 6, W = 6, KH = 3, KW = 3, pad = 0;
    const int ho = H + 2 * pad - KH + 1, wo = W + 2 * pad - KW + 1;
    const int hp = H + 2 * pad, wp = (W + 2 * pad + 15) & ~15;
    const int M = ho * wp, K = Cin * KH * KW;
    printf("-- im2col(row) Cin=%d H=%d W=%d -> M=%d K=%d wp=%d\n", Cin, H, W, M, K, wp);
    float *x = malloc((size_t)Cin * H * W * 4);
    for (size_t i = 0; i < (size_t)Cin * H * W; ++i) x[i] = frand();
    float *padimg = calloc((size_t)Cin * hp * wp, 4);
    for (int c = 0; c < Cin; ++c)
        for (int y = 0; y < H; ++y)
            memcpy(padimg + ((size_t)c * hp + y + pad) * wp + pad,
                   x + ((size_t)c * H + y) * W, (size_t)W * 4);
    vc4_buf bX, bA;
    vc4_alloc(ctx, (size_t)Cin * H * W * 4, &bX);
    vc4_alloc(ctx, (size_t)M * K * 4, &bA);
    vc4_upload(ctx, &bX, x, (size_t)Cin * H * W * 4);
    int rows;
    vc4_status st = vc4_im2col(ctx, &bX, Cin, H, W, KH, KW, pad, &bA, &rows);
    CHECK(st == VC4_OK, "im2col: %s", vc4_strerror(st));
    if (st != VC4_OK) return 1;
    CHECK(rows == M, "rows %d != %d", rows, M);
    float *got = malloc((size_t)M * K * 4);
    vc4_download(ctx, &bA, got, (size_t)M * K * 4);
    check_t ck = {0};
    for (int p = 0; p < M; ++p)
        for (int t = 0; t < KH * KW; ++t) {
            int ky = t / KW, kx = t % KW;
            for (int c = 0; c < Cin; ++c) {
                double ref = padimg[((size_t)c * hp + (p / wp) + ky) * wp +
                                   (p % wp) + kx];
                check_elem(&ck, got[(size_t)p * K + t * Cin + c], ref, 0, 0);
            }
        }
    printf("%s im2col(row) vs oracle: max_abs=%.3g mism=%ld\n",
           ck.mismatches ? "FAIL" : "PASS", ck.max_abs, ck.mismatches);
    if (ck.mismatches) g_fail++;
    free(x); free(padimg); free(got);
    return 0;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    const char *only = argc > 1 ? argv[1] : "all";
    setvbuf(stdout, NULL, _IONBF, 0);
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    printf("diag start (only=%s) -- unbuffered\n", only);
    if (!strcmp(only, "all") || !strcmp(only, "im2col")) t_im2col(ctx);
    if (!strcmp(only, "all") || !strcmp(only, "col2im")) t_col2im(ctx);
    if (!strcmp(only, "all") || !strcmp(only, "pool")) t_pool(ctx);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(g_fail ? "DIAG FAILED\n" : "DIAG PASSED\n");
    return g_fail != 0;
}
