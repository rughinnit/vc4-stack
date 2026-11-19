/* diag_im2col.c - is the QPU im2col panel what the GEMM expects?
 * usage: diag_im2col Cin H W KH KW pad
 * builds the padded image exactly as vc4_conv2d does, runs the im2col kernel,
 * and compares against the CPU reference
 *   PANEL[(rb*K + k)*8 + r], k = tap*Cin + ci, tap = ky*KW + kx
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int Cin = argc > 1 ? atoi(argv[1]) : 16;
    int H = argc > 2 ? atoi(argv[2]) : 8;
    int W = argc > 3 ? atoi(argv[3]) : 8;
    int KH = argc > 4 ? atoi(argv[4]) : 1;
    int KW = argc > 5 ? atoi(argv[5]) : 1;
    int pad = argc > 6 ? atoi(argv[6]) : 0;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    const int ho = (H + 2 * pad - KH) + 1, wo = (W + 2 * pad - KW) + 1;
    const int hp = H + 2 * pad;
    const int wp = (int)vc4__align16((uint32_t)(W + 2 * pad));
    const uint32_t K = (uint32_t)Cin * (uint32_t)(KH * KW);
    const uint32_t nk = vc4__align16(K);
    const uint32_t M = (uint32_t)ho * (uint32_t)wp;
    printf("Cin=%d H=%d W=%d K=%d M=%u ho=%d wo=%d wp=%d hp=%d\n",
           Cin, H, W, K, M, ho, wo, wp, hp);

    /* reference image, values encode (c,y,x) */
    float *img = malloc((size_t)Cin * H * W * 4);
    for (int c = 0; c < Cin; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                img[((size_t)c * H + y) * W + x] =
                    (float)(c * 1000000 + y * 1000 + x);

    vc4__scratch_reset(ctx);
    float *pad_img = vc4__scratch_alloc(ctx, (size_t)Cin * hp * wp * 4 + 64, NULL);
    memset((uint8_t *)pad_img, 0, (size_t)Cin * hp * wp * 4 + 64);
    for (int c = 0; c < Cin; ++c)
        for (int y = 0; y < H; ++y)
            memcpy((uint8_t *)pad_img +
                       ((size_t)c * hp * wp + (size_t)(y + pad) * wp + pad) * 4,
                   (uint8_t *)img + ((size_t)c * H + (size_t)y) * W * 4,
                   (size_t)W * 4);
    const uint32_t pad_bus =
        ctx->gpu_bus + (uint32_t)((uint8_t *)pad_img - ctx->arena);

    vc4_buf pb;
    float *panel = vc4__scratch_alloc(ctx, (size_t)M * nk * 4, &pb);
    if (!panel) { printf("no panel\n"); return 1; }

    char ncin[16], nk_[16], nkh[8], nkw[8], name[64];
    char *kargv[] = { (char *)"im2col", (char *)"--cin", ncin, (char *)"--k", nk_,
                      (char *)"--kh", nkh, (char *)"--kw", nkw, NULL };
    snprintf(ncin, sizeof(ncin), "%d", Cin);
    snprintf(nk_, sizeof(nk_), "%u", K);
    snprintf(nkh, sizeof(nkh), "%d", KH);
    snprintf(nkw, sizeof(nkw), "%d", KW);
    snprintf(name, sizeof(name), "diagcol_%d_%u_%d_%d", Cin, K, KH, KW);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, kargv, 4, &k);
    if (st != VC4_OK) { printf("kernel: %s\n", vc4_strerror(st)); return 1; }
    const uint32_t tr = M / 16u;
    int pnq = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
    const uint32_t per = tr / (uint32_t)pnq;
    for (int i = 0; i < pnq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = (uint32_t)hp * wp * 4u;
        u[1] = (uint32_t)i * per * 64u;
        u[2] = pad_bus;
        u[3] = 64u;
        u[4] = ctx->gpu_bus + pb.off;
        u[5] = per;
        u[6] = (uint32_t)i;
        u[7] = 256u;
        u[8] = 64u * nk;
        u[9] = (uint32_t)wp * 4u;
        u[10] = (uint32_t)i * per * 64u * nk;
        u[11] = 32u * nk;
        u[12] = nk * 4u - 32u;
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, pnq, &ns);
    printf("im2col launch nq=%d -> %s (%.4f ms)\n", pnq, vc4_strerror(st), ns / 1e6);
    if (st != VC4_OK) return 1;
    vc4__l2_flush(ctx);

    /* reference */
    float *ref = calloc((size_t)M * nk, 4);
    /* pixel-major: REF[pixel*nk + k] */
    int taps = KH * KW;
    for (uint32_t rb = 0; rb < M / 8; ++rb)
        for (int ci = 0; ci < Cin; ++ci)
            for (int t = 0; t < taps; ++t) {
                int ky = t / KW, kx = t % KW;
                int kk = t * Cin + ci;
                for (int r = 0; r < 8; ++r) {
                    uint32_t p = rb * 8u + (uint32_t)r;
                    int x = (int)(p % (uint32_t)wp), y = (int)(p / (uint32_t)wp);
                    int sy = y + ky, sx = x + kx;
                    float v = 0;
                    if (sy < hp && sx < wp)
                        v = pad_img[((size_t)ci * hp + sy) * wp + sx];
                    ref[p * nk + (uint32_t)kk] = v;
                }
            }

    printf("first block, r=0:  k  panel[..]           ref[..]\n");
    for (uint32_t kk = 0; kk < nk && kk < 20; ++kk)
        printf("  k=%2u  %10.1f  %10.1f\n", kk, panel[(size_t)kk * 8],
               ref[(size_t)kk * 8]);
    long bad = 0, shown = 0;
    for (size_t i = 0; i < (size_t)M * nk; ++i)
        if (((uint32_t *)panel)[i] != ((uint32_t *)ref)[i]) {
            if (shown < 16) {
                printf("  idx %zu (pixel %zu k %zu): got %.6g ref %.6g\n", i,
                       i / nk, i % nk, panel[i], ref[i]);
                shown++;
            }
            bad++;
        }
    printf("panel: %ld/%zu mismatches -> %s\n", bad, (size_t)M * nk,
           bad ? "FAIL" : "PASS");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return bad ? 1 : 0;
}
