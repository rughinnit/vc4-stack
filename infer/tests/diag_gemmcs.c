/* diag_gemmcs.c - does vc4__gemm3_run_c honour a C row stride > N? */
#include "vc4_int.h"
#include "util.h"

int main(void)
{
    int lock = gpu_lock();
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("init failed\n");
        return 1;
    }
    const uint32_t M = 32, N = 16, K = 16, CS = 32;
    float *A = malloc(M * K * 4), *B = malloc(K * N * 4);
    float *C = malloc(M * CS * 4), *got = malloc(M * CS * 4);
    double *ref = calloc(M * CS, sizeof(double));
    for (uint32_t i = 0; i < M * K; ++i) A[i] = frand();
    for (uint32_t i = 0; i < K * N; ++i) B[i] = frand();
    for (uint32_t i = 0; i < M * CS; ++i) C[i] = 12345.0f;
    /* k-major A panel: PANEL[(rb*K + k)*16 + p] */
    float *AP = malloc(M * K * 4);
    for (uint32_t rb = 0; rb < M / 16; ++rb)
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t p = 0; p < 16; ++p)
                AP[(rb * K + k) * 16 + p] = A[(rb * 16 + p) * K + k];
    float *BP = malloc(K * N * 4);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t c = 0; c < N; ++c)
            BP[k * 16 + c] = B[k * N + c];
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c) {
            double s = 0;
            for (uint32_t k = 0; k < K; ++k)
                s += (double)A[r * K + k] * B[k * N + c];
            ref[r * CS + c] = s;
        }
    vc4_buf ba, bb, bc, bap, bbp;
    vc4_alloc(ctx, M * K * 4, &ba);
    vc4_alloc(ctx, K * N * 4, &bb);
    vc4_alloc(ctx, M * CS * 4, &bc);
    vc4_alloc(ctx, M * K * 4, &bap);
    vc4_alloc(ctx, K * N * 4, &bbp);
    vc4_upload(ctx, &ba, A, M * K * 4);
    vc4_upload(ctx, &bb, B, K * N * 4);
    vc4_upload(ctx, &bc, C, M * CS * 4);
    vc4_upload(ctx, &bap, AP, M * K * 4);
    vc4_upload(ctx, &bbp, BP, K * N * 4);
    vc4_status st = vc4__gemm3_run_c(ctx, M, N, K, 1, vc4_bus_addr(ctx, &bbp),
                                     vc4_bus_addr(ctx, &bc),
                                     vc4_bus_addr(ctx, &bap), CS);
    printf("st=%d %s\n", st, vc4_last_error(ctx));
    vc4_download(ctx, &bc, got, M * CS * 4);
    double maxerr = 0, maxpad = 0;
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < CS; ++c) {
            if (c < N) {
                double d = fabs((double)got[r * CS + c] - ref[r * CS + c]);
                if (d > maxerr) maxerr = d;
            } else {
                double d = fabs((double)got[r * CS + c] - 12345.0);
                if (d > maxpad) maxpad = d;
            }
        }
    printf("maxerr %.4g  max pad change %.4g\n", maxerr, maxpad);
    printf("row0: ");
    for (uint32_t c = 0; c < CS; ++c) printf("%.3f ", got[c]);
    printf("\nref : ");
    for (uint32_t c = 0; c < CS; ++c) printf("%.3f ", ref[c]);
    printf("\n");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return maxerr < 1e-5 && maxpad == 0 ? 0 : 1;
}
