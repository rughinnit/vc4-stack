/* diag_mm.c - arbitrary-shape matmul check vs a double oracle.
 * usage: diag_mm M N K [reps] */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

static int one(vc4_ctx *ctx, int M, int N, int K)
{
    float *A = malloc((size_t)M * K * 4), *B = malloc((size_t)N * K * 4);
    float *C = malloc((size_t)M * N * 4);
    double *R = malloc((size_t)M * N * sizeof(double));
    for (int i = 0; i < M * K; ++i) A[i] = frand();
    for (int i = 0; i < N * K; ++i) B[i] = frand();
    vc4_buf ba, bb, bc;
    if (vc4_alloc(ctx, (size_t)M*K*4, &ba) || vc4_alloc(ctx, (size_t)N*K*4, &bb) ||
        vc4_alloc(ctx, (size_t)M*N*4, &bc)) { printf("alloc fail\n"); return 1; }
    vc4_upload(ctx, &ba, A, (size_t)M*K*4);
    vc4_upload(ctx, &bb, B, (size_t)N*K*4);
    vc4_status st = vc4_matmul(ctx, &ba, &bb, &bc, M, N, K, VC4_LAYOUT_ROW_MAJOR);
    if (st != VC4_OK) { printf("M=%d N=%d K=%d: %s (%s)\n", M,N,K, vc4_strerror(st), vc4_last_error(ctx)); return 1; }
    vc4_download(ctx, &bc, C, (size_t)M*N*4);
    ref_gemm(A, B, R, M, N, K);
    long mm = 0; double worst = 0;
    for (int i = 0; i < M*N; ++i) {
        double e = fabs(C[i] - R[i]);
        if (e > worst) worst = e;
        if (e > 1e-4 * (fabs(R[i]) > 1 ? fabs(R[i]) : 1)) {
            if (mm < 3) printf("  mm[%d] got %.6g ref %.6g\n", i, C[i], R[i]);
            mm++;
        }
    }
    printf("M=%3d N=%3d K=%3d: mismatches=%ld/%d worst=%.3g %s\n",
           M, N, K, mm, M*N, worst, mm ? "FAIL" : "PASS");
    free(A); free(B); free(C); free(R);
    return mm ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = gpu_lock(); if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init fail\n"); return 2; }
    int rc = 0;
    if (argc >= 4) {
        rc = one(ctx, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    } else {
        rc |= one(ctx, 16,16,16);
        rc |= one(ctx, 16,16,32);
        rc |= one(ctx, 16,32,16);
        rc |= one(ctx, 32,16,16);
        rc |= one(ctx, 32,32,32);
        rc |= one(ctx, 48,16,16);
    }
    vc4_shutdown(ctx); gpu_unlock(lock);
    return rc;
}
