/* diag_g6.c - run asm10's PROVEN gemm6 kernel through this engine's launcher.
 * usage: diag_g6 [N] [nqpu]
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int N = argc > 1 ? atoi(argv[1]) : 64;
    int nq = argc > 2 ? atoi(argv[2]) : 12;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    size_t np = (size_t)N * N, nb = (size_t)N * N, nc = (size_t)N * N;
    float *A = malloc(np * 4), *P = malloc(np * 4), *B = malloc(nb * 4);
    float *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    for (size_t i = 0; i < np; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();
    for (int rb = 0; rb < N / 8; ++rb)
        for (int k = 0; k < N; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * N + k) * 8 + r] = A[(size_t)(rb * 8 + r) * N + k];
    ref_gemm(A, B, R, N, N, N);
    vc4_buf bp, bb, bc;
    if (vc4_alloc(ctx, np * 4, &bp) != VC4_OK || vc4_alloc(ctx, nb * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bc) != VC4_OK) { printf("alloc failed\n"); return 1; }
    vc4_upload(ctx, &bp, P, np * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    memset(C, 0xCD, nc * 4);
    vc4_upload(ctx, &bc, C, nc * 4);

    char path[128];
    snprintf(path, sizeof(path), "/home/raspi/work/asm10/gemm6_%d.bin", N);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("no %s\n", path); return 1; }
    uint8_t blob[16384];
    size_t len = fread(blob, 1, sizeof(blob), f);
    fclose(f);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel_load_file(ctx, "gemm6", blob, len, nq, &k);
    if (st != VC4_OK) { printf("load: %s\n", vc4_strerror(st)); return 1; }
    for (int i = 0; i < nq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = (uint32_t)N;
        u[1] = vc4_bus_addr(ctx, &bc);
        u[2] = 0;
        u[3] = vc4_bus_addr(ctx, &bb);
        u[4] = vc4_bus_addr(ctx, &bp);
        u[5] = (uint32_t)i;
        u[6] = (uint32_t)nq;
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, nq, &ns);
    printf("gemm6 %d^3 nq=%d -> %s (%.4f ms)\n", N, nq, vc4_strerror(st), ns / 1e6);
    if (st != VC4_OK) return 1;
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, nc * 4);
    long bad = 0;
    for (size_t i = 0; i < nc; ++i) {
        double e = fabs(C[i] - R[i]);
        double den = fabs(R[i]) > 1.0 ? fabs(R[i]) : 1.0;
        if (e > 1e-4 * den) { if (bad < 4) printf("  i=%zu got %.6g ref %.6g\n", i, C[i], R[i]); bad++; }
    }
    printf("gemm6: %ld/%zu mismatches -> %s\n", bad, nc, bad ? "FAIL" : "PASS");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return bad ? 1 : 0;
}
