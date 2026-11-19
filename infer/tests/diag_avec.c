/* diag_avec.c - dump the A vector the GEMM's TMU returns (probe build).
 * usage: diag_avec M N K [kpair]
 * Uses kcache/gemm10_M_N_K_q1.bin built with --probe prime_a.
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    float *A = malloc((size_t)M * K * 4), *P = malloc((size_t)M * K * 4);
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (float)(r * 100 + k);
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * K + k) * 8 + r] = A[(size_t)(rb * 8 + r) * K + k];
    vc4_buf bp, bb, bc;
    vc4_alloc(ctx, (size_t)M * K * 4, &bp);
    vc4_alloc(ctx, (size_t)K * N * 4, &bb);
    vc4_alloc(ctx, (size_t)M * N * 4, &bc);
    vc4_upload(ctx, &bp, P, (size_t)M * K * 4);
    float *C = calloc((size_t)M * N, 4);
    vc4_upload(ctx, &bc, C, (size_t)M * N * 4);

    char name[80];
    snprintf(name, sizeof(name), "gemm10_%d_%d_%d_q1", M, N, K);
    vc4_kernel *kern = NULL;
    char *argv2[] = { (char *)"gemm", (char *)"--M", (char *)"8", (char *)"--N",
                      (char *)"64", (char *)"--K", (char *)"16",
                      (char *)"--nqpu", (char *)"1", (char *)"--probe",
                      (char *)"prime_a", NULL };
    vc4_status st = vc4__kernel(ctx, name, argv2, 1, &kern);
    if (st != VC4_OK) { printf("kernel: %s\n", vc4_strerror(st)); return 1; }
    uint32_t *u = vc4__uni(ctx, kern, 0);
    u[0] = vc4_bus_addr(ctx, &bp);
    u[1] = vc4_bus_addr(ctx, &bc);
    u[2] = vc4_bus_addr(ctx, &bb);
    u[3] = 0;
    u[4] = 32u * (uint32_t)K;
    u[5] = 0;
    u[6] = 8u * (uint32_t)N * 4u;
    u[7] = 0;
    uint64_t ns = 0;
    st = vc4__launch(ctx, kern, 1, &ns);
    printf("probe launch -> %s (%.4f ms)\n", vc4_strerror(st), ns / 1e6);
    if (st != VC4_OK) return 1;
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, (size_t)M * N * 4);
    printf("A vector lanes (expect the panel for k=0 then k=1):");
    for (int i = 0; i < 16; ++i) printf(" %4.0f", C[i]);
    printf("\n");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
