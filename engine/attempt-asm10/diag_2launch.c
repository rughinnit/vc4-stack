/* diag_2launch.c - pack + gemm in one process, on scratch buffers, the way
 * vc4_matmul does it.  usage: diag_2launch M N K [checkpanel]
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 16;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int chk = argc > 4 ? atoi(argv[4]) : 0;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    size_t np = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *A = malloc(np * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    for (size_t i = 0; i < np; ++i) A[i] = 1.0f;
    for (size_t i = 0; i < nb; ++i) B[i] = 1.0f;
    vc4_buf ba, bb, bc;
    vc4_alloc(ctx, np * 4, &ba);
    vc4_alloc(ctx, nb * 4, &bb);
    vc4_alloc(ctx, nc * 4, &bc);
    vc4_upload(ctx, &ba, A, np * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    memset(C, 0xCD, nc * 4);
    vc4_upload(ctx, &bc, C, nc * 4);

    vc4__scratch_reset(ctx);
    vc4_buf pb;
    float *panel = vc4__scratch_alloc(ctx, np * 4, &pb);
    printf("panel at arena off 0x%x (A off 0x%x B 0x%x C 0x%x)\n",
           pb.off, ba.off, bb.off, bc.off);
    vc4_status st = vc4__pack_run(ctx, vc4_bus_addr(ctx, &ba), (uint32_t)K,
                                  vc4_bus_addr(ctx, &pb), (uint32_t)M / 8);
    printf("pack -> %s\n", vc4_strerror(st));
    if (chk) {
        float *P = malloc(np * 4);
        vc4__l2_flush(ctx);
        vc4_download(ctx, &pb, P, np * 4);
        long bad = 0;
        for (int rb = 0; rb < M / 8; ++rb)
            for (int k = 0; k < K; ++k)
                for (int r = 0; r < 8; ++r)
                    if (P[((size_t)rb * K + k) * 8 + r] != 1.0f) bad++;
        printf("panel ones: %ld/%zu bad, P[0..7]=%.1f %.1f %.1f %.1f\n",
               bad, np, P[0], P[1], P[2], P[3]);
        free(P);
    }
    for (int rep = 0; rep < 2; ++rep) {
        st = vc4__gemm_run(ctx, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                           vc4_bus_addr(ctx, &bb), vc4_bus_addr(ctx, &bc),
                           vc4_bus_addr(ctx, &pb));
        printf("gemm rep%d -> %s\n", rep, vc4_strerror(st));
        vc4__l2_flush(ctx);
        vc4_download(ctx, &bc, C, nc * 4);
        printf("rep%d C[0][0..3] = %.3f %.3f %.3f %.3f  (expect %d)  C[8][0]=%.3f\n",
               rep, C[0], C[1], C[2], C[3], K, C[8 * N]);
    }
    for (int r = 0; r < M; ++r) {
        printf("  r%2d:", r);
        for (int c = 0; c < N; ++c) putchar(C[(size_t)r * N + c] != 0.0f ? '#' : '.');
        printf("  (%.3f %.3f)\n", C[(size_t)r * N], C[(size_t)r * N + N - 1]);
    }
    long nz = 0;
    for (size_t i = 0; i < nc; ++i) if (C[i] != 0.0f) nz++;
    printf("nonzero=%ld/%zu\n", nz, nc);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
