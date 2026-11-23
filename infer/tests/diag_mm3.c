/* diag_mm3.c - one matmul shape per run, checked against the CPU oracle.
 * usage: diag_mm3 M N K [reps]
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

#define TOL 1e-4

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    if (argc < 4) { printf("usage: %s M N K [reps]\n", argv[0]); return 2; }
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    int reps = argc > 4 ? atoi(argv[4]) : 1;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    float *A = malloc(na * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    for (size_t i = 0; i < na; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();
    vc4_buf ba, bb, bc;
    vc4_alloc(ctx, na * 4, &ba); vc4_alloc(ctx, nb * 4, &bb); vc4_alloc(ctx, nc * 4, &bc);
    vc4_upload(ctx, &ba, A, na * 4); vc4_upload(ctx, &bb, B, nb * 4);
    ref_gemm(A, B, R, M, N, K);
    int rc = 0;
    for (int rep = 0; rep < reps; ++rep) {
        memset(C, 0, nc * 4);
        vc4_upload(ctx, &bc, C, nc * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        vc4_status st = vc4_matmul(ctx, &ba, &bb, &bc, (uint32_t)M, (uint32_t)N,
                                   (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL vc4_matmul(%d,%d,%d): %s (%s)\n", M, N, K,
                   vc4_strerror(st), vc4_last_error(ctx));
            rc = 1; break;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bc, C, nc * 4);
        check_t ck = { 0, 0, 0, 1 };
        for (size_t i = 0; i < nc; ++i) check_elem(&ck, C[i], R[i], TOL, i < 8);
        double pk = (double)(s1.pack_ns - s0.pack_ns) / 1e6;
        double kn = (double)(s1.kernel_ns - s0.kernel_ns) / 1e6;
        printf("  %dx%dx%d rep%d: mismatches=%ld/%zu maxabs=%.3g | pack %.4f + "
               "gemm %.4f = %.4f ms  %.2f GFLOP/s (gemm-only %.2f) wall %.3f\n",
               M, N, K, rep, ck.mismatches, nc, ck.max_abs, pk, kn, pk + kn,
               2.0 * M * N * K / ((pk + kn) * 1e6),
               kn > 0 ? 2.0 * M * N * K / (kn * 1e6) : 0.0,
               (double)(t1 - t0) / 1e6);
        if (ck.mismatches) rc = 1;
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return rc;
}
