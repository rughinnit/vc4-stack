/* bench_matmul.c - wall-clock and GPU-time GFLOP/s for one square matmul.
 *
 * Runs on either backend, so the same binary measures the direct root path
 * (VC4INFER_DAEMON unset) and the unprivileged daemon path
 * (VC4INFER_DAEMON=1).  Both numbers are printed:
 *
 *   gpu GFLOP/s   from the engine's pack+kernel timers (the daemon's
 *                 kick->idle wall time in remote mode, the ARM poll loop in
 *                 direct mode) - the QPU work itself;
 *   wall GFLOP/s  2*M*N*K / (t1-t0) around vc4_matmul() - what the caller
 *                 waits for, including protocol and cache maintenance.
 *
 * The overhead of the daemon path is the difference between the two wall
 * numbers; the GPU numbers should be identical to within counter noise.
 *
 * usage: bench_matmul [size] [reps]
 */
#include "vc4infer.h"
#include "util.h"

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int N = M, K = M;
    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;

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

    const char *dm = getenv("VC4INFER_DAEMON");
    int daemon = dm && *dm && strcmp(dm, "0") != 0;
    printf("backend=%s arena=%.1f MiB  %dx%dx%d reps=%d\n",
           daemon ? "vc4d (unprivileged)" : "direct (/dev/mem, root)",
           (double)vc4_arena_bytes(ctx) / (1 << 20), M, N, K, reps);

    float *A = malloc(na * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    vc4_buf ba, bb, bc;
    if (!A || !B || !C || !R ||
        vc4_alloc(ctx, na * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, nb * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bc) != VC4_OK) {
        printf("FAIL: setup\n");
        vc4_shutdown(ctx);
        gpu_unlock(lock);
        return 2;
    }
    for (size_t i = 0; i < na; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();
    vc4_upload(ctx, &ba, A, na * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    ref_gemm(A, B, R, M, N, K);

    double best_wall_ms = 1e30, best_gpu_ms = 1e30, best_gpu_gf = 0;
    long mismatches = 0;
    for (int rep = 0; rep < reps; ++rep) {
        memset(C, 0, nc * 4);
        vc4_upload(ctx, &bc, C, nc * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        st = vc4_matmul(ctx, &ba, &bb, &bc, (uint32_t)M, (uint32_t)N,
                        (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: vc4_matmul: %s (%s)\n", vc4_strerror(st),
                   vc4_last_error(ctx));
            vc4_shutdown(ctx);
            gpu_unlock(lock);
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bc, C, nc * 4);

        check_t ck = { 0, 0, 0, 0 };
        for (size_t i = 0; i < nc; ++i)
            check_elem(&ck, C[i], R[i], 1e-4, i < 4);
        mismatches += ck.mismatches;

        double wall_ms = (double)(t1 - t0) / 1e6;
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double flops = 2.0 * M * N * K;
        if (wall_ms < best_wall_ms)
            best_wall_ms = wall_ms;
        if (gpu_ms < best_gpu_ms)
            best_gpu_ms = gpu_ms;
        double ggf = flops / (gpu_ms * 1e6);
        if (ggf > best_gpu_gf)
            best_gpu_gf = ggf;
        printf("  rep%d: mismatches=%ld/%zu  gpu %.3f ms  wall %.3f ms"
               "  (gpu %.2f, wall %.2f GFLOP/s)\n",
               rep, ck.mismatches, nc, gpu_ms, wall_ms,
               ggf, flops / (wall_ms * 1e6));
    }
    double flops = 2.0 * M * N * K;
    printf("RESULT %dx%dx%d %s: mismatches=%ld  gpu %.3f ms = %.2f GFLOP/s"
           "  wall %.3f ms = %.2f GFLOP/s\n",
           M, N, K, daemon ? "vc4d" : "direct", mismatches,
           best_gpu_ms, best_gpu_gf, best_wall_ms, flops / (best_wall_ms * 1e6));

    free(A); free(B); free(C); free(R);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return mismatches ? 1 : 0;
}
