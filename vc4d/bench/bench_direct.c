/*
 * bench_direct.c - the same 256^3 GEMM through the engine's direct root path.
 *
 * This is the baseline for the vc4d overhead comparison: libvc4infer mmaps
 * /dev/mem itself, so it must run as root, and it takes no protocol round
 * trips.  It is built against /home/raspi/work/engine-fast2 (read-only use).
 *
 * usage: bench_direct [reps] [size]
 */
#include "vc4infer.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

#define TOL 1e-4
static uint32_t xs = 123456789u;

static uint32_t xs_next(void)
{
    uint32_t x = xs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return xs = x;
}

static float frand(void)
{
    int32_t v = (int32_t)((xs_next() >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* The board-wide GPU lock, opened the way the engine's tests do it (an
 * O_CREAT on a /tmp file owned by another user trips fs.protected_regular, and
 * flock does not need write access). */
static int gpu_lock(void)
{
    int fd = open("/tmp/vc4-gpu.lock", O_RDWR | O_CREAT, 0666);
    if (fd < 0)
        fd = open("/tmp/vc4-gpu.lock", O_RDONLY);
    if (fd < 0)
        return -1;
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void ref_gemm(const float *A, const float *B, double *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    int M = argc > 2 ? atoi(argv[2]) : 256;
    int N = M, K = M;

    int lock = gpu_lock();
    if (lock < 0) {
        fprintf(stderr, "bench_direct: cannot take /tmp/vc4-gpu.lock\n");
        return 2;
    }

    vc4_ctx *ctx = NULL;
    const char *cache = getenv("VC4D_ENGINE_CACHE");
    if (!cache) cache = "/home/raspi/work/engine-fast2/kcache";
    vc4_status st = vc4_init(&ctx, cache, 0);
    if (st != VC4_OK) {
        fprintf(stderr, "bench_direct: vc4_init: %s (need root)\n", vc4_strerror(st));
        return 2;
    }
    printf("direct root path: libvc4infer, arena %.1f MiB, no daemon\n",
           (double)vc4_arena_bytes(ctx) / (1 << 20));

    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    float *A = malloc(na * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    for (size_t i = 0; i < na; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();

    vc4_buf ba, bb, bc;
    if (vc4_alloc(ctx, na * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, nb * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bc) != VC4_OK) {
        fprintf(stderr, "bench_direct: alloc failed\n");
        return 1;
    }
    vc4_upload(ctx, &ba, A, na * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    ref_gemm(A, B, R, M, N, K);

    int clean = 0;
    double best_gpu = 1e9, best_wall = 1e9, best_gf = 0;
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
            fprintf(stderr, "bench_direct: matmul: %s (%s)\n",
                    vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bc, C, nc * 4);

        long mism = 0;
        double max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < nc; ++i) {
            double err = fabs((double)C[i] - R[i]);
            double den = fabs(R[i]) > 1.0 ? fabs(R[i]) : 1.0;
            if (err > max_abs) max_abs = err;
            if (err / den > max_rel) max_rel = err / den;
            if (err > TOL * den) mism++;
        }
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall = (double)(t1 - t0) / 1e6;
        double gf = 2.0 * M * N * K / (gpu_ms * 1e6);
        if (!mism) clean++;
        if (gpu_ms < best_gpu) { best_gpu = gpu_ms; best_wall = wall; best_gf = gf; }
        printf("rep %d: mismatches=%ld/%zu maxabs=%.3g maxrel=%.2g | "
               "gpu %.3f ms  %.2f GFLOP/s | wall %.3f ms, overhead %.3f ms | "
               "cycles %llu\n",
               rep, mism, nc, max_abs, max_rel, gpu_ms, gf, wall, wall - gpu_ms,
               (unsigned long long)(s1.gpu_cycles - s0.gpu_cycles));
    }
    printf("%s %dx%dx%d: %d/%d clean, best %.3f ms gpu = %.2f GFLOP/s "
           "(wall %.3f ms)\n", clean == reps ? "PASS" : "FAIL", M, N, K,
           clean, reps, best_gpu, best_gf, best_wall);

    vc4_shutdown(ctx);
    flock(lock, LOCK_UN);
    close(lock);
    free(A); free(B); free(C); free(R);
    return clean == reps ? 0 : 1;
}
