/* tests/util.h - test helpers.
 *
 * The CPU reference implementations in this file are the *verifier*, not a
 * compute path: they are compiled into the test binaries only and are never
 * linked into libvc4infer.
 */
#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

static uint32_t g_xs = 123456789u;
#define UTIL_UNUSED __attribute__((unused))

UTIL_UNUSED static uint32_t xs_next(void)
{
    uint32_t x = g_xs;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_xs = x;
    return x;
}

/* value in [-1,1), exactly representable in fp32 */
UTIL_UNUSED static float frand(void)
{
    int32_t v = (int32_t)((xs_next() >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

UTIL_UNUSED static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* One GPU user at a time: the whole board hangs if two workloads race.
 * With VC4INFER_DAEMON=1 every launch goes through vc4d, which takes this
 * same lock for the duration of each launch itself; holding it here for the
 * whole test would make the daemon's bounded flock() wait fail (E_BUSY). */
static int gpu_lock(void)
{
    const char *dm = getenv("VC4INFER_DAEMON");
    if (dm && *dm && strcmp(dm, "0") != 0) {
        int fd = open("/dev/null", O_RDONLY);
        return fd >= 0 ? fd : -1;
    }
    /* fs.protected_regular refuses O_CREAT on a /tmp file owned by another
       user, so fall back to O_RDONLY (flock does not need write access) */
    int fd = open("/tmp/vc4-gpu.lock", O_RDWR | O_CREAT, 0666);
    if (fd < 0)
        fd = open("/tmp/vc4-gpu.lock", O_RDONLY);
    if (fd < 0) {
        perror("open /tmp/vc4-gpu.lock");
        return -1;
    }
    if (flock(fd, LOCK_EX) != 0) {
        perror("flock");
        close(fd);
        return -1;
    }
    return fd;
}

static void gpu_unlock(int fd)
{
    if (fd >= 0) {
        flock(fd, LOCK_UN);
        close(fd);
    }
}

/* double-precision reference GEMM, row-major, used only for verification */
UTIL_UNUSED static void ref_gemm(const float *A, const float *B, double *C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

typedef struct {
    long   mismatches;
    double max_abs;
    double max_rel;
    int    printed;
} check_t;

UTIL_UNUSED static void check_elem(check_t *c, double got, double ref, double tol, int first)
{
    double err = fabs(got - ref);
    double den = fabs(ref) > 1.0 ? fabs(ref) : 1.0;
    double rel = err / den;
    if (err > c->max_abs)
        c->max_abs = err;
    if (rel > c->max_rel)
        c->max_rel = rel;
    if (err > tol * den) {
        if (first && c->printed < 8) {
            printf("    MISMATCH got %.9g ref %.9g err %.3g\n", got, ref, err);
            c->printed++;
        }
        c->mismatches++;
    }
}

/* reference conv2d, NCHW, stride 1, zero padding */
UTIL_UNUSED static void ref_conv(const float *in, int Cin, int H, int W,
                     const float *wt, int Cout, int KH, int KW,
                     const float *bias, int pad, double *out, int ho, int wo)
{
    for (int co = 0; co < Cout; ++co)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < wo; ++x) {
                double acc = bias ? (double)bias[co] : 0.0;
                for (int ci = 0; ci < Cin; ++ci)
                    for (int ky = 0; ky < KH; ++ky)
                        for (int kx = 0; kx < KW; ++kx) {
                            int iy = y - pad + ky, ix = x - pad + kx;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                continue;
                            acc += (double)in[((size_t)ci * H + iy) * W + ix] *
                                   (double)wt[(((size_t)co * Cin + ci) * KH + ky) * KW + kx];
                        }
                out[((size_t)co * ho + y) * wo + x] = acc;
            }
}

#define OK(cond, msg) do {                                              \
        if (!(cond)) {                                                  \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);       \
            return 1;                                                   \
        }                                                               \
    } while (0)

#endif /* TEST_UTIL_H */
