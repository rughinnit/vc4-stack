/* test_coherency.c - the arena mapping is cacheable; the engine must maintain
 * the client's D-cache around every host touch.
 *
 * The dma-buf mapping the client gets is cacheable (the daemon's /dev/mem view
 * is not).  Two things follow, and both used to be wrong:
 *
 *   1. after the QPU writes a range, a host read must invalidate it first, or
 *      the ARM serves the copy it cached before the launch;
 *   2. before the host writes part of a cache line that the QPU has written,
 *      the line must be dropped, or the clean that precedes the launch writes
 *      the stale rest-of-line back over the QPU's data.
 *
 * (2) is the one that corrupts *results*: it needs no host read at all, just a
 * host write into a recycled pool offset.  Each case below uses the engine's
 * own paths (vc4_download, vc4__write, vc4_upload), so this test fails if any
 * of them loses its maintenance.
 *
 * cc with -Isrc: vc4__write/vc4__invalidate_off are internal (vc4_int.h).
 */
#include "vc4_int.h"
#include "util.h"

static void civac(void *p, size_t n)
{
#if defined(__aarch64__)
    uintptr_t line = (uintptr_t)p & ~63ul;
    uintptr_t end = (uintptr_t)p + n;
    for (; line < end; line += 64)
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)p; (void)n;
#endif
}

static int g_fail;

/* host writes `word3`, then the QPU copies A to B; everything else in A must
 * be the QPU's 5.0, so a stale line anywhere near word 3 shows up. */
static void partial_host_write(vc4_ctx *ctx, vc4_buf *A, vc4_buf *B, size_t n,
                               int via_api, const char *what)
{
    const uint32_t nv = (uint32_t)(n / 16 / 4);
    float *p = malloc(n * 4);
    for (size_t i = 0; i < n; ++i)
        p[i] = 1.0f;
    vc4_upload(ctx, A, p, n * 4);
    vc4_fill2d(ctx, B, nv, 4, 256, 64, 3.0f);         /* flush dirty A */
    float *pa = vc4_host_ptr(ctx, A);
    volatile float sink = 0;
    for (size_t i = 0; i < n; ++i)
        sink += pa[i];                                /* cache A (clean) */
    (void)sink;
    vc4_fill2d(ctx, A, nv, 4, 256, 64, 5.0f);         /* QPU: 5.0 */
    float seven = 7.0f;
    if (via_api) {
        vc4__write(ctx, A->off + 3 * 4, &seven, 4);
    } else {
        /* what a naive engine does: write through the mapping, then clean */
        pa[3] = seven;
        vc4__invalidate_off(ctx, A->off & ~63u, 64);
    }
    vc4_copy2d(ctx, A, B, nv, 4, 256, 256, 64, 64, 4);
    float *got = p;
    vc4_download(ctx, B, got, n * 4);
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        float want = (i == 3) ? 7.0f : 5.0f;
        bad += got[i] != want;
    }
    printf("%s %-46s %3zu/%zu words wrong\n", bad ? "FAIL" : "PASS", what,
           bad, n);
    if (bad)
        ++g_fail;
    free(p);
}

/* the download path must see what the QPU wrote, even though the ARM read the
 * same buffer before the launch (the stale-clean-line case) */
static void download_after_qpu_write(vc4_ctx *ctx, vc4_buf *A, size_t n)
{
    const uint32_t nv = (uint32_t)(n / 16 / 4);
    float *p = malloc(n * 4), *got = malloc(n * 4);
    for (size_t i = 0; i < n; ++i)
        p[i] = 1.0f;
    vc4_upload(ctx, A, p, n * 4);
    civac(vc4_host_ptr(ctx, A), n * 4);
    const float *pa = vc4_host_ptr(ctx, A);
    volatile float sink = 0;
    for (size_t i = 0; i < n; ++i)
        sink += pa[i];
    (void)sink;
    vc4_fill2d(ctx, A, nv, 4, 256, 64, 5.0f);
    vc4_download(ctx, A, got, n * 4);
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i)
        bad += got[i] != 5.0f;
    printf("%s %-46s %3zu/%zu words wrong\n", bad ? "FAIL" : "PASS",
           "download after a QPU write (stale-arm-cache)", bad, n);
    if (bad)
        ++g_fail;
    free(p);
    free(got);
}

int main(void)
{
    int lock = gpu_lock();
    if (lock < 0)
        return 2;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL vc4_init: %s\n", vc4_strerror(st));
        return 1;
    }
    const size_t n = 4096;
    vc4_buf A, B;
    if (vc4_alloc(ctx, n * 4 + 64, &A) != VC4_OK ||
        vc4_alloc(ctx, n * 4 + 64, &B) != VC4_OK) {
        printf("FAIL alloc\n");
        return 1;
    }
    download_after_qpu_write(ctx, &A, n);
    partial_host_write(ctx, &A, &B, n, 1, "host word via vc4__write");
    partial_host_write(ctx, &A, &B, n, 1, "host word via vc4__write (again)");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(g_fail ? "%d FAILURES\n" : "ALL COHERENCY CASES PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
