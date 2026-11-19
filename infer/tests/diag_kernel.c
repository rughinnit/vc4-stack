/* diag_kernel.c - launch one pool/col2im task directly, with explicit nqpu,
 * to isolate kernel hangs from the task-enumeration loop. */
#include "vc4_int.h"
#include "util.h"
#include <math.h>

static void run_fwd(vc4_ctx *ctx, int nq, int tasks)
{
    const int C = 1, H = 16, W = 16, op = 16, Ho = 8;
    size_t nx = (size_t)C * H * W, np = (size_t)C * Ho * op;
    float *x = malloc(nx * 4 + 64), *o = malloc(np * 4);
    for (size_t i = 0; i < nx; ++i) x[i] = frand();
    memset(x + nx, 0, 64);
    vc4_buf bx, bo;
    vc4_alloc(ctx, nx * 4 + 64, &bx);
    vc4_alloc(ctx, np * 4, &bo);
    vc4_upload(ctx, &bx, x, nx * 4 + 64);
    vc4_buf z = {0, 0};
    (void)z;
    char nh[8], nw[8], no[8], nc[8];
    snprintf(nh, sizeof(nh), "%d", H);
    snprintf(nw, sizeof(nw), "%d", W);
    snprintf(no, sizeof(no), "%d", op);
    snprintf(nc, sizeof(nc), "%d", C);
    char *argv[] = { (char *)"pool", (char *)"--h", nh, (char *)"--w", nw,
                     (char *)"--op", no, (char *)"--c", nc,
                     (char *)"--mode", (char *)"fwd", NULL };
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, "pool_16_16_16_c1_fwd", argv, 12, &k);
    printf("kernel: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
    if (st != VC4_OK) return;
    const uint32_t in_bus = vc4_bus_addr(ctx, &bx), out_bus = vc4_bus_addr(ctx, &bo);
    int n = nq < tasks ? nq : tasks;
    for (int i = 0; i < n; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = in_bus + 0;
        u[1] = out_bus + 0;
        u[2] = (uint32_t)i;
        u[3] = 0;
        printf("uni[%d] = %08x %08x %08x %08x\n", i, u[0], u[1], u[2], u[3]);
    }
    uint64_t ns = 0;
    printf("launch nq=%d...\n", n);
    fflush(stdout);
    st = vc4__launch(ctx, k, n, &ns);
    printf("launch: %s ns=%llu\n", vc4_strerror(st), (unsigned long long)ns);
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bo, o, np * 4);
    for (int i = 0; i < 16; ++i) printf("%.4f ", o[i]);
    printf("\n");
    free(x); free(o);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int nq = argc > 1 ? atoi(argv[1]) : 1;
    int tasks = argc > 2 ? atoi(argv[2]) : 1;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    printf("diag_kernel: fwd nq=%d tasks=%d\n", nq, tasks);
    run_fwd(ctx, nq, tasks);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf("done\n");
    return 0;
}
