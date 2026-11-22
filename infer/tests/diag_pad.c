/* diag_pad.c - minimal reproducer for pad_bwd (crop) returning zeros.
 * usage: diag_pad            (VC4INFER_DAEMON=1 for the daemon path)
 */
#include <stdio.h>
#include "gpu_gate.h"
#include <stdlib.h>
#include <string.h>

#include "vc4infer.h"

int main(void)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("init: %s\n", vc4_strerror(st));
        return 1;
    }
    printf("arena %.1f MiB, backend %s\n", vc4_arena_bytes(ctx) / 1048576.0,
           getenv("VC4INFER_DAEMON") ? "vc4d" : "direct");
    const int N = 1, C = 2, H = 3, W = 4, pt = 1, pl = 2;
    const size_t n_dy = (size_t)N * C * (H + 2 * pt) * (W + 2 * pl);
    const size_t n_dx = (size_t)N * C * H * W;
    float *dy = malloc(n_dy * 4), *dx = malloc(n_dx * 4);
    for (size_t i = 0; i < n_dy; ++i)
        dy[i] = 1.0f + (float)i;
    memset(dx, 0xCD, n_dx * 4);
    vc4_buf bdy, bdx;
    if (vc4_alloc(ctx, n_dy * 4, &bdy) != VC4_OK ||
        vc4_alloc(ctx, n_dx * 4, &bdx) != VC4_OK) {
        printf("alloc: %s\n", vc4_last_error(ctx));
        return 1;
    }
    vc4_upload(ctx, &bdy, dy, n_dy * 4);
    vc4_upload(ctx, &bdx, dx, n_dx * 4);
    st = vc4_pad_bwd(ctx, &bdy, &bdx, N, C, H, W, pt, pl);
    printf("pad_bwd: %s\n", vc4_strerror(st));
    vc4_download(ctx, &bdx, dx, n_dx * 4);
    double maxabs = 0;
    for (size_t i = 0; i < n_dx; ++i) {
        /* expected: dy at ((c)*(Hp)+(y+pt))*(Wp) + (x+pl) */
        int c = (int)(i / (size_t)(H * W));
        int y = (int)((i / (size_t)W) % (size_t)H);
        int x = (int)(i % (size_t)W);
        double want = dy[(((size_t)c * (H + 2 * pt) + (y + pt)) *
                          (W + 2 * pl)) + (x + pl)];
        double err = dx[i] - want;
        if (err < 0) err = -err;
        if (err > maxabs) maxabs = err;
        printf("  dx[%zu] = %.6f want %.6f\n", i, (double)dx[i], want);
    }
    printf("%s max|err| = %.3g\n", maxabs < 1e-6 ? "PASS" : "FAIL", maxabs);
    vc4_shutdown(ctx);
    return maxabs < 1e-6 ? 0 : 1;
}
