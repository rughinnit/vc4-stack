/* raw.c - run an arbitrary GEMM blob on a CPU-built panel.
 * usage: raw <blob> <M> <N> <K> <nqpu> [amode] [bmode] [dumprows]
 *   A[r][k] = 1000*r + k   (amode 0, default)
 *   A[r][k] = (k == kd ? 1 : 0) for amode 1 (kd = 7)
 *   B[k][c] = 1 if k==c else 0    (bmode 0)
 *   B[k][c] = 1                   (bmode 1)
 *   B[k][c] = 1000*k + c          (bmode 2)
 * prints C rows 0..dumprows-1 (default 1) as integers
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    if (argc < 6) { printf("usage: raw <blob> M N K nq [amode] [bmode] [rows]\n"); return 2; }
    const char *path = argv[1];
    int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), nq = atoi(argv[5]);
    int amode = argc > 6 ? atoi(argv[6]) : 0;
    int bmode = argc > 7 ? atoi(argv[7]) : 0;
    int rows = argc > 8 ? atoi(argv[8]) : 1;
    int panoff = argc > 9 ? atoi(argv[9]) : 0;

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    vc4_buf bp, bb, bc;
    vc4_alloc(ctx, (size_t)M * K * 4, &bp);
    vc4_alloc(ctx, (size_t)K * N * 4, &bb);
    vc4_alloc(ctx, (size_t)M * N * 4, &bc);

    float *P = malloc((size_t)M * K * 4);
    float *B = malloc((size_t)K * N * 4);
    float *C = malloc((size_t)M * N * 4);
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r) {
                float v;
                if (amode == 0) v = (float)(1000 * (rb * 8 + r) + k);
                else v = (k == 7) ? 1.0f : 0.0f;
                P[((size_t)rb * K + k) * 8 + r] = v;
            }
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < N; ++c) {
            float v;
            if (bmode == 0) v = (k == c) ? 1.0f : 0.0f;
            else if (bmode == 1) v = 1.0f;
            else v = (float)(1000 * k + c);
            B[(size_t)k * N + c] = v;
        }
    memset(C, 0xCD, (size_t)M * N * 4);
    vc4_upload(ctx, &bp, P, (size_t)M * K * 4);
    vc4_upload(ctx, &bb, B, (size_t)K * N * 4);
    vc4_upload(ctx, &bc, C, (size_t)M * N * 4);

    FILE *f = fopen(path, "rb");
    if (!f) { printf("no %s\n", path); return 1; }
    static uint8_t blob[1 << 20];
    size_t len = fread(blob, 1, sizeof(blob), f);
    fclose(f);
    vc4_kernel *kern = NULL;
    vc4_status st = vc4__kernel_load_file(ctx, "raw", blob, len, nq, &kern);
    if (st != VC4_OK) { printf("load: %s\n", vc4_strerror(st)); return 1; }

    uint32_t a_rb = 32u * K, c_rb = 8u * N * 4u;
    for (int i = 0; i < nq; ++i) {
        uint32_t *u = vc4__uni(ctx, kern, i);
        u[0] = vc4_bus_addr(ctx, &bp) + (uint32_t)panoff;
        u[1] = vc4_bus_addr(ctx, &bc);
        u[2] = vc4_bus_addr(ctx, &bb);
        u[3] = (uint32_t)i * a_rb;
        u[4] = (uint32_t)nq * a_rb;
        u[5] = (uint32_t)i * c_rb;
        u[6] = (uint32_t)nq * c_rb;
        u[7] = (uint32_t)i;
    }
    printf("host view of panel: [0]=%08x [1]=%08x [64]=%08x [65]=%08x\n",
           ((uint32_t *)(ctx->arena + bp.off))[0],
           ((uint32_t *)(ctx->arena + bp.off))[1],
           ((uint32_t *)(ctx->arena + bp.off))[64],
           ((uint32_t *)(ctx->arena + bp.off))[65]);
    printf("panel %08x  B %08x  C %08x\n",
           vc4_bus_addr(ctx, &bp), vc4_bus_addr(ctx, &bb),
           vc4_bus_addr(ctx, &bc));
    uint64_t ns = 0;
    st = vc4__launch(ctx, kern, nq, &ns);
    printf("launch -> %s (%.4f ms)\n", vc4_strerror(st), ns / 1e6);
    if (st != VC4_OK) { vc4_shutdown(ctx); gpu_unlock(lock); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, (size_t)M * N * 4);

    long bad = 0, nzero = 0;
    for (int i = 0; i < M * N; ++i) {
        if (((uint32_t *)C)[i] == 0xCDCDCDCDu) bad++;
        if (((uint32_t *)C)[i] == 0) nzero++;
    }
    printf("C: %ld/%d words still 0xCDCDCDCD, %ld zero\n", bad, M * N, nzero);
    if (rows == -3) {
        const uint8_t *p = (const uint8_t *)C;
        printf("bytes:");
        for (int i = 0; i < 48; ++i) printf(" %02x", p[i]);
        printf("\n");
    } else if (rows == -2) {
        /* search the whole arena for 0x3e8 and report offsets vs C */
        const uint32_t pat = 0x3e8u;
        int hits = 0;
        uint32_t cbase = (uint32_t)((uint8_t *)C - (uint8_t *)ctx->arena) / 4;
        printf("C is at word %u of the arena; arena=%zu words\n", cbase,
               ctx->arena_bytes / 4);
        for (size_t i = 0; i + 16 <= ctx->arena_bytes / 4; ++i) {
            const uint32_t *q = (const uint32_t *)ctx->arena + i;
            if (q[0] == pat && q[1] == pat && q[15] == pat) {
                if (hits < 40)
                    printf("  16x1000 run at word %zu (C %+ld)\n", i,
                           (long)i - (long)cbase);
                hits++;
            }
        }
        printf("pattern runs: %d\n", hits);
    } else if (rows < 0) {
        /* hex map: 16 words per line */
        for (int i = 0; i < M * N; i += 16) {
            printf("%4d:", i);
            for (int j = 0; j < 16 && i + j < M * N; ++j)
                printf(" %08x", ((uint32_t *)C)[i + j]);
            printf("\n");
        }
    } else {
        for (int r = 0; r < rows && r < M; ++r) {
            printf("r%-2d:", r);
            for (int c = 0; c < N; ++c) {
                float v = C[(size_t)r * N + c];
                if (c < 24) printf(" %8.1f", v);
            }
            printf("\n");
        }
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
