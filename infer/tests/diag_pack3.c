/* diag_pack3 - verify vc4__pack_run's panel against the CPU oracle. */
#include "vc4_int.h"
#include "gpu_gate.h"
#include "util.h"

static int one(vc4_ctx *ctx, uint32_t TR, uint32_t K)
{
    uint32_t rows = TR * 16, n = rows * K;
    float *A = malloc(n * 4), *Pref = malloc(n * 4);
    if (getenv("SEQ"))
        for (uint32_t i = 0; i < n; ++i) A[i] = (float)i;
    else
        for (uint32_t i = 0; i < n; ++i) A[i] = frand();
    for (uint32_t rb = 0; rb < TR; ++rb)
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t r = 0; r < 16; ++r)
                Pref[(rb * K + k) * 16 + r] = A[(rb * 16 + r) * K + k];

    vc4_buf ba, bp;
    vc4_alloc(ctx, n * 4, &ba);
    vc4_alloc(ctx, n * 4, &bp);
    vc4_upload(ctx, &ba, A, n * 4);
    float *P = vc4_host_ptr(ctx, &bp);
    memset(P, 0xCD, n * 4);
    vc4__l2_flush(ctx);

    printf("  TR=%u K=%u src off=0x%x bus=0x%08x dst off=0x%x bus=0x%08x A[0]=%08x\n",
           TR, K, ba.off, vc4_bus_addr(ctx,&ba), bp.off, vc4_bus_addr(ctx,&bp),
           *(uint32_t *)&A[0]);
    vc4_status st;
    const char *blob = getenv("ASM3BLOB");
    if (blob) {
        FILE *f = fopen(blob, "rb");
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *b = malloc(sz); if (fread(b,1,sz,f)!=(size_t)sz) return 1; fclose(f);
        vc4_kernel *kk = NULL;
        st = vc4__kernel_load_file(ctx, "asm3pack", b, sz, 4, &kk);
        if (st == VC4_OK) {
            uint32_t *u = vc4__uni(ctx, kk, 0);
            u[0]=K; u[1]=0; u[2]=vc4_bus_addr(ctx,&ba); u[3]=64u*K;
            u[4]=vc4_bus_addr(ctx,&bp); u[5]=TR; u[6]=0;
            uint64_t ns=0; st = vc4__launch(ctx, kk, 1, &ns);
        }
    } else {
        st = vc4__pack_run(ctx, vc4_bus_addr(ctx, &ba), K, 64u * K,
                                  vc4_bus_addr(ctx, &bp), 64u * K, TR);
    }
    {
        int waited = -1;
        for (int t = 0; t < 4000; ++t) {
            vc4__l2_flush(ctx);
            if (*(uint32_t *)&P[0] != 0xCDCDCDCDu) { waited = t; break; }
            usleep(50);
        }
        printf("  panel visible after ~%d us\n", waited * 50);
    }
    vc4__l2_flush(ctx);
    printf("  P[0..7]:"); for (int q=0;q<8;q++) printf(" %.1f", P[q]); printf("\n");
    printf("  ref[0..7]:"); for (int q=0;q<8;q++) printf(" %.1f", Pref[q]); printf("\n");
    long bad = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (memcmp(&P[i], &Pref[i], 4) != 0) {
            if (bad < 2) printf("    P[%u] got %08x ref %08x\n", i,
                                *(uint32_t *)&P[i], *(uint32_t *)&Pref[i]);
            bad++;
        }
    printf("pack TR=%u K=%u (pnq=%d): %s %ld/%u\n", TR, K,
           (int)(TR % 4 == 0 ? 4 : TR % 2 == 0 ? 2 : 1),
           st == VC4_OK ? vc4_strerror(st) : vc4_strerror(st), bad, n);
    free(A); free(Pref);
    return bad ? 1 : 0;
}

int main(void)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = gpu_lock(); if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init fail\n"); return 2; }
    int rc = 0;
    rc |= one(ctx, 1, 16); rc |= one(ctx, 2, 16); rc |= one(ctx, 3, 16);
    rc |= one(ctx, 4, 16); rc |= one(ctx, 8, 16); rc |= one(ctx, 4, 32);
    vc4_shutdown(ctx); gpu_unlock(lock);
    return rc;
}
