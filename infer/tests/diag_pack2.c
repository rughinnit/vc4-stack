/* diag_pack2: (1) does my pack blob produce the right panel bytes?
 *             (2) does the verified asm3 GEMM blob run under my launcher?
 * Debug tool only; builds against the internal header.
 */
#include "../src/vc4_int.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

static uint8_t *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)s);
    if (fread(b, 1, (size_t)s, f) != (size_t)s) exit(1);
    fclose(f); *n = (size_t)s; return b;
}

static uint32_t xs = 123456789u;
static float fr(void)
{
    xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5;
    int32_t v = (int32_t)((xs >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

int main(void)
{
    int lock = open("/tmp/vc4-gpu.lock", O_RDWR | O_CREAT, 0666);
    if (lock < 0) lock = open("/tmp/vc4-gpu.lock", O_RDONLY);
    if (lock >= 0 && flock(lock, LOCK_EX) != 0) perror("flock");

    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 1; }
    printf("arena bus=0x%08x phys=0x%08x\n", ctx->gpu_bus, ctx->gpu_phys);

    const uint32_t N = 128, nn = N * N, tr = N / 16;
    float *A = vc4__scratch_alloc(ctx, nn * 4, NULL);
    float *B = vc4__scratch_alloc(ctx, nn * 4, NULL);
    float *C = vc4__scratch_alloc(ctx, nn * 4, NULL);
    vc4_buf pbuf;
    float *P = vc4__scratch_alloc(ctx, nn * 4, &pbuf);
    float *Pref = malloc(nn * 4);
    float *Cref = malloc(nn * sizeof(double));
    double *Cd = malloc(nn * sizeof(double));
    for (uint32_t i = 0; i < nn; ++i) { A[i] = fr(); B[i] = fr(); C[i] = 0; }
    for (uint32_t rb = 0; rb < tr; ++rb)
        for (uint32_t k = 0; k < N; ++k)
            for (uint32_t r = 0; r < 16; ++r)
                Pref[(rb * N + k) * 16 + r] = A[(rb * 16 + r) * N + k];

    const uint32_t A_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)A - ctx->arena);
    const uint32_t B_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)B - ctx->arena);
    const uint32_t C_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)C - ctx->arena);
    const uint32_t P_bus = ctx->gpu_bus + pbuf.off;

    /* ---- (1) my pack blob, pnq = 1, 2, 4 ------------------------------- */
    size_t len = 0;
    uint8_t *myblob = slurp("kcache/pack.bin", &len);
    vc4_kernel *kp = NULL;
    if (vc4__kernel_load_file(ctx, "mypack", myblob, len, 4, &kp) != VC4_OK) return 1;
    for (int pnq = 1; pnq <= 4; pnq *= 2) {
        memset(P, 0, nn * 4);
        for (int i = 0; i < pnq; ++i) {
            uint32_t *u = vc4__uni(ctx, kp, i);
            u[0] = N; u[1] = (uint32_t)i * (tr / pnq) * 64u * N;
            u[2] = A_bus; u[3] = 64u * N; u[4] = P_bus;
            u[5] = tr / pnq; u[6] = (uint32_t)i;
            u[7] = 1024; u[8] = 64u * N; u[9] = N / 16;
        }
        uint64_t ns = 0;
        vc4_status st = vc4__launch(ctx, kp, pnq, &ns);
        long bad = 0;
        for (uint32_t i = 0; i < nn; ++i)
            if (memcmp(&P[i], &Pref[i], 4) != 0) bad++;
        printf("pack pnq=%d -> %s %llu us  panel mismatches=%ld/%u errs=%s\n",
               pnq, vc4_strerror(st), (unsigned long long)(ns / 1000), bad, nn,
               st ? vc4_last_error(ctx) : "-");
        if (st == VC4_ERR_TIMEOUT) { vc4_shutdown(ctx); return 1; }
    }

    /* ---- (2) asm3's verified GEMM on a correct panel -------------------- */
    len = 0;
    uint8_t *gblob = slurp("/home/raspi/work/asm3/gemm2_128_d3_v1.bin", &len);
    vc4_kernel *kg = NULL;
    if (vc4__kernel_load_file(ctx, "asm3gemm", gblob, len, 12, &kg) != VC4_OK) return 1;
    memcpy(P, Pref, nn * 4);
    for (uint32_t i = 0; i < nn; ++i) C[i] = 0;
    const int nq = 12;
    for (int i = 0; i < nq; ++i) {
        uint32_t *u = vc4__uni(ctx, kg, i);
        u[0] = N; u[1] = C_bus; u[2] = A_bus; u[3] = B_bus; u[4] = P_bus;
        u[5] = (uint32_t)i; u[6] = (uint32_t)nq;
    }
    uint64_t ns = 0;
    vc4_status st = vc4__launch(ctx, kg, nq, &ns);
    printf("asm3 gemm -> %s %llu us %s\n", vc4_strerror(st),
           (unsigned long long)(ns / 1000), st ? vc4_last_error(ctx) : "-");
    if (st == VC4_OK) {
        for (uint32_t i = 0; i < N; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                double acc = 0;
                for (uint32_t k = 0; k < N; ++k)
                    acc += (double)A[i * N + k] * (double)B[k * N + j];
                Cd[i * N + j] = acc;
            }
        long mm = 0; double mx = 0;
        for (uint32_t i = 0; i < nn; ++i) {
            double e = Cd[i] - (double)C[i];
            if (e < 0) e = -e;
            if (e > mx) mx = e;
            if (e > 1e-4 * (Cd[i] > 1 ? Cd[i] : 1)) mm++;
        }
        printf("asm3 gemm result: mismatches=%ld/%u maxabs=%g\n", mm, nn, mx);
    }
    vc4_shutdown(ctx);
    if (lock >= 0) close(lock);
    return 0;
}
