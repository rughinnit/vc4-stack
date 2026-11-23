/* A/B: launch asm3's verified pack blob and mine through the same launcher. */
#include "../src/vc4_int.h"
#include "gpu_gate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

static uint8_t *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(s);
    if (fread(b, 1, s, f) != (size_t)s) exit(1);
    fclose(f); *n = s; return b;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = open("/tmp/vc4-gpu.lock", O_RDWR | O_CREAT, 0666);
    if (lock < 0) lock = open("/tmp/vc4-gpu.lock", O_RDONLY);
    if (lock >= 0 && flock(lock, LOCK_EX) != 0) perror("flock");

    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) { printf("init failed\n"); return 1; }

    const uint32_t N = 128, tr = 8, pnq = 4, per = 2;
    for (int pass = 0; pass < 3; ++pass) {
        const char *path = pass == 0 ? "/home/raspi/work/asm3/pack_128_vp1_full.bin"
                                     : "kcache/pack.bin";
        size_t len = 0; uint8_t *blob = slurp(path, &len);
        vc4_kernel *k = NULL;
        char nm[32]; snprintf(nm, sizeof(nm), "ab%d", pass);
        st = vc4__kernel_load_file(ctx, nm, blob, len, 4, &k);
        if (st != VC4_OK) { printf("load: %s\n", vc4_strerror(st)); return 1; }
        printf("== %s (%u words)\n", path, k->code_words);
        for (int i = 0; i < (int)pnq; ++i) {
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = N;
            u[1] = (uint32_t)i * per * 64u * N;
            u[2] = ctx->gpu_bus + 0x200000;
            u[3] = 64u * N;
            u[4] = ctx->gpu_bus + 0x400000;
            u[5] = per;
            u[6] = (uint32_t)i;
            u[7] = 1024;
            u[8] = 64u * N;
            u[9] = N / 16;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, (int)pnq, &ns);
        printf("   launch -> %s  %llu us\n", vc4_strerror(st),
               (unsigned long long)(ns / 1000));
        if (st != VC4_OK) {
            printf("   err: %s\n", vc4_last_error(ctx));
            if (st == VC4_ERR_TIMEOUT) { printf("   context dead, stopping\n"); break; }
        }
        free(blob);
    }
    vc4_shutdown(ctx);
    if (lock >= 0) close(lock);
    return 0;
}
