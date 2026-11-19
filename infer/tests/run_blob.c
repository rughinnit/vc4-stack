/* run_blob.c - load and launch an arbitrary QPU blob with explicit uniforms.
 *
 * usage: run_blob <blob.bin> <nqpu> <n> <u0> <u1> <u2> <u3> <u4> <u5> [nout]
 *   u* are written literally into each QPU's 64 B uniform block.
 *   Buffers a (u0-style) and b are filled with frand; out is 0xCD.
 *   If nout>0, the first nout floats of the *first* buffer are printed.
 * Takes the GPU lock itself.
 */
#include "vc4_int.h"
#include "util.h"

int main(int argc, char **argv)
{
    if (argc < 10) {
        fprintf(stderr, "usage: %s blob nqpu n u0..u5 [nout]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int nq = atoi(argv[2]);
    size_t n = strtoul(argv[3], NULL, 0);
    uint32_t u[6];
    for (int i = 0; i < 6; ++i)
        u[i] = (uint32_t)strtoul(argv[4 + i], NULL, 0);
    size_t nout = argc > 10 ? strtoul(argv[10], NULL, 0) : 0;

    int lock = gpu_lock();
    if (lock < 0)
        return 2;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("init: %s\n", vc4_strerror(st));
        return 2;
    }

    size_t bytes = n * 4 ? n * 4 : 4096;
    vc4_buf ba, bb, bo;
    vc4_alloc(ctx, bytes, &ba);
    vc4_alloc(ctx, bytes, &bb);
    vc4_alloc(ctx, bytes, &bo);
    float *a = vc4_host_ptr(ctx, &ba), *b = vc4_host_ptr(ctx, &bb), *o = vc4_host_ptr(ctx, &bo);
    for (size_t i = 0; i < bytes / 4; ++i) { a[i] = 0.5f + frand(); b[i] = 2.0f * frand(); }
    memset(o, 0xCD, bytes);
    vc4__l2_flush(ctx);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc(sz);
    if (fread(blob, 1, sz, f) != (size_t)sz) return 2;
    fclose(f);

    vc4_kernel *k = NULL;
    st = vc4__kernel_load_file(ctx, "blob", blob, sz, nq, &k);
    if (st != VC4_OK) { printf("load: %s\n", vc4_strerror(st)); return 2; }

    uint32_t fixed[3] = { vc4_bus_addr(ctx, &ba), vc4_bus_addr(ctx, &bb),
                          vc4_bus_addr(ctx, &bo) };
    int ewmode = getenv("EWMODE") != NULL;
    uint64_t ngroups = (n + 255) / 256;
    for (int i = 0; i < nq; ++i) {
        uint32_t *up = vc4__uni(ctx, k, i);
        for (int j = 0; j < 6; ++j)
            up[j] = (j < 3) ? fixed[j] : u[j];
        if (ewmode) {
            uint64_t cnt = (ngroups > (uint64_t)i)
                         ? (ngroups - (uint64_t)i + (uint64_t)nq - 1) / (uint64_t)nq : 0;
            up[3] = (uint32_t)cnt;
            up[4] = (uint32_t)i;
            up[5] = (uint32_t)nq;
        }
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, nq, &ns);
    printf("launch %d qpu: %s (%s) wall=%llu us\n", nq, vc4_strerror(st),
           vc4_last_error(ctx), (unsigned long long)(ns / 1000));
    printf("a=0x%08x b=0x%08x out=0x%08x\n", fixed[0], fixed[1], fixed[2]);
    if (getenv("POLL")) {
        size_t wi = n ? n - 1 : 0;
        uint32_t want = *(uint32_t *)&o[wi];
        int found = -1;
        for (int k = 0; k < 4000; ++k) {
            vc4__l2_flush(ctx);
            if (*(uint32_t *)&o[wi] != 0xCDCDCDCDu || want != 0xCDCDCDCDu) { found = k; break; }
            usleep(50);
        }
        printf("POLL out[%zu]=0x%08x after ~%d us (raw first read)\n", wi, *(uint32_t *)&o[wi], found * 50);
    }
    usleep(200000);            /* let an asynchronous VPM DMA drain */
    vc4__l2_flush(ctx);
    printf("errors after settle: 0x%08x\n", ctx->v3d[0x0F20/4]);
    if (nout) {
        printf("out:");
        for (size_t i = 0; i < nout; ++i)
            printf(" %08x(%.5g)", *(uint32_t *)&o[i], (double)o[i]);
        printf("\n");
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return st == VC4_OK ? 0 : 1;
}
