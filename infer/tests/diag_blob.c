/* diag_blob.c - load an arbitrary QPU blob and launch it (bisect harness). */
#include "vc4_int.h"
#include "gpu_gate.h"
#include "util.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { printf("usage: diag_blob <file.bin> [nqpu] [u0] [u1]\n"); return 2; }
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("no blob\n"); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc(n);
    if (fread(blob, 1, n, f) != (size_t)n) { printf("read\n"); return 2; }
    fclose(f);
    int nq = argc > 2 ? atoi(argv[2]) : 1;
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel_load_file(ctx, "diagblob", blob, (size_t)n, 12, &k);
    printf("load: %s (%s) words=%ld\n", vc4_strerror(st), vc4_last_error(ctx), n / 8);
    if (st != VC4_OK) return 2;

    vc4_buf b;
    if (vc4_alloc(ctx, 1 << 16, &b) != VC4_OK) return 2;
    float *h = malloc(1 << 16);
    for (int i = 0; i < (1 << 14); ++i) h[i] = (float)(i + 1);
    vc4_upload(ctx, &b, h, 1 << 16);
    uint32_t base = vc4_bus_addr(ctx, &b);
    for (int i = 0; i < nq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = base;
        u[1] = (uint32_t)i;
        u[2] = (uint32_t)i;
        u[3] = 0;
        printf("u[%d]=%08x %08x %08x %08x\n", i, u[0], u[1], u[2], u[3]);
    }
    uint64_t ns = 0;
    printf("launch nq=%d ...\n", nq);
    uint64_t t0 = now_ns();
    st = vc4__launch(ctx, k, nq, &ns);
    printf("launch: %s ns=%llu wall=%.3f ms\n", vc4_strerror(st),
           (unsigned long long)ns, (double)(now_ns() - t0) / 1e6);
    vc4__l2_flush(ctx);
    vc4_download(ctx, &b, h, 1 << 16);
    for (int i = 0; i < 40; ++i) printf("%.1f ", h[i]);
    printf("\n");
    free(h); free(blob);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf("done\n");
    return 0;
}
