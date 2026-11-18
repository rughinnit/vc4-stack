/* diag_ew.c - minimal elementwise bring-up / arena dump.
 *
 * usage: diag_ew <op> <n> [reps]
 *   op: 0=add 1=mul 2=relu 3=sigmoid 4=silu
 * Takes the GPU lock itself (like the other test binaries).
 *
 * Prints the bus addresses and the arena contents of a, b and out before and
 * after the launch, plus the CPU reference, so it is possible to prove whether
 * the kernel read/wrote the intended addresses (QRUN_DUMP style).
 */
#include "vc4infer.h"
#include "util.h"

static void dump_words(const char *tag, const float *p, size_t n)
{
    printf("    %s:", tag);
    for (size_t i = 0; i < n && i < 16; ++i)
        printf(" %08x(%.4g)", *(const uint32_t *)&p[i], (double)p[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    int sel = argc > 1 ? atoi(argv[1]) : 2;
    static const vc4_op OPMAP[5] = { VC4_OP_ADD, VC4_OP_MUL, VC4_OP_RELU,
                                     VC4_OP_SIGMOID, VC4_OP_SILU };
    int op = (sel >= 0 && sel < 5) ? (int)OPMAP[sel] : (int)VC4_OP_RELU;
    size_t n = argc > 2 ? (size_t)strtoul(argv[2], NULL, 0) : 256;
    int reps = argc > 3 ? atoi(argv[3]) : 1;

    int lock = gpu_lock();
    if (lock < 0)
        return 2;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL vc4_init: %s\n", vc4_strerror(st));
        gpu_unlock(lock);
        return 2;
    }
    printf("arena %.1f MiB bus=0x%08x\n",
           (double)vc4_arena_bytes(ctx) / (1 << 20), vc4_bus_addr(ctx, &(vc4_buf){0, 0}));

    float *a = malloc(n * 4), *b = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        a[i] = 0.5f + frand();
        b[i] = 2.0f * frand();
    }
    vc4_buf ba, bb, bo;
    vc4_alloc(ctx, n * 4, &ba);
    vc4_alloc(ctx, n * 4, &bb);
    vc4_alloc(ctx, n * 4, &bo);
    vc4_upload(ctx, &ba, a, n * 4);
    vc4_upload(ctx, &bb, b, n * 4);
    memset(o, 0xCD, n * 4);
    vc4_upload(ctx, &bo, o, n * 4);

    printf("offsets: a=0x%06x b=0x%06x out=0x%06x (n=%zu, groups=%zu)\n",
           ba.off, bb.off, bo.off, n, (n + 255) / 256);
    printf("bus:     a=0x%08x b=0x%08x out=0x%08x\n",
           vc4_bus_addr(ctx, &ba), vc4_bus_addr(ctx, &bb), vc4_bus_addr(ctx, &bo));

    const float *pa = vc4_host_ptr(ctx, &ba);
    const float *pb = vc4_host_ptr(ctx, &bb);
    float *po = vc4_host_ptr(ctx, &bo);
    printf("before: a[0..] / b[0..] / out[0..]\n");
    dump_words("a  ", pa, n);
    dump_words("b  ", pb, n);
    dump_words("out", po, n);

    for (int r = 0; r < reps; ++r) {
        st = vc4_elementwise(ctx, (vc4_op)op, &ba, (sel == 0 || sel == 1) ? &bb : NULL,
                             &bo, n);
        printf("rep%d: status=%s (%s)\n", r, vc4_strerror(st), vc4_last_error(ctx));
        if (st != VC4_OK)
            break;
    }

    printf("after:  out[0..] (host, before download)\n");
    dump_words("out", po, n);
    vc4_download(ctx, &bo, o, n * 4);
    dump_words("out", o, n);

    long mm = 0;
    for (size_t i = 0; i < n; ++i) {
        double ref = sel == 0 ? (double)a[i] + b[i]
                   : sel == 1 ? (double)a[i] * b[i]
                   : sel == 2 ? (a[i] > 0 ? (double)a[i] : 0.0)
                   : sel == 3 ? 1.0 / (1.0 + exp(-(double)a[i]))
                   : (double)a[i] / (1.0 + exp(-(double)a[i]));
        float fref = (float)ref;
        int bad;
        if (sel <= 2)
            bad = memcmp(&o[i], &fref, 4) != 0;
        else
            bad = fabs((double)o[i] - ref) > 1e-5;
        if (bad) {
            if (mm < 8)
                printf("    mm[%zu] a=%08x b=%08x got %08x(%.9g) ref %08x(%.9g)\n",
                       i, *(uint32_t *)&a[i], *(uint32_t *)&b[i],
                       *(uint32_t *)&o[i], (double)o[i],
                       *(uint32_t *)&fref, (double)fref);
            mm++;
        }
    }
    printf("mismatches=%ld/%zu\n", mm, n);
    if (mm) {
        long hist[32] = {0};
        for (size_t i = 0; i < n; ++i) {
            size_t g = i / 256; if (g < 32) hist[g]++;
        }
        printf("    per-group mismatches:");
        for (int g = 0; g < 32 && (size_t)g * 256 < n; ++g)
            printf(" %ld", hist[g]);
        printf("\n");
    }
    printf("a unchanged: %s, b unchanged: %s\n",
           memcmp(pa, a, n * 4) ? "NO" : "yes",
           memcmp(pb, b, n * 4) ? "NO" : "yes");

    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return mm ? 1 : 0;
}
