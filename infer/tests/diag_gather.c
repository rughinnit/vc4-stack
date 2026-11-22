/* diag_gather.c - identity table, print every element. */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    size_t n = argc > 1 ? (size_t)atoi(argv[1]) : 16;
    float fill = argc > 2 ? (float)atof(argv[2]) : 0.0f;
    vc4_ctx *c = NULL;
    if (vc4_init(&c, NULL, 0) != VC4_OK) {
        printf("init: %s\n", vc4_last_error(c));
        return 1;
    }
    size_t nsrc = 2 * n + 16;
    float *src = malloc(nsrc * 4), *o = calloc(n, 4);
    int32_t *tab = malloc(n * 4);
    for (size_t i = 0; i < nsrc; ++i)
        src[i] = (float)(i + 1) * 0.25f;
    vc4_buf bs = { 0, 0 }, bt = { 0, 0 }, bo = { 0, 0 };
    if (vc4_alloc(c, nsrc * 4, &bs) != VC4_OK ||
        vc4_alloc(c, n * 4, &bt) != VC4_OK ||
        vc4_alloc(c, n * 4, &bo) != VC4_OK) {
        printf("alloc: %s\n", vc4_last_error(c));
        return 1;
    }
    int mode = argc > 3 ? atoi(argv[3]) : 0;
    for (size_t i = 0; i < n; ++i) {
        if (mode == 0)
            tab[i] = (int32_t)(bs.off + (uint32_t)(i * 4));
        else if (mode == 1)
            tab[i] = (int32_t)bs.off;             /* every entry -> src[0] */
        else if (mode == 2)
            tab[i] = (int32_t)(bs.off + 64);      /* every entry -> src[16] */
        else
            tab[i] = -1;                          /* fill */
    }
    vc4_upload(c, &bs, src, nsrc * 4);
    vc4_upload(c, &bt, tab, n * 4);
    vc4_upload(c, &bo, o, n * 4);
    printf("src.off=%u tab.off=%u out.off=%u bus(src)=0x%08x gpu_bus?\n",
           bs.off, bt.off, bo.off, vc4_bus_addr(c, &bs));
    /* is the table in DRAM where the GPU can see it?  Copy it with the
     * verified blit (copy2d) and compare the bits. */
    {
        vc4_buf bc = { 0, 0 };
        float *chk = malloc(n * 4);
        if (vc4_alloc(c, n * 4, &bc) == VC4_OK) {
            vc4_status cs = vc4_blit(c, &bt, &bc, n * 4);
            vc4_download(c, &bc, chk, n * 4);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i)
                if (memcmp(&chk[i], &tab[i], 4) != 0)
                    ++bad;
            printf("table visible to the GPU via blit: %zu/%zu bad"
                   " (st=%d), chk[0] bits=0x%08x\n", bad, n, (int)cs,
                   ((const unsigned *)chk)[0]);
            free(chk);
        }
    }
    vc4_status st = vc4_gather(c, &bt, &bo, n, fill);
    printf("status %d %s\n", (int)st, vc4_last_error(c));
    vc4_download(c, &bo, o, n * 4);
    for (size_t i = 0; i < n && i < 48; ++i)
        printf("  [%zu] got %.7g  (table %d)  want %.7g\n", i,
               (double)o[i], tab[i], (double)src[i]);
    vc4_shutdown(c);
    return 0;
}
