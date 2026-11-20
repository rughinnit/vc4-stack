/* diag_blit.c - what does vc4_blit actually write for n groups? */
#include "vc4infer.h"
#include "util.h"

int main(int argc, char **argv)
{
    size_t bytes = argc > 1 ? (size_t)atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 1;
    int chain = argc > 3 ? atoi(argv[3]) : 0;
    vc4_ctx *c = NULL;
    if (vc4_init(&c, NULL, 0) != VC4_OK) {
        printf("init failed: %s\n", vc4_last_error(c));
        return 1;
    }
    size_t n = bytes / 4;
    float *a = malloc(bytes), *o = malloc(bytes);
    for (size_t i = 0; i < n; ++i)
        a[i] = (float)i;
    vc4_buf ba = { 0, 0 }, bo = { 0, 0 };
    if (vc4_alloc(c, bytes, &ba) != VC4_OK ||
        vc4_alloc(c, bytes, &bo) != VC4_OK) {
        printf("alloc %zu failed: %s\n", bytes, vc4_last_error(c));
        return 1;
    }
    printf("a.off=%u o.off=%u bus(a)=0x%08x\n", ba.off, bo.off,
           vc4_bus_addr(c, &ba));
    for (int r = 0; r < reps; ++r) {
        vc4_upload(c, &ba, a, bytes);
        for (size_t i = 0; i < n; ++i)
            o[i] = -1.0f;
        vc4_upload(c, &bo, o, bytes);
        vc4_status st = vc4_blit(c, &ba, &bo, bytes);
        if (st != VC4_OK) {
            printf("rep %d: blit failed: %s\n", r, vc4_last_error(c));
            break;
        }
        vc4_download(c, &bo, o, bytes);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i)
            if (o[i] != a[i])
                ++bad;
        printf("rep %d: %zu/%zu bad  o[0..7]=", r, bad, n);
        for (size_t i = 0; i < 8 && i < n; ++i)
            printf("%.0f ", (double)o[i]);
        printf(" o[%zu..%zu]=", n / 2, n / 2 + 3);
        for (size_t i = n / 2; i < n / 2 + 4 && i < n; ++i)
            printf("%.0f ", (double)o[i]);
        printf("\n");
        if (bad) {
            for (size_t i = 0; i < n; ++i)
                if (o[i] != a[i]) {
                    printf("  first bad [%zu]: got %.0f want %.0f\n", i,
                           (double)o[i], (double)a[i]);
                    break;
                }
            /* one line per group of 16 */
            for (size_t g = 0; g * 16 < n; ++g) {
                size_t gb = 0;
                for (size_t k = 0; k < 16 && g * 16 + k < n; ++k)
                    if (o[g * 16 + k] != a[g * 16 + k])
                        ++gb;
                if (gb)
                    printf("  group %zu (%zu bad) got0=%.0f want0=%.0f\n", g,
                           gb, (double)o[g * 16], (double)a[g * 16]);
            }
        }
    }
    if (chain) {
        /* the shape test_gated uses: several launches in one process, each
         * with freshly allocated (and leaked) buffers */
        static const size_t more[] = { 64, 256, 448, 1024, 832, 6400 };
        for (size_t s = 0; s < sizeof(more) / sizeof(more[0]); ++s) {
            size_t m = more[s], mn = m / 4;
            float *ma = malloc(m), *mo = malloc(m);
            for (size_t i = 0; i < mn; ++i)
                ma[i] = (float)i + 1000.0f * (float)s;
            vc4_buf ba2 = { 0, 0 }, bo2 = { 0, 0 };
            if (vc4_alloc(c, m, &ba2) != VC4_OK ||
                vc4_alloc(c, m, &bo2) != VC4_OK) {
                printf("%zu: alloc failed %s\n", m, vc4_last_error(c));
                break;
            }
            vc4_upload(c, &ba2, ma, m);
            for (size_t i = 0; i < mn; ++i) mo[i] = -1.0f;
            vc4_upload(c, &bo2, mo, m);
            /* did the upload land where the ARM can see it? */
            {
                float *chk = malloc(m);
                vc4_download(c, &ba2, chk, m);
                size_t u = 0;
                for (size_t i = 0; i < mn; ++i)
                    if (chk[i] != ma[i])
                        ++u;
                if (u)
                    printf("  ! source upload check: %zu/%zu bad\n", u, mn);
                free(chk);
            }
            printf("  [a.off=%u o.off=%u]", ba2.off, bo2.off);
            vc4_status st = vc4_blit(c, &ba2, &bo2, m);
            if (st != VC4_OK) {
                printf("%5zu B: blit failed: %s\n", m, vc4_last_error(c));
                continue;
            }
            vc4_download(c, &bo2, mo, m);
            size_t bad = 0, first = mn;
            for (size_t i = 0; i < mn; ++i)
                if (mo[i] != ma[i]) { ++bad; if (first == mn) first = i; }
            printf("%5zu B: %3zu/%zu bad", m, bad, mn);
            if (bad)
                printf("  first[%zu] got %.0f want %.0f", first,
                       (double)mo[first], (double)ma[first]);
            if (bad) {
                int shown = 0;
                for (size_t i = 0; i < mn && shown < 4; ++i)
                    if (mo[i] != ma[i]) {
                        printf(" | [%zu] got %.0f", i, (double)mo[i]);
                        ++shown;
                    }
            }
            printf("\n");
            free(ma); free(mo);
        }
    }
    vc4_shutdown(c);
    return 0;
}
