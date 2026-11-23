/* test_softmax_probe.c - bring-up aid for the fused softmax kernel.
 * Runs the kernel in probe mode (max / exp / sum / p) and compares each
 * intermediate against a CPU oracle. */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include <math.h>

int main(void)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { gpu_unlock(lock); return 2; }
    const uint32_t R = 8, C = 16;
    float z[8 * 16], out[8 * 16], ref[8 * 16];
    int32_t lab[8];
    for (uint32_t r = 0; r < R; ++r) {
        lab[r] = (int32_t)(r % 3);
        for (uint32_t c = 0; c < C; ++c)
            z[r * C + c] = frand() * 3.0f;
    }
    vc4_buf bz, bl, bd, bp;
    vc4_alloc(ctx, sizeof(z) + 64, &bz);
    vc4_alloc(ctx, sizeof(lab) + 64, &bl);
    vc4_alloc(ctx, sizeof(out) + 64, &bd);
    vc4_alloc(ctx, 12 * 64, &bp);
    vc4_upload(ctx, &bz, z, sizeof(z));
    vc4_upload(ctx, &bl, lab, sizeof(lab));
    const char *modes[] = { "cor" };
    int rc = 0;
    for (int mi = 0; mi < 1; ++mi) {
        setenv("VC4_SOFTMAX_PROBE", modes[mi], 1);
        float zero[8 * 16] = {0};
        vc4_upload(ctx, &bd, zero, sizeof(zero));
        vc4_upload(ctx, &bp, zero, 12 * 64);
        vc4_status st = vc4_softmax_ce(ctx, &bz, &bl, &bd, &bp, R, 1.0f / R);
        if (st != VC4_OK) { printf("FAIL %s: %s\n", modes[mi], vc4_strerror(st)); rc = 1; continue; }
        vc4_download(ctx, &bd, out, sizeof(out));
        for (uint32_t r = 0; r < R; ++r) {
            double m = -1e30, s = 0;
            for (uint32_t c = 0; c < C; ++c) m = fmax(m, z[r * C + c]);
            for (uint32_t c = 0; c < C; ++c) { double e = exp(z[r*C+c]-m); s += e; }
            for (uint32_t c = 0; c < C; ++c) {
                double e = exp(z[r*C+c]-m);
                const char *md = modes[mi];
                if (strcmp(md, "cor") == 0) {
                    int am = 0; double mm2 = -1e30;
                    for (int q = 0; q < 16; ++q) if (z[r*C+q] > mm2) { mm2 = z[r*C+q]; am = q; }
                    ref[r*C+c] = (double)(am == lab[r]);
                } else if (strcmp(md, "l2") == 0) {
                    ref[r*C+c] = log2(s);
                } else if (strcmp(md, "zy") == 0) {
                    ref[r*C+c] = z[r*C + lab[r]];
                } else if (!strncmp(md, "red", 3)) {
                    int k = atoi(md + 3);
                    double acc = 0;
                    int off[4] = {0,0,0,0};
                    int nset = 0;
                    /* offsets covered after k steps (right rotates 8,4,2,1) */
                    int offs[16]; int no = 1; offs[0] = 0;
                    int ss[4] = {8,4,2,1};
                    for (int st = 0; st < k; ++st) {
                        int add = ss[st]; int nn = no;
                        for (int q = 0; q < nn; ++q) { int v = offs[q] - add; v &= 15; int dup=0; for (int q2=0;q2<no;q2++) if (offs[q2]==v) dup=1; if (!dup) offs[no++]=v; }
                    }
                    for (int q = 0; q < no; ++q) acc += exp(z[r*C + ((c + offs[q]) & 15)] - m);
                    (void)off; (void)nset;
                    ref[r*C+c] = acc;
                } else {
                    ref[r*C+c] = (double)(strcmp(md, "max")==0 ? m :
                                  strcmp(md, "exp")==0 ? e :
                                  strcmp(md, "sum")==0 ? s :
                                  strcmp(md, "raw")==0 ? z[r*C+c] : e/s);
                }
            }
        }
        double ma = 0; long mm = 0;
        for (uint32_t i = 0; i < R*C; ++i) {
            double e = fabs(out[i] - ref[i]);
            if (e > ma) ma = e;
            if (e > 1e-4 * (fabs(ref[i]) > 1 ? fabs(ref[i]) : 1)) mm++;
        }
        printf("%s probe %-4s: max_abs=%.3g mismatches=%ld  sample got=%.6g ref=%.6g\n",
               mm==0?"PASS":"FAIL", modes[mi], ma, mm, out[0], ref[0]);
        if (mm) {
            for (uint32_t r = 0; r < 2; ++r) {
                printf("  row %u got:", r);
                for (uint32_t c = 0; c < C; ++c) printf(" %.4g", out[r*C+c]);
                printf("\n  row %u ref:", r);
                for (uint32_t c = 0; c < C; ++c) printf(" %.4g", ref[r*C+c]);
                printf("\n");
            }
        }
        rc |= mm != 0;
    }
    unsetenv("VC4_SOFTMAX_PROBE");
    vc4_shutdown(ctx); gpu_unlock(lock);
    return rc;
}
