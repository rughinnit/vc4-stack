/*
 * demo_gemm.c - the verified asm10 256^3 GEMM, run entirely through vc4d.
 *
 * Runs as an unprivileged user: it connects to /run/vc4d.sock, gets buffer
 * fds from the daemon, uploads the two QPU programs (packer + GEMM), launches
 * them (4 QPUs pack, 12 QPUs GEMM) and checks every element of C against a
 * double-precision CPU oracle.  The CPU never multiplies anything except in
 * the verifier.
 *
 * usage: demo_gemm [--reps N] [--size N] [--blobs DIR] [--socket PATH]
 *                  [--hold SEC] [--no-check]
 */
#define _GNU_SOURCE
#include "vc4client.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TOL 1e-4

static uint32_t xs = 123456789u;

static uint32_t xs_next(void)
{
    uint32_t x = xs;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return xs = x;
}

static float frand(void)
{
    int32_t v = (int32_t)((xs_next() >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void dsb(void)
{
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static double now_ms(void) { return (double)now_ns() / 1e6; }

static void ref_gemm(const float *A, const float *B, double *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

/* ---- in-process privilege proof ---------------------------------------- */
static void print_privs(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    printf("client privilege state:\n");
    while (f && fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "Uid:", 4) || !strncmp(line, "Gid:", 4) ||
            !strncmp(line, "CapEff:", 7) || !strncmp(line, "CapPrm:", 7) ||
            !strncmp(line, "CapBnd:", 7) || !strncmp(line, "NoNewPrivs:", 11))
            printf("  %s", line);
    }
    if (f)
        fclose(f);
    int rawmem = 0;
    DIR *d = opendir("/proc/self/fd");
    if (d) {
        struct dirent *de;
        while ((de = readdir(d))) {
            char p[300], t[512];
            snprintf(p, sizeof(p), "/proc/self/fd/%s", de->d_name);
            ssize_t n = readlink(p, t, sizeof(t) - 1);
            if (n <= 0)
                continue;
            t[n] = 0;
            if (strstr(t, "/dev/mem"))
                rawmem = 1;
        }
        closedir(d);
    }
    printf("  /dev/mem open in client: %s\n", rawmem ? "YES (FAIL)" : "no");
}

int main(int argc, char **argv)
{
    int reps = 3, M = 256, N = 256, K = 256, check = 1, hold = 0, pack_only = 0;
    const char *blobs = "blobs";
    const char *socket_path = "/run/vc4d.sock";
    const char *pidfile = NULL;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) M = N = K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blobs") && i + 1 < argc) blobs = argv[++i];
        else if (!strcmp(argv[i], "--socket") && i + 1 < argc) socket_path = argv[++i];
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc) hold = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pidfile") && i + 1 < argc) pidfile = argv[++i];
        else if (!strcmp(argv[i], "--pack-only")) pack_only = 1;
        else if (!strcmp(argv[i], "--no-check")) check = 0;
        else {
            fprintf(stderr, "usage: %s [--reps N] [--size N] [--blobs DIR] "
                            "[--socket PATH] [--hold SEC] [--pidfile PATH] "
                            "[--pack-only] [--no-check]\n", argv[0]);
            return 2;
        }
    }
    if (M % 8 || N % 64 || K % 16) {
        fprintf(stderr, "demo_gemm: size must be a multiple of 8/64/16\n");
        return 2;
    }

    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, socket_path);
    if (rc != 0) {
        fprintf(stderr, "FAIL: connect %s: %s\n", socket_path, vc4c_strerror(rc));
        return 2;
    }
    struct vc4d_hello_rsp info;
    vc4c_info(c, &info);
    printf("vc4d protocol v%u, arena %.1f MiB, V3D IDENT0=0x%08x IDENT1=0x%08x, "
           "flags=%s%s\n", info.version, (double)info.arena_bytes / (1 << 20),
           info.ident0, info.ident1,
           (info.flags & VC4D_F_RAW_SRQ) ? "raw-srq " : "",
           (info.flags & VC4D_F_ALLOWLIST) ? "allowlist " : "");
    print_privs();

    /* program names match the engine's kcache layout */
    int tr = M / 8;
    int nq = tr < (int)VC4D_MAX_QPU ? tr : (int)VC4D_MAX_QPU;
    int pnq = tr < 4 ? tr : 4;
    char ppath[512], gpath[512];
    snprintf(ppath, sizeof(ppath), "%s/pack10_%d.bin", blobs, K);
    snprintf(gpath, sizeof(gpath), "%s/gemm10_%d_%d_%d_q%d.bin", blobs, M, N, K, nq);

    vc4c_prog pack, gemm;
    rc = vc4c_upload_file(c, ppath, (unsigned)pnq, 8, &pack);
    if (rc != 0) {
        fprintf(stderr, "FAIL: upload %s: %s\n", ppath, vc4c_strerror(rc));
        return 1;
    }
    rc = vc4c_upload_file(c, gpath, (unsigned)nq, 8, &gemm);
    if (rc != 0) {
        fprintf(stderr, "FAIL: upload %s: %s\n", gpath, vc4c_strerror(rc));
        return 1;
    }
    printf("programs: pack %u words (4 QPUs), gemm %u words (%d QPUs)\n",
           pack.words, gemm.words, nq);

    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    vc4c_buf A, B, C, P;
    if ((rc = vc4c_alloc(c, na * 4, VC4D_ALLOC_ZERO, &A)) ||
        (rc = vc4c_alloc(c, nb * 4, VC4D_ALLOC_ZERO, &B)) ||
        (rc = vc4c_alloc(c, nc * 4, VC4D_ALLOC_ZERO, &C)) ||
        (rc = vc4c_alloc(c, na * 4, VC4D_ALLOC_ZERO, &P))) {
        fprintf(stderr, "FAIL: alloc: %s\n", vc4c_strerror(rc));
        return 1;
    }
    printf("buffers: A@0x%06x B@0x%06x C@0x%06x panel@0x%06x (%zu KiB each)\n",
           A.off, B.off, C.off, P.off, na * 4 / 1024);

    float *ha = malloc(na * 4), *hb = malloc(nb * 4);
    double *ref = malloc(nc * sizeof(double));
    if (!ha || !hb || !ref)
        return 1;
    for (size_t i = 0; i < na; ++i) ha[i] = frand();
    for (size_t i = 0; i < nb; ++i) hb[i] = frand();
    memcpy(A.map, ha, na * 4);
    memcpy(B.map, hb, nb * 4);
    dsb();
    vc4c_sync(c, &A, VC4D_SYNC_W2G);
    vc4c_sync(c, &B, VC4D_SYNC_W2G);
    if (check) {
        printf("computing double-precision oracle...\n");
        ref_gemm(ha, hb, ref, M, N, K);
    }

    /* uniform templates: index 0..7 per QPU, rebuilt per QPU below */
    const uint32_t unitb = 32u * K;      /* pack: bytes per 8-row unit   */
    const uint32_t a_rb = 32u * K;       /* gemm: panel row-block stride */
    const uint32_t c_rb = 8u * N * 4u;   /* gemm: C 8-row stride         */
    const uint32_t units = (uint32_t)(M / 8);

    double rtt_us = 0;
    {
        const int n = 200;
        uint64_t t0 = now_ns();
        for (int i = 0; i < n; ++i) {
            rc = vc4c_ping(c);
            if (rc) { fprintf(stderr, "FAIL ping: %s\n", vc4c_strerror(rc)); return 1; }
        }
        rtt_us = (double)(now_ns() - t0) / (double)n / 1000.0;
        printf("round trip: %.1f us/message (%d PINGs)\n", rtt_us, n);
    }

    int clean = 0, total = 0;
    double best_gpu_ms = 1e9, best_wall_ms = 1e9, best_gflops = 0;
    double sum_pack = 0, sum_gemm = 0, sum_wall = 0;

    for (int rep = 0; rep < reps; ++rep) {
        /* clear C so a stale frame can never look like a result */
        memset(C.map, 0, nc * 4);
        dsb();
        vc4c_sync(c, &C, VC4D_SYNC_W2G);

        vc4c_uni u[VC4D_MAX_QPU * 8];
        memset(u, 0, sizeof(u));

        /* ---- pack: PANEL[(rb*K+k)*8+r] = A[rb*8+r][k], grid-stride on 4 QPUs */
        for (int q = 0; q < pnq; ++q) {
            uint32_t cnt = (units - (uint32_t)q + (uint32_t)pnq - 1u) / (uint32_t)pnq;
            vc4c_uni *e = &u[q * 8];
            e[0] = VC4C_LIT(K);
            e[1] = VC4C_LIT((uint32_t)q * unitb);
            e[2] = VC4C_PTR(&A, 0, (uint32_t)na * 4);
            e[3] = VC4C_LIT(unitb);
            e[4] = VC4C_PTR(&P, 0, (uint32_t)na * 4);
            e[5] = VC4C_LIT(cnt);
            e[6] = VC4C_LIT(q);
            e[7] = VC4C_LIT(unitb * (uint32_t)pnq);
        }
        double t0 = now_ms();
        if ((rc = vc4c_launch(c, &pack, (unsigned)pnq, 8, u, 5000))) {
            fprintf(stderr, "FAIL: launch pack: %s\n", vc4c_strerror(rc));
            return 1;
        }
        vc4c_result rp;
        if ((rc = vc4c_wait(c, &rp)))
            fprintf(stderr, "WARN: wait pack: %s\n", vc4c_strerror(rc));
        if ((rc = vc4c_sync(c, &P, VC4D_SYNC_G2R)))
            fprintf(stderr, "WARN: sync panel: %s\n", vc4c_strerror(rc));

        if (pack_only) {
            /* verify PANEL[(rb*K+k)*8+r] == A[rb*8+r][k] for every element */
            const float *pa = P.map;
            long bad = 0;
            for (uint32_t rb = 0; rb < (uint32_t)M / 8; ++rb)
                for (uint32_t k = 0; k < (uint32_t)K; ++k)
                    for (uint32_t r = 0; r < 8; ++r) {
                        float want = ((const float *)A.map)[((size_t)rb * 8 + r) * K + k];
                        float got = pa[((size_t)rb * K + k) * 8 + r];
                        if (got != want && bad < 8) {
                            printf("  panel MISMATCH rb=%u k=%u r=%u got %.9g want %.9g\n",
                                   rb, k, r, got, want);
                            bad++;
                        }
                    }
            printf("pack-only: panel %s (%ld bad), pack wall %.3f ms\n",
                   bad ? "FAIL" : "OK", bad, (double)rp.wall_ns / 1e6);
            vc4c_close(c);
            return bad ? 1 : 0;
        }

        /* ---- gemm: 8x64 tiles, 12 QPUs, 8 uniforms each */
        memset(u, 0, sizeof(u));
        for (int q = 0; q < nq; ++q) {
            vc4c_uni *e = &u[q * 8];
            e[0] = VC4C_PTR(&P, 0, (uint32_t)na * 4);
            e[1] = VC4C_PTR(&C, 0, (uint32_t)nc * 4);
            e[2] = VC4C_PTR(&B, 0, (uint32_t)nb * 4);
            e[3] = VC4C_LIT((uint32_t)q * a_rb);
            e[4] = VC4C_LIT((uint32_t)nq * a_rb);
            e[5] = VC4C_LIT((uint32_t)q * c_rb);
            e[6] = VC4C_LIT((uint32_t)nq * c_rb);
            e[7] = VC4C_LIT(q);
        }
        if ((rc = vc4c_launch(c, &gemm, (unsigned)nq, 8, u, 5000))) {
            fprintf(stderr, "FAIL: launch gemm: %s\n", vc4c_strerror(rc));
            return 1;
        }
        vc4c_result rg;
        if ((rc = vc4c_wait(c, &rg)))
            fprintf(stderr, "WARN: wait gemm: %s\n", vc4c_strerror(rc));
        if ((rc = vc4c_sync(c, &C, VC4D_SYNC_G2R)))
            fprintf(stderr, "WARN: sync C: %s\n", vc4c_strerror(rc));
        double t1 = now_ms();

        const float *Cg = C.map;
        long mism = 0;
        double max_abs = 0, max_rel = 0;
        int printed = 0;
        for (size_t i = 0; i < nc; ++i) {
            double got = Cg[i], r = ref[i];
            double err = fabs(got - r);
            double den = fabs(r) > 1.0 ? fabs(r) : 1.0;
            double rel = err / den;
            if (err > max_abs) max_abs = err;
            if (rel > max_rel) max_rel = rel;
            if (check && err > TOL * den) {
                if (printed < 6) {
                    printf("  MISMATCH [%zu] (%zu,%zu) got %.9g ref %.9g err %.3g\n",
                           i, i / N, i % N, got, r, err);
                    printed++;
                }
                mism++;
            }
        }
        double wall = t1 - t0;
        double gpu_ms = (double)(rp.wall_ns + rg.wall_ns) / 1e6;
        double gf = 2.0 * M * N * K / (gpu_ms * 1e6);
        total++;
        if (!check || mism == 0)
            clean++;
        sum_pack += (double)rp.wall_ns / 1e6;
        sum_gemm += (double)rg.wall_ns / 1e6;
        sum_wall += wall;
        if (gpu_ms < best_gpu_ms) {
            best_gpu_ms = gpu_ms;
            best_wall_ms = wall;
            best_gflops = gf;
        }
        printf("rep %d: mismatches=%ld/%zu maxabs=%.3g maxrel=%.2g | "
               "gpu %.3f ms = pack %.3f + gemm %.3f  %.2f GFLOP/s | "
               "wall %.3f ms, overhead %.3f ms (%.1f%%) | cycles %u+%u\n",
               rep, check ? mism : 0, nc, max_abs, max_rel, gpu_ms,
               (double)rp.wall_ns / 1e6, (double)rg.wall_ns / 1e6, gf,
               wall, wall - gpu_ms, 100.0 * (wall - gpu_ms) / (wall > 0 ? wall : 1),
               rp.cycles, rg.cycles);
    }

    printf("%s %dx%dx%d: %d/%d clean, best %.3f ms gpu = %.2f GFLOP/s "
           "(wall %.3f ms, mean overhead %.3f ms)\n",
           clean == total ? "PASS" : "FAIL", M, N, K, clean, total,
           best_gpu_ms, best_gflops, best_wall_ms,
           sum_wall / total - (sum_pack + sum_gemm) / total);

    vc4c_stats s;
    if (vc4c_get_stats(c, &s) == 0)
        printf("daemon stats: launches=%llu failed=%llu rejects=%llu "
               "arena_used=%llu/%llu KiB programs=%u clients=%u dead=%u\n",
               (unsigned long long)s.launches, (unsigned long long)s.launches_failed,
               (unsigned long long)s.rejects,
               (unsigned long long)s.arena_used / 1024,
               (unsigned long long)s.arena_bytes / 1024,
               s.programs, s.clients, s.dead);

    if (pidfile) {
        FILE *pf = fopen(pidfile, "w");
        if (pf) {
            fprintf(pf, "%d\n", (int)getpid());
            fclose(pf);
        }
    }
    if (hold > 0) {
        printf("PID %d holding %d s for /proc inspection\n", (int)getpid(), hold);
        fflush(stdout);
        sleep(hold);
    }

    vc4c_free(c, &P);
    vc4c_free(c, &C);
    vc4c_free(c, &B);
    vc4c_free(c, &A);
    vc4c_close(c);
    free(ha); free(hb); free(ref);
    return clean == total ? 0 : 1;
}
