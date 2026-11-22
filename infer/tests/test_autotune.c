/* test_autotune.c - CPU-only checks for the per-shape autotuner.
 *
 * Nothing here opens a device: the search space, the legality filter, the
 * record key and the cache reader/writer are all pure functions.  What is
 * proved:
 *
 *   1. every enumerated candidate is a launch the kernels can run: QPU counts
 *      within the 12-QPU ceiling and the row-block count, the auxiliary
 *      kernel's work exactly partitioned, and the field the other op uses
 *      left zero;
 *   2. the coarse list starts with the executor's default and contains no
 *      duplicate;
 *   3. the fine pass stays around the winner and adds nothing that was
 *      already measured;
 *   4. the record key separates ops, shapes and generator digests, and is
 *      stable for the same triple;
 *   5. a record round-trips, and a record whose generator digest differs, a
 *      truncated file, a foreign file and a record with an illegal parameter
 *      set are all misses (the executor then runs its default);
 *   6. the cache writer creates its directory and replaces a record
 *      atomically (the .tmp file does not survive a successful write).
 *
 * `--space` and `--refine` print the enumeration in a stable, machine
 * readable form for tests/test_autotune_space.py, which cross-checks the
 * Python mirror of the same tables against the C library.
 */
#include "vc4_int.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* The Python binding re-declares these three structs in ctypes.  Pin the
 * layout here so a field added on the C side without the mirror is a build
 * failure, not silent garbage in Engine.autotune's return value. */
_Static_assert(sizeof(vc4_tune_shape) == 48, "vc4_tune_shape layout");
_Static_assert(sizeof(vc4_tune_params) == 12, "vc4_tune_params layout");
_Static_assert(sizeof(vc4_tune_result) == 136, "vc4_tune_result layout");
_Static_assert(offsetof(vc4_tune_result, kernel) == 52,
               "vc4_tune_result.kernel offset");

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                           \
        checks++;                                                       \
        if (!(cond)) {                                                  \
            failures++;                                                 \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
            fprintf(stderr, __VA_ARGS__);                               \
            fputc('\n', stderr);                                        \
        }                                                               \
    } while (0)

static vc4_tune_shape mm(uint32_t M, uint32_t N, uint32_t K)
{
    vc4_tune_shape s;
    vc4__tune_shape_mm(&s, M, N, K);
    return s;
}

static vc4_tune_shape conv(uint32_t cin, uint32_t cout, uint32_t h, uint32_t w,
                           uint32_t k, uint32_t pad, int bias)
{
    vc4_tune_shape s;
    vc4__tune_shape_conv(&s, cin, cout, h, w, k, k, pad, 1, bias);
    return s;
}

static int same_params(const vc4_tune_params *a, const vc4_tune_params *b)
{
    return vc4__tune_params_same(a, b);
}

static int in_list(const vc4_tune_params *v, int n, const vc4_tune_params *p)
{
    for (int i = 0; i < n; ++i)
        if (same_params(&v[i], p))
            return 1;
    return 0;
}

/* ------------------------------------------------------------------ space */

static void check_space(vc4_op op, const vc4_tune_shape *s, int want_min,
                        const char *what)
{
    vc4_tune_params c[32];
    vc4_tune_params d;
    int n = vc4__tune_space(op, s, c, 32);

    CHECK(n >= want_min, "%s: %d candidates, expected at least %d", what, n,
          want_min);
    CHECK(n > 0, "%s: no candidates", what);
    for (int i = 0; i < n; ++i) {
        CHECK(vc4__tune_params_legal(op, s, &c[i]),
              "%s: candidate %d (%u/%u/%u) is not a legal launch", what, i,
              c[i].pack_q, c[i].gemm_q, c[i].im2col_q);
        for (int j = i + 1; j < n; ++j)
            CHECK(!same_params(&c[i], &c[j]),
                  "%s: candidates %d and %d are the same launch", what, i, j);
    }
    /* the default is first: the measurement needs its baseline before it can
     * compare anything to it */
    {
        int tr = (int)(op == VC4_OP_CONV2D ? s->m / 16u : s->m / 8u);
        d.pack_q = d.gemm_q = d.im2col_q = 0;
        d.gemm_q = tr < VC4_MAX_QPU ? (uint32_t)tr : VC4_MAX_QPU;
        if (op == VC4_OP_CONV2D)
            d.im2col_q = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
        else
            d.pack_q = tr < 4 ? (uint32_t)tr : 4;
        CHECK(same_params(&c[0], &d), "%s: first candidate is not the default",
              what);
    }
}

static void test_space(void)
{
    vc4_tune_shape s;
    vc4_tune_params c[32], fine[32], p;
    int n, nf;

    /* 512^3: 64 row blocks, every QPU count legal, every packer count legal */
    s = mm(512, 512, 512);
    CHECK(s.m == 512 && s.n == 512 && s.k == 512, "512^3 normalised wrong");
    check_space(VC4_OP_MATMUL, &s, 9, "matmul 512^3");
    n = vc4__tune_space(VC4_OP_MATMUL, &s, c, 32);
    CHECK(n == 9, "matmul 512^3: %d candidates, expected 9", n);

    /* 128^3: 16 row blocks, still more than 12 QPUs */
    s = mm(128, 128, 128);
    n = vc4__tune_space(VC4_OP_MATMUL, &s, c, 32);
    CHECK(n == 9, "matmul 128^3: %d candidates, expected 9", n);

    /* 64x64x16: 8 row blocks, so the 12-QPU entry cannot appear */
    s = mm(64, 64, 16);
    n = vc4__tune_space(VC4_OP_MATMUL, &s, c, 32);
    CHECK(n == 8, "matmul 64^3/16: %d candidates, expected 8", n);
    for (int i = 0; i < n; ++i)
        CHECK(c[i].gemm_q <= 8, "matmul 64 rows: gemm_q %u exceeds 8 row blocks",
              c[i].gemm_q);

    /* 16x64x16: two row blocks, no packer split of 2 into 4 */
    s = mm(16, 64, 16);
    n = vc4__tune_space(VC4_OP_MATMUL, &s, c, 32);
    CHECK(n == 3, "matmul 16 rows: %d candidates, expected 3", n);
    for (int i = 0; i < n; ++i) {
        CHECK(c[i].gemm_q <= 2, "matmul 16 rows: gemm_q %u exceeds 2",
              c[i].gemm_q);
        CHECK(c[i].pack_q <= 2, "matmul 16 rows: pack_q %u exceeds 2",
              c[i].pack_q);
    }

    /* the shape is padded to the generator tiles before it is keyed */
    s = mm(100, 100, 100);
    CHECK(s.m == 104 && s.n == 128 && s.k == 112,
          "100^3 padded to %ux%ux%u, expected 104x128x112", s.m, s.n, s.k);

    /* conv 32x32, 3x3, pad 1: the plane is padded to align16(32+2) = 48
     * floats, so 32 output rows of 48 = 1536 gemm rows = 96 row blocks */
    s = conv(16, 32, 32, 32, 3, 1, 1);
    CHECK(s.m == 1536 && s.n == 32 && s.k == 160,
          "conv shape %ux%ux%u, expected 1536x32x160 (144 + bias block)",
          s.m, s.n, s.k);
    check_space(VC4_OP_CONV2D, &s, 9, "conv 32x32x3x3");

    /* conv 6x14, 3x3, pad 1: 6 row blocks, which 4 does not divide */
    s = conv(16, 16, 6, 14, 3, 1, 1);
    CHECK(s.m == 96, "conv 6-row m = %u, expected 96", s.m);
    n = vc4__tune_space(VC4_OP_CONV2D, &s, c, 32);
    CHECK(n == 6, "conv 6-row: %d candidates, expected 6", n);
    for (int i = 0; i < n; ++i) {
        CHECK(c[i].gemm_q <= 6, "conv 6-row: gemm_q %u exceeds 6", c[i].gemm_q);
        CHECK(c[i].im2col_q == 1 || c[i].im2col_q == 2,
              "conv 6-row: im2col_q %u does not divide 6", c[i].im2col_q);
    }

    /* a conv without the bias k block is a different layer */
    {
        vc4_tune_shape nb = conv(16, 32, 32, 32, 3, 1, 0);
        vc4_tune_shape wb = conv(16, 32, 32, 32, 3, 1, 1);
        CHECK(nb.k == 144 && wb.k == 160, "bias block not in the conv k");
        CHECK(!(nb.m == wb.m && nb.k == wb.k && nb.flags == wb.flags),
              "bias and no-bias conv normalised to the same shape");
    }

    /* the fine pass: neighbours of the winner, nothing already measured */
    s = mm(512, 512, 512);
    n = vc4__tune_space(VC4_OP_MATMUL, &s, c, 32);
    p.pack_q = 4;
    p.gemm_q = 4;
    p.im2col_q = 0;
    nf = vc4__tune_refine(VC4_OP_MATMUL, &s, &p, c, n, fine, 32);
    /* the coarse pass already measured 4's immediate neighbours (3 and 6),
     * so the fine pass contributes what is left around it */
    CHECK(nf >= 1, "refine: %d candidates around gemm_q=4", nf);
    for (int i = 0; i < nf; ++i) {
        CHECK(vc4__tune_params_legal(VC4_OP_MATMUL, &s, &fine[i]),
              "refine: candidate %d is not legal", i);
        CHECK(!in_list(c, n, &fine[i]), "refine: candidate %d was already "
              "measured in the coarse pass", i);
        CHECK(fine[i].gemm_q >= 2 && fine[i].gemm_q <= 6,
              "refine: gemm_q %u is not a neighbour of 4", fine[i].gemm_q);
    }
}

static void test_legality(void)
{
    vc4_tune_shape s = mm(128, 128, 128);
    vc4_tune_params p = { 4, 12, 0 };

    CHECK(vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p), "default is legal");

    p.gemm_q = 13;                          /* above the hardware ceiling */
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p), "13 QPUs accepted");
    p.gemm_q = 12;
    p.pack_q = 3;                           /* 16 row blocks / 3 is not whole */
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p), "pack_q=3 accepted");
    p.pack_q = 4;
    p.im2col_q = 4;                         /* the other op's field must be 0 */
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p),
          "im2col_q set on a matmul accepted");
    p.im2col_q = 0;
    p.gemm_q = 0;
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p), "gemm_q=0 accepted");

    /* a two-row-block shape cannot use four QPUs or a four-way packer */
    s = mm(16, 64, 16);
    p.pack_q = 4; p.gemm_q = 2; p.im2col_q = 0;
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p),
          "pack_q=4 on 2 row blocks accepted");
    p.pack_q = 2; p.gemm_q = 3;
    CHECK(!vc4__tune_params_legal(VC4_OP_MATMUL, &s, &p),
          "gemm_q=3 on 2 row blocks accepted");

    /* conv: 6 row blocks.  Three divides 6 and is a launch the kernel can
     * run, but four is not a partition and five exceeds the registered
     * maximum, so both must be refused. */
    s = conv(16, 16, 6, 14, 3, 1, 1);
    p.pack_q = 0; p.gemm_q = 6; p.im2col_q = 3;
    CHECK(vc4__tune_params_legal(VC4_OP_CONV2D, &s, &p),
          "im2col_q=3 on 6 row blocks refused");
    p.im2col_q = 4;
    CHECK(!vc4__tune_params_legal(VC4_OP_CONV2D, &s, &p),
          "im2col_q=4 on 6 row blocks accepted");
    p.im2col_q = 5;
    CHECK(!vc4__tune_params_legal(VC4_OP_CONV2D, &s, &p),
          "im2col_q=5 accepted (above the 4 a blob is registered for)");
    p.im2col_q = 2;
    CHECK(vc4__tune_params_legal(VC4_OP_CONV2D, &s, &p),
          "im2col_q=2 on 6 row blocks refused");
}

/* ------------------------------------------------------------------- cache */

static void make_tmp(char *out, size_t n)
{
    const char *base = getenv("TMPDIR");
    if (!base || !*base)
        base = getenv("HOME");
    if (!base || !*base)
        base = ".";
    snprintf(out, n, "%s/vc4infer-autotune-test-%ld", base, (long)getpid());
}

static void test_cache(void)
{
    char base[512], path[640], path_b[640], other[704];
    char digest_a[65], digest_b[65], kernel[80], got_kernel[80];
    vc4_tune_shape s = mm(128, 128, 128);
    vc4_tune_params best, got, got_b;
    vc4_tune_trial trials[2];
    uint64_t key = 0, key_b = 0;
    double ms = 0.0, dms = 0.0;
    FILE *f;

    make_tmp(base, sizeof(base));
    mkdir(base, 0777);

    vc4__generator_digest(digest_a);
    CHECK(strlen(digest_a) == 64, "generator digest is %zu chars, expected 64",
          strlen(digest_a));
    /* a second copy, different by one character */
    snprintf(digest_b, sizeof(digest_b), "%s", digest_a);
    digest_b[0] = digest_b[0] == 'f' ? 'e' : 'f';

    vc4__tune_key(base, VC4_OP_MATMUL, &s, digest_a, &key, path, sizeof(path));
    vc4__tune_key(base, VC4_OP_MATMUL, &s, digest_a, &key_b, other,
                  sizeof(other));
    CHECK(strcmp(path, other) == 0 && key == key_b,
          "the same op/shape/digest produced two keys");
    vc4__tune_key(base, VC4_OP_MATMUL, &s, digest_b, &key_b, other,
                  sizeof(other));
    CHECK(strcmp(path, other) != 0, "a different digest produced the same key");
    vc4__tune_key(base, VC4_OP_CONV2D, &s, digest_a, &key_b, other,
                  sizeof(other));
    CHECK(strcmp(path, other) != 0, "a different op produced the same key");
    {
        vc4_tune_shape s2 = mm(128, 128, 256);
        vc4__tune_key(base, VC4_OP_MATMUL, &s2, digest_a, &key_b, other,
                      sizeof(other));
        CHECK(strcmp(path, other) != 0, "a different shape produced the same "
              "key");
    }

    /* no record yet */
    CHECK(vc4__tune_cache_read(path, VC4_OP_MATMUL, &s, digest_a, &got, NULL, 0,
                               NULL, NULL) < 0, "an absent record was a hit");

    best.pack_q = 4;
    best.gemm_q = 8;
    best.im2col_q = 0;
    trials[0].p = best;
    trials[0].ms = 1.25;
    trials[1].p.gemm_q = 12;
    trials[1].p.pack_q = 4;
    trials[1].p.im2col_q = 0;
    trials[1].ms = 2.5;
    snprintf(kernel, sizeof(kernel), "gemm10_128_128_128_q8");

    CHECK(vc4__tune_cache_write(path, VC4_OP_MATMUL, &s, digest_a, trials, 2,
                                &best, kernel, 1.25, 2.5) == VC4_OK,
          "cache write failed");
    snprintf(other, sizeof(other), "%s.tmp", path);
    CHECK(access(other, F_OK) != 0, "the .tmp file survived a successful write");

    memset(&got, 0, sizeof(got));
    CHECK(vc4__tune_cache_read(path, VC4_OP_MATMUL, &s, digest_a, &got,
                               got_kernel, sizeof(got_kernel), &ms, &dms) == 0,
          "the record just written was a miss");
    CHECK(vc4__tune_params_same(&got, &best), "record round-trip: %u/%u/%u",
          got.pack_q, got.gemm_q, got.im2col_q);
    CHECK(strcmp(got_kernel, kernel) == 0, "record kernel %s, expected %s",
          got_kernel, kernel);
    CHECK(ms > 1.24 && ms < 1.26 && dms > 2.49 && dms < 2.51,
          "record times %.6f/%.6f, expected 1.25/2.5", ms, dms);

    /* the same file, read for a different generator: a miss, and the executor
     * keeps its default instead of using a choice from another generator */
    CHECK(vc4__tune_cache_read(path, VC4_OP_MATMUL, &s, digest_b, &got, NULL, 0,
                               NULL, NULL) < 0,
          "a record from another generator was a hit");

    /* a different shape at the same path: also a miss (the shape line is
     * checked, not just the hashed file name) */
    {
        vc4_tune_shape s2 = mm(128, 128, 256);
        CHECK(vc4__tune_cache_read(path, VC4_OP_MATMUL, &s2, digest_a, &got,
                                   NULL, 0, NULL, NULL) < 0,
              "a record for another shape was a hit");
    }

    /* a record whose best is not a legal launch for this shape is a miss */
    vc4__tune_key(base, VC4_OP_MATMUL, &s, digest_b, &key_b, path_b,
                  sizeof(path_b));
    f = fopen(path_b, "w");
    CHECK(f != NULL, "cannot write %s", path_b);
    if (f) {
        fprintf(f, "%s\nop matmul\nshape %u %u %u 0 0 0 0 0 0 0 0 0\n"
                   "gen %s\nbest 4 13 0\nkernel x\n",
                "vc4infer autotune 1", s.m, s.n, s.k, digest_b);
        fclose(f);
    }
    CHECK(vc4__tune_cache_read(path_b, VC4_OP_MATMUL, &s, digest_b, &got_b,
                               NULL, 0, NULL, NULL) < 0,
          "a record with 13 QPUs was a hit");

    /* truncated and foreign files are misses, not crashes */
    f = fopen(path_b, "w");
    if (f) {
        fputs("vc4infer autotune 1\nop matmul\n", f);
        fclose(f);
    }
    CHECK(vc4__tune_cache_read(path_b, VC4_OP_MATMUL, &s, digest_b, &got_b,
                               NULL, 0, NULL, NULL) < 0,
          "a truncated record was a hit");
    f = fopen(path_b, "w");
    if (f) {
        fputs("kernel = whatever\nnonsense\n", f);
        fclose(f);
    }
    CHECK(vc4__tune_cache_read(path_b, VC4_OP_MATMUL, &s, digest_b, &got_b,
                               NULL, 0, NULL, NULL) < 0,
          "a foreign file was a hit");
    f = fopen(path_b, "w");
    if (f) {
        fprintf(f, "%s\nop matmul\nshape %u %u %u 0 0 0 0 0 0 0 0 0\ngen %s\n"
                   "best 4 8 0\nkernel k\n", "vc4infer autotune 1", s.m, s.n,
                s.k, digest_b);
        fclose(f);
    }
    CHECK(vc4__tune_cache_read(path_b, VC4_OP_MATMUL, &s, digest_b, &got_b,
                               NULL, 0, NULL, NULL) == 0,
          "a valid record with no timings was a miss");

    unlink(path);
    unlink(path_b);
    rmdir(base);
}

/* ------------------------------------------------------------------ digest */

static void test_digest(void)
{
    char a[65], b[65], c[65];
    vc4__sha256_hex("abc", 3, a);
    vc4__sha256_hex("abc", 3, b);
    vc4__sha256_hex("abd", 3, c);
    CHECK(strlen(a) == 64, "sha256 hex is %zu chars", strlen(a));
    /* FIPS-180 test vector for "abc" */
    CHECK(strcmp(a, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61"
                    "f20015ad") == 0, "sha256(\"abc\") = %s", a);
    CHECK(strcmp(a, b) == 0, "sha256 is not deterministic");
    CHECK(strcmp(a, c) != 0, "sha256 collided on a one-bit input change");
    vc4__generator_digest(a);
    vc4__generator_digest(b);
    CHECK(strlen(a) == 64, "generator digest is %zu chars", strlen(a));
    CHECK(strcmp(a, b) == 0, "the generator digest changed between calls");
}

/* ------------------------------------------------------- machine interface */

static int parse_op(const char *name, vc4_op *op)
{
    if (strcmp(name, "matmul") == 0) {
        *op = VC4_OP_MATMUL;
        return 0;
    }
    if (strcmp(name, "conv2d") == 0) {
        *op = VC4_OP_CONV2D;
        return 0;
    }
    return -1;
}

static void dump_params(const vc4_tune_params *p)
{
    printf("%u %u %u\n", p->pack_q, p->gemm_q, p->im2col_q);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--layout") == 0) {
        printf("%zu %zu %zu %zu\n", sizeof(vc4_tune_shape),
               sizeof(vc4_tune_params), sizeof(vc4_tune_result),
               offsetof(vc4_tune_result, kernel));
        return 0;
    }
    if (argc >= 14 && (strcmp(argv[1], "--space") == 0 ||
                       strcmp(argv[1], "--refine") == 0)) {
        vc4_op op;
        vc4_tune_shape s;
        vc4_tune_params c[32], fine[32], best;
        int n, nf;
        if (parse_op(argv[2], &op) != 0) {
            fprintf(stderr, "unknown op %s\n", argv[2]);
            return 2;
        }
        memset(&s, 0, sizeof(s));
        s.m = (uint32_t)strtoul(argv[3], NULL, 0);
        s.n = (uint32_t)strtoul(argv[4], NULL, 0);
        s.k = (uint32_t)strtoul(argv[5], NULL, 0);
        s.cin = (uint32_t)strtoul(argv[6], NULL, 0);
        s.cout = (uint32_t)strtoul(argv[7], NULL, 0);
        s.h = (uint32_t)strtoul(argv[8], NULL, 0);
        s.w = (uint32_t)strtoul(argv[9], NULL, 0);
        s.kh = (uint32_t)strtoul(argv[10], NULL, 0);
        s.kw = (uint32_t)strtoul(argv[11], NULL, 0);
        s.pad = (uint32_t)strtoul(argv[12], NULL, 0);
        s.stride = (uint32_t)strtoul(argv[13], NULL, 0);
        s.flags = argc > 14 ? (uint32_t)strtoul(argv[14], NULL, 0) : 0;
        if (op == VC4_OP_MATMUL) {
            vc4_tune_shape m2;
            vc4__tune_shape_mm(&m2, s.m, s.n, s.k);
            s = m2;
        } else {
            vc4_tune_shape c2;
            vc4__tune_shape_conv(&c2, s.cin, s.cout, s.h, s.w, s.kh, s.kw,
                                 s.pad, s.stride,
                                 (s.flags & VC4_TUNE_F_BIAS) != 0);
            s = c2;
        }
        n = vc4__tune_space(op, &s, c, 32);
        if (strcmp(argv[1], "--space") == 0) {
            for (int i = 0; i < n; ++i)
                dump_params(&c[i]);
            return 0;
        }
        if (argc < 18) {
            fprintf(stderr, "--refine needs pack_q gemm_q im2col_q\n");
            return 2;
        }
        best.pack_q = (uint32_t)strtoul(argv[15], NULL, 0);
        best.gemm_q = (uint32_t)strtoul(argv[16], NULL, 0);
        best.im2col_q = (uint32_t)strtoul(argv[17], NULL, 0);
        nf = vc4__tune_refine(op, &s, &best, c, n, fine, 32);
        for (int i = 0; i < nf; ++i)
            dump_params(&fine[i]);
        return 0;
    }

    test_digest();
    test_space();
    test_legality();
    test_cache();

    printf("test_autotune: %d checks, %d failure(s)\n", checks, failures);
    return failures ? 1 : 0;
}
