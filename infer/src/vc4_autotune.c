/*
 * vc4_autotune.c - per-shape kernel autotuning with a persistent cache.
 *
 * Two of the launch choices the generated kernels leave to the host are free
 * per shape: how many QPUs run the GEMM, and how many run the auxiliary packer
 * that feeds it (the A-panel packer for a matmul, the im2col gather for a
 * conv).  Neither has a single best answer.  A 512x512x512 matmul has 64 row
 * blocks and wants all 12 QPUs; a shape with 16 row blocks leaves 4 QPUs idle
 * at 12; a conv whose output has fewer row blocks than QPUs pays for the idle
 * ones, and one with so little packing work that four QPUs spend more on the
 * launch than the split saves.
 *
 * So this file does what a compiler's kernel selector does: enumerate the
 * legal choices for one (op, shape), measure them on the device with one
 * discarded warm-up run followed by a small number of timed repetitions, and
 * record the winner under the kernel cache directory so later runs read a
 * decision instead of making one.
 *
 * WHAT IS SEARCHED, AND WHAT IS NOT
 *
 * The generator also parameterises the tile shape and the vectors per group,
 * but those are part of each kernel's hand-verified instruction schedule (the
 * asm10 8x64 tile, the asm3 16x16 tile, build_ew's 16-vector group behind the
 * verified 256-float store).  Changing them produces a different program with
 * no element-level verification on this board, and a wrong QPU program is not
 * a recoverable error here.  The QPU counts are the part of the space that
 * changes loop bounds and uniform strides only, so every candidate runs the
 * same verified body over the same data.  The tuner still checks, after the
 * search, that its choice computes the same output as the executor's default
 * before it caches it: an unverified schedule that changes a number is exactly
 * the failure a cache would make permanent.
 *
 * MEASUREMENT
 *
 * One warm-up run of a candidate (which also generates and uploads its blob)
 * is discarded, then `reps` runs are timed with CLOCK_MONOTONIC and the median
 * recorded.  The median, not the mean: an ARM scheduler slice or a vc4d
 * round-trip outlier is common and the distribution is not normal.
 *
 * The search is a coarse coordinate sweep (the default, then every legal GEMM
 * QPU count with the default packer, then every legal packer with the default
 * GEMM) followed by a fine pass over the neighbours of the coarse winner.  It
 * is bounded twice, by a candidate cap and by a wall-clock budget that
 * includes blob generation, so a first run of a shape cannot take unbounded
 * time.  The blob cache means those generations are paid once per generator.
 *
 * CACHE
 *
 * One file per (op, shape, generator digest), <cache>/autotune/<key>.tune,
 * where the key is the first 8 bytes of sha256 over the op name, the canonical
 * shape and the generator digest.  The digest is the one the blob cache
 * already stamps into every .bin: sha256(kgen.py, kops.py beside it, the ABI
 * tag), so a generator change invalidates tuned choices through the existing
 * mechanism rather than a second one.  Inside the file each measured parameter
 * set is one `trial` record with its median time, and `best` names the chosen
 * set.  A stale, truncated or foreign file is a miss: the op runs with its
 * default and the next autotune call rewrites the record.
 *
 * OFF BY DEFAULT
 *
 * Nothing here runs unless VC4INFER_AUTOTUNE=1 or vc4_autotune() was called on
 * this context, so a graph's existing timings stay reproducible.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TUNE_MAGIC           "vc4infer autotune 1"
#define TUNE_KEY_DOMAIN      "vc4infer-autotune-1"
#define TUNE_DEFAULT_CAND    14
#define TUNE_DEFAULT_BUDGET  30000u
#define TUNE_DEFAULT_REPS    5
#define TUNE_MAX_REPS        9
#define TUNE_MAX_TOTAL       18

/* Measured descending: on any shape with enough row blocks the widest launch
 * wins, so the first candidate is usually the answer and the budget is spent
 * confirming it rather than walking up to it. */
static const uint32_t GEMM_LADDER[] = { 12, 8, 6, 4, 3, 2, 1 };
static const uint32_t AUX_LADDER[]  = { 4, 2, 1 };

/* ------------------------------------------------------------------ helpers */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static uint32_t env_u32(const char *name, uint32_t dflt)
{
    const char *e = getenv(name);
    long v;
    if (!e || !*e)
        return dflt;
    v = strtol(e, NULL, 0);
    return v > 0 ? (uint32_t)v : dflt;
}

int vc4_autotune_enabled(vc4_ctx *ctx)
{
    const char *e;
    if (ctx && ctx->autotune)
        return 1;
    e = getenv("VC4INFER_AUTOTUNE");
    return e && *e && strcmp(e, "0") != 0;
}

static void shape_str(const vc4_tune_shape *s, char *out, size_t n)
{
    snprintf(out, n, "%u %u %u %u %u %u %u %u %u %u %u %u",
             s->m, s->n, s->k, s->cin, s->cout, s->h, s->w, s->kh, s->kw,
             s->pad, s->stride, s->flags);
}

/* The one place a shape is normalised, so the executor and the tuner cannot
 * key the same layer differently (vc4_matmul pads to the generator's tiles
 * before it launches, and a conv's GEMM shape is derived, not given). */
void vc4__tune_shape_mm(vc4_tune_shape *s, uint32_t M, uint32_t N, uint32_t K)
{
    memset(s, 0, sizeof(*s));
    s->m = vc4__align8(M);
    s->n = vc4__align64(N);
    s->k = vc4__align16(K);
}

void vc4__tune_shape_conv(vc4_tune_shape *s, uint32_t Cin, uint32_t Cout,
                          uint32_t H, uint32_t W, uint32_t KH, uint32_t KW,
                          uint32_t pad, uint32_t stride, int bias)
{
    uint32_t ho, wp;
    memset(s, 0, sizeof(*s));
    s->cin = Cin;
    s->cout = Cout;
    s->h = H;
    s->w = W;
    s->kh = KH;
    s->kw = KW;
    s->pad = pad;
    s->stride = stride;
    s->flags = bias ? VC4_TUNE_F_BIAS : 0u;
    ho = (H + 2u * pad - KH) / (stride ? stride : 1u) + 1u;
    wp = vc4__align16(W + 2u * pad);
    s->m = ho * wp;
    s->n = Cout;
    s->k = vc4__align16(Cin * KH * KW) +
           (bias ? 16u : 0u);          /* the folded-bias k block */
}

static uint32_t aux_of(vc4_op op, const vc4_tune_params *p)
{
    return op == VC4_OP_CONV2D ? p->im2col_q : p->pack_q;
}

static void set_aux(vc4_op op, vc4_tune_params *p, uint32_t q)
{
    if (op == VC4_OP_CONV2D)
        p->im2col_q = q;
    else
        p->pack_q = q;
}

/* Row blocks one GEMM launch has work for: 8 rows per block in asm10
 * (matmul), 16 in asm3 (conv). */
static uint32_t row_blocks(vc4_op op, const vc4_tune_shape *s)
{
    return op == VC4_OP_CONV2D ? s->m / 16u : s->m / 8u;
}

/* ------------------------------------------------------- legality and space */

static int shape_ok(vc4_op op, const vc4_tune_shape *s)
{
    if (op == VC4_OP_MATMUL) {
        return s->m != 0 && s->n != 0 && s->k != 0 &&
               s->m % 8 == 0 && s->n % 64 == 0 && s->k % 16 == 0;
    }
    if (op == VC4_OP_CONV2D) {
        int hp, wp;
        if (s->cin == 0 || s->cout == 0 || s->h == 0 || s->w == 0 ||
            s->cin % 16 || s->cout % 16)
            return 0;
        if (s->kh != s->kw || (s->kh != 3 && s->kh != 1))
            return 0;
        if (s->kh == 1 && s->pad != 0)
            return 0;
        if (s->stride < 1 || s->h + 2 * s->pad < s->kh ||
            s->w + 2 * s->pad < s->kw)
            return 0;
        hp = (int)s->h + 2 * (int)s->pad;
        wp = (int)vc4__align16(s->w + 2u * s->pad);
        if ((size_t)hp * (size_t)wp * 4u > 8191u)   /* 13-bit VDR pitch */
            return 0;
        return s->m % 16 == 0;
    }
    return 0;
}

/* The executor's own choice, which is also the baseline every measurement is
 * compared against.  Keep these in step with vc4_gemm.c (pack_run, gemm_run,
 * gemm3_run_c) and vc4_conv.c (im2col): they are the defaults the engine has
 * always used, and a record replaces them only when it measured faster. */
static void shape_defaults(vc4_op op, const vc4_tune_shape *s,
                           vc4_tune_params *p)
{
    uint32_t tr = row_blocks(op, s);
    memset(p, 0, sizeof(*p));
    p->gemm_q = tr < VC4_MAX_QPU ? tr : VC4_MAX_QPU;
    if (op == VC4_OP_CONV2D)
        p->im2col_q = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
    else
        p->pack_q = tr < 4 ? tr : 4;
}

int vc4__tune_params_legal(vc4_op op, const vc4_tune_shape *s,
                           const vc4_tune_params *p)
{
    uint32_t tr = row_blocks(op, s);
    uint32_t aux = aux_of(op, p);
    if (p->gemm_q < 1 || p->gemm_q > VC4_MAX_QPU || p->gemm_q > tr)
        return 0;
    /* both auxiliary kernels partition their work exactly and have no
     * in-kernel guard; four is the count each blob is registered for */
    if (aux < 1 || aux > 4 || aux > tr || tr % aux)
        return 0;
    /* the field the other op uses must stay zero, so two records cannot
     * compare equal while meaning different launches */
    if (op == VC4_OP_CONV2D ? p->pack_q != 0 : p->im2col_q != 0)
        return 0;
    return 1;
}

int vc4__tune_params_same(const vc4_tune_params *a, const vc4_tune_params *b)
{
    return a->pack_q == b->pack_q && a->gemm_q == b->gemm_q &&
           a->im2col_q == b->im2col_q;
}

/* The coarse pass, in measurement order: the default first (it is the
 * baseline), then every legal GEMM count with the default packer, then every
 * legal packer with the default GEMM.  One axis moves at a time, so the coarse
 * pass cannot exceed 11 candidates. */
int vc4__tune_space(vc4_op op, const vc4_tune_shape *s, vc4_tune_params *out,
                    int max)
{
    vc4_tune_params d;
    int n = 0, ntr;
    shape_defaults(op, s, &d);
    if (max <= 0)
        return 0;
    out[n++] = d;

    ntr = (int)row_blocks(op, s);
    for (size_t i = 0; i < sizeof(GEMM_LADDER) / sizeof(GEMM_LADDER[0]); ++i) {
        uint32_t g = GEMM_LADDER[i];
        vc4_tune_params c = d;
        if (n >= max)
            break;
        if (g > VC4_MAX_QPU || (int)g > ntr || g == d.gemm_q)
            continue;
        c.gemm_q = g;
        out[n++] = c;
    }
    for (size_t i = 0; i < sizeof(AUX_LADDER) / sizeof(AUX_LADDER[0]); ++i) {
        uint32_t a = AUX_LADDER[i];
        vc4_tune_params c = d;
        if (n >= max)
            break;
        if ((int)a > 4 || (int)a > ntr || ntr % (int)a || a == aux_of(op, &d))
            continue;
        set_aux(op, &c, a);
        out[n++] = c;
    }
    return n;
}

/* The fine pass: the GEMM counts next to the coarse winner, then any auxiliary
 * count the coarse pass did not try.  Deduplicated against everything already
 * measured. */
int vc4__tune_refine(vc4_op op, const vc4_tune_shape *s,
                     const vc4_tune_params *best, const vc4_tune_params *seen,
                     int nseen, vc4_tune_params *out, int max)
{
    int n = 0, ntr = (int)row_blocks(op, s);

    for (int dist = 1; dist <= 2 && n < max; ++dist) {
        for (int sign = -1; sign <= 1 && n < max; sign += 2) {
            vc4_tune_params c = *best;
            int g = (int)best->gemm_q + sign * dist, dup = 0;
            if (g < 1 || g > VC4_MAX_QPU || g > ntr)
                continue;
            c.gemm_q = (uint32_t)g;
            for (int i = 0; i < nseen && !dup; ++i)
                if (vc4__tune_params_same(&seen[i], &c))
                    dup = 1;
            for (int i = 0; i < n && !dup; ++i)
                if (vc4__tune_params_same(&out[i], &c))
                    dup = 1;
            if (!dup)
                out[n++] = c;
        }
    }
    for (size_t i = 0; i < sizeof(AUX_LADDER) / sizeof(AUX_LADDER[0]) && n < max;
         ++i) {
        uint32_t a = AUX_LADDER[i];
        vc4_tune_params c = *best;
        int dup = 0;
        if ((int)a > 4 || (int)a > ntr || ntr % (int)a ||
            a == aux_of(op, best))
            continue;
        set_aux(op, &c, a);
        for (int j = 0; j < nseen && !dup; ++j)
            if (vc4__tune_params_same(&seen[j], &c))
                dup = 1;
        for (int j = 0; j < n && !dup; ++j)
            if (vc4__tune_params_same(&out[j], &c))
                dup = 1;
        if (!dup)
            out[n++] = c;
    }
    return n;
}

/* --------------------------------------------------------------- cache I/O */

void vc4__tune_key(const char *cache_dir, vc4_op op, const vc4_tune_shape *s,
                   const char *digest, uint64_t *key, char *path, size_t pn)
{
    char shape[160], buf[512], hex[65];
    shape_str(s, shape, sizeof(shape));
    snprintf(buf, sizeof(buf), "%s|%s|%s|%s", TUNE_KEY_DOMAIN,
             vc4_op_name(op), shape, digest);
    vc4__sha256_hex(buf, strlen(buf), hex);
    hex[16] = 0;
    if (key)
        *key = strtoull(hex, NULL, 16);
    snprintf(path, pn, "%s/autotune/%s.tune", cache_dir, hex);
}

/* Create the record's directory; the file is <cache>/autotune/<key>.tune and
 * the cache root itself is created by vc4__default_cache_dir. */
static int ensure_dir(const char *file_path)
{
    char dir[512];
    const char *slash = strrchr(file_path, '/');
    size_t n = slash ? (size_t)(slash - file_path) : 0;
    if (n == 0)
        return 0;
    if (n >= sizeof(dir))
        return -1;
    memcpy(dir, file_path, n);
    dir[n] = 0;
    if (mkdir(dir, 0777) == 0 || errno == EEXIST)
        return 0;
    return -1;
}

int vc4__tune_cache_read(const char *path, vc4_op op, const vc4_tune_shape *s,
                         const char *digest, vc4_tune_params *best,
                         char *kernel, size_t kn, double *best_ms,
                         double *default_ms)
{
    FILE *f = fopen(path, "r");
    char line[256], want[160], got[160];
    int header = 0, match = 1, have_best = 0;
    vc4_tune_params b;
    double bms = 0.0, dms = 0.0;
    char kname[80];

    if (!f)
        return -1;
    memset(&b, 0, sizeof(b));
    memset(kname, 0, sizeof(kname));
    shape_str(s, want, sizeof(want));

    while (fgets(line, sizeof(line), f)) {
        unsigned v[12];
        char word[80];
        if (!header) {
            if (strncmp(line, TUNE_MAGIC, strlen(TUNE_MAGIC)) != 0) {
                match = 0;
                break;
            }
            header = 1;
            continue;
        }
        if (sscanf(line, "op %79s", word) == 1) {
            if (strcmp(word, vc4_op_name(op)) != 0)
                match = 0;
        } else if (sscanf(line, "shape %u %u %u %u %u %u %u %u %u %u %u %u",
                          &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                          &v[7], &v[8], &v[9], &v[10], &v[11]) == 12) {
            snprintf(got, sizeof(got), "%u %u %u %u %u %u %u %u %u %u %u %u",
                     v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                     v[9], v[10], v[11]);
            if (strcmp(got, want) != 0)
                match = 0;
        } else if (sscanf(line, "gen %64s", word) == 1) {
            if (strcmp(word, digest) != 0)
                match = 0;
        } else if (sscanf(line, "best %u %u %u", &v[0], &v[1], &v[2]) == 3) {
            b.pack_q = v[0];
            b.gemm_q = v[1];
            b.im2col_q = v[2];
            have_best = 1;
        } else if (sscanf(line, "kernel %79s", word) == 1) {
            snprintf(kname, sizeof(kname), "%s", word);
        } else if (sscanf(line, "best_ms %lf", &bms) == 1) {
            /* recorded median of the chosen set */
        } else if (sscanf(line, "default_ms %lf", &dms) == 1) {
            /* recorded median of the executor's default */
        }
    }
    fclose(f);

    if (!header || !match || !have_best)
        return -1;
    if (!vc4__tune_params_legal(op, s, &b))
        return -1;
    if (best)
        *best = b;
    if (kernel && kn)
        snprintf(kernel, kn, "%s", kname);
    if (best_ms)
        *best_ms = bms;
    if (default_ms)
        *default_ms = dms;
    return 0;
}

vc4_status vc4__tune_cache_write(const char *path, vc4_op op,
                                 const vc4_tune_shape *s, const char *digest,
                                 const vc4_tune_trial *trials, int n,
                                 const vc4_tune_params *best, const char *kernel,
                                 double best_ms, double default_ms)
{
    char tmp[600], shape[160];
    FILE *f;

    if (ensure_dir(path) != 0)
        return VC4_ERR_IO;
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "w");
    if (!f)
        return VC4_ERR_IO;
    shape_str(s, shape, sizeof(shape));
    fprintf(f, "%s\n", TUNE_MAGIC);
    fprintf(f, "op %s\n", vc4_op_name(op));
    fprintf(f, "shape %s\n", shape);
    fprintf(f, "gen %s\n", digest);
    for (int i = 0; i < n; ++i)
        fprintf(f, "trial pack_q=%u gemm_q=%u im2col_q=%u ms=%.6f\n",
                trials[i].p.pack_q, trials[i].p.gemm_q, trials[i].p.im2col_q,
                trials[i].ms);
    fprintf(f, "best %u %u %u\n", best->pack_q, best->gemm_q, best->im2col_q);
    fprintf(f, "kernel %s\n", kernel && *kernel ? kernel : "-");
    fprintf(f, "best_ms %.6f\n", best_ms);
    fprintf(f, "default_ms %.6f\n", default_ms);
    if (fclose(f) != 0) {
        unlink(tmp);
        return VC4_ERR_IO;
    }
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        return VC4_ERR_IO;
    }
    return VC4_OK;
}

/* ------------------------------------------------ per-context lookup memo
 *
 * A graph runs the same matmul shape many times.  Reopening and parsing the
 * record per launch would be pure overhead, so the last 32 lookups are kept in
 * the context, keyed by the record key.  With autotuning off the whole lookup
 * is one getenv/strcmp, and a miss is one file open. */
static void memo_put(vc4_ctx *ctx, uint64_t key, const vc4_tune_params *p)
{
    int slot = (int)(key % VC4_TUNE_MEMO);
    ctx->tune_memo[slot].key = key;
    ctx->tune_memo[slot].p = *p;
    ctx->tune_memo[slot].valid = 1;
}

static int memo_get(vc4_ctx *ctx, uint64_t key, vc4_tune_params *p)
{
    int slot = (int)(key % VC4_TUNE_MEMO);
    if (!ctx->tune_memo[slot].valid || ctx->tune_memo[slot].key != key)
        return 0;
    *p = ctx->tune_memo[slot].p;
    return 1;
}

int vc4__tune_lookup(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *s,
                     vc4_tune_params *out)
{
    char path[512];
    uint64_t key;
    if (!ctx || !s || !out || !vc4_autotune_enabled(ctx))
        return 0;
    /* the generator digest hashes the generator sources: compute it once per
     * context, not once per launch */
    if (!ctx->tune_digest[0])
        vc4__generator_digest(ctx->tune_digest);
    vc4__tune_key(ctx->cache_dir, op, s, ctx->tune_digest, &key, path,
                  sizeof(path));
    if (memo_get(ctx, key, out))
        return 1;
    if (vc4__tune_cache_read(path, op, s, ctx->tune_digest, out, NULL, 0, NULL,
                             NULL) != 0)
        return 0;
    memo_put(ctx, key, out);
    return 1;
}

/* --------------------------------------------------------------- measuring */

struct tune_bufs {
    vc4_buf a, b, c;                 /* matmul: MxK, KxN, MxN                */
    vc4_buf in, wt, bias, out;       /* conv                                 */
    int      have_bias;
    uint32_t ho, wo;
};

/* Small deterministic values in [-2, 2): no denormals, no overflow in a K-long
 * dot product, and the same bytes on every run so two candidates are compared
 * on identical input. */
static void fill_pattern(float *p, size_t n)
{
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (float)((int)((x >> 16) & 0xFFu) - 128) / 64.0f;
    }
}

/* Upper bound on the scratch the op itself allocates inside the same arena.
 * Workspace buffers are carved down from the top of the arena and the op's
 * scratch grows up from the heap, so the tuner measures only when the gap
 * between them is large enough for both.  A collision would corrupt the input
 * a later candidate measured on, which is worse than refusing the shape. */
static size_t tune_scratch_bound(vc4_op op, const vc4_tune_shape *s)
{
    if (op == VC4_OP_MATMUL)
        return ((size_t)s->m * s->k + (size_t)s->m * s->n) * 4u;
    {
        uint32_t ho = (s->h + 2u * s->pad - s->kh) / s->stride + 1u;
        uint32_t wp = vc4__align16(s->w + 2u * s->pad);
        uint32_t m = ho * wp;
        size_t pad = (size_t)s->cin * ((size_t)s->h + 2u * s->pad) * wp * 4u;
        size_t panel = (size_t)m * s->k * 4u;
        size_t wr = (size_t)s->cout * s->cin * s->kh * s->kw * 4u;
        size_t wpanel = (size_t)s->cout * s->k * 4u;
        size_t cmat = (size_t)m * s->cout * 4u;
        return pad + panel + wr + wpanel + cmat + 256u;
    }
}

static vc4_status tune_prepare(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *s,
                               struct tune_bufs *tb)
{
    memset(tb, 0, sizeof(*tb));
    if (vc4__ws_free(ctx) < tune_scratch_bound(op, s)) {
        vc4__set_err(ctx, "autotune: %s %ux%ux%u needs %zu B of workspace for "
                     "its own scratch and the inputs; %zu B free",
                     vc4_op_name(op), s->m, s->n, s->k,
                     tune_scratch_bound(op, s), vc4__ws_free(ctx));
        return VC4_ERR_NOMEM;
    }
    if (op == VC4_OP_MATMUL) {
        size_t na = (size_t)s->m * s->k, nb = (size_t)s->k * s->n,
               nc = (size_t)s->m * s->n;
        float *a = vc4__ws_alloc(ctx, na * 4u, &tb->a);
        float *b = vc4__ws_alloc(ctx, nb * 4u, &tb->b);
        float *c = vc4__ws_alloc(ctx, nc * 4u, &tb->c);
        float *host = malloc((na > nb ? na : nb) * 4u);
        if (!a || !b || !c || !host) {
            free(host);
            vc4__set_err(ctx, "autotune: no workspace for the matmul buffers");
            return VC4_ERR_NOMEM;
        }
        fill_pattern(host, na);
        vc4_upload(ctx, &tb->a, host, na * 4u);
        fill_pattern(host, nb);
        vc4_upload(ctx, &tb->b, host, nb * 4u);
        free(host);
        memset(c, 0, nc * 4u);
        return VC4_OK;
    }

    {
        size_t nin = (size_t)s->cin * s->h * s->w;
        size_t nw = (size_t)s->cout * (size_t)s->cin * s->kh * s->kw;
        float *in, *wt, *out, *host;
        tb->ho = (s->h + 2u * s->pad - s->kh) / s->stride + 1u;
        tb->wo = (s->w + 2u * s->pad - s->kw) / s->stride + 1u;
        tb->have_bias = (s->flags & VC4_TUNE_F_BIAS) != 0;
        in = vc4__ws_alloc(ctx, nin * 4u, &tb->in);
        wt = vc4__ws_alloc(ctx, nw * 4u, &tb->wt);
        out = vc4__ws_alloc(ctx, (size_t)s->cout * tb->ho * tb->wo * 4u,
                            &tb->out);
        host = malloc((nin > nw ? nin : nw) * 4u);
        if (!in || !wt || !out || !host) {
            free(host);
            vc4__set_err(ctx, "autotune: no workspace for the conv buffers");
            return VC4_ERR_NOMEM;
        }
        fill_pattern(host, nin);
        vc4_upload(ctx, &tb->in, host, nin * 4u);
        fill_pattern(host, nw);
        vc4_upload(ctx, &tb->wt, host, nw * 4u);
        memset(out, 0, (size_t)s->cout * tb->ho * tb->wo * 4u);
        free(host);
        if (tb->have_bias) {
            float *bi = vc4__ws_alloc(ctx, (size_t)s->cout * 4u, &tb->bias);
            host = malloc((size_t)s->cout * 4u);
            if (!bi || !host) {
                free(host);
                vc4__set_err(ctx, "autotune: no workspace for the bias");
                return VC4_ERR_NOMEM;
            }
            fill_pattern(host, s->cout);
            vc4_upload(ctx, &tb->bias, host, (size_t)s->cout * 4u);
            free(host);
        }
        return VC4_OK;
    }
}

static vc4_status tune_run(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *s,
                           struct tune_bufs *tb, const vc4_tune_params *p)
{
    int ho = 0, wo = 0;
    if (op == VC4_OP_MATMUL)
        return vc4__matmul_tuned(ctx, &tb->a, &tb->b, &tb->c, s->m, s->n, s->k,
                                 VC4_LAYOUT_ROW_MAJOR, p);
    return vc4__conv2d_tuned(ctx, &tb->in, (int)s->cin, (int)s->h, (int)s->w,
                             &tb->wt, (int)s->cout, (int)s->kh, (int)s->kw,
                             tb->have_bias ? &tb->bias : NULL,
                             (int)s->stride, (int)s->pad, &tb->out, &ho, &wo,
                             p);
}

static int cmp_double(const void *pa, const void *pb)
{
    double a = *(const double *)pa, b = *(const double *)pb;
    return a < b ? -1 : a > b ? 1 : 0;
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    if (n & 1)
        return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* One warm-up (which also generates and loads the blob) then `reps` timed
 * runs.  A candidate that fails is not measured and not recorded: a cached
 * choice that cannot run is worse than no record. */
static double tune_measure(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *s,
                           struct tune_bufs *tb, const vc4_tune_params *p,
                           int reps, vc4_status *err)
{
    double v[TUNE_MAX_REPS];
    vc4_status st = tune_run(ctx, op, s, tb, p);
    if (st != VC4_OK) {
        *err = st;
        return -1.0;
    }
    for (int i = 0; i < reps; ++i) {
        double t0 = now_ms();
        st = tune_run(ctx, op, s, tb, p);
        v[i] = now_ms() - t0;
        if (st != VC4_OK) {
            *err = st;
            return -1.0;
        }
    }
    *err = VC4_OK;
    return median(v, reps);
}

static const vc4_buf *tune_output(vc4_op op, const struct tune_bufs *tb)
{
    return op == VC4_OP_MATMUL ? &tb->c : &tb->out;
}

static size_t tune_output_bytes(vc4_op op, const vc4_tune_shape *s,
                                const struct tune_bufs *tb)
{
    if (op == VC4_OP_MATMUL)
        return (size_t)s->m * s->n * 4u;
    return (size_t)s->cout * tb->ho * tb->wo * 4u;
}

/* The chosen launch must compute the same thing as the default.  QPU counts
 * change which QPU owns which row block and nothing else, so on this hardware
 * the two are expected to agree bit for bit; the check runs anyway, because
 * "expected" is not the standard the rest of this library holds, and a
 * schedule that silently changes a number is exactly the failure a cache would
 * make permanent.  Tolerance: the atol = rtol = 5e-4 of docs/ACCURACY.md. */
static void compare_outputs(vc4_ctx *ctx, const vc4_buf *a, const vc4_buf *b,
                            size_t bytes, double *max_abs, uint32_t *over_tol)
{
    size_t n = bytes / 4u;
    float *ha = malloc(bytes), *hb = malloc(bytes);
    double worst = 0.0;
    uint32_t over = 0;
    *max_abs = 0.0;
    *over_tol = 0;
    if (!ha || !hb || vc4_download(ctx, a, ha, bytes) != VC4_OK ||
        vc4_download(ctx, b, hb, bytes) != VC4_OK) {
        free(ha);
        free(hb);
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        double d = (double)ha[i] - (double)hb[i];
        double mag = d < 0 ? -d : d;
        double ref = (double)hb[i] < 0 ? -(double)hb[i] : (double)hb[i];
        double tol = 5e-4 + 5e-4 * (ref > 1.0 ? ref : 1.0);
        if (mag > worst)
            worst = mag;
        if (mag > tol)
            over++;
    }
    free(ha);
    free(hb);
    *max_abs = worst;
    *over_tol = over;
}

/* ------------------------------------------------------------------- entry */

vc4_status vc4_autotune(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *shape_in,
                        const vc4_tune_opts *opts_in, vc4_tune_result *out)
{
    vc4_tune_shape s;
    vc4_tune_opts o;
    vc4_tune_params def, best, cand[TUNE_MAX_TOTAL], fine[TUNE_MAX_TOTAL];
    vc4_tune_params seen[TUNE_MAX_TOTAL];
    vc4_tune_trial trials[TUNE_MAX_TOTAL];
    int ntrial = 0, ncand = 0, nfine = 0;
    char path[512], kernel[80];
    uint64_t key = 0;
    double best_ms = 0.0, default_ms = 0.0, t_start, max_abs = 0.0;
    uint32_t over = 0;
    vc4_status st, rc = VC4_OK;
    struct tune_bufs tb;
    vc4_buf ref;
    float *refbuf;
    size_t obytes, ws0;

    if (!ctx || !shape_in)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (op != VC4_OP_MATMUL && op != VC4_OP_CONV2D) {
        vc4__set_err(ctx, "autotune: %s has no tunable parameter set",
                     vc4_op_name(op));
        return VC4_ERR_UNSUPPORTED;
    }

    /* Normalise exactly the way the executor does. */
    if (op == VC4_OP_MATMUL) {
        vc4__tune_shape_mm(&s, shape_in->m, shape_in->n, shape_in->k);
    } else {
        vc4__tune_shape_conv(&s, shape_in->cin, shape_in->cout, shape_in->h,
                             shape_in->w, shape_in->kh, shape_in->kw,
                             shape_in->pad, shape_in->stride,
                             (shape_in->flags & VC4_TUNE_F_BIAS) != 0);
    }
    if (!shape_ok(op, &s)) {
        vc4__set_err(ctx, "autotune: %s shape %ux%ux%u is not one the engine "
                     "runs", vc4_op_name(op), s.m, s.n, s.k);
        return VC4_ERR_INVAL;
    }

    o.max_candidates = TUNE_DEFAULT_CAND;
    o.budget_ms = TUNE_DEFAULT_BUDGET;
    o.reps = TUNE_DEFAULT_REPS;
    o.force = 0;
    if (opts_in) {
        if (opts_in->max_candidates)
            o.max_candidates = opts_in->max_candidates;
        if (opts_in->budget_ms)
            o.budget_ms = opts_in->budget_ms;
        if (opts_in->reps)
            o.reps = opts_in->reps;
        o.force = opts_in->force;
    }
    o.max_candidates = env_u32("VC4INFER_AUTOTUNE_MAX", o.max_candidates);
    o.budget_ms = env_u32("VC4INFER_AUTOTUNE_BUDGET_MS", o.budget_ms);
    o.reps = env_u32("VC4INFER_AUTOTUNE_REPS", o.reps);
    if (o.max_candidates < 2)
        o.max_candidates = 2;
    if (o.max_candidates > TUNE_MAX_TOTAL - 1)
        o.max_candidates = TUNE_MAX_TOTAL - 1;
    if (o.reps < 1)
        o.reps = 1;
    if (o.reps > TUNE_MAX_REPS)
        o.reps = TUNE_MAX_REPS;

    if (!ctx->tune_digest[0])
        vc4__generator_digest(ctx->tune_digest);
    vc4__tune_key(ctx->cache_dir, op, &s, ctx->tune_digest, &key, path,
                  sizeof(path));

    if (!o.force &&
        vc4__tune_cache_read(path, op, &s, ctx->tune_digest, &best, kernel,
                             sizeof(kernel), &best_ms, &default_ms) == 0) {
        ctx->autotune = 1;
        if (out) {
            memset(out, 0, sizeof(*out));
            out->params = best;
            out->best_ms = best_ms;
            out->default_ms = default_ms;
            out->from_cache = 1;
            snprintf(out->kernel, sizeof(out->kernel), "%s", kernel);
        }
        return VC4_OK;
    }

    /* The measurement buffers live in the workspace (carved down from the top
     * of the arena) so the op's own per-launch scratch resets cannot free
     * them; the whole call nests inside one ws_begin/ws_end pair. */
    ws0 = vc4__ws_begin(ctx);
    rc = tune_prepare(ctx, op, &s, &tb);
    if (rc != VC4_OK)
        goto out;

    t_start = now_ms();
    shape_defaults(op, &s, &def);
    default_ms = tune_measure(ctx, op, &s, &tb, &def, (int)o.reps, &st);
    if (st != VC4_OK) {
        vc4__set_err(ctx, "autotune: the default launch failed (%s)",
                     vc4_strerror(st));
        rc = st;
        goto out;
    }
    best = def;
    best_ms = default_ms;
    trials[ntrial].p = def;
    trials[ntrial].ms = default_ms;
    seen[ntrial] = def;
    ntrial++;

    /* the default's output, kept for the correctness check */
    obytes = tune_output_bytes(op, &s, &tb);
    refbuf = vc4__ws_alloc(ctx, obytes, &ref);
    if (!refbuf ||
        vc4_download(ctx, tune_output(op, &tb), refbuf, obytes) != VC4_OK) {
        vc4__set_err(ctx, "autotune: cannot snapshot the default output");
        rc = VC4_ERR_NOMEM;
        goto out;
    }

    ncand = vc4__tune_space(op, &s, cand, TUNE_MAX_TOTAL);
    for (int i = 0; i < ncand; ++i) {
        double ms;
        if (ntrial >= (int)o.max_candidates)
            break;
        if (vc4__tune_params_same(&cand[i], &def))
            continue;
        if (now_ms() - t_start > (double)o.budget_ms)
            break;
        ms = tune_measure(ctx, op, &s, &tb, &cand[i], (int)o.reps, &st);
        if (st != VC4_OK)
            continue;
        trials[ntrial].p = cand[i];
        trials[ntrial].ms = ms;
        seen[ntrial] = cand[i];
        if (ms < best_ms) {
            best_ms = ms;
            best = cand[i];
        }
        ntrial++;
    }

    nfine = vc4__tune_refine(op, &s, &best, seen, ntrial, fine, TUNE_MAX_TOTAL);
    for (int i = 0; i < nfine; ++i) {
        double ms;
        if (ntrial >= (int)o.max_candidates)
            break;
        if (now_ms() - t_start > (double)o.budget_ms)
            break;
        ms = tune_measure(ctx, op, &s, &tb, &fine[i], (int)o.reps, &st);
        if (st != VC4_OK)
            continue;
        trials[ntrial].p = fine[i];
        trials[ntrial].ms = ms;
        seen[ntrial] = fine[i];
        if (ms < best_ms) {
            best_ms = ms;
            best = fine[i];
        }
        ntrial++;
    }

    /* Correctness before persistence. */
    if (!vc4__tune_params_same(&best, &def)) {
        if (tune_run(ctx, op, &s, &tb, &best) != VC4_OK) {
            vc4__set_err(ctx, "autotune: the chosen launch failed to run; "
                         "not cached");
            rc = VC4_ERR_KERNEL;
            goto out;
        }
        compare_outputs(ctx, tune_output(op, &tb), &ref, obytes, &max_abs,
                        &over);
        if (over != 0) {
            vc4__set_err(ctx, "autotune: %s q=%u/%u/%u disagrees with the "
                         "default (max |diff| %.3g, %u element(s) over 5e-4); "
                         "not cached", vc4_op_name(op), best.pack_q,
                         best.gemm_q, best.im2col_q, max_abs, over);
            rc = VC4_ERR_INVAL;
            goto out;
        }
    }

    if (op == VC4_OP_MATMUL)
        vc4__matmul_kernel_name(kernel, sizeof(kernel), s.m, s.n, s.k,
                                (int)best.gemm_q);
    else
        vc4__conv_kernel_name(kernel, sizeof(kernel), s.m, s.n, s.k,
                              (int)best.gemm_q);

    /* A cache directory the process cannot write (a read-only install) must
     * not fail an otherwise useful measurement. */
    if (vc4__tune_cache_write(path, op, &s, ctx->tune_digest, trials, ntrial,
                              &best, kernel, best_ms, default_ms) == VC4_OK)
        memo_put(ctx, key, &best);
    else
        vc4__set_err(ctx, "autotune: record for %s could not be written to %s",
                     vc4_op_name(op), path);

    ctx->autotune = 1;
    if (out) {
        memset(out, 0, sizeof(*out));
        out->params = best;
        out->best_ms = best_ms;
        out->default_ms = default_ms;
        out->max_abs = max_abs;
        out->candidates = (uint32_t)ntrial;
        out->over_tol = over;
        out->from_cache = 0;
        snprintf(out->kernel, sizeof(out->kernel), "%s", kernel);
    }
out:
    vc4__ws_end(ctx, ws0);
    return rc;
}
