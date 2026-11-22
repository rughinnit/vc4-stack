/*
 * vc4_kern.c - kernel blob cache with a JIT/cache on top of tools/kgen.py.
 *
 * Every distinct kernel blob gets its own code area and its own uniform area
 * inside the arena.  That is the asm3 "two-launch rule": a program written at
 * an address whose L2 line is still valid can be shadowed by that line, and the
 * QPU then runs the *previous* program (asm3 observed the packer executing with
 * the GEMM's uniforms).  Distinct offsets, plus an SLCACTL invalidate before
 * every launch (vc4_dev.c), remove the hazard.
 *
 * The GEMM blob's k loop is bit-identical to the verified asm3 kernel
 * gemm2_128_d3_v1.bin; tests/test_matmul.c reproduces that comparison.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef VC4_KGEN_DEFAULT
#define VC4_KGEN_DEFAULT "/home/raspi/work/vc4-stack/infer/tools/kgen.py"
#endif

/* Default blob cache: never the caller's cwd (a system-installed client may run
 * from /), and created on demand so a clean install can JIT.  $HOME/.cache is
 * user-writable, which matters because vc4d clients run unprivileged. */
void vc4__default_cache_dir(char *out, size_t n)
{
    const char *e = getenv("VC4INFER_CACHE");
    if (e && *e) {
        snprintf(out, n, "%s", e);
    } else {
        const char *xdg  = getenv("XDG_CACHE_HOME");
        const char *home = getenv("HOME");
        if (xdg && *xdg)
            snprintf(out, n, "%s/vc4infer/kcache", xdg);
        else if (home && *home)
            snprintf(out, n, "%s/.cache/vc4infer/kcache", home);
        else
            snprintf(out, n, "/tmp/vc4infer-kcache");
    }
    /* mkdir -p */
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", out);
    for (char *p = tmp + 1; *p; ++p)
        if (*p == '/') { *p = 0; mkdir(tmp, 0777); *p = '/'; }
    mkdir(tmp, 0777);
}


/* ------------------------------------------------------ generator stamps
 *
 * A blob is reused only when the sha256 of the generator (kgen.py plus the
 * kops.py beside it), the ABI tag and the generator argument vector match the
 * sidecar written when the blob was produced.  A changed generator therefore
 * invalidates every blob it built instead of silently reusing an old kernel -
 * a stale blob is what wedged the QPU during the op-long-tail work.
 */
#define VC4_ABI_TAG "vc4infer-kernel-abi-2"

typedef struct {
    uint32_t h[8];
    uint64_t len;
    uint8_t buf[64];
    size_t n;
} sha256_ctx;

static const uint32_t sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha_block(sha256_ctx *c, const uint8_t *p)
{
    uint32_t w[64], a, b, d, e, f, g, h, cc, s0, s1, t1, t2;
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3];
    e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
    for (int i = 0; i < 64; ++i) {
        s1 = ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25);
        t1 = h + s1 + ((e & f) ^ (~e & g)) + sha_k[i] + w[i];
        s0 = ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22);
        t2 = s0 + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d;
    c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}

static void sha_init(sha256_ctx *c)
{
    c->h[0] = 0x6a09e667; c->h[1] = 0xbb67ae85;
    c->h[2] = 0x3c6ef372; c->h[3] = 0xa54ff53a;
    c->h[4] = 0x510e527f; c->h[5] = 0x9b05688c;
    c->h[6] = 0x1f83d9ab; c->h[7] = 0x5be0cd19;
    c->len = 0;
    c->n = 0;
}

static void sha_update(sha256_ctx *c, const void *data, size_t len)
{
    const uint8_t *p = data;
    c->len += len;
    while (len) {
        size_t take = 64 - c->n;
        if (take > len)
            take = len;
        memcpy(c->buf + c->n, p, take);
        c->n += take;
        p += take;
        len -= take;
        if (c->n == 64) {
            sha_block(c, c->buf);
            c->n = 0;
        }
    }
}

static void sha_final(sha256_ctx *c, uint8_t out[32])
{
    uint64_t bits = c->len * 8;
    uint8_t pad = 0x80, lenb[8];
    sha_update(c, &pad, 1);
    pad = 0;
    while (c->n != 56)
        sha_update(c, &pad, 1);
    for (int i = 0; i < 8; ++i)
        lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(c, lenb, 8);
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(c->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(c->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(c->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)(c->h[i]);
    }
}

static const char *generator_path(void)
{
    const char *kg = getenv("VC4INFER_KGEN");
    return kg ? kg : VC4_KGEN_DEFAULT;
}

static void sha_file(sha256_ctx *c, const char *path, const char *tag)
{
    sha_update(c, tag, strlen(tag));
    FILE *f = fopen(path, "rb");
    if (!f) {
        sha_update(c, "absent", 6);
        return;
    }
    uint8_t buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        sha_update(c, buf, n);
    fclose(f);
}

/* sha256 of the generator itself: kgen.py, kops.py beside it, the ABI tag.
 * This is the prefix of every blob's stamp and the invalidation key of the
 * autotune records (vc4_autotune.c), so one generator change discards both. */
static void generator_hash(sha256_ctx *c)
{
    const char *kg = generator_path();
    char kops[512];
    const char *slash = strrchr(kg, '/');
    size_t dlen = slash ? (size_t)(slash - kg) : 0;
    sha_file(c, kg, "kgen:");
    if (dlen + 16 < sizeof(kops)) {
        memcpy(kops, kg, dlen);
        snprintf(kops + dlen, sizeof(kops) - dlen, "/kops.py");
        sha_file(c, kops, "kops:");
    }
    sha_update(c, VC4_ABI_TAG, strlen(VC4_ABI_TAG));
}

static void digest_hex(const uint8_t h[32], char out[65])
{
    static const char *digits = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[2 * i] = digits[h[i] >> 4];
        out[2 * i + 1] = digits[h[i] & 15];
    }
    out[64] = 0;
}

void vc4__generator_digest(char out[65])
{
    sha256_ctx c;
    uint8_t h[32];
    sha_init(&c);
    generator_hash(&c);
    sha_final(&c, h);
    digest_hex(h, out);
}

/* The ABI tag is part of the generator digest above; it is exposed so an
 * engine artifact can stamp and check it without duplicating the string in
 * Python (a copy there would silently disagree on the next ABI bump). */
const char *vc4_kernel_abi_tag(void)
{
    return VC4_ABI_TAG;
}

void vc4_generator_digest(char out[65])
{
    vc4__generator_digest(out);
}

int vc4_loaded_kernels(const vc4_ctx *ctx, char *buf, size_t bytes)
{
    if (!ctx || (!buf && bytes))
        return -1;
    size_t used = 0;
    for (int i = 0; i < ctx->nseen; ++i) {
        const char *name = ctx->seen[i];
        size_t len = strlen(name);
        if (used + len + 1 > bytes)
            return -1;
        memcpy(buf + used, name, len + 1);
        used += len + 1;
    }
    return ctx->nseen;
}

void vc4__sha256_hex(const void *data, size_t len, char out[65])
{
    sha256_ctx c;
    uint8_t h[32];
    sha_init(&c);
    sha_update(&c, data, len);
    sha_final(&c, h);
    digest_hex(h, out);
}

/* sha256(generator hash, argv) */
static void kernel_digest(char *const gen_argv[], uint8_t out[32])
{
    sha256_ctx c;
    sha_init(&c);
    generator_hash(&c);
    for (int i = 0; gen_argv[i]; ++i) {
        sha_update(&c, gen_argv[i], strlen(gen_argv[i]));
        sha_update(&c, "\0", 1);
    }
    sha_final(&c, out);
}

static int stamp_ok(const char *path, const uint8_t want[32])
{
    char sp[640], hex[80], expect[65];
    static const char *digits = "0123456789abcdef";
    snprintf(sp, sizeof(sp), "%s.stamp", path);
    FILE *f = fopen(sp, "r");
    if (!f)
        return 0;
    size_t n = fread(hex, 1, sizeof(hex) - 1, f);
    fclose(f);
    hex[n] = 0;
    for (int i = 0; i < 32; ++i) {
        expect[2 * i] = digits[want[i] >> 4];
        expect[2 * i + 1] = digits[want[i] & 15];
    }
    expect[64] = 0;
    return strncmp(hex, expect, 64) == 0;
}

static void stamp_write(const char *path, const uint8_t want[32])
{
    char sp[640];
    static const char *digits = "0123456789abcdef";
    snprintf(sp, sizeof(sp), "%s.stamp", path);
    FILE *f = fopen(sp, "w");
    if (!f)
        return;                     /* read-only cache: regenerate next time */
    for (int i = 0; i < 32; ++i) {
        fputc(digits[want[i] >> 4], f);
        fputc(digits[want[i] & 15], f);
    }
    fputc('\n', f);
    fclose(f);
}

static vc4_status jit(vc4_ctx *ctx, char *const gen_argv[], const char *path)
{
    /* Resolution order: explicit override, then the path baked at build time
     * (correct for a source tree), then the copy the .deb installs.  Without
     * this, a package built in a dev tree carries a path that no longer exists
     * once the tree moves, and every uncached kernel fails to JIT. */
    char kbuf[512];
    const char *kgen = getenv("VC4INFER_KGEN");
    if (!kgen || !*kgen) {
        if (access(VC4_KGEN_DEFAULT, R_OK) == 0) {
            kgen = VC4_KGEN_DEFAULT;
        } else if (access("/usr/lib/vc4infer/kgen/kgen.py", R_OK) == 0) {
            kgen = "/usr/lib/vc4infer/kgen/kgen.py";
        } else {
            snprintf(kbuf, sizeof(kbuf), "%s", VC4_KGEN_DEFAULT);
            kgen = kbuf;
            vc4__set_err(ctx, "no kernel generator: tried $VC4INFER_KGEN, %s and "
                              "/usr/lib/vc4infer/kgen/kgen.py", VC4_KGEN_DEFAULT);
            return VC4_ERR_KERNEL;
        }
    }

    pid_t pid = fork();
    if (pid < 0) {
        vc4__set_err(ctx, "kernel jit: fork: %s", strerror(errno));
        return VC4_ERR_KERNEL;
    }
    if (pid == 0) {
        char *a[64];
        int n = 0;
        a[n++] = (char *)"python3";
        a[n++] = (char *)kgen;
        a[n++] = gen_argv[0];            /* kernel kind */
        a[n++] = (char *)path;           /* output blob */
        for (int i = 1; gen_argv[i] && n < 62; ++i)
            a[n++] = gen_argv[i];
        a[n] = NULL;
        execvp("python3", a);
        _exit(127);
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0)
        return VC4_ERR_KERNEL;
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        vc4__set_err(ctx, "kernel jit failed for %s (kgen=%s, status=%d)",
                     gen_argv[0], kgen, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        return VC4_ERR_KERNEL;
    }
    return VC4_OK;
}

static vc4_status blob_read(vc4_ctx *ctx, const char *path,
                            uint8_t **out, size_t *len)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        vc4__set_err(ctx, "kernel blob %s: %s", path, strerror(errno));
        return VC4_ERR_KERNEL;
    }
    if (st.st_size <= 0 || (st.st_size % 8) != 0) {
        vc4__set_err(ctx, "kernel blob %s: bad size %lld", path, (long long)st.st_size);
        return VC4_ERR_KERNEL;
    }
    if ((size_t)st.st_size > VC4_CODE_SLOT) {
        vc4__set_err(ctx, "kernel blob %s: %lld B does not fit the %u B code slot",
                     path, (long long)st.st_size, VC4_CODE_SLOT);
        return VC4_ERR_KERNEL;
    }
    FILE *f = fopen(path, "rb");
    if (!f)
        return VC4_ERR_KERNEL;
    uint8_t *buf = malloc((size_t)st.st_size);
    if (!buf) {
        fclose(f);
        return VC4_ERR_NOMEM;
    }
    size_t got = fread(buf, 1, (size_t)st.st_size, f);
    fclose(f);
    if (got != (size_t)st.st_size) {
        free(buf);
        return VC4_ERR_KERNEL;
    }
    *out = buf;
    *len = got;
    return VC4_OK;
}

/* ------------------------------------------------------- kernel cache slots
 *
 * VC4_MAX_KERNELS entries, each owning a code slot and a uniform area (direct
 * mode) or a daemon program slot (remote mode).  A session that touches more
 * distinct kernels than that used to fail with "kernel cache full (128
 * entries)" and every later shape was unusable until the process restarted
 * (G3).  The cache is an LRU now: the least recently loaded/launched entry is
 * replaced.  Callers use a vc4_kernel* only between returning it and launching
 * it - every op loads, fills uniforms and launches in one function - so
 * replacing the oldest entry cannot invalidate a live pointer. */
vc4_kernel *vc4__kern_slot(vc4_ctx *ctx)
{
    int slot = -1;
    if (ctx->nkern < VC4_MAX_KERNELS) {
        slot = ctx->nkern++;
    } else {
        for (int i = 0; i < ctx->nkern; ++i) {
            if (!ctx->k[i].used)
                continue;
            if (slot < 0 || ctx->k[i].used_at < ctx->k[slot].used_at)
                slot = i;
        }
        if (slot < 0)
            return NULL;
        free(ctx->k[slot].code);        /* remote: the re-upload copy */
    }
    vc4_kernel *k = &ctx->k[slot];
    memset(k, 0, sizeof(*k));
    k->used_at = ++ctx->kern_clock;
    return k;
}

vc4_status vc4__kernel_load_file(vc4_ctx *ctx, const char *name,
                                 const uint8_t *words, size_t bytes,
                                 int nqpu, vc4_kernel **out)
{
    if (ctx->remote)
        return vc4__remote_load(ctx, name, words, bytes, nqpu, out);
    vc4_kernel *k = vc4__kern_slot(ctx);
    if (!k) {
        vc4__set_err(ctx, "kernel cache: no slot (all %d in use)",
                     VC4_MAX_KERNELS);
        return VC4_ERR_NOMEM;
    }
    int slot = (int)(k - ctx->k);
    k->code_off = VC4_CODE_BASE + (uint32_t)slot * VC4_CODE_SLOT;
    k->uni_off = VC4_UNI_BASE + (uint32_t)slot * VC4_UNI_SLOT;
    if (k->uni_off + VC4_UNI_SLOT > VC4_DATA_BASE) {
        vc4__set_err(ctx, "kernel cache: uniform area overlaps the data heap");
        return VC4_ERR_NOMEM;
    }
    k->code_words = (uint32_t)(bytes / 8);
    k->nqpu = (uint32_t)nqpu;
    k->used = 1;
    snprintf(k->name, sizeof(k->name), "%s", name);

    vc4__write(ctx, k->code_off, words, bytes);
    memset(ctx->arena + k->code_off + bytes, 0, VC4_CODE_SLOT - bytes);
    memset(ctx->arena + k->uni_off, 0, VC4_UNI_SLOT);
    __sync_synchronize();

    *out = k;
    return VC4_OK;
}

/* Remember every distinct blob name, for the engine artifact's kernel list. */
static void note_seen(vc4_ctx *ctx, const char *name)
{
    if (ctx->nseen >= VC4_MAX_SEEN)
        return;
    for (int i = 0; i < ctx->nseen; ++i)
        if (strcmp(ctx->seen[i], name) == 0)
            return;
    snprintf(ctx->seen[ctx->nseen], sizeof(ctx->seen[0]), "%s", name);
    ctx->nseen++;
}

vc4_status vc4__kernel(vc4_ctx *ctx, const char *name, char *const gen_argv[],
                       int nqpu, vc4_kernel **out)
{
    note_seen(ctx, name);
    for (int i = 0; i < ctx->nkern; ++i) {
        if (strcmp(ctx->k[i].name, name) == 0) {
            if (nqpu > (int)ctx->k[i].nqpu) {
                vc4__set_err(ctx, "kernel %s was generated for %u QPUs, need %d",
                             name, ctx->k[i].nqpu, nqpu);
                return VC4_ERR_INVAL;
            }
            ctx->k[i].used_at = ++ctx->kern_clock;
            *out = &ctx->k[i];
            return VC4_OK;
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", ctx->cache_dir, name);

    uint8_t want[32];
    kernel_digest(gen_argv, want);

    struct stat st;
    int have = (stat(path, &st) == 0);
    if (have && !stamp_ok(path, want)) {
        /* an unstamped or stale blob must never run: a kernel from an older
         * generator can wedge the QPU, which is not recoverable from the ARM */
        vc4__set_err(ctx, "kernel %s: blob does not match the current "
                     "generator (kgen=%s), regenerating", name,
                     generator_path());
        unlink(path);
        have = 0;
    }
    if (!have) {
        if (!ctx->allow_jit) {
            vc4__set_err(ctx, "kernel %s not in the cache (or stale) and JIT "
                         "is disabled", name);
            return VC4_ERR_UNSUPPORTED;
        }
        vc4_status js = jit(ctx, gen_argv, path);
        if (js != VC4_OK)
            return js;
        stamp_write(path, want);
    }

    uint8_t *blob = NULL;
    size_t len = 0;
    vc4_status rs = blob_read(ctx, path, &blob, &len);
    if (rs != VC4_OK)
        return rs;
    rs = vc4__kernel_load_file(ctx, name, blob, len, nqpu, out);
    free(blob);
    return rs;
}
