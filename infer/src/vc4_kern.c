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

static vc4_status jit(vc4_ctx *ctx, char *const gen_argv[], const char *path)
{
    const char *kgen = getenv("VC4INFER_KGEN");
    if (!kgen)
        kgen = VC4_KGEN_DEFAULT;

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

vc4_status vc4__kernel_load_file(vc4_ctx *ctx, const char *name,
                                 const uint8_t *words, size_t bytes,
                                 int nqpu, vc4_kernel **out)
{
    if (ctx->remote)
        return vc4__remote_load(ctx, name, words, bytes, nqpu, out);
    if (ctx->nkern >= VC4_MAX_KERNELS) {
        vc4__set_err(ctx, "kernel cache full (%d entries)", VC4_MAX_KERNELS);
        return VC4_ERR_NOMEM;
    }
    vc4_kernel *k = &ctx->k[ctx->nkern];
    k->code_off = VC4_CODE_BASE + (uint32_t)ctx->nkern * VC4_CODE_SLOT;
    k->uni_off = VC4_UNI_BASE + (uint32_t)ctx->nkern * VC4_UNI_SLOT;
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

    ctx->nkern++;
    *out = k;
    return VC4_OK;
}

vc4_status vc4__kernel(vc4_ctx *ctx, const char *name, char *const gen_argv[],
                       int nqpu, vc4_kernel **out)
{
    for (int i = 0; i < ctx->nkern; ++i) {
        if (strcmp(ctx->k[i].name, name) == 0) {
            if (nqpu > (int)ctx->k[i].nqpu) {
                vc4__set_err(ctx, "kernel %s was generated for %u QPUs, need %d",
                             name, ctx->k[i].nqpu, nqpu);
                return VC4_ERR_INVAL;
            }
            *out = &ctx->k[i];
            return VC4_OK;
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", ctx->cache_dir, name);

    struct stat st;
    if (stat(path, &st) != 0) {
        if (!ctx->allow_jit) {
            vc4__set_err(ctx, "kernel %s not in the cache and JIT is disabled", name);
            return VC4_ERR_UNSUPPORTED;
        }
        vc4_status js = jit(ctx, gen_argv, path);
        if (js != VC4_OK)
            return js;
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
