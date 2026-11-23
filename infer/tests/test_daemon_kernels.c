/*
 * test_daemon_kernels.c - G3 from the engine's side: a session may touch more
 * than VC4_MAX_KERNELS (128) distinct kernels.
 *
 * Before 1.2.7 the cache was a fixed 128-entry array with no eviction, and the
 * daemon's program table had 127 usable slots, so the 129th kernel failed with
 * "kernel cache full (128 entries)" (or E_NOMEM from UPLOAD) and every shape
 * that was not already loaded failed afterwards.  A full YOLOv8 "util" run and
 * a long training session both reach that.
 *
 * Now both caches evict LRU-wise and the remote backend keeps each kernel's
 * code so a launch whose daemon slot was evicted is re-uploaded transparently.
 * This test loads NPROG=200 distinct kernels (the default is 128), launches
 * every one, then runs the first one again.
 *
 * usage: test_daemon_kernels [N]        (needs VC4INFER_DAEMON=1)
 */
#include "vc4_int.h"
#include "gpu_gate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPROG 200

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static const uint32_t head[4] = {
    0x9e70009eu, 0x30000000u,      /* thrend */
    0x9e70009eu, 0x10000000u,
};
static const uint32_t nop[2] = { 0x9e70009eu, 0x10000000u };

/* distinct code per index: thrend + delay nop + i extra nops */
static size_t make_code(uint32_t *out, size_t cap, int i)
{
    size_t n = 4;
    memcpy(out, head, sizeof(head));
    for (int k = 0; k < i + 1 && n + 2 <= cap; ++k) {
        memcpy(out + n, nop, sizeof(nop));
        n += 2;
    }
    return n * 4;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int nprog = argc > 1 ? atoi(argv[1]) : NPROG;
    if (nprog < 1 || nprog > 400)
        nprog = NPROG;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s (%s)\n", vc4_strerror(st),
               ctx ? vc4_last_error(ctx) : "");
        return 1;
    }
    if (!ctx->remote) {
        printf("FAIL: run with VC4INFER_DAEMON=1 (this is the daemon path)\n");
        vc4_shutdown(ctx);
        return 1;
    }
    printf("arena %.1f MiB, asking for %d distinct kernels (cache holds %d)\n",
           vc4_arena_bytes(ctx) / 1048576.0, nprog, VC4_MAX_KERNELS);

    uint32_t code[520];
    char name[96];
    int loaded = 0, launched = 0, load_failed = -1, launch_failed = -1;
    uint64_t ns = 0;

    for (int i = 0; i < nprog; ++i) {
        size_t bytes = make_code(code, sizeof(code) / 4, i);
        snprintf(name, sizeof(name), "lrk%04d", i);
        vc4_kernel *k = NULL;
        st = vc4__kernel_load_file(ctx, name, (const uint8_t *)code, bytes, 1,
                                   &k);
        if (st != VC4_OK) {
            load_failed = i;
            printf("  load #%d (%s) failed: %s (%s)\n", i, name,
                   vc4_strerror(st), vc4_last_error(ctx));
            break;
        }
        loaded++;
        st = vc4__launch(ctx, k, 1, &ns);
        if (st != VC4_OK) {
            launch_failed = i;
            printf("  launch #%d (%s) failed: %s (%s)\n", i, name,
                   vc4_strerror(st), vc4_last_error(ctx));
            break;
        }
        launched++;
    }
    CHECK(loaded == nprog, "%d/%d distinct kernels loaded (first failure at "
          "#%d)", loaded, nprog, load_failed);
    CHECK(launched == nprog, "%d/%d launched (first failure at #%d)",
          launched, nprog, launch_failed);

    /* the first one again: its cache entry was evicted long ago */
    {
        size_t bytes = make_code(code, sizeof(code) / 4, 0);
        vc4_kernel *k = NULL;
        vc4_status ls = vc4__kernel_load_file(ctx, "lrk0000",
                                              (const uint8_t *)code, bytes, 1,
                                              &k);
        if (ls == VC4_OK)
            ls = vc4__launch(ctx, k, 1, &ns);
        CHECK(ls == VC4_OK, "the first kernel runs again at the end (%s)",
              vc4_strerror(ls));
    }

    printf("  engine launches=%llu gpu_cycles=%llu\n",
           (unsigned long long)ctx->st.launches,
           (unsigned long long)ctx->st.gpu_cycles);
    vc4_shutdown(ctx);
    printf(fails ? "G3 ENGINE FAIL (%d)\n" : "G3 ENGINE PASS\n", fails);
    return fails ? 1 : 0;
}
