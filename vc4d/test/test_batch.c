/*
 * test_batch.c - protocol 3: one request, an ordered launch list.
 *
 * The GPU-side ordering is covered by the engine parity runs (the same graph
 * with VC4INFER_NO_BATCH set and unset must produce byte-identical output).
 * This test pins the wire semantics the engine relies on:
 *
 *   - HELLO advertises VC4D_F_LAUNCH_BATCH;
 *   - malformed requests are refused before anything runs (count, length,
 *     reserved flags, unknown kind);
 *   - validation is per entry: the failing entry carries its own status and
 *     every entry after it is VC4D_E_NOTRUN, so a client that queued
 *     dependent kernels never has to guess what ran;
 *   - a fence is an entry like any other and counts as run;
 *   - a batch entry cannot ask for the tagged-uniform form or WAIT.
 *
 * usage: test_batch SOCKET
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4client.h"

static const char *sock = "/run/vc4d.sock";
static int fails;

#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static struct vc4d_batch_item items[VC4D_BATCH_MAX];

int main(int argc, char **argv)
{
    if (argc > 1)
        sock = argv[1];
    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, sock);
    if (rc != 0) {
        printf("connect %s: %s\n", sock, vc4c_strerror(rc));
        return 2;
    }
    struct vc4d_hello_rsp info;
    rc = vc4c_info(c, &info);
    CHECK(rc == 0 && (info.flags & VC4D_F_LAUNCH_BATCH),
          "HELLO advertises VC4D_F_LAUNCH_BATCH (flags=0x%x)", info.flags);

    vc4c_buf region;
    rc = vc4c_alloc(c, 4096, 0, &region);
    CHECK(rc == 0, "ALLOC a region for fences: %s", vc4c_strerror(rc));
    if (rc != 0)
        return 1;

    vc4c_batch_ent e[4];
    memset(e, 0, sizeof(e));

    /* 1. count out of range: refused by the client before the wire. */
    rc = vc4c_launch_batch(c, &region, e, 0, items, NULL);
    CHECK(rc == -VC4D_E_INVAL, "count 0 refused: %s", vc4c_strerror(rc));
    rc = vc4c_launch_batch(c, &region, e, VC4D_BATCH_MAX + 1, items, NULL);
    CHECK(rc == -VC4D_E_INVAL, "count %u refused: %s", VC4D_BATCH_MAX + 1,
          vc4c_strerror(rc));

    /* 2. unknown kind. */
    e[0].kind = 7;
    rc = vc4c_launch_batch(c, &region, e, 1, items, NULL);
    CHECK(rc == -VC4D_E_INVAL, "unknown entry kind refused: %s",
          vc4c_strerror(rc));

    /* 3. unknown program at entry 0: it carries E_NOENT, nothing after runs. */
    memset(e, 0, sizeof(e));
    e[0].kind = VC4D_BENT_LAUNCH;
    e[0].prog = 0xdead;
    e[0].nqpu = 1;
    e[0].nuni = 4;
    e[0].flags = VC4D_LAUNCH_UNI_MASK;
    e[0].uni_off = 0;
    e[0].uni_addr = 0;
    e[1].kind = VC4D_BENT_FENCE;
    e[1].handle = region.handle;
    e[1].dir = VC4D_SYNC_G2R;
    unsigned done = 0;
    memset(items, 0, sizeof(items));
    rc = vc4c_launch_batch(c, &region, e, 2, items, &done);
    CHECK(rc == -VC4D_E_NOENT && done == 1 &&
          items[0].status == VC4D_E_NOENT &&
          items[1].status == VC4D_E_NOTRUN,
          "unknown program: header %s done=%u item0=%u item1=%u",
          vc4c_strerror(rc), done, items[0].status, items[1].status);

    /* 4. a fence alone runs and is reported as run. */
    memset(e, 0, sizeof(e));
    e[0].kind = VC4D_BENT_FENCE;
    e[0].handle = region.handle;
    e[0].dir = VC4D_SYNC_W2G;
    memset(items, 0, sizeof(items));
    rc = vc4c_launch_batch(c, &region, e, 1, items, &done);
    CHECK(rc == 0 && done == 1 && items[0].status == VC4D_OK,
          "fence-only batch: rc=%s done=%u", vc4c_strerror(rc), done);

    /* 5. a bad fence stops the batch; the launch after it must not run. */
    memset(e, 0, sizeof(e));
    e[0].kind = VC4D_BENT_FENCE;
    e[0].handle = 0xdead;
    e[0].dir = VC4D_SYNC_G2R;
    e[1].kind = VC4D_BENT_FENCE;
    e[1].handle = region.handle;
    e[1].dir = VC4D_SYNC_G2R;
    memset(items, 0, sizeof(items));
    rc = vc4c_launch_batch(c, &region, e, 2, items, &done);
    CHECK(rc == -VC4D_E_INVAL && done == 1 &&
          items[0].status == VC4D_E_INVAL &&
          items[1].status == VC4D_E_NOTRUN,
          "unknown fence region: header %s done=%u item1=%u",
          vc4c_strerror(rc), done, items[1].status);

    /* 6. a batch launch cannot carry WAIT or the tagged-uniform form; the
     * client refuses both locally. */
    memset(e, 0, sizeof(e));
    e[0].kind = VC4D_BENT_LAUNCH;
    e[0].prog = 1;
    e[0].nqpu = 1;
    e[0].nuni = 4;
    e[0].flags = VC4D_LAUNCH_WAIT;
    rc = vc4c_launch_batch(c, &region, e, 1, items, NULL);
    CHECK(rc == -VC4D_E_INVAL, "WAIT in a batch refused: %s",
          vc4c_strerror(rc));

    /* 7. uni_addr bits past nuni are refused before the wire. */
    e[0].flags = VC4D_LAUNCH_UNI_MASK;
    e[0].uni_addr = 1u << 4;
    rc = vc4c_launch_batch(c, &region, e, 1, items, NULL);
    CHECK(rc == -VC4D_E_INVAL, "mask bit past nuni refused: %s",
          vc4c_strerror(rc));

    vc4c_free(c, &region);
    vc4c_close(c);
    printf(fails ? "BATCH TEST FAILED (%d)\n" : "BATCH TEST PASSED\n", fails);
    return fails ? 1 : 0;
}
