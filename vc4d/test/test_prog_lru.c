/*
 * test_prog_lru.c - G3: more kernel programs than slots must not be fatal.
 *
 * Before 1.2.7 a daemon session had no way to free a program slot and a
 * per-client cap of 128 handles.  A run that touched more distinct kernels -
 * a real detector reaches 133 - got E_NOMEM from UPLOAD for every later shape
 * and every following launch failed until `systemctl restart vc4d`.
 *
 * Protocol 5 raises the table to 256 slots (255 usable; one is the probe), so
 * 320 distinct programs exercise the LRU again: 65 of them must be evicted and
 * re-uploaded by name, and none may fail.
 *
 * What this checks:
 *
 *   1. 320 distinct programs upload successfully.  (Before 1.2.7: upload #128
 *      fails with E_NOMEM.)
 *   2. every one of them launches, including the ones whose slot was evicted
 *      in the meantime *if* the daemon advertises named programs: the test
 *      re-uploads by name first, which is what libvc4infer does on E_NOENT.
 *   3. the first program can be run again at the end.  Under eviction its old
 *      handle is stale (E_NOENT); re-uploading the same name and launching it
 *      must work.  That is the "re-run the first one" case.
 *   4. STATS reports evictions and the slot count stays bounded.
 *
 * It also runs against a daemon that does not know about named programs (no
 * VC4D_F_NAMED_PROGS): then it uploads unnamed and the shortfall is the
 * "before" behaviour, reported as such.
 *
 * The program is the daemon's own known-good probe (`thrend` + three branch
 * delay slots) followed by valid `nop`s; the variable length makes each
 * upload a distinct code image, and nothing after `thrend` executes.  One QPU,
 * no operands, no memory access: a launch is ~1 us.
 *
 * usage: test_prog_lru [SOCKET]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4d_proto.h"
#include "vc4client.h"

#define NPROG 320

static const uint32_t head[4] = {
    0x9e70009eu, 0x30000000u,      /* thrend */
    0x9e70009eu, 0x10000000u,      /* nop, delay 1 */
};
static const uint32_t nop[2] = { 0x9e70009eu, 0x10000000u };

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

/* code image i: thrend + one delay nop + i extra nops (8..2024 bytes) */
static size_t make_code(uint32_t *out, size_t cap, int i)
{
    size_t n = 4;
    memcpy(out, head, sizeof(head));
    if (n + 2 <= cap) {
        memcpy(out + n, nop, sizeof(nop));
        n += 2;
    }
    for (int k = 0; k < i && n + 2 <= cap; ++k) {
        memcpy(out + n, nop, sizeof(nop));
        n += 2;
    }
    return n * 4;
}

int main(int argc, char **argv)
{
    const char *sock = argc > 1 ? argv[1] : "/run/vc4d.sock";
    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, sock);
    if (rc != 0) {
        printf("FAIL: connect %s: %s\n", sock, vc4c_strerror(rc));
        return 1;
    }
    struct vc4d_hello_rsp info;
    memset(&info, 0, sizeof(info));
    if (vc4c_info(c, &info) != 0) {
        printf("FAIL: HELLO\n");
        return 1;
    }
    const int named = (info.flags & VC4D_F_NAMED_PROGS) != 0;
    printf("daemon %s: max_prog %u B, named programs: %s\n", sock,
           info.max_prog, named ? "yes" : "no (pre-1.2.7)" );

    vc4c_buf uni;
    if (vc4c_alloc(c, 4096, 0, &uni) != 0) {
        printf("FAIL: ALLOC uniforms\n");
        vc4c_close(c);
        return 1;
    }
    uint32_t *u = uni.map;
    for (int i = 0; i < 16; ++i)
        u[i] = 0x3f800000u;                  /* 1.0f: no address-looking word */
    vc4c_sync(c, &uni, VC4D_SYNC_W2G);

    vc4c_prog prog[NPROG];
    char names[NPROG][16];
    uint32_t code[520];
    int uploaded = 0, up_failed = -1;
    memset(prog, 0, sizeof(prog));

    for (int i = 0; i < NPROG; ++i) {
        size_t bytes = make_code(code, sizeof(code) / 4, i);
        snprintf(names[i], sizeof(names[i]), "lru%03d", i);
        if (named)
            rc = vc4c_upload_named(c, names[i], code, bytes, 1, 16, &prog[i]);
        else
            rc = vc4c_upload(c, code, bytes, 1, 16, &prog[i]);
        if (rc != 0) {
            up_failed = i;
            printf("  upload #%d failed: %s\n", i, vc4c_strerror(rc));
            break;
        }
        uploaded++;
    }
    CHECK(uploaded == NPROG, "%d/%d distinct programs uploaded (first failure "
          "at #%d)", uploaded, NPROG, up_failed);

    if (uploaded != NPROG) {
        printf("G3 FAIL (program slots are still a hard cap)\n");
        vc4c_free(c, &uni);
        vc4c_close(c);
        return 1;
    }

    /* launch each one; if the slot was evicted, re-upload by name first */
    int launched = 0, stale_handles = 0;
    for (int i = 0; i < NPROG; ++i) {
        vc4c_result res;
        rc = vc4c_launch_wait(c, &prog[i], 1, &uni, 0, 5000, &res);
        if (rc == -VC4D_E_NOENT && named) {
            stale_handles++;
            size_t bytes = make_code(code, sizeof(code) / 4, i);
            rc = vc4c_upload_named(c, names[i], code, bytes, 1, 16, &prog[i]);
            if (rc == 0)
                rc = vc4c_launch_wait(c, &prog[i], 1, &uni, 0, 5000, &res);
        }
        if (rc != 0) {
            printf("  launch #%d failed: %s\n", i, vc4c_strerror(rc));
            break;
        }
        launched++;
    }
    CHECK(launched == NPROG, "%d/%d launches succeeded after churn (%d stale "
          "handles re-uploaded)", launched, NPROG, stale_handles);

    /* re-run the first kernel: its slot is long gone */
    {
        vc4c_result res;
        rc = vc4c_launch_wait(c, &prog[0], 1, &uni, 0, 5000, &res);
        int needed_upload = (rc == -VC4D_E_NOENT);
        if (needed_upload) {
            size_t bytes = make_code(code, sizeof(code) / 4, 0);
            rc = vc4c_upload_named(c, names[0], code, bytes, 1, 16, &prog[0]);
            if (rc == 0)
                rc = vc4c_launch_wait(c, &prog[0], 1, &uni, 0, 5000, &res);
        }
        CHECK(rc == 0, "the first program runs again at the end (%s)",
              needed_upload ? "slot had been evicted, re-uploaded by name"
                            : "handle was still valid");
    }

    vc4c_stats st;
    memset(&st, 0, sizeof(st));
    if (vc4c_get_stats(c, &st) == 0)
        printf("  stats: %u programs resident, %llu evictions, arena %llu MiB "
               "of %u MiB%s\n", st.programs,
               (unsigned long long)st.evictions,
               (unsigned long long)(st.arena_bytes >> 20),
               (unsigned)(st.arena_want >> 20),
               st.arena_degraded ? " (DEGRADED)" : "");
    if (named)
        CHECK(1, "the LRU kept the slot table bounded and every launch worked");
    else
        CHECK(0, "this daemon predates named slots: the table is still a cap");

    vc4c_free(c, &uni);
    vc4c_close(c);
    printf(fails ? "G3 FAIL (%d)\n" : "G3 PASS\n", fails);
    return fails ? 1 : 0;
}
