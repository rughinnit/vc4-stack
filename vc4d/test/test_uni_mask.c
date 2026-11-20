/*
 * test_uni_mask.c - G4: a float whose bits look like a bus address.
 *
 * Every fp32 value <= -2.0f has a bit pattern >= 0xC0000000, which is the GPU
 * bus window.  vc4d validates the uniform words a client materialises in its
 * own arena, and before protocol v2.0.1 the only rule available was "a word
 * that is not inside this arena may not look like an address", so -2.0f,
 * -3.0f, -inf and every other such value were refused with
 * `DENY ... arena uniform word is outside the arena` (G4).
 *
 * The fix is VC4D_LAUNCH_UNI_MASK: the launcher declares, per uniform slot,
 * which words are GPU bus addresses.  A declared address must be inside the
 * client's arena; a word without the bit is a value and is not compared with
 * any address window.  A client that sends no flag keeps the old rule.
 *
 * The program is the daemon's known-good `thrend` probe (one QPU, no operands,
 * ~12 cycles), so every accepted launch really runs on the QPU and the test
 * proves the *acceptance*, not just the parse.  Values used:
 *     -2.0f = 0xC0000000   -3.5f = 0xC0600000   -inf = 0xFF800000
 *
 * usage: test_uni_mask [SOCKET]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4d_proto.h"
#include "vc4client.h"

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static const uint32_t probe_code[8] = {
    0x9e70009eu, 0x30000000u,      /* thrend */
    0x9e70009eu, 0x10000000u,      /* nop (delay 1) */
    0x9e70009eu, 0x10000000u,      /* nop (delay 2) */
    0x9e70009eu, 0x10000000u,      /* nop (delay 3) */
};

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
    const int have_mask = (info.flags & VC4D_F_UNI_MASK) != 0;
    printf("daemon %s: uniform address mask: %s\n", sock,
           have_mask ? "yes" : "no (pre-1.2.7)");

    vc4c_buf uni;
    if (vc4c_alloc(c, 4096, VC4D_ALLOC_ZERO, &uni) != 0) {
        printf("FAIL: ALLOC\n");
        vc4c_close(c);
        return 1;
    }
    vc4c_prog p;
    if (vc4c_upload(c, probe_code, sizeof(probe_code), 1, 16, &p) != 0) {
        printf("FAIL: UPLOAD probe\n");
        vc4c_free(c, &uni);
        vc4c_close(c);
        return 1;
    }

    uint32_t *u = uni.map;
    vc4c_result res;

    /* ---- 1. no flag: the old rule still applies ---------------------- */
    for (int i = 0; i < 16; ++i)
        u[i] = 0x3f800000u;
    u[0] = 0xc0000000u;                        /* -2.0f */
    vc4c_sync(c, &uni, VC4D_SYNC_W2G);
    rc = vc4c_launch_wait(c, &p, 1, &uni, 0, 5000, &res);
    CHECK(rc == -VC4D_E_DENIED,
          "no mask: a word that looks like an address is still refused (%s)",
          vc4c_strerror(rc));

    /* ---- 2. the mask: these are values, not addresses ----------------- */
    u[0] = 0xc0000000u;                        /* -2.0f  */
    u[1] = 0xc0600000u;                        /* -3.5f  */
    u[2] = 0xff800000u;                        /* -inf   */
    u[3] = 0xbf800000u;                        /* -1.0f (always worked) */
    for (int i = 4; i < 16; ++i)
        u[i] = 0x3f800000u;
    vc4c_sync(c, &uni, VC4D_SYNC_W2G);
    if (!have_mask) {
        CHECK(0, "daemon does not support VC4D_LAUNCH_UNI_MASK");
    } else {
        rc = vc4c_launch_wait_u(c, &p, 1, &uni, 0, 5000, &res, 0);
        CHECK(rc == 0, "mask=0 (all values): -2.0f/-3.5f/-inf launch (%s)",
              vc4c_strerror(rc));
        if (rc == 0)
            CHECK(res.errs == 0, "the launch raised no V3D error (errs=0x%x)",
                  res.errs);
    }

    /* ---- 3. a declared address inside the arena is accepted ------------ */
    uint32_t arena_addr = uni.bus;              /* this buffer's bus address */
    u[0] = arena_addr;
    vc4c_sync(c, &uni, VC4D_SYNC_W2G);
    if (have_mask) {
        rc = vc4c_launch_wait_u(c, &p, 1, &uni, 0, 5000, &res, 1u);
        CHECK(rc == 0, "mask=bit0: a declared address in the arena is accepted "
              "(%s)", vc4c_strerror(rc));

        /* ---- 4. a declared address outside it is still refused --------- */
        u[0] = 0xc0000000u;                    /* looks like an address, is -2.0f */
        vc4c_sync(c, &uni, VC4D_SYNC_W2G);
        rc = vc4c_launch_wait_u(c, &p, 1, &uni, 0, 5000, &res, 1u);
        CHECK(rc == -VC4D_E_DENIED,
              "mask=bit0 with a word outside the arena: refused (%s)",
              vc4c_strerror(rc));

        /* ---- 5. a mask bit past nuni is a protocol error ---------------- */
        rc = vc4c_launch_wait_u(c, &p, 1, &uni, 0, 5000, &res, 1u << 20);
        CHECK(rc == -VC4D_E_INVAL, "mask with a bit past nuni: E_INVAL (%s)",
              vc4c_strerror(rc));
    }

    vc4c_free(c, &uni);
    vc4c_close(c);
    printf(fails ? "G4 FAIL (%d)\n" : "G4 PASS\n", fails);
    return fails ? 1 : 0;
}
