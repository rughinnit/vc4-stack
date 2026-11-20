/*
 * test_daemon_recover.c - G1: a launch timeout must not leave vc4d dead.
 *
 * Before the fix, one timed-out launch set the daemon's `dead` flag and it
 * stayed set: every later client failed with
 *     -12 daemon is dead (GPU timeout earlier)
 * until someone ran `systemctl restart vc4d`.  The QPU itself was usually
 * fine (SRQCS back to 0, ERRSTAT back to 0x1000 VCD-idle).
 *
 * This test forces a *real* timeout against a real daemon and then asks the
 * same daemon to run a known-good launch:
 *
 *   1. upload a program with no `thrend` signal (two nops; the daemon zeroes
 *      the rest of the slot, so the QPU hits sig=0 "break" and the queue
 *      never completes) and launch it with a 100 ms timeout  => E_TIMEOUT;
 *   2. STATS must report dead == 0: the context was reset and verified;
 *   3. the 4-instruction probe program must launch, complete with no error
 *      bits and report cycles  => the daemon is serving again;
 *   4. steps 1-3 twice more: recovery is a repeated capability, not a
 *      one-shot.
 *
 * The timeout program halts itself (sig=BREAK), so it cannot leave the QPU
 * spinning: a wedged window here would already be a wedged board.
 *
 * usage: test_daemon_recover [SOCKET]      (default /run/vc4d.sock)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vc4d_proto.h"
#include "vc4client.h"

/* thrend + three delay nops: the tail every kgen kernel ends with. 1 QPU,
 * no operands, ~1 us. */
static const uint32_t good_prog[8] = {
    0x9e70009eu, 0x30000000u,
    0x9e70009eu, 0x10000000u,
    0x9e70009eu, 0x10000000u,
    0x9e70009eu, 0x10000000u,
};
/* two nops and no end signal: the queue never reports completion */
static const uint32_t hang_prog[4] = {
    0x9e70009eu, 0x10000000u,
    0x9e70009eu, 0x10000000u,
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) {                                                    \
            printf("  PASS  ");                                        \
        } else {                                                       \
            printf("  FAIL  ");                                        \
            fails++;                                                   \
        }                                                              \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

int main(int argc, char **argv)
{
    const char *sock = argc > 1 ? argv[1] : "/run/vc4d.sock";
    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, sock);
    if (rc) {
        printf("FAIL: connect %s: %s\n", sock, vc4c_strerror(rc));
        return 1;
    }
    struct vc4d_hello_rsp info;
    if (vc4c_info(c, &info)) {
        printf("FAIL: HELLO\n");
        return 1;
    }
    printf("daemon %s: arena %u B, ident0 0x%08x\n", sock, info.arena_bytes,
           info.ident0);

    vc4c_buf uni;
    rc = vc4c_alloc(c, 4096, VC4D_ALLOC_ZERO, &uni);
    if (rc) {
        printf("FAIL: alloc: %s\n", vc4c_strerror(rc));
        return 1;
    }
    vc4c_prog hang, good;
    if ((rc = vc4c_upload(c, hang_prog, sizeof(hang_prog), 1, 1, &hang)) ||
        (rc = vc4c_upload(c, good_prog, sizeof(good_prog), 1, 1, &good))) {
        printf("FAIL: upload: %s\n", vc4c_strerror(rc));
        return 1;
    }

    /* A known-good launch must work *before* the timeout, so a failure later
     * cannot be blamed on the program. */
    vc4c_result r;
    rc = vc4c_launch_wait(c, &good, 1, &uni, 0, 1000, &r);
    CHECK(rc == 0 && r.errs == 0,
          "control launch before the timeout: rc=%d (%s) errs=0x%x",
          rc, vc4c_strerror(rc), r.errs);
    if (rc)
        return 1;

    for (int round = 1; round <= 3; ++round) {
        printf("-- timeout round %d\n", round);
        memset(&r, 0, sizeof(r));
        rc = vc4c_launch_wait(c, &hang, 1, &uni, 0, 100, &r);
        CHECK(rc == -VC4D_E_TIMEOUT,
              "launch without an end signal times out: rc=%d (%s)",
              rc, vc4c_strerror(rc));
        if (rc == -VC4D_E_TIMEOUT)
            printf("        (client sees %s, as it should)\n",
                   vc4c_strerror(rc));

        struct vc4d_stats_rsp st;
        memset(&st, 0, sizeof(st));
        rc = vc4c_get_stats(c, &st);
        CHECK(rc == 0 && st.dead == 0,
              "daemon reports dead=%u after the timeout (0 = recovered)",
              st.dead);

        /* The next launch is the proof: it has to complete on the reset
         * context, with no new error bits. */
        uint64_t t0 = now_ns();
        memset(&r, 0, sizeof(r));
        rc = vc4c_launch_wait(c, &good, 1, &uni, 0, 2000, &r);
        double ms = (double)(now_ns() - t0) / 1e6;
        CHECK(rc == 0 && r.errs == 0,
              "known-good launch after recovery: rc=%d (%s) errs=0x%x "
              "cycles=%u in %.1f ms", rc, vc4c_strerror(rc), r.errs,
              r.cycles, ms);
        if (rc)
            break;
    }

    vc4c_close(c);
    printf(fails ? "G1 FAIL (%d)\n" : "G1 PASS\n", fails);
    return fails ? 1 : 0;
}
