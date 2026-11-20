/*
 * test_daemon_recover.c - G1: a launch timeout must not leave vc4d dead.
 *
 * Before the fix one timed-out launch set the daemon's `dead` flag and it
 * stayed set: every later client failed with
 *     -12 daemon is dead (GPU timeout earlier)
 * until someone ran `systemctl restart vc4d`, although the QPU was fine.
 *
 * The timeout is produced the way the driver produces them in the field: a
 * legitimate, *bounded* kernel (a 256^3 GEMM, ~5 ms of QPU work on this board)
 * launched with a 1 ms deadline.  The kernel always terminates by itself, so
 * the daemon's reset finds the queue idle a few milliseconds later - this is
 * a transient timeout, not a wedged QPU, and it cannot wedge the board.
 *
 * Each round:
 *   1. a fresh client runs the control GEMM with a generous deadline  => OK
 *   2. the same client re-runs it with a 1 ms deadline              => TIMEOUT
 *   3. a *new* client runs the GEMM with a generous deadline: this is the
 *      proof that the daemon reset the context and is serving again (before
 *      the fix this failed with E_DEAD, and STATS.dead stayed 1)
 *
 * usage: test_daemon_recover SOCKET
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vc4infer.h"
#include "vc4d_proto.h"
#include "vc4client.h"

#define M 256
#define BYTES (M * M * 4)

static const char *sock = "/run/vc4d.sock";
static int fails;

#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* One engine client, as the CLI and the tests create them. */
static vc4_ctx *open_client(const char *tmo_ms)
{
    setenv("VC4INFER_DAEMON", "1", 1);
    setenv("VC4D_SOCKET", sock, 1);
    setenv("VC4INFER_TMO_MS", tmo_ms, 1);
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK)
        return NULL;
    return ctx;
}

static int run_gemm(vc4_ctx *ctx, vc4_buf *a, vc4_buf *b, vc4_buf *c,
                    double *ms, char *err, size_t errn)
{
    uint64_t t0 = now_ns();
    vc4_status st = vc4_matmul(ctx, a, b, c, M, M, M, VC4_LAYOUT_ROW_MAJOR);
    *ms = (double)(now_ns() - t0) / 1e6;
    if (st != VC4_OK)
        snprintf(err, errn, "%s (%s)", vc4_strerror(st), vc4_last_error(ctx));
    return st == VC4_OK ? 0 : -1;
}

static int stats_dead(unsigned *dead)
{
    vc4c_conn *c = NULL;
    if (vc4c_connect(&c, sock))
        return -1;
    struct vc4d_stats_rsp st;
    memset(&st, 0, sizeof(st));
    int rc = vc4c_get_stats(c, &st);
    vc4c_close(c);
    if (rc == 0 && dead)
        *dead = st.dead;
    return rc;
}

int main(int argc, char **argv)
{
    if (argc > 1)
        sock = argv[1];

    char err[512];
    double ms = 0;

    /* ---- round 0: control, with a generous deadline ------------------- */
    vc4_ctx *ctx = open_client("20000");
    if (!ctx) {
        printf("FAIL: cannot open a client on %s (is vc4d running?)\n", sock);
        return 1;
    }
    vc4_buf a, b, c;
    if (vc4_alloc(ctx, BYTES, &a) || vc4_alloc(ctx, BYTES, &b) ||
        vc4_alloc(ctx, BYTES, &c)) {
        /* Say what the client actually got: an arena with no data space is a
         * client-side sizing/ABI problem, not a daemon one (B12). */
        printf("FAIL: arena alloc of 3 x %d B: %s\n"
               "      client arena %.2f MiB (%zu B used), %s\n",
               BYTES, vc4_last_error(ctx),
               vc4_arena_bytes(ctx) / 1048576.0, vc4_arena_used(ctx),
               vc4_arena_bytes(ctx) <= (3u << 20)
                   ? "the grant has no data space - rebuild client and daemon "
                     "from the same tree" : "see the message above");
        return 1;
    }
    float *pa = calloc(M * M, 4), *pb = calloc(M * M, 4);
    for (int i = 0; i < M * M; ++i) {
        pa[i] = 1e-3f;
        pb[i] = 1e-3f;
    }
    vc4_upload(ctx, &a, pa, BYTES);
    vc4_upload(ctx, &b, pb, BYTES);
    CHECK(run_gemm(ctx, &a, &b, &c, &ms, err, sizeof(err)) == 0,
          "control GEMM with a 20 s deadline: %.1f ms", ms);

    for (int round = 1; round <= 3; ++round) {
        printf("-- timeout round %d\n", round);

        /* the same client, now with a 1 ms deadline: a real transient
         * timeout, because the kernel needs milliseconds */
        setenv("VC4INFER_TMO_MS", "1", 1);
        int rc = run_gemm(ctx, &a, &b, &c, &ms, err, sizeof(err));
        CHECK(rc != 0 && strstr(err, "timeout") != NULL,
              "1 ms deadline on a ~5 ms GEMM times out: %s", err);

        unsigned dead = 99;
        CHECK(stats_dead(&dead) == 0 && dead == 0,
              "daemon reports dead=%u after the timeout (0 = it reset and "
              "verified the context)", dead);

        vc4_shutdown(ctx);           /* the timed-out client gives up */

        /* a *new* client is the proof that launches work again */
        uint64_t t0 = now_ns();
        ctx = open_client("20000");
        double reopen_ms = (double)(now_ns() - t0) / 1e6;
        if (!ctx) {
            CHECK(0, "a new client can open after the timeout");
            break;
        }
        if (vc4_alloc(ctx, BYTES, &a) || vc4_alloc(ctx, BYTES, &b) ||
            vc4_alloc(ctx, BYTES, &c)) {
            CHECK(0, "the new client gets its arena: %s (arena %.2f MiB)",
                  vc4_last_error(ctx), vc4_arena_bytes(ctx) / 1048576.0);
            break;
        }
        vc4_upload(ctx, &a, pa, BYTES);
        vc4_upload(ctx, &b, pb, BYTES);
        rc = run_gemm(ctx, &a, &b, &c, &ms, err, sizeof(err));
        CHECK(rc == 0, "the next client's GEMM completes: %.1f ms "
              "(reconnect %.1f ms)", ms, reopen_ms);
        if (rc)
            break;
    }

    if (ctx)
        vc4_shutdown(ctx);
    printf(fails ? "G1 FAIL (%d)\n" : "G1 PASS\n", fails);
    return fails ? 1 : 0;
}
