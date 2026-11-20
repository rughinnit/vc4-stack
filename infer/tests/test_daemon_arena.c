/*
 * test_daemon_arena.c - G9: the arena must come back when a client's process
 * is gone, even if its socket is not.
 *
 * Three cases, all against one daemon:
 *
 *   A. churn: N short-lived clients, half of which exit without FREE at all.
 *      The daemon must reclaim every region; after each batch STATS shows
 *      arena_used == 0 and every client got its allocation.
 *
 *   B. orphan: a client ALLOCs, forks a child that inherits the connection and
 *      sleeps, then the parent (the pid recorded at connect time) exits.  The
 *      daemon used to keep the region until that *socket* closed, so it was
 *      pinned by a process that no longer existed.  The next client must be
 *      able to use the arena: with a small --arena-mb the region is most of it.
 *
 *   C. after everything is gone, arena_used must be exactly 0 - the property a
 *      free list with a hard cap could not offer.
 *
 * usage: test_daemon_arena [SOCKET]
 *
 * Needs a daemon whose arena is small enough for B to be provable
 * (run_tests.sh starts one with --arena-mb 16); with a big arena B still runs
 * but proves less.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

#include "vc4d_proto.h"
#include "vc4client.h"

static const char *sock = "/run/vc4d.sock";
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

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int stats_used(vc4c_conn **keep, uint64_t *used, uint32_t *clients)
{
    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, sock);
    if (rc)
        return rc;
    struct vc4d_stats_rsp st;
    memset(&st, 0, sizeof(st));
    rc = vc4c_get_stats(c, &st);
    if (rc == 0) {
        *used = st.arena_used;
        if (clients)
            *clients = st.clients;
    }
    if (keep)
        *keep = c;               /* caller closes it */
    else
        vc4c_close(c);
    return rc;
}

/* One short-lived client: connect, ALLOC hold_mb, (maybe) FREE, close. */
static int short_client(size_t bytes, int do_free, int *_rc)
{
    pid_t p = fork();
    if (p < 0)
        return -1;
    if (p == 0) {
        vc4c_conn *c = NULL;
        if (vc4c_connect(&c, sock))
            _exit(2);
        vc4c_buf b;
        int rc = vc4c_alloc(c, bytes, 0, &b);
        if (rc == 0 && do_free)
            vc4c_free(c, &b);
        vc4c_close(c);
        _exit(rc == 0 ? 0 : 3);
    }
    int status = 0;
    if (waitpid(p, &status, 0) != p)
        return -1;
    int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (_rc)
        *_rc = ok ? 0 : 1;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1)
        sock = argv[1];
    uint64_t used = 0;
    uint32_t clients = 0;
    if (stats_used(NULL, &used, &clients)) {
        printf("FAIL: connect %s\n", sock);
        return 1;
    }
    printf("daemon %s: %u client(s), %llu B in use at start\n", sock,
           (unsigned)clients, (unsigned long long)used);
    CHECK(used == 0, "arena is empty before the test (%llu B)",
          (unsigned long long)used);

    /* ---- A. churn ---------------------------------------------------- */
    const int N = 24;
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        int rc = 0;
        /* a 1 MiB region; every third client exits without FREE */
        short_client(1u << 20, (i % 3) != 0, &rc);
        if (rc)
            bad++;
    }
    CHECK(bad == 0, "%d short-lived clients all got their allocation "
          "(%d failed)", N, bad);
    used = 1;
    stats_used(NULL, &used, &clients);
    CHECK(used == 0, "after the churn the daemon has reclaimed everything "
          "(arena_used=%llu B)", (unsigned long long)used);
    CHECK(clients <= 1, "no stale client entries (%u besides this probe)",
          (unsigned)clients);

    /* ---- B. orphaned connection -------------------------------------- */
    /* The carrier asks for a region, hands the connection to a child and
     * exits.  The daemon's peer pid is the parent's, so the region must be
     * reclaimed while the socket is still open in the child. */
    int pipes[2];
    if (pipe(pipes)) {
        perror("pipe");
        return 1;
    }
    pid_t carrier = fork();
    if (carrier == 0) {
        vc4c_conn *c = NULL;
        if (vc4c_connect(&c, sock))
            _exit(2);
        vc4c_buf b;
        int rc = vc4c_alloc(c, 8u << 20, 0, &b);
        if (rc)
            _exit(3);
        /* tell the parent the allocation succeeded, then exit with the
         * connection still referenced by a sleeping child */
        if (write(pipes[1], "x", 1) != 1)
            _exit(4);
        pid_t holder = fork();
        if (holder == 0) {
            sleep(8);                 /* keeps c->fd open, does no GPU work */
            _exit(0);
        }
        _exit(0);
    }
    char ch = 0;
    (void)!read(pipes[0], &ch, 1);
    int wst = 0;
    waitpid(carrier, &wst, 0);
    printf("-- orphan: carrier allocated 8 MiB and exited (holder child "
           "keeps the socket for 8 s)\n");

    vc4c_conn *c = NULL;
    if (vc4c_connect(&c, sock)) {
        printf("FAIL: connect\n");
        return 1;
    }
    /* The daemon sweeps once per poll iteration (<= 1 s); retry for 5 s. */
    int got = 0;
    double t0 = now_s();
    for (int tries = 0; tries < 100 && !got; ++tries) {
        vc4c_buf b[8];
        int n = 0;
        for (int i = 0; i < 8; ++i)
            if (vc4c_alloc(c, 4u << 20, 0, &b[n]) == 0)
                n++;
        if (n >= 3) {                 /* 8 MiB pinned would leave < 3 x 4 MiB
                                         only if the base arena is ~16 MiB */
            got = n;
            break;
        }
        usleep(50 * 1000);
    }
    CHECK(got >= 3, "the next client can use the arena the dead carrier held "
          "(%d x 4 MiB after %.1f s)", got, now_s() - t0);
    vc4c_close(c);

    /* ---- C. exact accounting ----------------------------------------- */
    double t1 = now_s();
    do {
        stats_used(NULL, &used, &clients);
        if (!used)
            break;
        usleep(50 * 1000);
    } while (now_s() - t1 < 5.0);
    CHECK(used == 0, "arena_used is exactly 0 with every client gone "
          "(%llu B)", (unsigned long long)used);

    printf(fails ? "G9 FAIL (%d)\n" : "G9 PASS\n", fails);
    return fails ? 1 : 0;
}
