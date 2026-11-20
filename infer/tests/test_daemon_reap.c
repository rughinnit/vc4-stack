/*
 * test_daemon_reap.c - the liveness sweep must not unbalance the poll loop.
 *
 * The daemon builds its pollfd table from the client table, then serves the
 * ready slots one by one.  A HELLO (and an accept that has to free a slot)
 * runs the liveness sweep *inside* that loop, so it can destroy a client whose
 * slot is still to come.  The loop used to hold `struct client *`, which then
 * dangled; with an out-of-range/garbage index it could index `clients[]` out of
 * bounds.
 *
 * This drives that interleaving on purpose: half the connections are held open
 * by a forked child after the allocating process exits (so they are reapable),
 * and the other half keep sending HELLO/ALLOC/FREE/STATS, so a sweep happens
 * while reapable slots are still queued in the same poll round.
 *
 * usage: test_daemon_reap [SOCKET] [SECONDS]
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

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* connect, allocate, then leave the connection to a sleeping child */
static int spawn_orphan(void)
{
    pid_t p = fork();
    if (p != 0) {
        int ws = 0;
        waitpid(p, &ws, 0);
        return WIFEXITED(ws) ? WEXITSTATUS(ws) : 1;
    }
    vc4c_conn *c = NULL;
    if (vc4c_connect(&c, sock))
        _exit(2);
    vc4c_buf b;
    if (vc4c_alloc(c, 1u << 20, 0, &b))
        _exit(3);
    if (vc4c_ping(c))
        _exit(4);
    pid_t holder = fork();
    if (holder == 0) {
        usleep(400 * 1000);          /* keep the socket, do no work */
        _exit(0);
    }
    _exit(0);
}

int main(int argc, char **argv)
{
    if (argc > 1)
        sock = argv[1];
    double secs = argc > 2 ? atof(argv[2]) : 5.0;

    vc4c_conn *live[4] = { NULL };
    for (int i = 0; i < 4; ++i) {
        if (vc4c_connect(&live[i], sock)) {
            printf("FAIL: connect %d\n", i);
            return 1;
        }
        vc4c_buf b;
        if (vc4c_alloc(live[i], 4096, 0, &b)) {
            printf("FAIL: alloc on live %d\n", i);
            return 1;
        }
    }

    int rounds = 0, died = 0;
    double t0 = now_s();
    while (now_s() - t0 < secs) {
        /* a batch of orphanable connections, then traffic on the live ones */
        for (int i = 0; i < 3; ++i) {
            if (spawn_orphan())
                died++;
        }
        for (int i = 0; i < 4; ++i) {
            /* HELLO is the op that sweeps from inside the poll loop */
            if (vc4c_ping(live[i])) {
                died++;
                break;
            }
            struct vc4d_stats_rsp st;
            if (vc4c_get_stats(live[i], &st)) {
                died++;
                break;
            }
            vc4c_buf b;
            if (vc4c_alloc(live[i], 64 << 10, 0, &b) == 0)
                vc4c_free(live[i], &b);
        }
        rounds++;
    }

    /* the daemon must still answer, with every region of ours released */
    for (int i = 0; i < 4; ++i)
        vc4c_close(live[i]);
    vc4c_conn *probe = NULL;
    int rc = vc4c_connect(&probe, sock);
    uint64_t used = 1;
    if (rc == 0) {
        double t1 = now_s();
        do {
            struct vc4d_stats_rsp st;
            memset(&st, 0, sizeof(st));
            if (vc4c_get_stats(probe, &st) == 0)
                used = st.arena_used;
            if (!used)
                break;
            usleep(50 * 1000);
        } while (now_s() - t1 < 3.0);
        vc4c_close(probe);
    }

    int fails = 0;
    if (rc)
        fails++;
    printf("  %s  %d rounds of orphan+HELLO traffic, daemon %s\n",
           (!rc && !died) ? "PASS" : "FAIL", rounds,
           rc ? "is GONE" : "answered");
    if (died)
        printf("  %s  %d requests failed during the run\n",
               died ? "FAIL" : "PASS", died);
    printf("  %s  arena in use after the run: %llu B\n",
           used == 0 ? "PASS" : "FAIL", (unsigned long long)used);
    if (used)
        fails++;

    printf(fails ? "REAP FAIL (%d)\n" : "REAP PASS\n", fails);
    return fails ? 1 : 0;
}
