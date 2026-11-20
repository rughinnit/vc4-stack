/*
 * arena_size.c - report what arena a vc4d is actually serving (G2).
 *
 * Connects, HELLO gives the arena bytes in use, STATS gives the size the
 * daemon asked for at startup and whether it is degraded.  The G2 test uses
 * the exit status to wait for a state:
 *
 *   0  arena == want           (full)
 *   3  arena <  want           (degraded)
 *   4  arena >  want           (unexpected)
 *   1  cannot talk to the daemon
 *
 * usage: arena_size [SOCKET] [WANT_MB]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4d_proto.h"
#include "vc4client.h"

int main(int argc, char **argv)
{
    const char *sock = argc > 1 ? argv[1] : "/run/vc4d.sock";
    unsigned long want_mb = argc > 2 ? strtoul(argv[2], NULL, 0) : 0;
    vc4c_conn *c = NULL;
    int rc = vc4c_connect(&c, sock);
    if (rc != 0) {
        printf("arena_size: connect %s: %s\n", sock, vc4c_strerror(rc));
        return 1;
    }
    struct vc4d_hello_rsp info;
    memset(&info, 0, sizeof(info));
    rc = vc4c_info(c, &info);
    if (rc != 0) {
        printf("arena_size: HELLO: %s\n", vc4c_strerror(rc));
        vc4c_close(c);
        return 1;
    }
    vc4c_stats st;
    memset(&st, 0, sizeof(st));
    rc = vc4c_get_stats(c, &st);
    vc4c_close(c);
    if (rc != 0) {
        /* A daemon older than 1.2.7 has a shorter STATS reply (no evictions /
         * want / degraded fields).  HELLO is enough to report the size, which
         * is what the "before" side of the G2 comparison needs. */
        printf("arena_size: arena=%llu MiB want=? MiB degraded=? (no STATS: %s)\n",
               (unsigned long long)(info.arena_bytes >> 20), vc4c_strerror(rc));
        if (!want_mb)
            return 0;
        unsigned long got_mb = (unsigned long)(info.arena_bytes >> 20);
        return got_mb < want_mb ? 3 : (got_mb > want_mb ? 4 : 0);
    }
    printf("arena_size: arena=%llu MiB want=%llu MiB degraded=%u flags=0x%x\n",
           (unsigned long long)(st.arena_bytes >> 20),
           (unsigned long long)(st.arena_want >> 20), st.arena_degraded,
           info.flags);
    if (!want_mb)
        return 0;
    unsigned long arena_mb = (unsigned long)(st.arena_bytes >> 20);
    if (arena_mb < want_mb)
        return 3;
    if (arena_mb > want_mb)
        return 4;
    return 0;
}
