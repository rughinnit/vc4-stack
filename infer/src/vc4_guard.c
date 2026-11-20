/*
 * The unverified kernel gate.
 *
 * Four kernels in this tree have never been exercised by a test and are
 * confirmed to break the board:
 *
 *   vc4_gather   hands the TMU1 setup word an address taken from an
 *                accumulator (areg(100) encodes a different register) and its
 *                per-iteration load/pop does not match build_ew's 2-deep
 *                pipeline.  Both candidate fixes still hang the QPU.
 *   vc4_reduce   walks a fixed 256-output group, so it reads and writes past
 *                the caller's `outer` elements (a TMU read past the end of the
 *                arena hangs the board), and it seeds register 36 (r4) instead
 *                of r0.
 *   vc4_blit     returns VC4_OK and copies nothing; the 4-vector store path it
 *                uses (build_ew with nvec != 16) writes no output either.
 *   the uop/ewb nvec=4 store path
 *                same: no output is written, so a caller silently gets stale
 *                data instead of a result.
 *
 * Until each one is fixed it fails fast here, so no caller can launch it by
 * accident.  VC4_ALLOW_UNVERIFIED_KERNELS=1 re-enables them for the person
 * fixing them (they can then hang the board - that is the point of the gate).
 */
#include "vc4_int.h"

#include <stdlib.h>
#include <string.h>

int vc4__unverified_allowed(void)
{
    const char *e = getenv("VC4_ALLOW_UNVERIFIED_KERNELS");
    return e && *e && strcmp(e, "0") != 0;
}

vc4_status vc4__refuse_unverified(vc4_ctx *ctx, const char *what,
                                  const char *why)
{
    if (vc4__unverified_allowed())
        return VC4_OK;
    vc4__set_err(ctx, "%s is not verified on this board: %s.  It is refused "
                 "rather than launched (set VC4_ALLOW_UNVERIFIED_KERNELS=1 to "
                 "override while fixing it).", what, why);
    return VC4_ERR_UNSUPPORTED;
}
