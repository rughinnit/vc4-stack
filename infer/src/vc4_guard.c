/*
 * The unverified kernel gate.
 *
 * These kernels are reachable from the public API, have never been exercised
 * by a test, and are confirmed to break the board.  Each one either hangs the
 * QPU (which kills the vc4d context for every later client on the machine) or
 * silently returns wrong results:
 *
 *   vc4_gather   hands the TMU1 setup word an address taken from an
 *                accumulator (areg(100) encodes a different register) and its
 *                per-iteration load/pop does not match build_ew's 2-deep
 *                pipeline.  Both candidate fixes still hang the QPU.
 *   vc4_reduce   walks a fixed 256-output group, so it reads and writes past
 *                the caller's `outer` elements (a TMU read past the end of the
 *                arena hangs the board), and it seeds register 36 (r4) instead
 *                of r0.  Measured 2026-11-20, one launch per process: every
 *                shape, including outer=256 red=16, kills the context.
 *   vc4_cumsum   uses the same TMU setup as vc4_gather; every shape kills the
 *                context (measured 2026-11-20, one launch per process).
 *   vc4_select3  WHERE returns VC4_OK with every element wrong (the
 *                pad_stream/TMU lane pattern it shares with vc4_reduce).
 *   vc4_blit     returns VC4_OK and copies nothing; the 4-vector store path it
 *                uses (build_ew with nvec != 16) writes no output either.
 *   the ewb nvec=4 store path
 *                the unary (uop) path was fixed and verified on 2026-11-20
 *                (docs/NUMERICS.md: 4q write row, B8 constant exclusion); the
 *                backward bodies still need the same verification.
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

/*
 * Scalar parameters travel in the uniform block, and vc4d denies any uniform
 * word >= 0xC0000000 because such a word may name a GPU bus address (its
 * literal_forbidden()).  Every float <= -2.0f lands in that window: -2.0f is
 * 0xC0000000, -3.5f is 0xC0600000, -inf is 0xFF800000.  So Pow(x, -2), a Clip
 * limit of -3 or a LeakyRelu slope of -2 are legal in direct mode and
 * impossible through the daemon.
 *
 * The daemon cannot tell a value from an address in the materialised-uniform
 * (UNI_IN_ARENA) launch the engine uses - there is no per-word kind - so the
 * engine refuses with the real reason here, instead of letting the user read
 * "uniform is a bus/arena/MMIO address".  See docs/RELEASE.md for the
 * proposed protocol change (a per-word value flag); the engine has no
 * directory of "which uniform word is an address" to send today.
 */
vc4_status vc4__check_uniform_value(vc4_ctx *ctx, const char *what, float v)
{
    uint32_t b;
    if (!ctx->remote)
        return VC4_OK;                  /* direct /dev/mem has no such rule */
    memcpy(&b, &v, 4);
    if (b < 0xC0000000u)
        return VC4_OK;
    vc4__set_err(ctx,
                 "%s: parameter %g has bit pattern 0x%08x, which vc4d refuses "
                 "as a possible GPU bus address (every float <= -2.0 is in "
                 "0xC0000000..0xFFFFFFFF).  The value cannot be carried in a "
                 "uniform through the daemon; run with VC4INFER_DAEMON=0, or "
                 "avoid parameters at or below -2.0 on this path.",
                 what, (double)v, b);
    return VC4_ERR_UNSUPPORTED;
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
