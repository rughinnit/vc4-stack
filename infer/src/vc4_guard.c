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
 * Scalar parameters travel in the uniform block, and up to 1.2.6 this file
 * refused any parameter whose fp32 bit pattern was >= 0xC0000000, because vc4d
 * could not tell such a word from a GPU bus address (its literal_forbidden()).
 * Every float <= -2.0f lands in that window (-2.0f is 0xC0000000, -3.5f is
 * 0xC0600000), so Pow(x, -2), a Clip limit of -3 and a LeakyRelu slope of -2
 * were refused in managed mode while working in direct mode (G4).
 *
 * That refusal is gone: protocol v2 gained VC4D_LAUNCH_UNI_MASK, a per-slot
 * mask in which the launcher declares which uniform words are addresses.  The
 * remote backend sends it on every launch (vc4_dev_remote.c: remote_uni_mask),
 * so a value word is never compared with the bus window and the daemon still
 * range-checks every real address.  See docs/VC4D.md and docs/BACKENDS.md.
 */
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
