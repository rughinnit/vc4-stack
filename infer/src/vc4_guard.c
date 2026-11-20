/* The unverified-kernel gate.
 *
 * This used to refuse six entry points.  All of them are now verified on
 * silicon, so no call site uses it: vc4_blit rides copy2d, and
 * vc4_gather/vc4_reduce/vc4_cumsum/vc4_select3 pass their element oracles
 * (tests/test_gated.c).  What they all had in common was a broken VPM store:
 *
 *   - build_copy2d's register block sat at module scope and rebound the
 *     shared B_VPMQ7 (23 -> 8) and B_C64 (21 -> 10), so the four kernels
 *     loaded their per-vector advance into B10 and wrote the VDW read row to
 *     B8 while emit_store16 read B23.  The store then read a VPM row the QPU
 *     had not written, and the caller kept the previous launch's bytes.
 *   - vc4_reduce's manual prologue put the group index in the VDW row field
 *     (qid<<7 is row qid, not row 4*qid), so every QPU after the first read
 *     another QPU's VPM sector.
 *   - the TMU pop was read on the word right after SIG_LOAD, before the
 *     response had landed in accumulator 4.
 *   - the DMA store address carried the per-lane offset; the address is a
 *     scalar and every lane must present the same value.
 *   - vc4_cumsum restarted its carry every 256 columns; the host now chains
 *     the chunks with the previous chunk's last output word.
 *
 * VC4_ALLOW_UNVERIFIED_KERNELS=1 still overrides the gate, which is kept for
 * future kernels: a new QPU blob that has not been element-verified should
 * fail fast here rather than wedge the board.
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
