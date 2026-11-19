# attempt-asm10 — diagnostics and notes for the asm10 GEMM port

**This port is now shipped** (`../RESULTS-fast2.md`): the generalised tile
addressing is fixed, `REPS=5 ./run_tests.sh` is green 5/5 on every shape, and
matmul at 256^3 is 8.50 GFLOP/s (engine-fix: 6.24) with the GEMM itself faster
than asm10's own kernel through the same launcher (3.55 vs 4.12 ms).

## What the two real bugs were

1. `build_gemm10`'s prologue read a uniform into a GP register and used it in
   the *next* instruction.  A GP register written by a uniform read is not
   visible one instruction later on this silicon: the A row-block base came out
   255 bytes high, which the TMU rounds to +256, i.e. exactly 8 k pairs into the
   row block — the `C[r][c] = A[r][c+8]` symptom.  All eight uniforms are now
   read first and combined >= 3 words later.
2. The row-block guard had two nops between `add.setf` and `brr.4`; the branch
   then saw cleared flags, the row-block body was never entered, and the kernel
   fell through to the end-program path storing nothing (the ~30 us runs).  The
   compare and the branch are now adjacent, as in asm10.

The k loop, prime, store and tail were already byte-identical to
`asm10/gemm6.py` (`diffbody.py`: 0 of 302 words differ).

## Diagnostics kept here (they work on the fixed tree)

| tool | what it does |
|---|---|
| `raw.c` | run any blob with a CPU-built panel/B/C and dump C as floats, hex words, raw bytes, or search the arena for a pattern. `raw <blob> M N K nq amode bmode rows [panoff]` — `rows` -1 = word map, -2 = pattern search, -3 = byte dump; `panoff` shifts u0 (used to prove the A base offset). |
| `diag_conv.c` | `vc4_conv2d` against a CPU oracle for any Cin/H/W/Cout/KH/KW/pad/bias (the engine's own test_conv covers the accepted shapes). |
| `patch_dumpreg.py`, `patch_dumparb.py` | rewrite the 32 accumulator-init words so the kernel's own store dumps any address register (`ra24` = panel row-block base, `ra20` = AADR, ...) — the only non-self-referential register probe found. |
| `patch_bisect.py` | same, optionally neutering one prologue add, to bisect an address term. |
| `probe_unif8.py` | dump u0..u7 (and constants) through a *correct* store; use this before trusting any probe that writes registers. |
| `g6probe.py`, `diag_g6.c` | asm10's own kernel through this engine (harness control). |
| `diffbody.py`, `g10sim.py`, `check_math.py.ported` | byte-diff against asm10, control-flow/address simulation, GPU-free models. |

Two important traps when using these:

* **kernel names are the cache key.** A probe build written under a normal name
  poisons `kcache/`; always `sudo -n rm -f kcache/*.bin` after changing
  `tools/kgen.py`, and give probes their own names.
* probes that store a *register* through `g10_store` must set every register the
  store uses (CADR, CADR2 and B r24 = 64), otherwise the store's byte
  alignment/column advance is wrong and the dump looks shifted by 1 byte.

## Conv

`src/vc4_conv.c` is back on the **asm3** path (asm3 im2col + asm3 16x16 GEMM +
k-major B panel).  The 8-row im2col needs a `horizontal=1` VDM, which is only
exact at 16x16 words on this silicon; the 8x8 `horizontal=1` shape silently
ignores the VPM row/col offsets (measured), so the ported 4-VDM im2col was
wrong for every conv shape.  Two workable routes and why (a) was chosen are in
`../RESULTS-fast2.md` §6.
