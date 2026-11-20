# engine-fast - asm10 GEMM port: status, what was verified, and where it stopped

Work tree `/home/raspi/work/engine-fast` (copy of `engine-fix`, which is
`f41e61f` in `vc4-stack`).  Board: Pi 3B, stock clocks (V3D 300 MHz,
`get_throttled=0x0`).  Every GPU sequence: one launch at a time under
`flock -w 900 /tmp/vc4-gpu.lock`, `VC4INFER_TMO_MS` short, board state checked
after every run.  No board reset occurred in this run.

## Verdict

**Not shipped.**  The tree is the *unmodified* `engine-fix` engine: `REPS=5
./run_tests.sh` is green (exit 0, `ALL TESTS PASSED`, 0 mismatches), and the
matmul numbers are the asm3 numbers (no improvement).

The asm10 **packer port works** and is byte-identical to asm10's proven blob,
and the asm10 **GEMM's k loop, prime and store are byte-identical** to asm10's
proven kernel - but the *generalised tile addressing* I wrote for the engine
(arbitrary M, N, K instead of asm10's square N) produces wrong results, and I
could not isolate the cause within this run.  Details, evidence and the
ruled-out list are below so a follow-up can start where this stopped.

## 1. What was built (kept in `attempt-asm10/`)

| file | content |
|---|---|
| `kgen.py.ported` | engine generator with `build_gemm10()` (asm10 8x64 tile, 3-phase k loop, drain group for K/2 % 3 == 0, uniform-driven row-block addressing), `build_pack10()` and a rewritten `build_im2col()` (8-row panel) |
| `vc4_gemm.c.ported` | `vc4_matmul`/`vc4__pack_run`/`vc4__gemm_run` for the new ABI (M%8, N%64, K%16, row-major B, 4-QPU grid-stride pack) |
| `vc4_conv.c.ported` | conv on the 8-row panel: im2col writes four 256 B VDMs per 16x16 block, weights transposed to a row-major [K][Cout] matrix on the ARM, bias folded as one k row |
| `check_math.py.ported` | GPU-free models of the new im2col layout and the new tile addressing |
| `g10sim.py` | scalar control-flow/address simulator for the generated GEMM: proves termination, 8 uniforms per QPU, and that every TMU address and every store address stays inside its buffer.  9 shapes pass, including the drain path |
| `diag_*.c`, `g6probe.py`, `diffbody.py` | the probes described below |

## 2. What was verified on hardware

1. **The packer is correct and byte-identical to asm10's.**
   `build_pack10(K)` emits exactly `asm10/pack10_<K>_k16_single_vp1_us.bin`
   (`cmp` clean for K = 64, 128, 256), and through the engine's launcher the
   panel is bit-exact: `diag_pack10 16 16` -> `0/256 mismatches`, K=16/64/128
   shapes all pass.  Pack wall time at K=16, 2 QPUs: 1.3 us.
2. **asm10's own GEMM kernel runs correctly through this engine.**
   `diag_g6 N nq` loads `asm10/gemm6_<N>.bin` with `vc4__kernel_load_file` and
   launches it with asm10's uniforms on a CPU-built panel:
   `64^3 nq=1 -> 0/4096`, `nq=4 -> 0/4096`, `nq=12 -> 0/4096`.  So the arena,
   the launcher, the L2/SLCACTL discipline, `VPMBASE=16`, multi-QPU stores and
   the 8-row panel layout are all compatible with the asm10 kernel.
3. **My generated k loop is byte-identical to asm10's.**
   `diffbody.py` byte-compares the emitted 302-word k-loop body (6 phases +
   2 boundaries + 3 delay slots) against `asm10/gemm6.py`: **0 differing
   words**.  The prime (issue/pop/copy/rot) is also identical, and the store
   differs only in which register holds `4*qpu`.
4. **The k/KPAIRS arithmetic and the drain path generate and simulate.**
   `build_gemm10` passes its own schedule verifier and `g10sim.py` for
   M,N,K = (16,64,16), (24,64,48) [drain], (8,64,16), (128,128,128),
   (256,256,256), (512,64,288) [drain], (16,128,48), (64,64,32), (40,192,80).
   The drain group (K/2 % 3 == 0, i.e. K = 48/144/288/576 - the conv shapes)
   was derived so that every TMU pop has a matching issue; asm10 had no path
   for it.

## 3. Where it fails

With my kernel: `diag_gemm11 M N K` (CPU-built 8-row panel, row-major B,
double-precision oracle) fails for every shape, including a single tile with
one QPU:

* `8 64 16` -> 509/512..1024/1024 wrong, values of the right magnitude;
* only QPU 0's 8 C rows are written in the multi-QPU cases (rows 8.. untouched,
  i.e. still the test's 0xCDCDCDCD fill);
* `B = identity` -> `C[r][c] = A[r][c+8]`, i.e. the A/B k pairing or the B
  column base is off by 8 k / 32 bytes;
* `A = delta at k0`, `B = ones` (C should be 1) -> 8 (k0 < 8) / 9 (k0 >= 8)
  contributions, i.e. the same A value is accumulated about once per k pair.

The probe path was validated on asm10's kernel first (`g6probe.py` copies
`acc0` into the store's row-0 register and jumps to the store; with gemm6's
kernel it dumps the exact panel vector `0 1000 2000 ... 7000 | 1 1001 ...`),
so the probe technique itself is sound.  With my kernel the same probe dumps
all zeros, which points at the A read address - but every address-dump probe I
then wrote depends on the same (suspect) address registers, so those results
are self-referential and were not conclusive.

### Ruled out (measured, not assumed)

* the engine's launcher / arena / L2 flush / SLCACTL / VPMBASE (asm10's own
  kernel: 0 mismatches);
* the panel layout and the packer (bit-exact panel; asm10's kernel consumes my
  CPU-built panel correctly);
* multi-QPU launches and stores (asm10's kernel at 12 QPUs: 0 mismatches);
* the k loop, prime, tail structure and store instruction sequence (byte-diff
  against asm10: identical);
* the uniform ABI/order (a probe that dumps u0..u7 through the proven store
  returns the values the ARM wrote);
* control flow / wild addresses of my kernel (the simulator: terminates, all
  TMU/store addresses in range, 8 uniforms per QPU);
* QPU register file hazards (asm10's own verifier *and* a stricter
  "GP RAW gap >= 3 words" check: 0 violations).

### Prime suspect for a follow-up

The whole difference between my kernel and asm10's is the tile addressing:
asm10 computes `rowoff = rowblk << log2(32*N)` (so its panel and C row-block
strides are the same power of two), while the engine needs three independent
strides (panel `32*K`, B `N*4`, C `8*N*4`) and therefore a nested
(row block x column block) loop with the strides in uniforms.  The next thing
to try is to keep asm10's *per-tile* address computation and make the strides
power-of-two-compatible by padding - the panel row-block stride to
`32*Kpad` (Kpad = next power of two; the k loop is unchanged, only the panel
memory grows) - which requires the *C* row stride to equal it, i.e. `Np ==
Kpad`; that is only free when K <= N (fine for 128^3, 256^3) and costs up to
`Np/K` extra GEMM work otherwise.  The alternative is to find why my uniform
addressing reads the wrong A line, which is a small, well-localised bug: the
A vector read at the prime is the only thing the probes ever showed as wrong.

## 4. Numbers (this run, same binary, same board)

Before = engine-fix/engine-fast baseline, measured at the start of the session
(right after boot, V3D 300 MHz, `get_throttled=0x0`):

| shape | pack ms | gemm ms | total ms | GFLOP/s |
|---|---|---|---|---|
| 128^3 | 0.066 | 0.651 | 0.717 | **5.85** |
| 256^3 | 0.383 (asm10 pack measured separately) | 5.306 | 5.689 | **5.90** |
| 512x64x288 | ~0.30 | 2.49 | 2.788 | **6.77** |

After the port was reverted the same binary was run again at the end of the
session (the SoC core clock drops 400 -> 250 MHz under sustained 12-QPU load -
see `engine-fix/RESULTS.md` §3 - so these are *slower*, not a regression):

| shape | total ms | GFLOP/s |
|---|---|---|
| 128^3 | 1.252 | 3.35 |
| 256^3 | 7.270 | 4.62 |
| 512x64x288 | 4.683 | 4.03 |

The pack pass of the engine's asm3 path is 63-68 us at 128^3 and ~0.38-0.46 ms
at 256^3 in the fast state; asm10's packer saves 16 % of that, which is 2.7 %
of the total at 256^3 - the GEMM is the part that matters, and it is unchanged.

## 5. Suite

`REPS=5 ./run_tests.sh` (reverted tree) -> exit 0:

```
=== maths (no GPU)          sigmoid/silu/im2col/gemm models: PASS
=== test_dispatch           PASS
=== test_ew                 add/mul/relu/add+tail/sigmoid/silu 5/5 clean
=== test_matmul             128^3, 256^3, 64x48x32, 20x36x44, 16x16x16,
                            512x64x288: 5/5 clean each, 0 mismatches
=== test_conv               3x3 pad1 +bias (Cin 16/32), 1x1 (with and without
                            bias), 3x3 pad0, Cin=64 K=576: 5/5 clean each
ALL TESTS PASSED
```
