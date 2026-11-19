# attempt-asm10 - the asm10 GEMM/packer port (not shipped)

`../RESULTS-fast.md` has the full story.  Short version:

* **Works and is verified on hardware**: the asm10 *packer*
  (`build_pack10` in `kgen.py.ported`) - byte-identical to
  `asm10/pack10_<K>_k16_single_vp1_us.bin` for K = 64/128/256 and bit-exact
  through the engine's launcher (`diag_pack10.c`: 0/256); the *drain-free*
  parts of the GEMM (k loop, prime, store are byte-identical to asm10's
  `gemm6.py`), the GPU-free models in `check_math.py.ported`, and the
  simulator `g10sim.py`.
* **Does not work**: the generalised tile addressing (`vc4_gemm.c.ported` +
  `build_gemm10`) produces wrong C values on every shape, so the tree is
  reverted to the asm3 engine.

## To pick this up

```sh
cd /home/raspi/work/engine-fast
mkdir -p /tmp/port && cp attempt-asm10/kgen.py.ported tools/kgen.py
cp attempt-asm10/vc4_gemm.c.ported src/vc4_gemm.c
cp attempt-asm10/vc4_conv.c.ported src/vc4_conv.c
cp attempt-asm10/vc4_int.h.ported src/vc4_int.h
cp attempt-asm10/check_math.py.ported tools/check_math.py
cp attempt-asm10/test_matmul.c.ported tests/test_matmul.c
make clean && make -j4 && sudo -n rm -f kcache/*.bin
# then the probes (build them by hand, they include ../src/vc4_int.h):
gcc -O2 -std=gnu11 -Iinclude -Itests attempt-asm10/diag_g6.c build/libvc4infer.a -o build/diag_g6 -lm
```

Useful probes, in the order that was informative:

| probe | what it proves |
|---|---|
| `diag_g6 N nq` | asm10's own `gemm6_<N>.bin` launched by this engine: 0 mismatches for nq = 1/4/12.  Run this first after any change - if it fails, the harness broke, not the port. |
| `diag_pack10 K M` | the panel builder against the reference panel |
| `diag_mm3 M N K` | `vc4_matmul` end to end against the CPU oracle |
| `diag_gemm11 M N K` | `vc4__gemm_run` with a CPU-built panel (isolates the GEMM) |
| `diag_probe M N K [r0]` | delta panels: which k / row reaches which C cell |
| `diag_probe2`, `diag_probe3` | identity-B and B=k probes: column/row mapping |
| `g6probe.py` + `diag_avec.c` | dumps the A vector a kernel *reads*, using the proven C store; validated against `gemm6_64.bin` (dumps `0 1000 2000 ... 7000 | 1 1001 ...`) |
| `diffbody.py` | byte-diff of the emitted k-loop body against `asm10/gemm6.py` (currently 0 words differ) |
| `g10sim.py` | scalar run of the generated GEMM: termination, uniform count, TMU/store address ranges |

## The one open symptom

`diag_gemm11 8 64 16` (single 8x64 tile, one QPU, CPU panel):
`C[r][c] = A[r][c+8]` with B = identity, and 8/9 instead of 1 with a delta
panel, i.e. the A/B k pairing is off by 8 k (or the B base by 32 bytes) while
every emitted word in the loop is identical to asm10's.  The address-register
dumps were all self-referential (they store through the same CADR they are
measuring), which is why they were inconclusive; dumping through a *fixed*
address (u8) needs the uniform port to accept a 9th word, which was not
confirmed.
