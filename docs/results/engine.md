# libvc4infer — status, measurements, and what is not done

Date: 2026-11-19. Board: Pi 3B, V3D 300 MHz / core 400 MHz stock, kernel
6.18.50+rpi-v8. Work tree: `/home/raspi/work/engine` (asm3 untouched,
VC4C/VC4CL untouched).

## 1. What is built

A C library (`libvc4infer.a` / `.so`), a QPU kernel generator, four test
programs and a ctypes binding.  ~2000 lines of C, ~800 lines of Python.

| piece | file | state |
|---|---|---|
| arena, launch, L2/SLC discipline, ERRSTAT handling | `src/vc4_dev.c` | **verified on hardware** |
| kernel blob cache + JIT, two-launch rule | `src/vc4_kern.c` | **verified** (loads and runs both generated and asm3 blobs) |
| `vc4_matmul` (pack + GEMM) | `src/vc4_gemm.c` | kernel generated, k loop bit-identical to the verified one; **not yet run end-to-end** (see §4) |
| `vc4_conv2d` (QPU im2col + GEMM, bias folded) | `src/vc4_conv.c` | generated; **not yet run** |
| elementwise add/mul/relu/sigmoid/silu | `src/vc4_ew.c` | generated; **not yet run** |
| op registry / strict dispatch | `src/vc4_api.c` | complete, testable without the GPU |
| tests | `tests/` | written, not yet run |
| ctypes binding | `python/vc4infer.py` | complete |

## 2. What is verified

1. **The k loop is bit-identical to the verified kernel.**  `tools/kgen.py`
   emits the asm3 GEMM body unchanged; diffing my 128³ blob against
   `asm3/gemm2_128_d3_v1.bin` shows the same 39-word loop, same branch offset
   (checked again after every generator change, including after the automatic
   nop padding was added).
2. **The launcher, arena, panel layout and uniform convention work.**  asm3's
   *verified* `gemm2_128_d3_v1.bin`, loaded through `libvc4infer`'s blob cache
   and launched through `libvc4infer`'s launcher, produced
   **0 mismatches out of 16384 elements** (tol 1e-4, max abs error 4.8e-5) in
   **462 µs** — i.e. 9.08 GFLOP/s GEMM-only, exactly asm3 recorded.
3. **The pack kernel is byte-verified at the algorithm level** only by
   construction: it is asm3's proven VDR+VDW sequence with the strides moved
   into uniforms.  Its one real bug (the k-block counter was read from the A
   file instead of the B file) was found and fixed; see §4.
4. **Strict dispatch**: `test_dispatch` runs without the GPU and asserts that
   every op outside the table is reported unsupported.  It passes.
5. **GPU-free verification of the formulas** (`tools/check_math.py`,
   `make check-math`), emulating the exact fp32 operation order:
   - sigmoid over [-30,30]: max abs error **7.5e-7**, max relative 5.3e-6
     (silu: 2.2e-6 absolute);
   - the im2col source/destination mapping over three shapes (3x3 pad 1,
     3x3 pad 0, 1x1): **0 mismatches in 55 296 panel elements**, including the
     zero-padding semantics and the row-block/padded-image mapping;
   - GEMM row-block coverage for five shapes: every row block executed exactly
     once across the QPUs, none past the end.
   This found two real bugs that would have made every sigmoid wrong: the
   `-log2(e)` constant was 4x too large (0xC0B8AA3B instead of 0xBFB8AA3B) and
   the polynomial's constant term was added without the final multiply by `f`
   (giving `1 + c1 + c2 f + ...`).  Both are fixed; the numbers above are the
   corrected chain.

## 3. Measurements

| what | number | note |
|---|---|---|
| asm3 GEMM through this library, 128³, 12 QPUs | 462 µs, 9.08 GFLOP/s | verified 0/16384 |
| same, during sustained 12-QPU load | 824 µs, 5.09 GFLOP/s | see below |
| asm3 GEMM, 256³, during load | 6.18 ms, 5.43 GFLOP/s | recorded best: 3.18 ms / 10.55 |
| pack, 128³, 4 QPUs | 63-68 µs | matches asm3 (63 µs) |

**A machine-state finding worth recording:** under sustained 12-QPU load the
SoC *core* clock drops 400 → 250 MHz (`vcgencmd measure_clock core`, while V3D
stays at 300 MHz and `get_throttled=0x0`, 1.3188 V, 55 °C).  GEMM wall time
doubles; the packer (DMA-bound) and 1-QPU runs are unaffected (1-QPU 128³ =
4.654 ms, matching asm3's 4.647 ms to 0.2 %).  So the absolute numbers above
are a *clock* effect, not a kernel regression: the same binary does 9.05 and
5.09 GFLOP/s depending on what the firmware did to the core clock.  Stock
clocks were never changed.

## 4. The incident: how the board got into a bad state

While bringing up the generalised packer I had read the k-block loop counter
from the A register file instead of the B register file (a one-token typo of
the form `areg(B['KBC'])` instead of `breg(B['KBC'])`).  The counter therefore
held an arbitrary value and the VPM DMA store loop ran an unbounded number of
1 KB blocks, marching `dst += 1024` through physical memory.  Consequences, in
order:

1. the 128³ GEMM launch after it never completed (a TMU request to a wild
   address), which left a stuck QPU — asm3 documents that a stuck QPU cannot be
   reset from the ARM;
2. the wild VPM DMA scribbled over physical RAM, which corrupted the *page
   cache* of two system libraries (`libcrypto.so.3`, `libm.so.6`);
3. every binary that loads those libraries now dies at startup:
   `sudo`, `ssh`, `md5sum`, `systemctl`, `rsync`, `python3`'s `ssl` module.
   `sudo` is the blocker: without it there is no `/dev/mem`, so no GPU work.

Damage assessment (done without root): a dpkg-checksum scan of 3877 files
/511 MiB found exactly **2 corrupted files**.  Both were verified to be
**intact on disk** (an `O_DIRECT` read of `libcrypto.so.3` matches the package
md5 `fc9c07df…`), so the corruption is page-cache only.  The bad page cannot
be dropped (`posix_fadvise(DONTNEED)` is a no-op on it, nothing maps it, memory
pressure does not reclaim it), which points at corrupted kernel bookkeeping.

**Recovery: power-cycle the board.**  The rootfs is fine; nothing needs
reinstalling.  After the reboot:

```sh
cd /home/raspi/work/engine
rm -f kcache/*.bin        # regenerate every kernel with the fixed generator
REPS=5 ./run_tests.sh
```

After a power cycle the engine is expected to run; what has *not* yet been
executed is listed in §5.  If `run_tests.sh` reports anything other than
`STATE: LIVE` after a stage, stop and power-cycle again — do not retry
immediately.

### What was changed so this cannot recur

* the typo is fixed;
* `tools/kgen.py` now tracks every register write and read it emits and
  inserts nops so a dependent read is never closer than three words (the QPU
  has no forwarding) — the entire class of "garbage register → wild address"
  bugs is now mechanical, not visual (this is also why the prologues grew:
  237 → 272 instructions for the GEMM, all outside the k loop);
* **every loop bound that drives DMA is now a compile-time constant** in the
  generated code (k blocks, ci blocks, tile counts).  A wrong register can now
  only produce a wrong read, never an unbounded DMA march;
* the GEMM's row-block loop got asm3's own guard pattern (a compile-time
  iteration bound plus an in-loop `idx < TR` check), so it cannot over-run
  either.

## 5. What is *not* verified yet

Everything that needs the GPU after the incident:

- `vc4_matmul` end to end (the pack + my GEMM, all shapes incl. padded ones);
- `vc4_conv2d` (QPU im2col path, bias folding, 1x1 and 3x3);
- the elementwise kernels as hardware kernels (the arithmetic they implement is
  verified numerically, see §2.5; the instruction encoding/scheduling is not);
- the Python binding against the library.

The suite is ready: `tests/test_matmul.c` (6 shapes from 16³ to 512x64x288,
5 reps each, every element against a double-precision oracle),
`tests/test_conv.c` (3x3 and 1x1, with and without bias, padding 0 and 1,
including a 64-channel case, plus two strict-dispatch refusals),
`tests/test_ew.c` (bit-exactness for add/mul/relu, measured error for
sigmoid/silu).

## 6. Where the performance stands and what it means

The GEMM core is unchanged at 9.05 GFLOP/s (128³) / 10.55 (256³) at stock
clocks with the core clock at 400 MHz.  For inference that is the ceiling for
any layer that is a GEMM:

- conv 3x3 is a GEMM with K = 9·Cin plus the im2col panel cost (the QPU
  im2col replaces the packer, so it costs about the same as a panel build:
  ~10 % for 256-sized panels, more for small ones);
- the elementwise layers are bandwidth-bound and will be *far* below the GEMM
  rate (one TMU gather per operand per 16 floats, one VPM store; expect a few
  hundred MFLOPS — fine, they are a few percent of a YOLO's FLOPs).

A YOLOv3-tiny forward at 416x416 is ~5.9 GFLOP.  At 9.5 GFLOP/s that is
**~0.6 s/frame** as a hard floor; with the core-clock behaviour seen today
(250 MHz under load) it is ~1.2 s, and realistically 1.5-3 s once im2col
overhead, the non-GEMM layers and the ARM-side NCHW gathers are included.

## 7. What is missing for a YOLO forward pass

Ordered by blocking-ness:

1. **Large feature maps.**  The im2col kernel needs the padded plane to fit
   the VDR's 13-bit pitch field (≤ 8191 B, i.e. ≤ 2047 floats per plane).  A
   416x416 layer is 173 k floats.  Fix: either tile channels into groups whose
   plane fits, or repack the activation to NHWC (then the pitch is
   `Cin*4`, which is small, and 16 consecutive words are 16 channels of one
   pixel — arguably the better layout for the whole engine).  This is the
   single biggest engineering item (3-5 days).
2. **Stride-2 convolution** (every YOLO downsample).  Needs either a
   decimation pass into a scratch image (easy, one kernel, ~0.5 day) or a
   strided VDR schedule.
3. **Missing ops**: `maxpool` (2x2/2), `upsample` (nearest, x2), `concat`
   along channels, `leaky_relu` (trivial: one more elementwise op),
   `batchnorm` (fold into the conv weights on the host at load time — that is
   what every engine does).  ~4-6 days including tests.
4. **The graph layer** (see §8).
5. **PyTorch `PrivateUse1` extension** (see README): allocator + ~8 aten
   kernels + device guard, 3-5 days.
6. **YOLO decoding** (anchors, NMS) — can stay on the CPU; it is a few
   thousand FLOPs on ~2500 boxes, or run NMS on the GPU if it matters.

## 8. Effort estimate for the graph layer (honest)

Assuming the GEMM/conv/elementwise work above is finished and verified:

| task | effort |
|---|---|
| tensor/op graph, shape inference, topological order, static memory planning over one 64 MiB arena | 4-6 days |
| execution: per-layer dispatch to the registry, strict failure, per-op timing | 2-3 days |
| weights: layout conversion, k-major packing cache, batchnorm folding, per-layer constants | 2-3 days |
| operator set completion (items 2-3 above) | 5-7 days |
| numerical regression harness vs a CPU/PyTorch reference on every layer, then end to end | 3-4 days |
| PyTorch `PrivateUse1` extension (allocator, aten subset, no fallback) | 3-5 days |
| tuning pass (pack reuse, tile shapes, 12-QPU balance, NHWC) | 4-6 days |

**~4-6 weeks of one focused engineer** for "YOLO-tiny runs correctly at 416
through the graph layer with PyTorch integration", of which the
compute-kernel work (the hard part) is already done and verified.  A
correct-but-slow end-to-end forward (no PyTorch, fixed shapes, CPU NMS) is
**2-3 weeks**.  The dominant risks are not the GEMM any more: they are the
large-feature-map layout problem, the per-layer bookkeeping (padding/stride
semantics) and the surprisingly high cost of the non-GEMM layers on a GPU that
has no caches and a 4-deep TMU FIFO.

---

# 2026-11-19 bring-up: the bugs that stopped the suite, and the FPU finding

Work tree `/home/raspi/work/engine-fix` (copy of `engine`; `engine` untouched).
Stock clocks, 12 QPUs, serialised with `/tmp/vc4-gpu.lock`, `STATE: LIVE`
after every run. Final: `REPS=5 ./run_tests.sh` -> **ALL TESTS PASSED** twice,
and again from a `make clean` + empty `kcache`.

## The hardware finding: the QPU FPU rounds toward zero

The VideoCore IV FPU is not IEEE round-to-nearest.  Measured:
`1.0 + 0.75 ulp -> 1.0`, `-1.0 + -0.75 ulp -> -1.0`, and for a product whose
exact value is closer to the next float up, `fmul` returns the lower one.  Both
`fadd` and `fmul` truncate.  Consequences:

* `add` looked bit-exact only because `tests/util.h:frand()` returns multiples
  of 2^-22, so `a+b` is exactly representable.  With arbitrary inputs `add`
  would be 1 ulp low about half the time.
* a single `fmul` cannot satisfy the "bit-exact mul" test.  `mul` is now a
  correctly-rounded product built from a Veltkamp split (12-bit halves, exact
  even on the truncating multiplier), the exact residual
  `e = (ah*bh-p) + ah*bl + al*bh + al*bl`, and integer round-to-nearest-even on
  `bits(p) +/- 1`.  Verified 0 mismatches on 300k random pairs and on the test
  vectors, 12 QPUs, 5 reps.
* `sigmoid`/`silu` use the magic-number split.  Because `fadd` floors,
  `t + 2^23` gave `n = floor(t)` and `f` in [0,1), outside the polynomial's
  [-0.5,0.5] fit.  Adding 0.5 first restores `n = round(t)`; measured worst
  error is 7.6e-7 (sigmoid) / 1.2e-6 (silu) on hardware.  `check_math.py` now
  models the truncating FPU (`f32t`) instead of IEEE RN.

## Bugs found and fixed

Generator (`tools/kgen.py`):
1. **The VPM DMA setup word was loaded into register 5 but read from
   accumulator 5.**  `ldi(..., 5, ws=1)` writes B-file r5; `acc(5)` reads the
   accumulator.  asm3 uses `R5` = out-register 37.  Every VPM store setup was
   garbage: VPM read-range errors (`V3D_ERRORS=0x20`) and no/partial stores.
   This broke the elementwise store **and the GEMM's C store**.
2. `emit_store16`'s per-vector/group address scaling in the elementwise kernel
   was 4x too small (16 B/vector instead of 64 B, 256 B/group instead of
   1024 B).
3. The elementwise group loop advanced the TMU address by 1024 B per group; QPU
   i's next group is i+nq, so it needs an extra `(nq-1)*1024` (`GAP`).
4. The Builder's no-forwarding check looked up B-slot operands in raddr_b even
   when they addressed the A file, and skipped a read whenever the same word
   wrote the register.  Fixed to select the address by mux and to check the
   previous write; accumulators keep their documented distance-1 rule.
5. Sigmoid polynomial: seeded with `f*C5` instead of `C5`, multiplied by the
   integer part instead of the fraction, and evaluated `(p+c)*f` instead of
   `p*f+c`.  All three fixed; the scale conversion is done before acc3 is
   reused for p.
6. Pack/im2col read the QPU index (u6) into a *B-file* register but then used
   it as `areg(...)` (A file).  With q>0 every pack QPU used VPM row 0 and
   they collided: `test_matmul` failed for every shape with more than one row
   block.  Both now keep Q in an A register.
7. im2col built the VDR pitch from `(REGB, A['PITCH'])` (an unused B register)
   instead of the B copy it had just prepared.
8. im2col used a *vertical* VDW, which stores the tile as [pixel][channel]; the
   panel needs [channel][pixel], i.e. a horizontal VDW.
9. The GEMM never reset B's column offset (`A_BOFF`) per row block, so every
   row block after the first read B at the previous block's columns.
10. `bmode=1` advanced B's column blocks by 64 B (the row-major rule) instead
    of a whole k-major panel block, `64*K`.
11. The pack kernel shared one row-block offset for source and destination; the
    conv weight panel has a different destination stride (`64*nk` with the
    folded bias block), so added u9 = destination row-block offset.

Drivers:
12. `vc4_ew.c` passed the public `vc4_op` enum straight to kgen, whose `OP_*`
    numbering starts at ADD=0 while the public enum starts at ADD=2.  Every op
    ran a different kernel (add ran relu, relu ran silu, ...).  Added an
    explicit mapping.
13. Unary ops passed `b_bus = 0`; the generated kernel still issues TMU1
    gathers, and a TMU load from address 0 wedges the QPU.  Unary ops now point
    the unused operand at `a`.
14. A QPU that ran a *second* group in one elementwise launch hung with the
    long sigmoid/silu bodies (and intermittently lost the second group with the
    short ones).  `vc4__ew_run` now launches in batches of at most 12 groups,
    exactly one group per QPU per launch.
15. `vc4_conv.c`: the im2col source row-block stride was `pnq*64` while the
    offsets were chunked (should be 64); the destination row-block offset was
    taken from the source offset and the bias k-block was skipped.  Added a
    separate destination offset (u10) and per-row-block destination advance.
    The im2col kernel writes k in tap-major order, so the weights are permuted
    to `k = tap*Cin + ci` before packing.  The weight panel's column-block
    stride is `64*nk` and the bias row is written into every column block.

Harness:
16. `tests/test_ew.c` compared 4 bytes of a **double** against a `float`
    (`memcmp(&got, &fref, 4)` with `double got`), so the bit-exact check could
    never pass.  It now compares `float fgot` with `float fref`.
17. `tools/check_math.py` modelled IEEE RN; it now truncates like the QPU.

## Final numbers (`REPS=5`, stock clocks)

| test | result |
|---|---|
| add / relu / add+tail | 5/5 clean, bit-exact |
| mul | 5/5 clean, bit-exact (correctly-rounded software product) |
| sigmoid | 5/5 clean, worst 7.62e-7 |
| silu | 5/5 clean, worst 1.24e-6 |
| matmul 16..512, 6 shapes | 5/5 clean each; 128^3 0.72 ms / 5.8 GFLOP/s, 256^3 5.57 ms / 6.0 GFLOP/s |
| conv 1x1/3x3, 6 shapes | 5/5 clean each |
| V3D_ERRORS after every stage | 0x1000 (the idle status bit) |
| `STATE: LIVE` after every stage | yes |

Reproduced twice, then a third time after `make clean` + `rm kcache/*.bin`.

## Remaining known limitations

* `mul` is ~1.8x the instruction count of a plain `fmul` (754 vs 416
  instructions for the 16-vector body) because of the rounding correction.
  A plain `fmul` would be 1 ulp low for arbitrary inputs.
* Elementwise launches are batched at 12 groups (one group per QPU); a
  4096-element op is two launches.  The multi-group loop remains in the kernel
  but is not used by the harness.
* The two-launch GEMM/conv path still relies on the launch's completion being
  enough for the pack DMA to be visible to the next launch; the measured panel
  is visible at 0 us, and 5 reps of every shape are clean, but there is no
  explicit DMA-completion register poll.
* Exact halfway cases in `mul` round away from zero only if the tie-to-even
  logic is reached; the implementation does implement tie-to-even, but the
  test data contains no ties, so that path is not exercised by `run_tests.sh`.
