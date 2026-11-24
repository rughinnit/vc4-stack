# The five kernels that were gated, and the exact erf-form GELU derivative

This file records the engine-level work on the kernels that `vc4_guard.c`
refused: `vc4_gather`, `vc4_blit`, `vc4_reduce`, `vc4_cumsum`, `vc4_select3`,
and the `Gelu`-erf backward that could not fit the 2,048-instruction code slot.

Status at the end of this run:

| entry point | status | evidence |
|---|---|---|
| `vc4_blit` | **fixed, ungated** | byte-exact on silicon: 0/1600 B and 0/208 B, `tests/test_gated blit` |
| `vc4_gather` | **fixed, ungated** | 0 bad at n = 1, 16, 17, 255, 256, 257, 1000, `tests/test_gated gather` |
| `vc4_reduce` | **fixed, ungated** | 0 bad for all 6 ops, outer 1..1024, red 1..64 |
| `vc4_cumsum` | **fixed, ungated** | 0 bad for 1x256, 2x256, 3x512, 5x1024 (max\\|err\\| 7.6e-05) |
| `vc4_select3` | **fixed, ungated** | 0 bad for where/clip3/prelu at n = 1..1000 |
| `VC4_BW_GELU_ERF` | **implemented** | one 1967-word kernel at 4 vectors/group; gradcheck rel_L2 2.45e-07, over_tol 0 |

All five entry points are element-verified on silicon.  The gate has no call
sites left; `vc4__refuse_unverified` stays as the fast-fail for a future blob
that has not been through `test_gated`.

## 1. The model that found the defects

`infer/tests/emu_alu.py` executes a generated blob word by word: the ALU
(round toward zero, denormals flushed), the accumulator file, the uniform
port, the TMU issue/pop queue, and the special write ports.  This run added
four things it was missing, each of which had hidden a real defect:

* **branches with flags** (`sf` sets the per-lane Z/N flags; relative targets
  are measured from the word *after* the three delay slots, verified against
  `Prog`'s label table).  Without them the per-vector reduction loop ran once.
* **`mul24`** (integer product) and **the 16-lane rotate** (`MUL_V8MIN` with a
  B small immediate 49..63).  Without them the integer address math and the
  prefix scan in `vc4_cumsum` were nonsense in the model.
* **the `rep_all` write port** (destination 37, `ws=1`: r5 gets lane 0 in
  every lane).
* **the store address**: `infer/tests/emu_gated.py` now checks that the
  per-lane DMA address points inside the caller's output buffer, not just that
  the result registers are right.

That last check is what proved the most valuable.  Registers can be perfect
while the store writes someone else's memory, and the model had been blind to
it.

## 2. Defects found in the four kernels (all fixed in `kops.py`)

### 2.1 The E-helper register convention was used in raw builder calls

`E.op`/`E.mul` take the pseudo indices `ACC0..ACC3` = 100..103 and map them
with `_d()` to accumulator words 32..35.  A **raw** `k.add_`/`k.add_or` call
encodes `add_out & 0x3F`, so 100..103 silently became the *special* writes
**36** (`tmu_noswap`), **37** (`rep_all`), **38** (irq) and **39** (NOP), and
`200 + n` became B-register `n + 8`.  Consequences:

* `kops._stream_base` wrote its staging accumulator to NOP, so `A_IB`/`A_OB2`
  lost the bus base entirely and the TMU read low physical addresses;
* `vc4_reduce` seeded its accumulator into `tmu_noswap` and its `lane*4`
  into irq;
* `vc4_cumsum` wrote its four scan masks and its carry into the wrong B
  registers.

Fix: `_aw()` maps every raw destination (`_stream_base`, `build_sel3`,
`build_reduce`, `build_cumsum`).

### 2.2 `A_CADR` was redefined at module scope

`build_copy2d`'s local register block ended with `A_TA, A_CADR = 26, 27`,
which **rebound the module-level `A_CADR`** that `build_gather`,
`build_sel3`, `build_cumsum` and `build_reduce` were using for the store
address.  Those four stored through `A27`, which is also the first A-file
temp their per-vector body allocates, so the store address became the
per-vector sign mask.  The copy2d block is now `A_TA2, A_CD2`.

### 2.3 `vc4_gather`: fill select and TMU address

* The fill mask was overwritten by the fill value before the gathered word
  was masked (`m = m & fill` then `~m & gathered`), so every negative table
  entry kept the junk fetched from the arena base.  The mask now stays in
  `m` and the fill goes through the dead `tg` register.
* The per-lane address is staged in an A-file register (a TMU setup word
  cannot take an accumulator), with explicit gaps.

### 2.4 `vc4_reduce`

* `argmin` seeded `-inf`, so `best - new` was `-inf` and the comparison never
  fired: every window reported index 0.
* The mean's `1/red` was read from uniform word 6, which carried the *op
  index*: a Mean multiplied by the bit pattern of the integer 1 (a denormal,
  flushed to zero), so every mean was 0.  Word 6 is `1/red` now.
* `min`/`argmin` used `sel(out, mask, a, out)`, which reads its own
  destination as the false operand; the builder's padding turns that read
  into `~mask`.  `keep_sel` is the safe form.
* The kernel's scalar registers (red, 1/red, the strides) were allocated from
  the same B registers `Ctx` hands to the per-vector body, so the argmax body
  overwrote the reduction count.  They are reserved now.

### 2.5 `vc4_cumsum`

* `ADD_ADD` (integer) added float bit patterns: the scan was meaningless
  whenever two prefixes had different exponents.
* The carry used a plain write to r5 instead of the `rep_all` port, so lane
  L got the rotated prefix instead of the vector total: the next vector added
  `a[0]`, `a[1]`, ... instead of the running sum.

### 2.6 `vc4_select3`

* WHERE tested `bits(a) & 0x80000000` and then `gt_mask`, i.e. "a < 0", so a
  0/1 condition selected the false operand for every element.  It now tests
  `bits(a) & 0x7FFFFFFF != 0` (ONNX: nonzero; `-0.0` is zero, NaN is not).
* The c operand lived in `ACC2`, which is the emitter's bank-conflict staging
  register; `eq0_mask`'s `opi` moves a B operand through it, so the false
  branch returned whatever had just been staged.  c moves to a temp first.

### 2.7 The 256-float store wrote only its first 64 bytes

`kops._prologue` loaded the DMA chunk step (`B_CSTEP`) with the *group* size
(`nvec*64` = 1024 B) instead of 256 B.  `emit_store16` advances the DMA
address by that step after each four-row chunk, so chunks 2..4 of every
256-float group landed at `out+1024/+2048/+3072` and the caller's
`out+256..1023` kept whatever was there.  The step is 256 B now.

### 2.8 The EWB gradient stream was read from A8, not B8

`ewb_body` called every body with `G = 8`, which the emitter reads as **A-file
r8**; the harness loads the second stream into **B8**.  Every binary/backward
kops body therefore multiplied by an unwritten register.  The training path's
`exp`/`log`/`sqrt`/`rcp` re-exports ignore `G`, which is why the suite never
saw it; the new GELU-erf body does not.  The argument is `208` (B8) now.

## 3. `vc4_blit` is fixed by not having its own kernel

The dedicated 64 B kernel produced another group's data nondeterministically
at more than one QPU per launch (48-word blocks, reproducible in a chain of
launches, not in a single fresh process; both backends).  Rather than keep a
fragile second copy path, `vc4_blit` now uses the verified `copy2d` kernel:
one row per 64 B, `vpr=1`, `units=1`, `depth=16`, `slane=4`.

Measured on silicon, daemon backend (peak 64.5 C, `throttled=0x0`):

```
PASS blit       64 B     0/16 bad    0.155 ms
PASS blit      256 B     0/64 bad    0.150 ms
PASS blit      448 B     0/112 bad   0.150 ms
PASS blit     1024 B     0/256 bad   0.152 ms
PASS blit      832 B     0/208 bad   0.157 ms
PASS blit     6400 B   0/1600 bad    0.156 ms
```

Launch count: `copy2d` spreads rows over at most 12 QPUs and loops inside the
kernel, so a 6400 B copy is **one launch**; the old per-group kernel needed
one group per 64 B (`bytes/64` groups, 12 per launch).  That is the shape the
`bytes % 64` contract promises and the shape callers get.

## 4. Why the four kernels returned another launch's bytes

The output was not a stale copy that some CPU path failed to invalidate: the
VPM->DMA store did not land.  The store sequence itself was already identical
to `build_ew`'s verified four-chunk store (confirmed word for word against a
working blob), so the defect had to be in the state the store was given.
Bisecting a marker that writes 42.0 to all 16 lanes found five independent
faults.

### 4.1 `build_copy2d`'s register block sat at module scope

`B_C4D, B_STRIDE, B_VPMQ, B_VPMQ7, B_FILL, B_C64, B_VPR = 5, ...` is
*module-level* code above `build_copy2d`, so importing `kops` rebound

* `B_VPMQ7` from 23 to **8**, and
* `B_C64` from 21 to **10**.

`_prologue` therefore loaded the per-vector table/source advance (64) into
B10, which the per-vector temp pool hands out and clobbers, and wrote the VDW
read row (`(4*qpu) << 7`) into B8 while `emit_store16` read B23 -- which
nothing had written.  The DMA read a VPM row the QPU had not stored, the
kernel finished normally, and the caller kept the previous launch's bytes.
This is the same failure mode as A16's `A_CADR` rebinding.  The copy2d
registers are `C2_*` now.

### 4.2 `vc4_reduce` put the group index in the VDW row field

`build_reduce` builds its prologue inline and computed
`B_VPMQ7 = A_QID << 7`.  The VDW row field is bits 7..12, so that selects row
`qid`, while the matching `vpw_generic` write setup selects row `4*qid`.
Every QPU after the first DMA-read another QPU's sector; measured with
`outer=512, red=1`, `out[256]` held `a[16]`, i.e. QPU 0's second store chunk,
not `a[256]`.  Both rows are `4*qpu` now, taken from the hardware QPU number
(`R_QPU_NUM`) rather than the uniform group index.

### 4.3 The TMU pop was read one word too early

`kops._load` issued `SIG_LOAD_TMU0/1` and read accumulator 4 on the very next
word.  The response lands in acc4 *on* the load word, and the QPU does not
forward, so the staging accumulator received the previous response -- the
previous vector inside a loop, the previous launch on the first vector.
`build_ew`'s verified pipeline keeps a nop between the load and the pop and
one after it; `_load` does the same now.

### 4.4 The DMA store address carried the lane offset

`_stream_base` is shared by the TMU streams, where the address is per-lane,
and by the output stream.  The VPM DMA address is a scalar: every lane
presents the same value.  `build_ew` and `copy2d` both build it without a
lane term; the four kernels carried `lane*4` into it.  The output stream is
built with `lane=False` now.

### 4.5 `vc4_cumsum`'s carry stopped at every 256 columns

The scan kernel covers one 256-word chunk per group and restarted its carry
at the group label, while the host spread a row's chunks over different
QPUs.  No shape with `cols > 256` could be right.  The kernel now takes a
carry-seed address (uniform 6), and the host walks the chunk positions in
order, passing the previous chunk's last output word as the seed (a zero word
in scratch for the first chunk).  Rows stay parallel: one row per QPU, so
`5x1024` is four launches.

### 4.6 What the model cannot see

`emu_alu` executes the register/ALU/TMU behaviour and checks the per-lane DMA
address, but it has no register-file forwarding model, no per-physical-QPU VPM
sector, and it cannot observe a module-scope Python rebinding at all, because
`emu_gated.py` calls the same `build_*` functions.  Each of 4.1-4.4 was exact
in the model and wrong on silicon, which is why the element checks in
`infer/tests/test_gated.c` are the verification that counts.

## 5. `Gelu`-erf backward

The 16-vector body is 4,222 words against 2,048; the split chosen is *not* two
kernels.  The derivative

```
d/dx [0.5 x (1 + erf(x/sqrt 2))] = 0.5(1 + erf(x/sqrt 2))
                                 + x exp(-x^2/2)/sqrt(2 pi)
```

is computed by one fused body at **4 vectors per group** (64 floats), which
fits: **1,967 words**, generated by `kops.build_ewb` because
`kops.EWB_HEAVY = {'gelu_erf_bwd'}` and the host's group size in `vc4__ewb`
is 64 floats for exactly that name.

Two things made it fit:

* `-x^2/2` is formed as `(x*x)*0.5` with the sign bit flipped, reusing
  `SIGNB` from the erf body instead of a `-0.5` constant;
* the erf half lives in `OUT` (the store register, not a temp), because the
  fused body needs seven live A registers and the pools leave seven.

Verification (`REPS=1 ./run_tests.sh`, `test_grad`, daemon backend):

```
ewb: 1967 instructions -> .../ewb_gelu_erf_bwd_q1.bin
PASS gelu_erf_bwd  n=64  rel_L2=2.45e-07 max_rel=2.88e-06
                         max_abs=3.89e-07 over_tol=0
```

and in the ALU/TMU model, 0/64 elements outside `2e-5` with no TMU address
outside the mapped arena.

It is wired into `vc4_bwop_run(VC4_BW_GELU_ERF)` (one launch; the previous
entry point returned `VC4_ERR_UNSUPPORTED`), and the registry row that already
advertised the op is now true.  `VC4_BW_GELU_ERF`'s old "use the tanh form"
refusal is gone.

## 6. What the ungated kernels replace

Measured launch counts of the compositions, from `docs/OPS-GEOMETRY.md` and
`docs/OPS-GAPS.md` (same board, same clock pin):

| op | composition today | launches |
|---|---|---|
| `GridSample` (`1x3x6x6`, `5x5` grid) | four predicated `copy2d` taps + QPU weights | 238 |
| `MaxRoiPool` (2 ROIs, `2x2`) | one window copy per channel plane | 238 |
| `DepthToSpace` (`1x4x4x4`, bs 2) | one laned gather per 16 output words | 32 |

A working `vc4_gather` covers 3,072 outputs per launch (12 QPUs × 256), so
the 900-element GridSample case above is one gather launch plus the index and
weight kernels, and `DepthToSpace`'s 1,024 output words are one gather launch.
Those two figures are projections from the measured capacity, not measured
compositions: `infer/python/vc4infer/_dispatch_*.py` is out of scope for this
change, so the runner still takes the composed path.

`vc4_blit` had no callers in the runner (it is a public API entry point); its
measured effect is per call: one launch for any 64 B multiple, against one
group per 64 B before.

`vc4_reduce` and `vc4_cumsum` would collapse the `sum`/`mean`/`global_avgpool`
compositions (`vc4_reduce` + `vc4_affine` per axis) and `CumSum`, and
`vc4_select3` the `Clip`/`PReLU`/`Where` compositions; those numbers are not
quoted here because they were not measured in this run.

## 7. The activation epilogue in the GEMM store

`conv -> relu` used to be two launches: the GEMM wrote `C`, then an
elementwise kernel read it back and wrote the activation.  The GEMM now applies
the activation to the value it stores, so the pair is one launch.  The bodies
are the *same bodies* the elementwise kernels use (`kops.act_apply` mirrors
`kgen._ew_compute`), which is why the result is bit-identical to the
two-launch path and not merely within tolerance.

### 7.1 The budget before writing any code

The code slot is 2,048 instructions (16 KiB, `docs/TRAINING-MEMORY.md`); the
JIT refuses a larger blob.  The daemon's protocol-5 program table holds 256
such slots, so a detector's 133 distinct kernels stay resident; a slot cannot
be widened to hold more than one kernel, which is why the table grew instead.  Both GEMMs have room to spare in their store
path, and the two have very different register situations:

| kernel | tile | C vectors live at the store | free B registers | free A registers |
|---|---|---:|---:|---|
| `gemm3` (asm3, the conv path) | 16x16 | 16 (A file r0..r15) | 19 (B0..B18) | 7 (the dead addressing state) |
| `gemm` (asm10, the matmul) | 8x64 | 32 (A r0..r15 **and** B r0..r15) | 1, and it is reloaded per column block | 3 |

The mul ALU writes only the A file or an accumulator, so every temporary an
activation body needs is an A-file register; a body that consumes a B-file C
vector must copy the result back with `ws=1`.

### 7.2 Measured word counts

`kgen.py gemm3 --M 25600 --N 16 --K 144 --bmode 1 --nqpu 8 [--act N]` and
`kgen.py gemm --M 512 --N 512 --K 512 --nqpu 8 [--act N]`:

| activation | `gemm3` words | delta | `gemm` (asm10) words | delta |
|---|---:|---:|---:|---:|
| none | 274 | - | 687 | - |
| relu | 290 | +16 | 722 | +35 |
| clip(0,6) | 310 | +36 | 756 | +69 |
| silu | 1100 | +826 | - | - |

`relu` is one word per C vector (`FMAX v, v, 0.0f`, the small immediate the ALU
sign-extends to `+0.0f`), plus 3 words in asm10 for the zero the B-file vectors
must read from an A register: a B-file operand and a small immediate cannot
share the raddr_b port in the same word.

`clip(0,6)` is that plus one `FMIN` per vector and one constant: 34 words for
the 16 conv vectors (the min reads the max's result, and a GP read needs three
words after the write, so three vectors are in flight at once; the single
leftover vector pays two nops), and 64 for the matmul plus 5 words of loads --
the matmul needs 6.0f in *both* files, because whichever file a C vector is in
is the one the constant cannot use.  The B-file copy goes into the column
block's 4-row step (B30), which the next column block reloads.

`silu` is the elementwise body's 50 words per C vector -- the 2^(-x log2 e)
reduction with its +/-90 clamp, the degree-5 `2^f` polynomial and the 3x new
Newton reciprocal -- plus one word to copy x into a B register (the final
multiply needs one A and one B operand), and the twelve constants cost 24
words of loads.  It fits the conv GEMM with 948 words to spare.

The counts are the same for every shape: the tile, not M/N/K, fixes them.

### 7.3 What does not fit where

The body cost is per C vector, so the 8x64 matmul tile pays twice the conv
tile's price for the same activation, and it starts from a worse register
position: its 32 C accumulators occupy *both* files completely, leaving one
reloadable B register and three dead A registers.

A body needs three things: A-file temporaries (3 available in asm10, 7 in
asm3), a B-file scratch for the `x` copy that keeps the final multiply on one
A and one B port, and float constants in the file the C vector is *not* in.
Whether a given activation fits is therefore a question about those three
pools, not about the instruction slot: at 32 vectors, even the longest body
fits 2,048 words, but it cannot find the registers to run.

### 7.4 The matmul tile cannot take silu

The asm10 tile's 32 C vectors take A-file r0..r15 and B-file r0..r15, the
three vector sets take A16..A19/A28/A29 and B16..B19/B28/B29, and the tile
loop's state takes the rest of both files.  After the k loop exactly one B
register is dead -- the column block's 4-row step (B30), which the next
column block reloads -- and three A registers.

`relu` needs neither: its bound is the small immediate 0.  `clip(0,6)` needs
6.0f in *both* files, because whichever file a C vector lives in is the one
the constant cannot use; B30 and A23 (the k counter, also reloaded) carry
them.  `silu` needs twelve, and the instruction slot is not the obstacle:
32 vectors x 37 words is 1,184, which would fit 2,048.  The register file
is.  `vc4_matmul_act` therefore returns `VC4_ERR_UNSUPPORTED` for silu
rather than running an unfused launch, and `kgen.py gemm --act 3` refuses to
generate it: a kernel that cannot be built must not be silently different.

The conv GEMM has 19 free B registers, which is why the model that needs
the fusion (YOLOv8n's 57 conv -> silu pairs) is covered and the one that
does not (a matmul followed by silu) is not.

`docs/OPTIMIZER.md` records which activations the two paths accept and why.

## 8. Reproducing

```
# model (no GPU): every kernel, including the store address
python3 infer/tests/emu_gated.py gather
python3 infer/tests/emu_gated.py reduce argmin
python3 infer/tests/emu_gated.py sel3 where
python3 infer/tests/emu_gated.py cumsum
python3 infer/tests/emu_gated.py gelu_erf_bwd

# silicon, one entry point per process
VC4INFER_DAEMON=1 ./infer/build/test_gated blit
VC4INFER_DAEMON=1 ./training/build/test_grad 1     # or infer/build/test_grad
```

`emu_gated.py` lives in the tests tree (it imports `kgen`, `kops` and
`emu_alu`); `test_gated.c` is built by the suite and `run_tests.sh` runs its
five subcommands after `test_copyops`.
## 9. Measurement caveat: the memory path's stall time is intermittent

Any GFLOP/s or kernel-ms figure from this board is only comparable with another
one taken in the same memory-path state, and that state is not a boot, a clock
or a temperature.  Measured on 2026-11-23, one private daemon, one client:

| quantity | fast sample | slow sample |
|---|---|---|
| 512^3 asm10 GEMM | 26.58 ms, 10.10 GFLOP/s | 34.17 ms, 7.86 GFLOP/s |
| cycle count per 8x64 tile-k | 3269 | 3269 |
| `core` / `v3d` / `sdram` clock | 400 / 300 / 166.7 MHz | 400 / 300 / 166.7 MHz |
| `get_throttled` | 0x0 | 0x0 |
| temperature | 47-49 C | 47-52 C |
| 256^3 pack (512 KiB) | 1.17 GB/s | 0.85 GB/s |

The cycle count is `gpu_cycles / ((M/8)*(N/64)*(K/16))`, where `gpu_cycles` is
V3D `EXECUTION_CYCLES` summed over the QPUs.  It was identical in every run of
this pass, fast or slow: the program, the QPU issue rate and the instruction
schedule did not change.  The wall time did, which means the QPU spent the
difference stalled on memory.  The docs' earlier 13.745 vs 8.12 GFLOP/s for the
same `gemm6_256.bin` is the same signature.

The state changes within a run, not only between boots.  The autotuner measured
one launch at 28.581 ms, then the same launch (same `pack_q=4 gemm_q=12`) at
34.130 ms ten seconds later, with the same cycle count.  Twelve consecutive
512^3 runs over 15 s, one second apart, sat at 26.58-27.55 ms; the inflated
observations are isolated minutes apart and do not follow a warm-up ramp.

What it is not:

* **not a clock change visible to the kernel.** `measure_clock core`, `v3d`,
  `arm` and the `clk_summary` rates for `fw-clk-core`, `fw-clk-v3d` and
  `sdram` were sampled in both states and did not differ; the firmware config
  is pinned (`core_freq=400`, `core_freq_min=400`, `gpu_freq=300`,
  `sdram_freq=450`).  `sdram` reports 166.7 MHz in both, so that field does
  not discriminate.
* **not thermal.** Every tabulated run stayed between 46 and 55 C,
  `throttled=0x0`; the slow observations were not the hot ones.
* **not the throttle/under-voltage latch.** A boot that showed
  `throttled=0x50000` (an earlier under-voltage event, nothing current)
  produced slow readings, but a boot with `0x0` produced both fast and slow
  readings, so the flag is a coincidence of the slow boot, not the cause.
* **not a degraded daemon arena.** A 48 MiB arena instead of 64 MiB fails a
  conv's scratch allocation outright (`out of arena memory`), it does not slow
  a kernel.

The remaining candidates are DRAM contention from another master (an ARM
process, the display pipe, a concurrent GPU client) and a firmware
memory-controller state that no exposed register reports.  I did not reproduce
it on demand with one client and the board idle, so it is recorded here as
nondeterministic rather than attributed.

Rules for any later measurement on this board:

1. print `cyc/tile-k` beside every ms; if it moved, a different program ran,
   if it did not, the difference is stall time;
2. interleave A/B (default, candidate, default) instead of measuring the
   baseline once and the candidate after a search;
3. never compare two absolute ms taken more than a minute apart without an
   interleaved reference;
4. record `get_throttled`, the temperature and `core`/`v3d` at the start and
   end of the run, `tools/thermal-guard.sh --max 75` does the latter.
