# The five gated kernels, and the exact erf-form GELU derivative

This file records the engine-level work on the kernels that `vc4_guard.c`
refused: `vc4_gather`, `vc4_blit`, `vc4_reduce`, `vc4_cumsum`, `vc4_select3`,
and the `Gelu`-erf backward that could not fit the 2,048-instruction code slot.

Status at the end of this run:

| entry point | status | evidence |
|---|---|---|
| `vc4_blit` | **fixed, ungated** | byte-exact on silicon: 0/1600 B and 0/208 B, `tests/test_gated blit` |
| `vc4_gather` | gated, defects fixed | exact in the ALU/TMU model; silicon still returns another launch's bytes |
| `vc4_reduce` | gated, defects fixed | exact in the ALU/TMU model (6 ops); silicon still returns another launch's bytes |
| `vc4_cumsum` | gated, defects fixed | exact in the ALU/TMU model; silicon still returns another launch's bytes |
| `vc4_select3` | gated, defects fixed | exact in the ALU/TMU model (3 ops); silicon still returns another launch's bytes |
| `VC4_BW_GELU_ERF` | **implemented** | one 1967-word kernel at 4 vectors/group; gradcheck rel_L2 2.45e-07, over_tol 0 |

The four remaining gates are honest about what they are: each names the
defects that were found and fixed *and* the measured silicon symptom.  The
element checks in `infer/tests/test_gated.c` are the test that turns green the
moment the last defect is closed; today they print `GATED`.

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

## 4. What the four remaining gates are waiting for

The model says the emitted code computes the right thing, including the store
address.  On silicon, with the same uniforms, the output is not the kernel's
own result:

* a marker body that stores `42.0` in every lane returns a *previous launch's*
  bytes, and the same value appears from the emulator's point of view only if
  the address arithmetic is right;
* `vc4_gather` at `n=1` (one 16-lane vector, one QPU) is wrong on the very
  first element, so this is not tail or multi-group arithmetic;
* `emu_kernel.run_kernel('tanh', nvec=4)` (the same harness that carries
  `VC4_BW_GELU_ERF`, which *is* exact on silicon) works, so the harness and
  the nv4 store are not the difference.

The remaining difference between those kernels and `build_ew`'s harness is in
`kops._prologue`/`_stream_base`/`emit_store16` as a whole: the nv16 store's
VPM row (`A_VPMQ`) and DMA row (`B_VPMQ7`) are both derived from the *uniform*
group index rather than the physical QPU, and the group's stream base is
staged through an accumulator.  The next step is to rebuild one of them
(`vc4_cumsum` is the smallest) directly on `build_ew`'s prologue and store,
which are element-verified on silicon, and then re-apply the algorithm.

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

## 6. Cost of the compositions that the gated kernels still serve

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

## 7. Reproducing

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
