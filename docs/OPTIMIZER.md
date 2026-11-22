# The inference optimiser: what TensorRT, MIGraphX and ncnn do, and what this engine needs

The engine runs a graph node by node.  TensorRT, MIGraphX and ncnn run a graph
after an ahead-of-time optimiser: fusion, layout assignment, memory planning,
weight prepacking, kernel selection and, on some targets, precision reduction.
This document is the plan for giving `vc4infer` the same class of optimiser,
with the constraints this hardware imposes.

Nothing here changes what a model computes.  Each pass is gated on the same
oracles the executor already uses: `infer/tools/op_matrix.py` for operator
coverage and the model corpus for end-to-end parity, at the tolerance recorded
in `docs/ACCURACY.md`.

## Baseline

On YOLOv8n at 160x160, 481.7 ms wall, 802 launches:

| component | ms | share |
|---|---:|---:|
| QPU kernel time | 263 | 55% |
| launch path | 222 | 46% |
| host-side dispatch, packing, conversion | included above | |

The QPU time alone is already 3.5x ncnn's total 74.8 ms, so fusion cannot
reach parity by removing launches alone; it must also remove passes over
memory.  The realistic target is 250-350 ms, with the gap to ncnn stated
rather than hidden.

## What the reference engines do

- **TensorRT**: layer fusion, kernel autotuning per shape, tensor-format
  selection, memory reuse, optional low precision.
- **MIGraphX**: graph rewrites in MIGraphX IR, target-specific fusion,
  layout-aware scheduling.
- **ncnn**: PNNX conversion with operator fusion, weight repacking, layout
  choices, and hand-tuned kernels per layer type.

The common core is: an explicit graph IR, a deterministic pass pipeline, and
measurement after every pass.

## Pass pipeline

Order matters; each pass is individually switchable so a regression can be
bisected to one rewrite.

| # | pass | what it does | expected gain |
|---|---|---|---|
1 | canonicalise | constant folding, dead-node removal, identity elimination | fewer nodes, no numeric change |
2 | elementwise fusion | merge chains (`add -> relu -> add`, `mul -> add`) into one EWB launch | removes launches; the biggest single win |
3 | epilogue fusion | bias and activation folded into the GEMM/conv that produces the tile | 2-3 launches per conv |
4 | layout assignment | keep packed NHWC between compatible nodes; convert only at boundaries | removes copy/transpose bandwidth |
5 | weight prepacking | pack weights once at plan time into the arena layout | removes per-launch host work and repeated packing |
6 | memory planning | one arena plan with buffer reuse and liveness, replacing per-op scratch growth | lowers peak arena; more models fit |
7 | launch batching | several kernels per daemon round-trip, or a persistent work queue | attacks the 222 ms directly |
8 | kernel autotuning | QPU counts per shape (GEMM and its packer), recorded per generator | recorded per shape; see "Pass 8" |

Passes 1-3 are graph rewrites on the Python side; 4 and 6 touch tensor layout
and arena planning; 7 needs runtime work in `vc4d` and the client; 8 is
measurement-driven and must be recorded per shape.

## Correctness gates

- every pass has an off switch, and the default stays off until it is measured;
- after each pass: `op_matrix.py` (152/152 expressible, 0 refused, 0 mismatch);
- after each pass: YOLOv8n 160 parity against ONNX Runtime CPU at the recorded
  tolerance, with detection count and box delta;
- fusing changes summation order, so tolerances are re-established per pass,
  not inherited;
- the optimiser never runs an operator the executor cannot, and never falls
  back to the ARM.

## Phases

| phase | content | exit criterion |
|---|---|---|
A | IR and pass harness, pass 1 | same numbers, passes switchable, tests green |
B | passes 2-3 | launches for v8n160 down by >= 40%, wall < 420 ms, parity held |
C | passes 4-6 | wall < 380 ms, arena peak down >= 25% |
D | pass 7 | wall < 320 ms |
E | pass 8 | recorded per-shape gains, no regression on the corpus |

Each phase is a separate merge, and each number is added to
`docs/BENCHMARKS.md` with its conditions.

## Pass 8: per-shape kernel autotuning

`vc4_autotune` picks the QPU counts the generated kernels leave to the host,
measures the legal alternatives on the device, and records the winner under the
kernel cache.  The executor reads the record for a shape when autotuning is
enabled and runs the launch it has always run when it is not.  Implementation:
`infer/src/vc4_autotune.c`; C API in `infer/include/vc4infer.h`; Python
`Engine.autotune`.

### The space

| op | axis | values tried | legality |
|---|---|---|---|
| `matmul` | `pack_q` | 4, 2, 1 | `<= 4` (the blob is registered for 4) and divides `M/8` |
| `matmul` | `gemm_q` | 12, 8, 6, 4, 3, 2, 1 | `1..12` and `<= M/8` |
| `conv2d` | `im2col_q` | 4, 2, 1 | `<= 4` and divides `M/16` |
| `conv2d` | `gemm_q` | 12, 8, 6, 4, 3, 2, 1 | `1..12` and `<= M/16` |

`M` is the padded output row count the GEMM iterates: `align8(M)` rows for a
matmul and `Ho * align16(Wo)` for a conv.  The auxiliary kernels (the A-panel
packer and the im2col gather) partition their row blocks exactly and have no
in-kernel guard, so their QPU count must divide the work; the GEMMs guard their
row-block loop with `rb < TR` and only need more row blocks than QPUs.  The
12-QPU ceiling is the hardware's; the kernel cache separately refuses to launch
a blob with more QPUs than it was registered for.

The tile shape and the vectors per group are *not* searched.  They are part of
each kernel's hand-verified instruction schedule (the asm10 8x64 tile, the asm3
16x16 tile, `build_ew`'s 16-vector group in front of the verified 256-float
store).  A different tile is a different program, and a wrong QPU program is not
a recoverable error on this board.  The QPU counts change loop bounds and
uniform strides only, so every candidate runs the same verified body over the
same data.

Coarse pass: the default first (it is the baseline), then each legal GEMM count
with the default packer, then each legal packer with the default GEMM - at most
11 candidates, one axis at a time.  Fine pass: the GEMM counts one and two
either side of the coarse winner, then auxiliary counts the coarse pass did not
try - at most 4 more, deduplicated.

### Measurement

One warm-up run per candidate (which also generates and uploads its blob) is
discarded, then `reps` runs are timed with `CLOCK_MONOTONIC` and the median
recorded: an ARM scheduler slice or a vc4d round trip is a common outlier and
the distribution is not normal.  The whole op is timed, so host-side packing and
the NCHW store of a conv are part of the number.

Bounded twice: at most 14 measured parameter sets, and a 30 s wall-clock budget
that starts before the first blob is generated.  `VC4INFER_AUTOTUNE_MAX`,
`VC4INFER_AUTOTUNE_BUDGET_MS` and `VC4INFER_AUTOTUNE_REPS` override the three
numbers.

Before a choice is recorded, the op is run again with it and the output compared
with the default launch's output at the tolerance of `docs/ACCURACY.md`
(`atol = rtol = 5e-4`).  A disagreement is reported and nothing is cached.  On
this hardware the chosen launch is expected to be bit-identical - the QPU count
moves row blocks between QPUs, not the k order - and the check is there because
"expected" is not a verification.

### The cache

- Path: `<kernel cache>/autotune/<key>.tune`, where the key is the first 8
  bytes of
  `sha256("vc4infer-autotune-1|" op "|" shape "|" digest)`.
- Digest: the same `sha256(kgen.py, kops.py beside it, ABI tag)` the blob cache
  writes into every `.bin.stamp`.  A generator change therefore invalidates
  tuned choices through the mechanism that already exists; there is no second
  hash to keep in step.
- Content: one `trial` line per measured parameter set with its median, then
  `best`, the blob name, and the two medians.  The record is written to a
  temporary file and renamed, so a power loss cannot leave a half record.
- Lookup: a 32-entry per-context memo, then one file open per shape.  With
  autotuning off, a lookup is one `getenv` and no file I/O, which is what keeps
  the existing numbers reproducible.
- A record that is stale (different digest), truncated, foreign, or names a
  launch the filter rejects is a miss: the op runs its default and the next
  `vc4_autotune` rewrites the record.

### API

```c
vc4_tune_shape sh = { .m = 512, .n = 512, .k = 512 };
vc4_tune_result r;
vc4_autotune(ctx, VC4_OP_MATMUL, &sh, NULL, &r);
/* r.params.gemm_q, r.params.pack_q, r.best_ms, r.default_ms, r.max_abs */
```

```python
eng.autotune("matmul", (512, 512, 512))
eng.autotune("conv2d", {"cin": 32, "cout": 32, "h": 32, "w": 32,
                        "kh": 3, "kw": 3, "pad": 1})
```

`VC4INFER_AUTOTUNE=1` makes every process consult the records without a call;
a successful `vc4_autotune` enables the same for its context.  Both are off by
default.

### Measured on the board

```
cd ~/work/vc4-autotune
export TMPDIR=$HOME/scratch/tmp
export VC4INFER_CACHE=$HOME/scratch/kcache-autotune      # an empty one: cold
make -C infer -j4
VC4INFER_DAEMON=1 tools/thermal-guard.sh --max 71 -- ./infer/build/bench_autotune 5
VC4INFER_DAEMON=1 tools/thermal-guard.sh --max 71 -- ./infer/build/bench_autotune 5
```

Not measured yet.  The QPU itself was idle, but two numpy reference runs and a
Python environment install held the CPU at 100% and the SoC at 70-82 C for the
whole window available for this change; a benchmark there measures a different
clock and the 71 C guard would stop it.  The commands above are the run, once
the board is idle and at ~58 C, one client at a time and no `flock` around the
client.  The same applies to the API check:

```
VC4INFER_CACHE=$HOME/scratch/kcache-autotune-api PYTHONPATH=infer/python \
    python3 infer/tests/test_autotune_api.py
```

### Cost of a cold search

The QPU half is pending with the table above.  The half that can be measured
without the QPU - blob generation, which happens once per (shape, generator) -
is the dominant term:

| `kgen.py gemm --M 512 --N 512 --K 512 --nqpu` | 12 | 8 | 6 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| generation, s | 0.92 | 1.04 | 0.89 | 0.79 | 0.78 | 0.77 | 0.88 |

Measured with the board at a load average of ~3.9 (other work), so these are
pessimistic.  A 512^3 cold search generates seven distinct GEMM blobs (one per
QPU count; the packer variants reuse one), i.e. roughly 6 s of generation, and
then runs 6 op executions per candidate - 10 coarse candidates and up to 4 fine
ones within the 14-set cap.  A generator edit re-pays this once per shape,
because the digest changes.

### Not covered

- the NHWC conv path (`vc4_conv_nhwc`) and the grouped/depthwise variants have
  their own launch geometry and are not keyed by this record; only `vc4_matmul`
  and the NCHW `vc4_conv2d` consult it;
- a generator edit changes the digest and re-measures every shape on next use,
  including a comment-only edit: the cost is the cold-search cost below, once
  per shape;
- elementwise kernels are not tuned: their group size is fixed by the store
  path, not by the launch.

## Non-goals

- no fp16 or int8 path in this work; the engine is fp32 and the board has no
  usable low-precision multiplier array for this purpose (`docs/int8-RESULTS.md`);
- no change to the operator semantics or the coverage denominator;
- no hidden host computation.
