# Training memory budget — what fits in the arena

This page answers the question a framework user actually has: *can my model
train on this board at all?*

Two kinds of numbers appear below: **measured** (what the board reported) and
**computed** (arithmetic over the allocator and the buffers a model takes).
Both are labelled.  The measured ceiling comes from the demo trainer, which
allocates its whole state in the arena and reports `vc4_arena_used()`.

## 1. The pool

| item | size | how |
|---|---|---|
| one dma-buf arena | 64 MiB default (direct), 8–96 MiB via `VC4INFER_ARENA_MB` / `vc4d --arena-mb` | `vc4_arena_bytes()` |
| layout | `0x000000` signature, `0x001000` 256 × 16 KiB kernel code slots, `0x410000` uniform areas, `0x510000` batch shadows, `0x600000` data heap | `vc4_int.h` |
| usable data heap | `arena_bytes − 0x600000` | measured: a 16 MiB arena reports 6.00 MiB used at init → 10.0 MiB of data (protocol 5; a protocol-4 daemon keeps the 3 MiB map) |
| persistent buffers | bump-allocated upward, never freed (`vc4_alloc`) | params, grads, optimiser state, saved activations, labels |
| per-op scratch | bump from `arena_top`, reset before/after every op | GEMM panels, padded tails |
| backward workspace | bump **downward** from the end of the arena (`vc4__ws_alloc`) | the temporaries of a composed backward |
| workspace left | `ws_top − max(arena_top, scratch)`; a composition that needs more returns `VC4_ERR_NOMEM` with the byte counts | |

```
data = arena_bytes - 3 MiB          (headers, code slots, uniforms)
free = data - params - grads - opt_state - activations - label_bufs - workspace
```

## 2. Cost of the state, per float32 parameter

| term | multiplier | notes |
|---|---|---|
| parameters | 1× | conv/fc weights, gamma, beta, running mean/var (BN adds 4·C) |
| gradients | 1× | the same shapes |
| SGD | 0× | in place |
| Adam | 2× | `m` and `v`, plus one full-size scratch for the update |
| activations saved for the backward | per layer | conv: the padded input (im2col is recomputed); BN: the raw conv output **and** the normalised output; relu: the BN output (shared); pool: the relu output; fc: the flattened features |

## 3. The demo net, measured

`test_cnntrain`: conv 16ch 3×3 (8 real) → BN(16) → relu → maxpool 2×2 →
fc 528×16, batch 8, 18×18 input.

```
params      conv 16*16*3*3 = 2304  +  fc 528*16 = 8448  +  bn 4*16 = 64
            = 10816 floats = 42 KiB
grads+Adam  4x params                    = 170 KiB
activations  batch 8:
  padded in   8*16*18*18 = 41472 floats  = 162 KiB
  conv out Z  8*16*16*16 = 32768 floats  = 128 KiB
  BN out      same                        = 128 KiB
  pool out    8*16*4*16  =  8192 floats  =  32 KiB
  fc features 8*528      =  4224 floats  =  17 KiB
  logits      8*16       =   256 floats  =   1 KiB
  total                                  = 468 KiB
sum                                      = 0.7 MiB
```

So the demo needs ~0.7 MiB of data and runs inside an 8 MiB arena.  The
exact `vc4_arena_used()` the demo prints is in section 5.

Measured (`test_cnntrain --batch 16`, which doubles the batch-dependent rows
above): **7.84 MiB of a 40 MiB arena used after 40 steps** (and of a 45 MiB
one in an earlier run), i.e. the whole model, its gradients, the SGD state,
the saved activations and the backward workspace fit in 8 MiB.  The rest is
scratch for the GEMM panels of the largest single op and the workspace
temporaries of the composite backwards.

Measured step time for that footprint: **1.29–1.37 steps/s** (40 steps in
29–31 s, 16 images per step ≈ 22 images/s) at stock clocks, including the
per-step loss readback.

## 4. Where the ceiling is

Activations dominate: `params + 3·params (Adam) + activations`.  For a 64 MiB
arena (61 MiB of data), computed:

| model | params | Adam state | activation budget left | largest batch (approx) |
|---|---|---|---|---|
| MLP 512→512 | 0.26 M floats (1 MiB) | 3 MiB | ~56 MiB | thousands of rows |
| the demo CNN | 11 k floats (42 KiB) | 170 KiB | ~60 MiB | hundreds of images |
| conv 32ch 32×32, fc 2048→16 | 0.6 M floats (2.4 MiB) | 7 MiB | ~50 MiB | ~500 images at 32×32×32 |
| conv 64ch 64×64, fc 16 | 2.4 M floats (9.4 MiB) | 28 MiB | ~20 MiB | ~60 images at 64×64×64 |
| conv 128ch 128×128 | 9.4 M floats (38 MiB) | 113 MiB | **does not fit** | — |

The last row is the honest stop: with fp32, no gradient checkpointing and no
f16, a 128-channel 128×128 conv net cannot train in a 64 MiB arena at any
batch size, because the Adam state alone is 113 MiB.  Workarounds live
outside this library: SGD instead of Adam (3× less state: 38 MiB of
params+grads, which does fit), a smaller batch, or gradient accumulation.

## 5. Measuring it yourself

```sh
# the allocator's view (the demo prints arena / used after its first step)
VC4INFER_ARENA_MB=16 ./build/test_cnntrain --steps 1
# per-op high-water: the bump cursor only grows
VC4INFER_DEBUG=1 ...
```

`vc4_arena_bytes()` and `vc4_arena_used()` are the two numbers to log: the
first is the pool, the second is what the model has taken.  Logging both after
the first step tells a user whether the model fits before an epoch is wasted.

## 6. Fixed-size resources that do not scale with the model

| resource | limit | where |
|---|---|---|
| kernel code slots | 32 × 16 KiB | the JIT refuses a blob > 16384 B (2048 instructions) |
| one kernel body | 2048 instructions at 16 vectors/group | why several derivatives are compositions |
| GEMM tile padding | M→8, N→64, K→16 | `vc4_matmul` pads in scratch, so a matvec costs `n_out·64·n_in` MACs |
| irregular-layout table | 32 MiB (`VC4_MATVEC_MAX`) | above it the op returns `VC4_ERR_UNSUPPORTED` |
| pooled-output pitch | `Wo` → 16, `Ho` → 4 rows | `vc4_maxpooln` / `vc4_maxpool2x2` (4-row VPM store) |
