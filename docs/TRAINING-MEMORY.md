# Training memory budget — what fits in the arena

This page answers the question a framework user actually has: *can my model
train on this board at all?*

**Read this first.**  The numbers marked *formula* below are computed from the
allocator and the op implementations.  The numbers marked *measured* are what
the hardware reported.  At the time of writing the device path for the
composite training ops (batchnorm, softmax, pooling) wedges the QPU, so the
end-to-end "largest net that trains" experiment could **not** be run; the
memcpy-and-allocator part of the budget *is* measured, and the activation part
is arithmetic over the buffers the demo allocates.  Treat the ceiling below as
an upper bound, not a demonstrated one.

## 1. The pool

| item | size | how |
|---|---|---|
| one dma-buf arena | 64 MiB default (`vc4infer` direct), 24–96 MiB via `VC4INFER_ARENA_MB` / the daemon's `--arena-mb` | `vc4_arena_bytes()` |
| layout | `0x000000` signature, `0x001000` 32 × 16 KiB kernel code slots, `0x210000` uniform areas, `0x300000` data heap | `vc4_int.h` |
| usable data heap | `arena_bytes − 0x300000` | measured: 24 MiB arena → `vc4_arena_used()` starts at 3.00 MiB, so 21.0 MiB of data |
| persistent buffers | bump-allocated upward, never freed (`vc4_alloc`) | params, grads, optimiser state, saved activations, label buffers |
| per-op scratch | bump-allocated upward from `arena_top`, reset before/after every op | GEMM panels, padded tails |
| backward workspace | bump-allocated **downward** from the end of the arena (`vc4__ws_alloc`) | the temporaries of a composed backward |
| workspace left | `ws_top − max(arena_top, scratch)` | a composed op that needs more returns `VC4_ERR_NOMEM` with the exact byte counts |

So the data budget is

```
data = arena_bytes - 3 MiB                       (headers + code + uniforms)
free = data - params - grads - opt_state - activations - label_bufs
```

and the *workspace* shares `free` with the persistent allocations: a backward
composition that needs `W` bytes of temporaries reduces the activation budget
by `W`.

## 2. Cost of the state, per float32 parameter

| term | multiplier | notes |
|---|---|---|
| parameters | 1× | conv weights, fc weights, gamma, beta, running mean/var (BN adds 4×C) |
| gradients | 1× | the same shapes as the parameters |
| SGD | 0× | in place |
| Adam | 2× | `m` and `v`, plus one full-size scratch for the update (`s1`) |
| activations saved for the backward | per layer | conv: the padded input (im2col is recomputed); BN: the normalised output **and** the raw conv output; relu: the BN output (shared with BN); pool: the relu output; fc: the flattened features |

The demo net (`conv 16ch 3x3` + `fc 528×16`):

```
conv weights  16*16*3*3      = 2304 floats =   9 KiB
fc weights    528*16         = 8448 floats =  33 KiB
bn             4*16          =   64 floats =  0.25 KiB
params total                 = 10816 floats = 42 KiB
grads + Adam m/v             = 4x params    = 170 KiB
activations, batch 8:
  padded input   8*16*18*18  = 41472 floats = 162 KiB
  conv out (Z)   8*16*16*16  = 32768 floats = 128 KiB
  BN out         same                        = 128 KiB
  pool out       8*16*4*16   =  8192 floats =  32 KiB
  fc features    8*528       =  4224 floats =  17 KiB
  logits/dlogits 8*16        =   256 floats =   1 KiB
  total activations                         = 468 KiB
sum                                        = ~0.7 MiB
```

The whole demo fits in ~0.7 MiB of the 21 MiB a 24 MiB arena gives, and runs
with a 8 MiB arena.  That is why the demo trains at all where a ResNet would
not.

## 3. Where the ceiling is

Activations dominate: `params + 3×params (Adam) + activations`.  As a rule of
thumb, with a 64 MiB arena (61 MiB of data):

| model | params | Adam state | activation budget left | largest batch (approx) |
|---|---|---|---|---|
| MLP 512→512 | 0.26 M floats (1 MiB) | 3 MiB | ~56 MiB | thousands of rows |
| the demo CNN | 11 k floats (42 KiB) | 170 KiB | ~60 MiB | hundreds of images |
| conv 32ch 32×32, fc 2048→16 | 0.6 M floats (2.4 MiB) | 7 MiB | ~50 MiB | ~500 images at 32×32×32 |
| conv 64ch 64×64, fc 16 | 2.4 M floats (9.4 MiB) | 28 MiB | ~20 MiB | ~60 images at 64×64×64 |
| conv 128ch 128×128 | 9.4 M floats (38 MiB) | 113 MiB | **does not fit** | — |

The last row is the honest stop: with fp32, no gradient checkpointing and no
f16, a 128-channel 128×128 conv net does not fit in a 64 MiB arena at any
batch size, because the Adam state alone is 113 MiB.  The workarounds are the
usual ones — SGD instead of Adam (3× less state: 38 MiB of params+grads, which
does fit), a smaller batch, or gradient accumulation — and they are *outside*
this library: a framework plugin must implement them.

## 4. Fixed-size resources that do not scale with the model

| resource | limit | where |
|---|---|---|
| kernel code slots | 32 × 16 KiB | `VC4_MAX_KERNELS`, `VC4_CODE_SLOT`; the JIT refuses a blob > 16384 B (`2048` instructions) |
| one kernel body | 2048 instructions at 16 vectors/group | the store granularity; this is why several activation derivatives are compositions |
| GEMM tile padding | M→8, N→64, K→16 | `vc4_matmul` pads in scratch, so a matvec costs `n_out·64·n_in` MACs |
| irregular-layout table | 32 MiB (`VC4_MATVEC_MAX`) | above it the op returns `VC4_ERR_UNSUPPORTED` |
| pooled-output pitch | `Wo` → 16, `Ho` → 4 rows | `vc4_maxpooln` / `vc4_maxpool2x2` (the 4-row VPM store) |

## 5. Measuring it yourself

```sh
# what the allocator sees
VC4INFER_ARENA_MB=64 ./build/test_cnntrain --steps 1   # prints arena MiB
# per-op arena high-water (after each op the bump cursor only grows)
VC4INFER_DEBUG=1 ...
```

`vc4_arena_bytes()` and `vc4_arena_used()` are the two numbers to log: the
first is the pool, the second is how much of it the current model has taken.
A training loop that logs both after the first step tells you immediately
whether the model fits, before it wastes an epoch.
