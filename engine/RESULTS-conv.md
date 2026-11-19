# Convolution backward, col2im and pooling on VC4 (2026-11-19)

Everything here was measured in `/home/raspi/work/train-conv` on a Pi 3B with
stock clocks: **core 400 MHz, V3D (frequency 46) 300 MHz**.  Every measurement
below was taken with `vcgencmd measure_clock core` before and after the run;
the V3D clock is reported where it matters.  One GPU job at a time under
`/tmp/vc4-gpu.lock`.

## Kernels added

| kernel | shape | what runs on the QPU |
|---|---|---|
| `col2im` | 3x3 / 1x1, Cin%16==0 | scatter-add of the im2col gradient; one task = one (row block, channel block) 16x16 tile; fixed tap order in 16 registers, one store |
| `im2col --vout r` | 3x3 / 1x1, Cin%16==0 | the existing VDR/VPM gather with a vertical VDW: row-major `[pixel][k]` (the dW path) instead of the GEMM's `[k][pixel]` panel |
| `pool fwd` | 2x2/2, H%8==0, W%16==0 | 4 output rows per task; 2x2 max + active-lane mask; verified 4-row `units=4/depth=16` VPM->DMA store |
| `pool bwd` | 2x2/2, H%4==0, W%16==0 | 4 input rows per task; 4 group loads + own + dy, argmax mask, 4-row store |
| `vc4_conv2d_backward` | stride 1, KH=KW in {1,3} | `dA = dY*W^T` and `dW = dY^T*A` via the asm3 GEMM (`pack3`+`gemm3`), dX via `col2im` |

Both new kernel families are **branch-free, one task per QPU**; the host
enumerates tasks and launches `ceil(tasks/12)` times.  There is no
read-modify-write and no cross-QPU accumulation: a destination tile is owned
by one task, so the result is independent of QPU scheduling.  All three
kernels were validated against a float64 CPU oracle on the board, and
before that against the repo's functional QPU simulator (`asmq/vsim.py`).

## Gradient checks (CPU checker only; forwards and gradients are QPU results)

`tests/test_convgrad`, h = 0.05, tol 1e-3, central differences of
`L = sum(dY * out)`.  "oracle" is a float64 reference; "FD" is the finite
difference.  Elements with |fd| <= 1e-2 are below the noise floor.

| check | n | rel L2 | max rel | over tol | notes |
|---|---|---|---|---|---|
| conv dW vs float64 oracle, 6x6 pad0 | 2304 | 3.96e-07 |  |  | max abs 2.1e-6 |
| conv dX vs float64 oracle, 6x6 pad0 | 576 | 4.85e-07 |  |  | max abs 3.7e-6 |
| conv dW FD, 6x6 pad0 | 2304 | 4.26e-06 | 4.6e-4 | 0 | 19 elements \|fd\|<=1e-2 |
| conv dX FD, 6x6 pad0 | 576 | 5.48e-06 | 3.3e-4 | 0 | 1 element \|fd\|<=1e-2 |
| conv dW vs float64 oracle, 6x6 pad1 | 2304 | 6.92e-07 |  |  | max abs 6.2e-6 |
| conv dX vs float64 oracle, 6x6 pad1 | 576 | 4.97e-07 |  |  | max abs 3.0e-6 |
| conv dW FD, 6x6 pad1 | 2304 | 3.60e-06 | 8.2e-4 | 0 | 8 elements \|fd\|<=1e-2 |
| conv dX FD, 6x6 pad1 | 576 | 5.24e-06 | 4.5e-4 | 0 | 1 element \|fd\|<=1e-2 |
| maxpool bwd vs oracle, 16x16 C=16 | 4096 | 0 (exact) | 0 | 0 | 0 mismatches |
| maxpool bwd FD, 16x16 C=16 | 4096 | 2.34e-07 | 3.6e-7 | 0 | 395 kink elements skipped, 2877 \|fd\|<=1e-2 |
| col2im vs float64 oracle, 4x4 out | 1536 |  | 5.8e-8 | 0 | max abs 2.4e-7 |
| im2col (row-major) vs oracle | 9216 | 0 (exact) | 0 | 0 | bit-exact gather |

The maxpool is piecewise linear: a central difference crosses a kink whenever
the 2x2 argmax changes within +-h.  Those elements are counted and excluded
from the FD row (395 of 4096); the direct oracle comparison covers every
element exactly, including ties and kinks.  The conv sums are fp32 with the
QPU's truncating `fadd`, so the FD agreement is limited by the forward fp32
noise, not by the backward formulas.

## Toy conv net

`conv 3x3/1 (1->8) -> relu -> maxpool 2x2 -> flatten 8x8x8=512 -> fc(10)`,
18x18 input, 640 train / 160 test deterministic synthetic samples (a Gaussian
blob at one of ten positions + noise), batch 16, full-batch FC/softmax.
Host work is layout only (1->16 channel padding with zeros, 8->16 filter
padding, pooled-row gather/scatter, the tap-major weight permutation); every
MAC, activation, pool, gradient and optimizer update runs on a QPU.

SGD lr 0.2 (150 steps):

```
initial: train loss=2.305825 acc=12.50%
step    1: loss=2.305825472 acc=12.50%
step   25: loss=0.258082092 acc=100.00%
step   50: loss=0.033654071 acc=100.00%
step   75: loss=0.016181692 acc=100.00%
step  100: loss=0.008949540 acc=100.00%
step  125: loss=0.007744060 acc=100.00%
step  150: loss=0.005039738 acc=100.00%
final: train loss=0.005180205 acc=100.00%  test loss=0.005131163 acc=100.00%
time 19.864 s for 150 steps, 7.56 steps/s, 121 samples/s (batch 16)
weights hash conv=1ff89d6f671f1e61 fc=5bf35e2e6e6dd927
```

Adam lr 0.01 (150 steps):

```
initial: train loss=2.305825 acc=12.50%
step   25: loss=0.079744421 acc=100.00%
step   50: loss=0.001666779 acc=100.00%
step   75: loss=0.001114126 acc=100.00%
step  100: loss=0.000645406 acc=100.00%
step  125: loss=0.000769413 acc=100.00%
step  150: loss=0.000533860 acc=100.00%
final: train loss=0.000556604 acc=100.00%  test loss=0.000570589 acc=100.00%
time 20.446 s for 150 steps, 7.34 steps/s, 117 samples/s (batch 16)
weights hash conv=06804c3e1b5447c4 fc=3315e0d75b2a7f74
```

Both were run twice; the printed losses and both weight hashes are
**bit-identical** across runs (timing aside).  Loss drops 445x (SGD) / 4140x
(Adam), 100% train and test accuracy.

Throughput is **launch-bound**: the engine API is single-image NCHW, so a
16-image step is ~26 launches/image (conv fwd 3, conv bwd 10, relu 2, pool 9)
= ~420 launches at ~0.3 ms each = ~130 ms/step.  The MLP (no per-image loop)
stays at its documented 103 steps/s (65.9k samples/s) when run alone at
300 MHz V3D; the 52 steps/s seen inside the full suite is sustained-load
clock droop, not a regression.

## Op coverage: 15/27 -> 21/34

Added as dispatchable: `im2col`, `col2im`, `conv2d/bwd_dx`, `conv2d/bwd_dw`,
`maxpool/2x2s2`, `maxpool_bwd`.  New honest "not implemented" rows:
`maxpool/3x3`, `avgpool`.  Still unsupported: conv 5x5/stride2/depthwise/
grouped, f16, batched matmul, layernorm, concat/slice, arbitrary permute,
upsample, quantize.

## Suite

`REPS=5 ./run_tests.sh` -> **ALL TESTS PASSED** (rc=0): maths, dispatch
(21/34), ew 5/5, matmul 128^3 5.95 GFLOP/s and 256^3 8.45 GFLOP/s, conv
(shapes) 5/5, train_ops, test_grad (GRADCHECK PASSED), test_convgrad
(CONV GRADCHECK PASSED), MLP trainer SGD/Adam (TRAIN PASSED), conv trainer
SGD/Adam (CONV TRAIN PASSED).  Board `STATE: LIVE` after every stage.

## Honest limits

- **Launch-bound throughput.**  A per-image conv loop is ~120 samples/s.  The
  next step is a batched im2col/col2im (batch as extra M row blocks with a
  per-image source/destination offset) to cut the launch count by the batch
  factor.
- Conv backward is Cin%Cout multiple of 16, stride 1, 3x3/1x1; the trainer
  pads 1->16 channels and 8->16 filters on the host with zeros.
- **No conv bias gradient** (conv bias is forward-only; the FC bias is an
  augmented column in the weight matrix, so it does get a gradient).
- maxpool: 2x2/2 only, H%8==0 for fwd and H%4==0 for bwd, W%16==0, output
  row pitch a multiple of 16.  Ties share the gradient (subgradient).
- col2im is deterministic but not IEEE: 9 truncating `fadd`s in fixed tap
  order, ~1e-7 relative to a float64 oracle.
- The 4-row `units=4/depth=16` VPM->DMA store is the only proven DMA shape;
  a single-row `units=1` store is not used.
- One wedge happened during development: an early pool-bwd wrapper was
  missing a uniform word, the kernel read a wild TMU address, and the V3D
  queue died (`STATE: LIVE`, `SRQCS` done never increments; `systemctl
  reboot` recovered it).  Fixed; all kernels are now simulated offline first.
