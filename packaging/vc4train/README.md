# vc4train

The engine's own training demo, installed as a command:

```
$ vc4train --opt adam --steps 300
libvc4infer 1.2.2 toy trainer: 64-32-10, opt=adam, seed=12345, arena 96.0 MiB
step    0: loss=2.302585093 acc=9.84%
step   50: loss=0.4213...
...
final: train loss=... acc=...%  test loss=... acc=...%
TRAIN PASSED
```

The point is verification, not benchmarks: every multiply-accumulate, every
activation, the fused softmax + cross-entropy gradient and the whole optimizer
step run as QPU kernels.  The ARM builds the synthetic dataset and the initial
weights, lays out the bias-augmented matrices (row copies), fills scalar
hyperparameters and prints the two numbers the GPU produced.  No arithmetic
falls back to the CPU.

The program exits non-zero unless the training loss drops at least 5x and the
test accuracy reaches 90%, so `vc4train` is a one-command "can this engine
train?" check.

## Options

| flag | meaning |
|---|---|
| `--opt sgd\|adam` | optimizer (default `sgd`) |
| `--steps N` | steps (default 300) |
| `--seed S` | PRNG seed (default 12345) |

## Backend

`VC4INFER_DAEMON=1` (set by `/etc/profile.d/vc4.sh`) routes every launch
through `/run/vc4d.sock`, unprivileged.  Without it the binary uses `/dev/mem`
and must run as root.  It never overclocks anything: the board stays at stock
core 400 MHz / V3D 300 MHz.

## Limits

This is a toy: fp32 only, no FMA, 76 MiB of device memory, full-batch.  It
demonstrates that the backward kernels are real; general training is out of
scope for this stack.
