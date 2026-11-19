# training/ — backward kernels and the toy trainer

The backward-side QPU kernels are compiled into `libvc4infer` (the op registry
in `infer/src/vc4_api.c` dispatches to them), but they are developed and tested
here:

```
src/vc4_train.c        transpose, C=A*B^T, relu mask, rsqrt, and the fused
                       softmax + cross-entropy backward kernel
src/vc4_convbwd.c      im2col/col2im, conv dX/dW, maxpool forward/backward
tests/test_train.c     full-batch 64-32-10 MLP trainer (SGD or Adam)
tests/test_train_ops.c training primitives
tests/test_grad.c      matmul/relu/softmax gradients vs central differences
tests/test_convgrad.c  conv dX/dW and pool, finite differences
tests/test_convnet.c   conv-net trainer
```

## Build and run

The library owns the build rules, so there is one entry point:

```sh
make -C infer -j4                                  # builds the lib + all tests
REPS=5 VC4INFER_DAEMON=1 infer/run_tests.sh        # includes every test here
VC4INFER_DAEMON=1 infer/build/test_train --opt adam --steps 300
VC4INFER_DAEMON=1 infer/build/test_convgrad 1
```

Run one GPU user at a time; `infer/run_tests.sh` serialises with
`/tmp/vc4-gpu.lock` and checks `STATE: LIVE` after each stage.

## Results

Numbers and limits: [`../docs/results/training.md`](../docs/results/training.md),
[`../docs/TRAINING-RESULTS.md`](../docs/TRAINING-RESULTS.md),
[`../docs/CONV-RESULTS.md`](../docs/CONV-RESULTS.md).

fp32 only, no FMA, 76 MiB device memory.  This demonstrates that the gradients
are computed on the QPUs; it is not a training framework, and general training
is out of scope.
