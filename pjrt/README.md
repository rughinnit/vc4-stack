# pjrt - JAX device plugin for the QPUs

`libpjrt_vc4.so` is a PJRT C API plugin: jaxlib loads it, `jax.devices()`
shows `Vc4Device(id=0)`, and fp32 StableHLO programs run on the QPUs through
`libvc4infer`.  `jax.grad` of `sum(relu(x @ w))` runs forward and backward on
the device.

    make -C pjrt
    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4 python3 -c 'import jax; print(jax.devices())'
    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4,cpu python3 pjrt/tests/test_pjrt.py

The op subset, the bytecode-reader design, the mismatch table, the gradient
result and the measured per-call cost are in `docs/PJRT.md`.
