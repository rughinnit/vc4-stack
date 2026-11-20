# vc4-toolkit

One command installs the whole VideoCore IV stack:

```sh
sudo apt install vc4-toolkit
```

| package | what it gives you |
|---|---|
| `vc4d` | privileged helper: `/dev/mem` + V3D dispatch, socket `/run/vc4d.sock`, `vc4` group, `Restart=always` |
| `vc4infer` | engine, Python API and CLI (inference on the QPUs) |
| `vc4infer-dev` | `/usr/include/vc4infer.h`, `libvc4infer.so`, `pkg-config`, CMake |
| `vc4-query` | the `deviceQuery` analogue |
| `vc4-smi` | the `nvidia-smi` analogue |
| `vc4train` | the engine's own training demo (backward kernels) |
| `vc4top` | realtime QPU telemetry |
| `vc4top-dkms` | the read-only V3D register/counter module |

After install:

```sh
vc4-query                 # what did I just install?
vc4-smi                   # is the GPU healthy?
vc4infer selftest         # 4/4 bundled models
vc4train --opt adam --steps 300   # training path, loss must fall 5x
```

`/etc/profile.d/vc4.sh` (shipped by `vc4d`) sets `VC4INFER_DAEMON=1`, so normal
users do not need root or `/dev/mem`.

Everything runs at stock clocks (core 400 MHz / V3D 300 MHz).  Nothing in this
stack changes a clock or a voltage.

For the same stack rebuilt from source with `-O3 -march=native` plus a
prewarmed kernel cache, install `vc4-toolkit-fast` instead.  See
`/usr/share/doc/vc4-toolkit-fast/README.md` and `docs/INSTALL.md` in the
source tree.
