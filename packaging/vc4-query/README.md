# vc4-query

The `deviceQuery` analogue for the VideoCore IV stack: one command, no
arguments, everything a first run needs to answer "what did I just install?".

```
$ vc4-query
vc4-query 1.0.0 -- VideoCore IV device query
--------------------------------------------------------------
engine:          libvc4infer 1.2.2
daemon:          vc4d 1.2.1, protocol 2, arena 100663296 bytes (96.0 MiB)
backend:         vc4d (/run/vc4d.sock)
kernel:          6.18.50+rpt-rpi-v8 (aarch64)
soc:             VideoCore IV (V3D) rev 1, 3 slices x 4 QPUs = 12 QPUs, 6 TMUs
clocks:          core 400 MHz, V3D 300 MHz  (stock; this stack never changes them)
temperature:     46.2 C
throttled:       0x0 (none)
cma:             256 MiB total, 49 MiB free (207 MiB used)
arena bytes:     100663296
ops:             40 registry entries, 34 dispatchable on the QPU, 6 not implemented
training:        yes -- 10/10 backward/training ops dispatchable, 5/5 forward ops with a backward
system kcache:   /usr/lib/vc4infer/kcache (612 blobs)
packages:        vc4infer 1.2.2, vc4d 1.2.1, ...
```

`--json` prints the same document as one JSON object.

## Where the numbers come from

- V3D revision/slices/QPUs: `IDENT0`/`IDENT1` from `/sys/kernel/vc4mon/ident`
  (vc4top's read-only module), or the vc4d HELLO reply when the module is not
  loaded.
- Clocks, temperature and `throttled`: `vcgencmd`, which talks to the firmware
  over `/dev/vcio`.
- Arena bytes: the vc4d HELLO reply.  No client arena is mapped.
- Op counts and training support: `vc4_supported_ops()` from `libvc4infer`.
  "Training" is the set of backward/optimizer primitives the engine exports
  (`matmul_bt`, `transpose`, `relu_bwd`, `rsqrt`, `softmax_ce_bwd`, `im2col`,
  `col2im`, `conv2d/bwd_dx`, `conv2d/bwd_dw`, `maxpool_bwd`) plus the forward
  ops that have a backward (`matmul`, `conv2d`, `relu`, `maxpool`, `softmax`).
- Package versions: `dpkg-query`.

`vc4-query` never opens `/dev/mem`; `strace -f -e trace=openat vc4-query`
shows only sysfs, /proc and the vc4d socket.

## Exit status

0 when the report was produced.  A stopped daemon is reported as
`backend: none (...)` rather than a failure: the point of the tool is to tell
you that.
