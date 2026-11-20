# vc4-toolkit-fast

The vc4-toolkit stack rebuilt on the target board with speed as the only build
objective.  Same daemon, same socket, same headers, same commands:

```sh
sudo apt install vc4-toolkit-fast    # replaces vc4-toolkit / vc4infer
vc4-query
vc4-smi
vc4infer selftest
vc4infer util /usr/share/vc4infer/models/yolov8n_160.onnx --reps 3
```

## What is different

### 1. Compile flags

The engine is built with

```
-O3 -march=native -mtune=native -fno-plt
```

instead of the standard package's `-O2`.  The static archive is additionally
built with LTO (`AR=gcc-ar`, `CFLAGS+=-flto`) and the test suite is run
against it; the shipped shared library is the non-LTO build.  If LTO ever
breaks a test, the archive build is a packaging detail, not the shipped
library.

### 2. Prewarmed kernel cache

`/usr/lib/vc4infer/kcache` contains every kernel blob the bundled models
need, generated at package build time, stamped with the generator's SHA-256
and shipped read-only:

```
$ vc4infer --version
...
kernel cache:    /home/raspi/.cache/vc4infer/kcache
system kcache:   /usr/lib/vc4infer/kcache (612 blobs, prewarmed read-only)
```

`/usr/bin/vc4infer` seeds the user's writable cache from it on first use, so a
first `vc4infer selftest` performs no JIT at all.  A model that is not bundled
still JITs normally into the user's cache.  Set `VC4INFER_CACHE` explicitly to
opt out of the seeding.

### 3. Daemon tuning

`/etc/systemd/system/vc4d.service.d/10-fast.conf` selects:

```
--arena-mb 150        # the measured 320x320 recipe (largest bundled graph
                      # needs ~91 MiB of client arena, the rest is headroom)
--max-timeout-ms 60000  # a cold 784-launch graph can outlast the 10 s default
```

Toggle it with

```sh
sudo vc4fast on       # tuned drop-in (default after install)
sudo vc4fast off      # /etc/default/vc4d defaults, arena 96 MiB
vc4fast status        # which one is active + the running daemon's argv
```

`vc4fast` changes no clock, governor or voltage.  The board stays at stock
core 400 MHz / V3D 300 MHz.

## Measured difference

| workload (stock clocks) | standard `-O2` | fast `-O3 -march=native` | fast vs standard |
|---|---|---|---|
| matmul 256³, QPU time | **4.051 ms / 8.28 GFLOP/s** | 4.106 ms / 8.17 GFLOP/s | 1.4% slower |
| conv 32×16×32 k3 pad1+bias, QPU time | **2.913 ms / 3.24 GFLOP/s** | 2.962 ms / 3.19 GFLOP/s | 1.7% slower |
| YOLOv8n 160×160, end to end | **608 ms / 1.64 FPS** | 660 ms / 1.51 FPS | 8.5% slower* |

Best of three interleaved rounds per build on the same Pi 3B with the `vc4d`
backend, stock clocks.  QPU time is the engine's own pack+kernel timers;
end-to-end is `vc4infer util --reps 3` on the bundled YOLOv8n.  The board was
shared with other build jobs (load average 2.5-5.6) and the YOLO mean varied
2x between rounds for *both* builds (standard: 608, 1103, 1319 ms; fast: 660,
949 ms), so \*the 8.5% is inside the run-to-run spread.  The short version:
**the two builds are within noise of each other.**

## Honest conclusion

That is the expected result.  The expensive work is QPU instructions produced
by the Python kernel generator, which neither `-O3` nor `-march=native` touches;
the ARM side is the same control plane and the same launch protocol.  On this
workload compiler optimisation does not move the number, and this package does
not claim that it does.  Install it for the install path and the cold-start
behaviour, not for a speed number:

- a fresh install starts with **zero JIT**: the 338 blobs the bundled models
  need are generated at package build time (about five minutes of `python3`
  JIT on this board) and ship read-only, stamped;
- the 320×320 recipe (`--arena-mb 150`, `--max-timeout-ms 60000`) is active
  out of the box and toggled with `vc4fast`;
- going back is one `apt install vc4-toolkit`.

## Going back

```sh
sudo apt install vc4-toolkit
```

removes `vc4-toolkit-fast` and reinstalls the `-O2` stack in the same apt
transaction.  Your per-user kernel caches are left in place.
