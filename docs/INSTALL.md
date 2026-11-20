# Installing the VideoCore IV stack

## The one-liner

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/install.sh | sudo sh
```

That adds the signed key and apt source, installs `vc4-toolkit`, enables the
`vc4d` helper, adds you to the `vc4` group, and then *verifies* the result with
`vc4-query`, `vc4-smi`, `vc4infer selftest` and `vc4train`.  It is idempotent,
never restarts a healthy daemon, and writes a log to `/var/log/vc4-install.log`.

| flag | effect |
|---|---|
| `--fast` | install `vc4-toolkit-fast` (rebuilt on the board, prewarmed kernel cache, 150 MiB arena drop-in) |
| `--torch` | also `pip install torch-vc4` (needs torch already importable) |
| `--pin-clocks` | set `core_freq=400` / `core_freq_min=400` — the **stock** maximum, not an overclock; without it the firmware can drop the core to 250 MHz mid-run (see `docs/CLOCK-VARIANCE.md`) |
| `--no-verify` | skip the selftest |
| `--dry-run` | print the commands |
| `--uninstall` | purge the packages and remove the source |

The same script lives at the root of this repository (`./install.sh`).

## After installing: two backends

The stack ships **`managed`** (through `vc4d`, unprivileged, validated, ~0.1 ms
per launch) and **`direct`** (root + `/dev/mem`, no per-launch overhead, no
validation). `managed` is the default and what the packages configure; `direct`
is opt-in for benchmarking:

```sh
vc4infer util model.onnx                  # managed
sudo -E vc4infer --backend direct util model.onnx
```

Measured on YOLOv8n 160: **488 ms (managed) vs 410 ms (direct)**, both 0/44,100
over tolerance. Details and the trade-off: [`BACKENDS.md`](BACKENDS.md).

## The two paths in detail

Two install paths, one repository:

| path | command | what you get |
|---|---|---|
| **standard** ("like CUDA/TensorRT") | `sudo apt install vc4-toolkit` | the whole stack built at `-O2`, JITs kernels on first use |
| **speed** | `sudo apt install vc4-toolkit-fast` | the same stack rebuilt on the board with `-O3 -march=native`, a prewarmed kernel cache and a tuned `vc4d` |

Both are stock-clock builds.  Nothing here changes a core/V3D frequency, a
governor or a voltage; see `docs/CLOCK-VARIANCE.md` for the board rules.

## The repository

```sh
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://rughinnit.github.io/vc4-apt/key.asc \
  | sudo gpg --dearmor -o /etc/apt/keyrings/vc4-apt.gpg
echo "deb [signed-by=/etc/apt/keyrings/vc4-apt.gpg] https://rughinnit.github.io/vc4-apt stable main" \
  | sudo tee /etc/apt/sources.list.d/vc4-apt.list
sudo apt update
```

The archive key is `E4CB3BA058DAAB2BE60EF5D6B736BB9A48B94108`
(`vc4top archive <rughinnit@users.noreply.github.com>`).  The same key signs
`dists/stable/InRelease`; `apt` verifies it on every update.

## Install like CUDA / TensorRT

```sh
sudo apt install vc4-toolkit
```

That single meta-package pulls in:

| package | role |
|---|---|
| `vc4d` | privileged helper: `/dev/mem` + V3D dispatch in one small root process that drops capabilities; clients use `/run/vc4d.sock` (group `vc4`) |
| `vc4infer` | engine (`libvc4infer.so.1`), Python API, CLI (run/bench/nodes/util/selftest) |
| `vc4infer-dev` | `/usr/include/vc4infer.h`, `libvc4infer.so`, `pkg-config`, CMake package |
| `vc4-query` | the `deviceQuery` analogue: driver/daemon versions, kernel, QPU count, clocks, temperature, throttle, CMA, arena, op counts, training support, package versions |
| `vc4-smi` | the `nvidia-smi` analogue: one-shot state/util/clocks/temp/throttle/memory/clients, non-zero exit when unhealthy |
| `vc4train` | the engine's own training demo (MLP 64-32-10, backward kernels) |
| `vc4top` | realtime QPU telemetry |
| `vc4top-dkms` | read-only V3D register/counter module (`/sys/kernel/vc4mon`) |

`/etc/profile.d/vc4.sh` (from `vc4d`) sets `VC4INFER_DAEMON=1`, so normal
users do not need root or `/dev/mem`.  Add a user to the group if the socket
is not reachable:

```sh
sudo usermod -aG vc4 "$USER"      # then log in again
```

### Verify the install

```sh
vc4-query                 # deviceQuery: everything above, one screen
vc4-smi                   # nvidia-smi: exits non-zero if the GPU is unhealthy
vc4infer selftest         # 4 bundled models, must be 4/4 with 0 over tolerance
```

`vc4-query` also prints the installed package versions, so a bug report can
start with its output.  `vc4-smi --json` is the machine-readable form.

### Build against the engine

`vc4infer-dev` installs the header and both build-system files:

```sh
pkg-config --cflags --libs vc4infer
# -I/usr/include -L/usr/lib/aarch64-linux-gnu -lvc4infer
```

```cmake
find_package(vc4infer 1.2 REQUIRED)
target_link_libraries(myapp PRIVATE vc4infer::vc4infer)
```

A minimal program:

```c
#include <vc4infer.h>
int main(void)
{
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK)   /* daemon when VC4INFER_DAEMON=1 */
        return 1;
    vc4_shutdown(ctx);
    return 0;
}
```

Every float operation runs on the QPUs.  An op the engine cannot dispatch
returns `VC4_ERR_UNSUPPORTED`; there is no CPU fallback.

### Verify training

The engine carries the backward kernels; `vc4-query` reports them before you
write a training loop:

```sh
$ vc4-query | grep training
training:        yes -- 10/10 backward/training ops dispatchable, 5/5 forward ops with a backward
```

Then run the engine's own trainer.  It prints the loss every step and exits
non-zero unless the loss drops at least 5x and test accuracy reaches 90%:

```sh
$ vc4train --opt adam --steps 300
libvc4infer 1.2.2 toy trainer: 64-32-10, opt=adam, seed=12345, arena 96.0 MiB
initial: train loss=2.302585093 acc=9.84% (batch of 640)
step    0: loss=2.289...
...
step  299: loss=0.0...
final: train loss=0.0... acc=100.00%  test loss=0.0... acc=100.00%
TRAIN PASSED
```

The QPU does every multiply-accumulate, the activation, the fused
softmax+cross-entropy gradient and the optimizer step; the ARM only prepares
buffers.  It is a demonstration that the backward path works, not a training
framework (fp32, full batch, toy scale).

### Python

The `vc4infer` .deb installs the Python package system-wide.  The same code is
also on the pip index:

```sh
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer
```

The wheel is pure Python; `libvc4infer.so.1` still comes from the apt package
(or from `vc4-toolkit-fast`).  In a venv, use
`python3 -m venv --system-site-packages` so `numpy`, `onnx` and `pillow` from
the apt packages are visible.

## Install for speed

```sh
sudo apt install vc4-toolkit-fast
```

It `Conflicts`/`Replaces` `vc4-toolkit` and `vc4infer`, so apt swaps the
standard stack for this one in the same transaction.  Same commands, same
socket, same headers.

What it actually does:

1. **Build flags**: the engine is compiled on the board with
   `-O3 -march=native -mtune=native -fno-plt` (the standard package is
   `-O2`).  The static archive is also built with LTO and the test suite is
   run against it; the shipped shared library is the non-LTO build.
2. **Prewarmed kernel cache**: `/usr/lib/vc4infer/kcache` is generated at
   package build time with every kernel blob the bundled models need, stamped
   with the generator's SHA-256 and shipped read-only.  A first-run
   `vc4infer selftest` therefore does no JIT.  `/usr/bin/vc4infer` seeds the
   per-user writable cache from it on first use; a model that is not bundled
   still JITs into the user's own cache.  `vc4infer --version` prints the
   system cache path and blob count.
3. **Daemon tuning**: `/etc/systemd/system/vc4d.service.d/10-fast.conf` sets
   `--arena-mb 150` (the measured 320×320 recipe; the largest bundled graph
   needs ~91 MiB of client arena and the headroom keeps it resident) and
   `--max-timeout-ms 60000` (a cold 784-launch graph can take longer than the
   10 s default before its kernels are cached; a timeout kills the daemon).
4. **`vc4fast on|off|status`** switches between that drop-in and the safe
   `vc4d` defaults, and prints the running daemon's argv.  It changes no
   clock.

### What it measurably buys

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

`vc4-query` shows the arena the daemon actually got.  `vc4d` halves the
request when CMA cannot supply a contiguous block (the display and the
firmware share the same 256 MiB CMA region), so on a board that has been
allocating for a while a 150 MiB request can land lower; a fresh boot gives
the full size.  That is a property of CMA on a 1 GB board, not of the
package.  If `vc4-smi` or `vc4infer selftest` reports a small arena, reboot
or lower `--arena-mb` and check `vc4-query` again.

`vc4fast` is deliberately not a clock switch.  We do not touch `cpufreq`: on
this board the ARM governor is `ondemand`, and pinning it to `performance`
would raise the ARM clock under load.  Every number above was taken at the
stock policy, stock core 400 MHz, stock V3D 300 MHz.

To go back:

```sh
sudo apt install vc4-toolkit        # removes vc4-toolkit-fast in the same run
```

## Framework support

Only what exists is listed.  "not published yet" means exactly that: no
package, no wheel, nothing to install.

| framework | how it is installed | status |
|---|---|---|
| **NumPy / Python (ONNX)** | `apt install vc4infer` + `pip install .../pip/simple/ vc4infer` | published |
| **C / C++** | `apt install vc4infer-dev` (header, `-lvc4infer`, pkg-config, CMake) | published |
| **PyTorch** | `pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ torch_vc4` | published (`torch_vc4-0.3.0`, cp313 aarch64 wheel) |
| **ONNX Runtime** | `apt install vc4infer` then `import vc4infer.ort` (the EP is inside the engine package; no separate wheel) | published as part of `vc4infer` 1.2.2 |
| **TVM** | — | not published yet |
| **TFLite / LiteRT** | — | not published yet |
| **ggml** | — | not published yet |

The pip index at `https://rughinnit.github.io/vc4-apt/pip/simple/` is a normal
PEP 503 index: each project gets `pip/simple/<project>/index.html` and the
top-level page lists the wheels that are actually present.  A project appears
there only when a wheel is published.

## Uninstall

```sh
sudo apt purge vc4-toolkit vc4-toolkit-fast   # whichever is installed
```

`vc4d`'s postrm stops the unit, removes `/run/vc4d.sock` and the drop-in, and
drops the `vc4` group only if it is empty.  The per-user kernel caches under
`~/.cache/vc4infer` are yours and are left alone; delete them by hand if you
want a cold start.

## If something is wrong

- `vc4-smi` exits non-zero with a reason (throttle bits, undervoltage, queue
  error, ERRSTAT).
- `vc4-query` says whether the daemon is reachable and what the backend is.
- `systemctl status vc4d` and `journalctl -u vc4d -n 50`.
- `vc4infer --version` prints the engine, backend, generator, cache and
  package versions.

**One GPU user at a time.**  The QPUs have no preemption; two concurrent
workloads or a wild kernel can hang V3D hard enough to need a power cycle.
`vc4d` serialises launches, but it is still one GPU.
