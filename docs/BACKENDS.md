# Backends: `managed` (default) or `direct`

There are two ways to reach the QPUs. They run the **same kernels** and produce
the **same results**; they differ in who is allowed to touch the hardware and
what that costs per launch.

| | **`managed`** | **`direct`** |
|---|---|---|
| path | client → `vc4d` (unix socket) → `/dev/mem` | client → `/dev/mem` |
| privilege | **unprivileged** (member of group `vc4`) | **root** |
| validation | every register value, every address, arena bounds, allowlist | **none** |
| isolation | a buggy/hostile client cannot take the board down (90/90 hostile-input cases rejected) | a wrong offset corrupts memory; the GPU may need a reboot |
| overhead | **~0.1 ms per launch** (IPC + validation + cache maintenance) | **[none](/)** |
| availability | default, what the packages configure | opt-in, for benchmarks and trusted single-user runs |

## Measured (stock clocks pinned at 400 MHz, YOLOv8n 160, 3 reps each)

| | ms/image | FPS | QPU duty | GPU ms |
|---|---:|---:|---:|---:|
| `managed` | **488.4** | 2.05 | 52.9 % | 258.3 |
| `direct` | **410.0** | 2.44 | 58.4 % | 239.6 |

So the daemon costs **16 % wall time** on a graph of 784 small launches, and
even the kernel time is ~7 % higher because the managed path does cache
maintenance per launch that the direct path does not.

At the other end of the scale the tax nearly vanishes: on a single 256³ GEMM
(two launches) `managed` is **4.264 ms** and `direct` **3.964 ms** — **7 %**.

That is the rule: **the overhead is per launch, not per FLOP.** One big kernel
barely notices the daemon; a graph of many small kernels pays on every node.

## How to choose

```sh
vc4infer util model.onnx            # managed (the default)
vc4infer --backend direct util ...  # needs root; do not do this unless you mean it
sudo -E vc4infer --backend direct util model.onnx
```

The flag works before or after the subcommand, and can be set for a process:

```sh
export VC4INFER_BACKEND=direct      # or "managed"
```

In Python it is the same switch:

```python
import vc4infer
from vc4infer import _runtime
_runtime.set_backend("direct")        # raises PermissionError if not root
eng = vc4infer.Engine("model.onnx", arena_mb=64)
```

`direct` owns the arena itself, so its size comes from `VC4INFER_ARENA_MB`
(default 64 MiB) — the daemon's `--arena-mb` does not apply.

## Which one should you use?

| situation | use |
|---|---|
| anything installed from the package, any multi-user board, anything you will leave running | **`managed`** |
| benchmarking, comparing kernels, a trusted single-user board where you accept the risk | `direct` |
| comparing two kernel versions | use the same backend for both — otherwise you are measuring the HAL |
| a graph with many small ops (YOLO, transformer blocks) | `direct` buys the most (~16 %); `managed` is still the default for the reasons above |

`vc4-smi` shows the daemon and the GPU; `vc4infer --version` prints which
backend the process would use.

One historical reason to pick `direct` is gone: before `vc4d` 1.2.7 a scalar
parameter at or below −2.0 (`LeakyRelu(-2)`, `Clip(-3,3)`) was refused in
managed mode, because its fp32 bit pattern (≥ `0xC0000000`) looks like a GPU bus
address and the protocol carried no way to say "this uniform word is a value".
`LAUNCH` now carries a per-slot address mask (`VC4D_LAUNCH_UNI_MASK`), the
engine sets it on every launch, and the two backends accept the same parameter
range.  Old clients that send no mask keep the old behaviour, so a 1.2.6
client against a 1.2.7 daemon is unchanged.

## Why `managed` is the default

The daemon is not overhead for its own sake. It is what makes the GPU usable
from an unprivileged process with **no `/dev/mem` in any client**, which is what
lets `vc4infer`, `torch_vc4` and the tests run as a normal user. Without it,
every client would need root, and one stray DMA from a buggy kernel can corrupt
the page cache of a running library — this project has already done that once.
The 16 % is the price of that guarantee, and it is stated rather than hidden.
