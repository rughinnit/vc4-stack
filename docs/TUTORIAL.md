# Tutorial: from a stock Pi to YOLO and PyTorch on the VideoCore IV QPUs

This is the practical entry point to `vc4-stack`: a Raspberry Pi 3B whose
VideoCore IV QPUs run ONNX graphs and a subset of PyTorch.  It starts from a
stock 64-bit Raspberry Pi OS install and ends with a YOLOv8n detection run, the
Python engine API, the Ultralytics adapter, the `torch_vc4` backend, and an
ONNX export you can deploy.

**Reference board for every number and command below:** Raspberry Pi 3 Model B
Rev 1.2 (BCM2837, VideoCore IV), Debian 13 arm64, kernel
`6.18.50+rpt-rpi-v8`, Python 3.13.5, **stock clocks** (core 400 MHz, V3D
300 MHz, `throttled=0x0`), one GPU client at a time through `vc4d`.
The documentation pass was taken on **2026-11-20**.

Every command in the walkthrough was run on that board and the output quoted
is the output it printed, except where a block is explicitly marked
`(not verified on this board)`.  Timings move by 15-25% between sessions on
this shared board (see [CLOCK-VARIANCE.md](CLOCK-VARIANCE.md)); launch counts,
the MAC share and the zero-host-byte invariant do not.

---

## 0. What you are building

```
your code (numpy / ONNX / PyTorch)
        |
        +-- vc4infer.Engine  -------- ONNX graph executor (fp32, static shapes)
        |       |
        |       +-- libvc4infer.so.1 -- QPU kernels (conv, GEMM, pool, softmax, ...)
        |               |
        |               +-- /run/vc4d.sock --> vc4d (root helper) --> V3D QPUs
        |
        +-- torch_vc4 (PrivateUse1 backend) -- same libvc4infer arena, same vc4d
```

The ARM cores do allocation, I/O and byte-only layout work.  Every multiply-
accumulate in the engine runs on the QPUs.  There is **no CPU arithmetic
fallback**: an operator, attribute or shape the engine cannot dispatch raises
and names the node.  `vc4infer util` is the command that proves the invariant
(`host_compute_nodes 0`, `bytes_copied_by_host 0 B`,
`arithmetic_on_vc4 100.0%`) rather than asking you to trust it.

A client process never opens `/dev/mem`.  The only privileged piece is the
small `vc4d` daemon; clients get one dma-buf mapping and one Unix socket, and
run with an empty capability set.

---

## 1. Prerequisites

### 1.1 The board: VideoCore IV, not VideoCore VI

The QPU instruction set in this stack is VideoCore IV (V3D 2.x), as found on
the Pi 1, Pi 2, Pi 3/3B+, Zero, Zero 2 W and CM1/CM3.  The reference board is
a **Pi 3B**.

A **Pi 4 is a VideoCore VI (V3D 4.2)** part.  The kernels in this repository
are not written for it, and nothing here claims support.  If you have a Pi 4,
stop: this is not the board this tutorial is for.

Check what you have:

```sh
$ cat /proc/device-tree/model
Raspberry Pi 3 Model B Rev 1.2
$ getconf LONG_BIT
64
```

A 64-bit Raspberry Pi OS (Debian 13 "trixie" arm64 here) is required: the
packaged engine is `aarch64`, and the `torch_vc4` extension is built for
`cp313`/`aarch64`.  A 32-bit userland will not load the packages.

### 1.2 Kernel with `vc4-kms-v3d`, and CMA

The V3D must be owned by the KMS driver, not by the legacy firmware mailbox.
On Raspberry Pi OS that is one line in `/boot/firmware/config.txt`:

```ini
dtoverlay=vc4-kms-v3d
```

```sh
$ grep vc4 /boot/firmware/config.txt
dtoverlay=vc4-kms-v3d
$ uname -r
6.18.50+rpt-rpi-v8
$ dmesg | grep 'bound 3fc00000.v3d'
[   14.185462] vc4-drm soc:gpu: bound 3fc00000.v3d (ops vc4_v3d_ops [vc4])
$ ls /dev/dri
by-path  card0  renderD128
$ lsmod | grep '^vc4'
vc4                   421888  2
```

CMA is the contiguous memory the dma-buf arena comes from.  **At least
256 MiB** is required for the default 64 MiB client arena and the DRM
framebuffers; 320x320 needs a larger arena (section 7.2).  Check it with:

```sh
$ grep -E 'CmaTotal|CmaFree' /proc/meminfo
CmaTotal:         262144 kB
CmaFree:            3580 kB
```

`CmaTotal: 262144 kB` is 256 MiB.  `CmaFree` is low here because the DRM
driver and the running daemon already hold their blocks; that is normal.  A
fresh boot with no GPU clients has more free CMA.  `vc4top` prints the same
number in MiB with the percentage used.

### 1.3 The QPUs are owned by KMS, and `vc4d` is the only privileged helper

The engine and `torch_vc4` never touch `/dev/mem`.  `vc4d` maps the V3D
register window and one CMA dma-buf once, drops to `nobody` with an empty
capability set, and serves launches over `/run/vc4d.sock`.  Check the unit:

```sh
$ systemctl is-active vc4d
active
$ systemctl cat vc4d | sed -n '1,20p'
[Unit]
Description=vc4d - VideoCore IV QPU helper daemon
...
[Service]
ExecStart=/usr/local/sbin/vc4d --socket /run/vc4d.sock --arena-mb 72 \
          --group vc4 --drop-user nobody --max-timeout-ms 10000
Environment=VC4D_GPU_LOCK=1
...
$ ls -l /run/vc4d.sock
srw-rw---- 1 root vc4 0 Nov 20 07:34 /run/vc4d.sock
```

The unit uses `--arena-mb 72`: 72 MiB of daemon arena gives the client its
full 64 MiB region (the daemon's data heap takes the first 2 MiB).  A 64 MiB
daemon arena leaves the client one megabyte short and fails late with
`E_NOMEM`; that was a real bug, not a tuning preference.

### 1.4 The `vc4` group

The socket is `root:vc4`, mode `0660`.  The user that runs inference must be in
the `vc4` group:

```sh
$ id raspi
uid=1000(raspi) gid=1000(raspi) groups=...,1001(vc4)
```

If `vc4` is missing:

```sh
sudo usermod -aG vc4 "$USER"
# log out and back in, then check: id
```

*(Not run on the reference board: `id` already listed `vc4`.  The command is
the standard group membership fix.)*

### 1.5 Telemetry: `vc4mon` and `vc4top`

`vc4top` is a read-only monitor.  Unprivileged it reads the V3D registers
through the `vc4mon` module (no `/dev/mem`); root falls back to `/dev/mem`.
Install the module and the tool from the same archive as the engine (section
2), then:

```sh
$ lsmod | grep vc4mon
vc4mon                 12288  0
$ ls /sys/kernel/vc4mon
ident  perf  perf_clear  perf_enable  regs
$ vc4top --once --no-color
```

The fields worth looking at first, with the values from this board:

```
arm clock:       1200000000 Hz (1200.000 MHz)
core clock:      400000000 Hz (400.000 MHz)
v3d clock:       300000000 Hz (300.000 MHz)
throttled:       0x0 (none)
temp:            56.4 °C
cma:             CMA 256 MiB / 63 MiB free
QPUs/slice:      4 QPUs/slice (total 12 QPUs)
SRQCS:           0x00010100 queue=0 requests=1 completed=1 qerr=0
ERRSTAT:         0x00001000 VCD-idle
```

* `core clock` must be 400 MHz and `v3d clock` 300 MHz for the numbers in this
  repository.  `throttled: 0x0` means the firmware has not throttled.
* `cma` tells you whether a larger arena can be allocated.  If `free` is
  smaller than the arena you want, `vc4d` will fall back or fail (section 9).
* `ERRSTAT` is the first thing to read after a hang.  `VCD-idle` is healthy;
  `0x40` set after a timeout means the V3D context is wedged.
* The primary counters (QPU total cycles, instruction-cache hits) are the
  hardware's own accounting; `vc4infer util` reports the engine's view
  (`qpu_duty`, launches) alongside them.

Do not pipe a GPU workload or a long `vc4top` run through `head`: `SIGPIPE`
can kill a client in the middle of a launch, which wedges the V3D (section
9.1).  Redirect to a file and read the file instead.

---

## 2. Install

### 2.1 The signed apt repository

The engine, the CLI, the kernel generator, the selftest models, `vc4top` and
the `vc4mon` DKMS module are in one signed archive:

```sh
curl -fsSL https://rughinnit.github.io/vc4-apt/key.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/vc4top-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/vc4top-archive-keyring.gpg] https://rughinnit.github.io/vc4-apt stable main" \
  | sudo tee /etc/apt/sources.list.d/vc4-apt.list
sudo apt update
sudo apt install vc4infer vc4top vc4top-dkms
```

The key is `vc4top archive <rughinnit@users.noreply.github.com>`, RSA-4096,
fingerprint `E4CB 3BA0 58DA AB2B E60E F5D6 B736 BB9A 48B9 4108`:

```sh
$ gpg --show-keys --with-fingerprint key.asc
pub   rsa4096 2026-11-19 [SC]
      E4CB 3BA0 58DA AB2B E60E  F5D6 B736 BB9A 48B9 4108
uid                      vc4top archive <rughinnit@users.noreply.github.com>
```

*(The source line was already applied on the reference board; the key was
re-fetched and its fingerprint checked.  The `echo ... | tee` line itself was
not re-run.)*

The dependency resolution was checked on the board; `vc4infer` brings
`python3-numpy`, `python3-onnx`, `python3-pil`, and the DKMS module builds for
the running kernel:

```sh
$ sudo apt-get install -s vc4infer vc4top vc4top-dkms
0 upgraded, 0 newly installed, 0 to remove and 54 not upgraded.
```

(The `0 newly installed` line is because this board already has them; a fresh
board installs the three packages and their dependencies.  The `apt install`
line itself was therefore not re-run during this pass, only simulated; the
installed versions are 1.2.1 / 1.0.5 / 1.0.5.)

The package layout is documented in [VC4INFER.md](VC4INFER.md).  The important
paths are:

```
/usr/bin/vc4infer                               CLI
/usr/lib/aarch64-linux-gnu/libvc4infer.so.1     engine
/usr/lib/python3/dist-packages/vc4infer/        Python package
/usr/share/vc4infer/models/                     selftest models
```

### 2.2 The pip index (pure-Python package, same engine library)

The wheel is pure Python and uses the `libvc4infer.so.1` that the .deb
installed.  In a venv (or with `--break-system-packages`, which you should not
need):

```sh
python3 -m venv ~/venvs/vc4
. ~/venvs/vc4/bin/activate
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer
```

Verified on the board:

```sh
$ pip download --no-deps --extra-index-url \
    https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer
Collecting vc4infer
  Downloading https://rughinnit.github.io/vc4-apt/pip/vc4infer-1.2.1-py3-none-any.whl (142 kB)
Saved ./vc4infer-1.2.1-py3-none-any.whl
$ python -c "import vc4infer; print(vc4infer.__version__)"
1.2.1
```

`pip install vc4infer` alone will not find the wheel on PyPI; the
`--extra-index-url` is the vc4-apt index, not PyPI.

### 2.3 From source: the engine

The engine is plain C and builds on the board.  With ccache warm and the
existing object files, this is what "nothing to do" looks like:

```sh
$ cd /home/raspi/work/vc4-stack
$ make -C infer -j2
make: Entering directory '/home/raspi/work/vc4-stack/infer'
make: Nothing to be done for 'all'.
make: Leaving directory '/home/raspi/work/vc4-stack/infer'
```

A full build is memory-bound, not CPU-bound.  The engine is safe at `-j4`
(171 MiB peak tree RSS, no swap); `tools/build-fast.sh` picks the largest job
count that fits `MemAvailable`, turns on ccache, and reports the choice:

```sh
$ tools/build-fast.sh infer
build-fast: /home/raspi/work/vc4-stack/infer
build-fast: mode=make jobs=4/4 mem_avail=637MiB reserve=200MiB mem_per_job=60MiB
build-fast: ccache=/usr/bin/ccache (CC/CXX launcher)
make: Nothing to be done for 'all'.
build-fast: rc=0 elapsed=0.0s mem_avail=639MiB swap_used=0MiB_delta
build-fast: ccache hits=50/64(78.12%) misses=14/64(21.88%)
```

`ccache` 4.11.2 is installed on this board.  If it is missing, the script still
works; it just runs slower.

**VC4C is a different story.**  It is a C++/LLVM-weight build; one translation
unit peaks near 400 MiB.  `-j4` on a 1 GB board pushes ~800 MiB into swap and
ends up slower than `-j2`; it can also reset the board.  Use `-j2` on a quiet
board, `-j1` if anything else heavy is running, and never `-j4`:

```sh
# VC4C, not the engine:
BUILD_FAST_TARGET=VC4CC tools/build-fast.sh /home/raspi/work/combined/build
# or, if you drive cmake yourself:
cmake --build . -j2
```

*(The VC4C command was not re-run during this documentation pass; the engine
build above was.  The `-j2` rule is measured in [BUILD.md](BUILD.md).)*

Full matrix and measurements: [BUILD.md](BUILD.md).

---

## 3. Verify the install

### 3.1 Version and backend

```sh
$ vc4infer --version
vc4infer 1.2.1
engine lib:      /usr/lib/aarch64-linux-gnu/libvc4infer.so.1
python:          3.13.5
onnx / numpy:    1.17.0 / 2.2.4
backend:         vc4d (/run/vc4d.sock, running)
kernel generator:/usr/lib/python3/dist-packages/vc4infer/kgen/kgen.py
kernel cache:    /home/raspi/.cache/vc4infer/kcache
```

If `backend:` says `vc4d is not running`, start it with
`sudo systemctl start vc4d`; do not run the CLI as root to "fix" it.  The
client is meant to be unprivileged.

### 3.2 `vc4infer selftest`

This runs the four shipped reference models and compares every element with a
stored CPU reference at `atol = rtol = 5e-4`.  It is the fastest end-to-end
check that the QPUs, the kernel cache and the daemon all work:

```sh
$ vc4infer selftest
vc4infer 1.2.1 selftest
  models:  /usr/share/vc4infer/models
  engine:  /usr/lib/aarch64-linux-gnu/libvc4infer.so.1
  backend: vc4d (/run/vc4d.sock, running)
  onnx 1.17.0, numpy 2.2.4, tolerance atol=rtol=5e-4 (over must be 0)

  modelA       PASS 1x10             38 launches    38.8 ms  max abs 4.84e-06  max rel 4.84e-06  0/10 over  [conv/relu/maxpool/gemm/softmax 32x32]
  modelB       PASS 1x16x32x32       19 launches    43.1 ms  max abs 4.41e-06  max rel 4.09e-06  0/16384 over  [YOLO-ish block: stride-2, depthwise, pool, concat, resize]
  modelC       PASS 1x4               9 launches    12.9 ms  max abs 4.06e-07  max rel 4.06e-07  0/4 over  [conv/gap/gemm 8x8]
  yolov8n_160  PASS 1x84x525        784 launches   646.0 ms  max abs 2.04e-02  max rel 2.23e-04  0/44100 over  [real YOLOv8n COCO, 1x3x160x160 -> 1x84x525]

SELFTEST PASSED: 4/4 cases, 0 elements over tolerance
```

The acceptance criterion is exactly that last line: **`4/4` cases** and
**`0 elements over tolerance`**.  Exit status is 0 on success, 1 on any
failure.  The `646.0 ms` includes a cold kernel cache and a loaded board; the
quiet-board number is in section 7.  The launch count (784 for YOLOv8n 160) is
stable across sessions; the wall time is not.

### 3.3 `vc4top` after a run

```sh
vc4top --once --no-color
```

Look at `throttled`, `core clock`, `v3d clock`, `temp`, `cma`, and `ERRSTAT`.
After a healthy run `ERRSTAT` is back to `VCD-idle`; `SRQCS ... qerr=0` says
the scheduler saw no queue error.  `vc4top` sampling itself uses only
`perf_enable`/`perf_clear`; it does not dispatch work.

### 3.4 `vc4infer util` on the bundled YOLOv8n

This is the invariant check.  It warms up the graph (JIT and weight panels),
then measures timed runs; the warm-up is not counted:

```sh
$ vc4infer util /usr/share/vc4infer/models/yolov8n_160.onnx --reps 3
model:  /usr/share/vc4infer/models/yolov8n_160.onnx
input:  random normal (rng 0), 1x3x160x160
mode:   hybrid (host_policy=device)
587.3 ms/image (mean), 1.70 FPS over 3 run(s)

  device invariant           OK
  host_compute_nodes         0
  bytes_copied_by_host       0 B
  arithmetic_on_vc4          100.0%  (273249600/273249600 MACs)
  qpu_duty                   47.0%  (275.7 / 587.3 ms)
  launches                   784  (4.45 per dispatched node)
  launch path                311.5 ms  (53.0% of wall)
  nodes                      233 vc4, 30 folded, 0 host-meta, 0 host-control, 0 host-unwritten
```

The run above was taken while another process was compiling on the ARM, which
is why 587.3 ms is above the quiet-board 478.7 ms in section 7.  What must not
move: `device invariant OK`, `host_compute_nodes 0`,
`bytes_copied_by_host 0 B`, `arithmetic_on_vc4 100.0%`, and 784 launches.

`nodes` counts folded metadata separately from runtime nodes.  `233 vc4 + 30
folded` is the 263-node graph; the 30 are `Constant`/`Shape`/`Gather` and a few
folded arithmetic nodes resolved at plan time, not runtime CPU arithmetic.  If `host-meta`, `host-control` or
`host-unwritten` is non-zero under `host_policy=device`, the command exits
non-zero and names the node (section 9.2).

---

## 4. YOLO, three ways

All three use the same QPU graph.  They differ in how much of the surrounding
pipeline you want to own.

### 4.1 (a) The CLI: `vc4infer run`

The shortest path.  Images are letterboxed to the model input, converted to
RGB, scaled to `[0,1]`, run as NCHW fp32, and the YOLO head is decoded and
NMS'd on the ARM (the head is an affine combination plus an argmax; there is
no convolution in the decode).  The annotated image is written next to the
input unless you pass `--no-save`:

```sh
vc4infer run /usr/share/vc4infer/models/yolov8n_160.onnx \
  --input ~/yolo/bus.jpg --imgsz 160 --no-save
```

Exact output on this board:

```
model:  /usr/share/vc4infer/models/yolov8n_160.onnx
input:  bus.jpg (810x1080 -> 160x160)  ->  1x3x160x160
output: (1, 84, 525)  float32
run:    2144.7 ms  (346 GPU ms, 848 launches; first run in a process, so it includes the one-time weight-panel packs)
mode:   strict
tiers:  233 vc4, 30 host-meta of 263
host_compute_nodes=30  bytes_copied_by_host=12635064
  host bytes: /model.7/conv/Conv                 Conv       upload          1180672 B x2
  ...
decode: 525 anchors, 80 classes, channels_first; 38 above conf 0.25 -> 4 after NMS (IoU 0.45); boxes in original image pixels
  #   class            conf        x1        y1        x2        y2
  1   person          0.778      39.0     409.2     228.7     901.7
  2   bus             0.703       1.8     226.9     800.6     758.8
  3   person          0.619     678.5     424.5     810.0     865.8
  4   person          0.597     206.2     391.5     353.2     863.2
```

Two things in that output need reading correctly:

* `848 launches` and `2144.7 ms` are **first run in a process**.  The 64
  extra launches are the one-time weight-panel packs, and the time includes
  the JIT and the cold cache.  The steady-state count is 784, the number
  `selftest` and `util` report.
* `host_compute_nodes=30` / `bytes_copied_by_host=12635064` in `run` are
  the process's first-run counters: the 30 plan-time metadata nodes and the
  one-time 12.6 MB weight upload.  `vc4infer util` warms up first and reports
  the per-run figure: `host_compute_nodes 0`, `bytes_copied_by_host 0 B`.
  Use `util` when you need the invariant, and `run` when you need a picture.

Changing the thresholds is two flags:

```sh
vc4infer run model.onnx --input img.jpg --imgsz 160 --conf 0.6 --iou 0.3
```

`--mode hybrid --host-policy device` is the explicit device-invariant mode;
it is what `util` uses.  Plain `--mode strict` (the CLI default) also computes
on the QPUs but reports the plan-time metadata tier separately.  Hybrid
timings are **not** VC4-only numbers and the tool says so.

The input can also be a `.npy` tensor (exact model shape, or that shape without
the batch axis) or a `.csv` of floats in NCHW order.  That is how you feed an
image you have already preprocessed.

### 4.2 (b) The Python engine API: raw outputs

`Engine.run` is numpy in, numpy out, and never post-processes.  This is the
right layer when you have your own decode, your own tracker, or a non-YOLO
head.

```python
import vc4infer
from PIL import Image
from vc4infer.postprocess import Letterbox

MODEL = "/usr/share/vc4infer/models/yolov8n_160.onnx"
eng = vc4infer.Engine(MODEL, vc4infer.Device("vc4"))
print(eng.inputs, eng.outputs)

x, meta = Letterbox(eng.inputs[0].shape[2])(Image.open("bus.jpg"))
out = eng.run({eng.inputs[0].name: x})
print(out["output0"].shape)          # (1, 84, 525), raw head

print(eng.stats)                     # {'launches': 848, 'gpu_ms': 339.4}
p = eng.profile(reps=2)              # warm-up + 2 timed runs
print(p["mean_ms"], p["fps_mean"], p["launches"], p["qpu_duty"])
print(len(eng.nodes()), len(eng.unsupported()), len(eng.host_debt()))
print(eng.node_backends()[0])
eng.close()
```

Output from the board (timings on a loaded session; the shape and the counts
are the stable part):

```
(TensorSpec('images', 1x3x160x160, float32),) (TensorSpec('output0', 1x84x525, float32),)
(1, 84, 525)
{'launches': 848, 'gpu_ms': 339.377808}
profile mean=779.3 ms  1.28 FPS  launches=784  gpu=292.9 ms  qpu_duty=37.6%
nodes 206 unsupported 0 host_debt 0
first node backend NodeBackend(name='/model.0/conv/Conv', op='Conv', backend='vc4', reason='3x3 s2 d1 g1 dense NHWC, batch 1')
```

Notes that save time:

* `eng.inputs` / `eng.outputs` are `TensorSpec(name, shape, dtype)`.  The
  shape is fixed at export time; `Engine.run` raises `ShapeError` if you feed
  a different one.
* `eng.run` accepts `{name: ndarray}` or one ndarray for a single-input
  model.  Inputs are converted to contiguous fp32; the graph expects exactly
  the shape it was exported with.
* `eng.profile(reps=2)` does a warm-up run that is **not** counted, then
  `reps` timed runs.  Use it when you want per-node wall/GPU time; use
  `eng.stats` for the last run only.
* `eng.nodes()` returns every node's static verdict (206 dispatched nodes for
  this graph; the graph has 263 nodes including folded constants).  `eng.
  unsupported()` is empty; `eng.host_debt()` is empty under the device
  policy.
* `eng.node_backends()` is the actual partition after a hybrid run, or the
  static partition before one.

#### Decode and NMS without the YOLO adapter

```python
from vc4infer.postprocess import decode_yolo
d = decode_yolo(out["output0"], imgsz=160, conf=0.25, iou=0.45)
print(len(d), d.boxes[:4], d.scores[:4], d.classes[:4])
```

`decode_yolo` handles the `[1, 4+nc, A]`, `[1, A, 4+nc]` and YOLOv5
`5+nc` layouts, per-class greedy NMS, and returns model-input pixels.  It is
pure numpy and is meant to be copied or replaced.

### 4.3 (c) The Ultralytics adapter

If you want the stock `YOLO(...).predict(...)` workflow, `vc4infer.contrib.
ultralytics` routes the `.onnx` backend to the engine.  Ultralytics' own
letterbox, `non_max_suppression` and `Results` stay in charge; only the graph
runs on the QPUs.

```python
import vc4infer.contrib.ultralytics as vu
vu.register()                       # AutoBackend._BACKEND_MAP["onnx"] = VC4Backend

from ultralytics import YOLO
model = YOLO("/usr/share/vc4infer/models/yolov8n_160.onnx")
r = model.predict("bus.jpg", imgsz=160, conf=0.25, iou=0.45, verbose=False)[0]
print(type(model.predictor.model).__name__)     # AutoBackend
print(len(r.boxes))
for xyxy, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                        r.boxes.cls.tolist()):
    print(r.names[int(cl)], round(cf, 3), [round(v, 1) for v in xyxy])
```

Verified output on this board (ultralytics 8.4.215, torch 2.15.0+cpu):

```
backend: AutoBackend
detections: 5
  person       0.767  [44.0, 411.3, 212.5, 895.3]
  person       0.613  [220.8, 399.3, 350.0, 862.2]
  bus          0.554  [13.0, 234.7, 794.1, 761.6]
  person       0.491  [681.1, 436.0, 810.0, 860.1]
  truck        0.391  [19.5, 223.1, 797.2, 756.0]
```

It reports five boxes where the CLI reported four because Ultralytics applies
its own `non_max_suppression` and class handling; the underlying QPU output is
the same.

**The two patches this board needs.**

1. **The backend map.**  Ultralytics 8.4 derives the format from the file
   suffix and consults `AutoBackend._BACKEND_MAP["onnx"]`.  There is no
   entry-point plugin API in 8.4.215, so selecting vc4 is one documented dict
   entry.  `register()` does exactly that and returns the previous class:

   ```python
   previous = vu.register()
   # ...
   vu.unregister(previous)         # restores ONNX Runtime
   ```

2. **The warmup shim.**  `AutoBackend.warmup()` unconditionally imports
   `torchvision` so NMS binds to `torchvision.ops.nms`.  This board's
   torchvision 0.29.2 registers no operators against its torch 2.15.0+cpu
   build:

   ```
   RuntimeError: operator torchvision::nms does not exist
   ```

   Stock warmup aborts before inference.  `register()` (with the default
   `patch_warmup=True`) swaps in a warmup that runs the same NMS check with
   Ultralytics' pure-torch implementation; `unregister()` restores the
   original.  On a normal torch/torchvision build, pass
   **`patch_warmup=False`** so the integration does not replace stock
   behaviour:

   ```python
   vu.register(patch_warmup=False)
   ```

   *(On this board `patch_warmup=False` would hit the torchvision bug, so that
   path was not exercised; the default `patch_warmup=True` path was, and is
   what the verified output above uses.)*

The adapter's `VC4Backend` is glue: torch tensor in, numpy through
`Engine.run`, numpy out.  `.pt`, `.engine` and other formats keep their normal
backends.  `is_registered()` tells you whether the `onnx` key is currently
vc4.

**Changing `conf`/`iou`, and a custom decode.**  The adapter's `Detector`
gives you the same knobs without Ultralytics objects:

```python
import vc4infer
from vc4infer.postprocess import decode_yolo, Detections

det = vc4infer.yolo.Detector("/usr/share/vc4infer/models/yolov8n_160.onnx")
r = det.predict("bus.jpg", imgsz=160, conf=0.60, iou=0.30)
print(len(r), r.scores)             # 3 [0.778 0.703 0.619]

def bus_only(raw, imgsz, conf, iou, max_det=300, agnostic=False):
    d = decode_yolo(raw, imgsz=imgsz, conf=conf, iou=iou)
    keep = d.classes == 5          # COCO bus
    return Detections(d.boxes[keep], d.scores[keep], d.classes[keep], d.meta)

r2 = det.predict("bus.jpg", imgsz=160, conf=0.25, postprocess=bus_only)
print(len(r2), r2.boxes)            # 1 [[1.8 226.9 800.6 758.8]]
```

Verified output:

```
conf 0.60 / iou 0.30 -> 3 [0.778, 0.703, 0.619]
custom postprocess -> 1 [0.703]
boxes: [[1.8, 226.9, 800.6, 758.8]]
```

The `postprocess=` callable receives `(raw_output, imgsz=, conf=, iou=,
max_det=, agnostic=)` and returns a `Detections` (or a dict with `boxes`,
`scores`, `classes`).  The adapter maps the boxes from model-input pixels back
to the original image; everything else is yours.

---

## 5. The PyTorch backend (`torch_vc4`)

`torch_vc4` is a real `PrivateUse1` device renamed `vc4`.  Tensors live in the
engine's dma-buf arena and every launch goes through `vc4d`; the Python
process is unprivileged.  It is a declared subset: an op outside the table
raises, there is no silent CPU fallback.

### 5.1 Install (read this before `pip install torch_vc4`)

**As of 2026-11-20 there is no `torch_vc4` wheel in the vc4-apt pip index.**
The index contains `vc4infer` only; the command fails with:

```sh
$ pip download --no-deps --extra-index-url \
    https://rughinnit.github.io/vc4-apt/pip/simple/ torch_vc4
ERROR: Could not find a version that satisfies the requirement torch_vc4 (from versions: none)
ERROR: No matching distribution found for torch_vc4
```

The backend is a compiled `cp313`/`aarch64` extension, so build it from the
`torch-vc4` checkout.  The version in the source tree is **0.3.0** (the
installed editable metadata on the reference board still says 0.2.0 because it
predates the bump).  The measured numbers in section 7 are from the 0.2.0
surface; the 0.3.0 tree was still landing while this document was written, and
`COMPLETENESS.md` is its authoritative per-op table.

```sh
sudo apt install vc4infer vc4top          # engine library + daemon client
cd ~/work/torch-vc4
python3 -m venv .venv
. .venv/bin/activate

# The +cpu wheel, not the PyPI aarch64 CUDA wheel (that one needs ~3 GB of
# nvidia libraries and does not import).
pip install --no-cache-dir --no-deps \
  /path/to/torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl
pip install --no-cache-dir typing_extensions filelock sympy networkx jinja2 \
  fsspec setuptools

ulimit -v 1800000          # the one-TU compile peaks near 1 GB
MAX_JOBS=1 pip install --no-build-isolation -e .
```

The extension links **statically** against `libvc4infer.a` and
`libvc4client.a`; that is why it is a binary wheel when one is published, and
why a later engine rebuild cannot break an already-built backend.  Build on
the Pi, not on a host.

*(The source-build commands above are the ones the existing editable install
was built with, recorded in the `torch-vc4` checkout's `RESULTS.md`; they were **not re-run**
during this documentation pass because a rebuild peaks near 1 GB and the board
was shared with another build.  The import and every PyTorch command below
were run against that installed backend.)*

The venv is large: about 816 MB, of which `site-packages/torch` is 688 MB.
`import torch` takes ~12 s warm, ~42 s cold from the SD card; `import
torch_vc4` adds 0.15-0.55 s.  This is a 1 GB board: do not build inside a
`/tmp` tmpfs (section 9.9).

Prerequisites: `vc4d` active, user in `vc4`, and an engine tree with
`build/libvc4infer.a` and `tools/kgen.py`.  `VC4_ENGINE_DIR=` and
`VC4D_DIR=` override the paths.  A 2 GB swapfile is useful for rebuilds; it is
not needed to run.

### 5.2 `torch_vc4.device()` and the device-string caveat

```python
import torch, torch_vc4 as tv
d = tv.device()
print(d)                            # vc4:0
print(torch.zeros(1, device=d).device)   # vc4:0
```

`torch.device("vc4")` **does not parse** in torch 2.15: the parser is populated
during `import torch`, before the `PrivateUse1` rename.  This is a documented
caveat, not a bug we are hiding:

```python
>>> torch.device("vc4")
RuntimeError: Invalid device string: 'vc4'
```

Use `tv.device()`, or the built-in spelling `torch.device("privateuseone")`.
Tensors still report `device(type='vc4')`.

### 5.3 A conv/pool/linear model moved to the device

```python
import torch, torch.nn as nn
import torch_vc4 as tv

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(8 * 8 * 8, 4)).to(tv.device())
print(next(model.parameters()).device)      # vc4:0

x = torch.randn(1, 1, 16, 16, device=tv.device())
print(model(x).shape)                       # torch.Size([1, 4])
```

`module.to(vc4)` works because the backend implements the allocator,
`copy_`, `empty_strided` and the module ops the standard layers use.  The
first forward JITs the kernels into the cache and prints lines such as
`im2col: 330 instructions -> .../kcache/im2col_16_144_3_3.bin`; later runs
reuse them.

### 5.4 A short training loop

The loss must stay on the device.  `F.mse_loss` decomposes into ops this
backend does not have; use `torch_vc4.mse_loss` (or `(a-b).pow(2).sum()`), and
update parameters by hand (`p.copy_(p - lr*p.grad)`) instead of
`torch.optim`, which is a host-side state machine but works only if every op
it touches is implemented.

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(8 * 8 * 8, 4)).to(tv.device())
x = torch.randn(1, 1, 16, 16, device=tv.device())
target = torch.randn(1, 4, device=tv.device())

losses = []
for step in range(20):
    out = model(x)
    loss = tv.mse_loss(out, target)
    for p in model.parameters():
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p - 5e-4 * p.grad)
    losses.append(float(loss.detach()))

print(losses[0], losses[-1])        # 2.126302 0.151130
```

Verified on the board:

```
loss first 2.126302 last 0.151130  min 0.151130  (12.99 s)
last 3 ['0.199630', '0.173699', '0.151130']
```

The whole loop is on the QPUs: im2col+GEMM conv, relu, argmax max-pool,
flatten view, GEMM linear, the backward kernels (`vc4_conv2d_backward`,
`vc4_relu_backward`, `vc4_maxpool2x2_backward`, `mm_bt`/`mm_at`), and the MSE
(`ones`-GEMM sum).  The only host arithmetic is the scalar multiply-add in the
parameter update, which is itself a QPU kernel plus a byte `copy_`.

### 5.5 `fold_bn_into_conv`

BatchNorm in inference mode is a chain of elementwise kernels in the engine
(ten launches).  If the BN follows a conv and will not change, fold it into
the conv weights once, offline:

```python
conv = nn.Conv2d(3, 4, 3, padding=1)
bn = nn.BatchNorm2d(4)
bn.eval()                           # fold uses running_mean/running_var
xx = torch.randn(1, 3, 6, 6)
with torch.no_grad():
    ref = bn(conv(xx))
    w, b = tv.fold_bn_into_conv(conv.weight, conv.bias, bn.running_mean,
                                bn.running_var, bn.weight, bn.bias)
    got = torch.nn.functional.conv2d(xx, w, b, padding=1)
print(float((got - ref).abs().max()))    # 2.38e-07
```

The function takes `[Cout][Cin][kH][kW]` weights and returns folded weight and
bias tensors.  Call it after `bn.eval()`; in training mode the batch statistics
are not the running statistics, and the comparison will be meaningless (we
measured a 1.76 absolute difference when we forgot `.eval()` - a useful
warning, not a backend bug).

### 5.6 The strictness contract

Every aten op that is not in the table raises, and the message names the op:

```python
>>> torch.sin(torch.zeros(4, device=tv.device()))
RuntimeError: torch_vc4: 'aten::sin' is not implemented for device vc4.  This backend
  has no CPU fallback by design; see README.md for the op table (matmul/linear,
  conv2d, pools, softmax, elementwise, batch_norm inference, views).
```

That is the design.  If a model needs `sin`, it will not silently run half on
the CPU and half on the GPU.  The alternative is to rewrite the op in terms of
the supported set, or to use the ONNX engine where a kernel exists.

Other representative raises:

```
torch.sin(vc4)          'aten::sin' is not implemented for device vc4
torch.mean(vc4, dim=1)  a dim reduction is not in the 0.2.0 surface (full mean is)
broadcast add           add requires identical shapes (0.2.0)
float64 on vc4          only float32 tensors are supported, got Double
mm cpu x vc4            mm(A): expected a vc4 tensor, got cpu
```

A non-contiguous view is a metadata alias; a compute op on it raises rather
than silently copying.  The exact op table, including the 0.3.0 work, is in
[API.md](API.md); `COMPLETENESS.md` is the $1 work's authoritative
per-op table.

---

## 6. Export a model to ONNX and run it

The engine consumes **opset 13, fp32, static shapes, batch 1** (the NHWC conv
path supports batch N, but the portable path is batch 1).  Export with the
legacy TorchScript exporter (`dynamo=False`) so you get the stable op set the
engine implements:

```python
import torch, torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(8 * 8 * 8, 10)).eval()
example = torch.randn(1, 3, 16, 16)          # static shape, batch 1

torch.onnx.export(
    model, example, "tiny_fp32.onnx",
    opset_version=13,
    input_names=["images"], output_names=["logits"],
    dynamo=False)
```

Verified on the board (the deprecation warning is expected on torch 2.15):

```
$ python export_test.py
.../torch/onnx/_internal/... DeprecationWarning: You are using the legacy
TorchScript-based ONNX export. ...
exported 22158 bytes
```

Check coverage **before** you run.  `vc4infer nodes` reads the graph,
attributes, initialiser shapes and inferred shapes; it does not touch the GPU:

```sh
$ vc4infer nodes tiny_fp32.onnx
model:  tiny_fp32.onnx  (0.0 MiB, opset 13, pytorch 2.15.0)
input:  images  1x3x16x16 FLOAT
output: logits  1x10 FLOAT
5 nodes: 5 supported, 0 unsupported, 0 unverified
tiers:  5 vc4, 0 folded, 0 host-meta, 0 host-control, 0 host-unwritten, 0 unsupported-graph
policy: mode=hybrid host_policy=device

  op                      nodes    vc4   fold   meta    ctl   unwr  implementation
  Conv                        1      1      0      0      0      0  QPU conv: NHWC 1x1/3x3/5x5 s1/s2, dilation 1..8, groups
  Flatten                     1      1      0      0      0      0  view (no copy)
  Gemm                        1      1      0      0      0      0  QPU gemm + Add
  MaxPool                     1      1      0      0      0      0  QPU maxpool_nhwc + QPU -inf border
  Relu                        1      1      0      0      0      0  QPU elementwise

unsupported by the VC4 executor: none

host debt: none

static report only; `vc4infer run` is authoritative
```

Then run it with a `.npy` of the exact input shape:

```sh
python -c "import numpy as np; np.save('x.npy', np.random.randn(1,3,16,16).astype('float32'))"
vc4infer run tiny_fp32.onnx --input x.npy
```

(That last `run` was not re-executed during this documentation pass; the same
graph was covered by `nodes`, and the shipped YOLOv8n/`modelA`/`modelB`/
`modelC` models were run by `selftest`.  It is the same command path.)

Export rules that avoid surprises:

* **Static shapes.**  `dynamic_axes` is not supported; every dimension is
  baked in.  A different input size means a new export.
* **Opset 13.**  Newer opsets introduce ops the engine does not implement.
* **No training-mode ops.**  `BatchNorm` in training mode inserts
  `BatchNormalization`/`ReduceMean` graphs the engine rejects; call
  `.eval()` first, or fold the BN (section 5.5).
* **Check `nodes` first.**  `Sin`, `Pow`, `Expand`, `BatchNormalization`,
  `LRN`, `ConvTranspose` and arbitrary kernels are not implemented; `nodes`
  names them before you spend time on a run.
* **Quantisation is not a shortcut.**  The stack is fp32; int8 has no exact
  packed byte MAC on this ISA ([int8-RESULTS.md](int8-RESULTS.md)).

---

## 7. Performance expectations

All numbers below are **stock clocks** (core 400 MHz / V3D 300 MHz), one GPU
client at a time, through `vc4d`.  They are measurements, not projections.
The board is shared; wall time moves 15-25% between sessions, the launch count
and the MAC share do not.

### 7.1 YOLOv8n 160x160 (the throughput mode)

Measured **2026-11-20**, quiet board, after `/tmp` was moved off tmpfs:

| metric | value |
|---|---:|
| wall / FPS | **478.7 ms / 2.09 FPS** |
| launches | **784** |
| arithmetic on VC4 | **100.0%** (273,249,600 / 273,249,600 MACs) |
| host compute nodes / host bytes | **0 / 0 B** |
| QPU duty | **54%** (259 / 478.7 ms) |
| output vs reference, 5e-4 | **0 / 44,100 elements over** |

The QPU time is 259 ms; the other ~220 ms is the launch path (socket write,
daemon validation, GPU lock, kick, `LAUNCH_WAIT`, cache maintenance).  That is
why the honest statement is "100% of MACs on the QPU, 54% QPU duty": the QPU
is idle while the ARM issues the next launch.

The same boot before `/tmp` was masked measured 539-543 ms / 1.84 FPS with
QPU duty 59%; the change was memory pressure, not a kernel change
([BUILD.md](BUILD.md)).  A loaded session during this documentation pass
measured 587.3 ms / 1.70 FPS and 646.0 ms in `selftest`; both still report 784
launches and the zero-host-byte invariant.

### 7.2 YOLOv8n 320x320 (the recall mode)

The 320 export needs a larger arena.  Measured **2026-11-20** with a 150 MiB
daemon arena and a 145 MiB client region:

| metric | value |
|---|---:|
| wall / FPS | **1,323 ms / 0.76 FPS** |
| launches | 1,936 (11.0 per dispatched node) |
| arithmetic on VC4 | **100.0%** (1,092,998,400 MACs) |
| host compute / host bytes | **0 / 0 B** |
| QPU duty | **71.9%** (950.8 / 1,323.1 ms) |
| output vs reference, 5e-4 | 1 / 176,400 over (max rel 1.003x tolerance) |

The recipe (exact commands from the measured run):

```sh
sudo systemctl stop vc4d
sudo /usr/local/sbin/vc4d --socket /run/vc4d.sock --arena-mb 150 \
     --group vc4 --drop-user nobody --max-timeout-ms 10000 &
sleep 1
VC4INFER_ARENA_MB=145 VC4_POOL_RESERVE_MB=32 \
  vc4infer bench yolov8n_320.onnx --reps 3
```

The client region is what the engine can see, so `VC4INFER_ARENA_MB` must match
what the daemon serves.  The recipe was **not re-run during this documentation
pass**: the board's `CmaFree` was only 63 MiB at the time (the DRM driver and
the daemon already held their blocks), and a 150 MiB contiguous CMA allocation
is not reliable then.  The 1,323 ms row is the recorded run from
[ONNX.md](ONNX.md) section 10.4.3.

**Why 640 does not fit.**  The 320 run already uses a 113 MiB graph pool and a
33.4 MiB peak, plus ~23 MiB of per-conv scratch; the 150 MiB daemon arena is
tight.  A 640 export is 4x the spatial area: scaling the measured 320 sizes,
the stem plane alone is ~26 MB, the im2col panel ~65 MB, and the activation
pool ~4x.  That is well past the 256 MiB CMA on this board (and past what the
1 GB ARM can hold alongside the DRM framebuffers).  No 640 export has been run
here; the honest statement is "320 is the largest measured size, and 640 is out
of the board's memory budget."

The default 72 MiB daemon / 64 MiB client fails 320 at plan time with:

```
vc4infer: unsupported: unsupported graph: graph pool exhausted:
589888 B needed, 7845248 B used of 8388608 B
```

(exit 3).  That is the arena, not a kernel bug; see section 9.6.

### 7.3 GEMM ceilings: engine vs hand-written asm10

| kernel | GFLOP/s | where measured |
|---|---:|---|
| `vc4infer` engine GEMM (through `vc4d`) | **8.5** | 256³, 2026-11-19/20, stock clocks, engine pack+kernel time |
| hand-written `asm10` GEMM, stand-alone | **13.7** | recorded in `kernels/wider-tile/asm10/RESULTS10.md`; the same kernel re-run on a busy board gave 8.12 because of memory latency, with identical cycle counts |

The engine number is what you get through the public API; 13.7 is the
hand-written kernel's recorded ceiling, not a `vc4infer` number.  Structural
peak at 300 MHz is 28.8 GFLOP/s (12 QPUs x 16 lanes x 2 ALU ops x 1/4
instruction per clock); see [ENGINE-RESULTS.md](ENGINE-RESULTS.md).

### 7.4 `torch_vc4` vs the CPU: where it is slower, and why

Measured **2026-11-20** on `torch_vc4` 0.2.0, torch 2.15.0+cpu, stock clocks.
CPU is the same torch build on the 4x Cortex-A53.

| op | vc4 | CPU | ratio |
|---|---:|---:|---:|
| `mm` 256³ | **4.98 ms** | **2.891 ms** | **1.7x slower** |
| conv 16→16, 3x3, 32x32 | **5.20 ms** | **2.30 ms** | **2.3x slower** |
| small ops (relu 1K, sum 1K, pool, softmax) | 0.2-5.5 ms | 0.02-0.14 ms | **5-180x slower** |

The small-op ratios are launch-bound and round-trip-bound: one launch costs
~0.15-0.25 ms wall (socket write, daemon validation, GPU lock, kick,
`LAUNCH_WAIT`, cache maintenance), and several ops are two or three launches.
`sum` of 4096 floats is a 16x64x1024 GEMM (1 M MACs for a 1 Ki reduction) and
measured 5.49 ms vs 0.029 ms on the CPU (187x); `linear 1x1024x4` pads the 4
outputs to 64 columns.  At 256³ the QPU GEMM is still 1.7x slower than torch's
four-core CPU GEMM, even though the engine's own kernel rate is 8.5 GFLOP/s.

**The honest summary:** on this board, eager PyTorch is *not* faster than the
CPU for small tensors.  Its value is that the work happens on the QPU at all,
and that the same code trains.  The graph engine (`vc4infer`) is where the
speed is: a whole YOLOv8n graph at 160x160 runs at 2.09 FPS with 100% of the
MACs on the VC4 and zero host bytes.  Use `torch_vc4` for training and
experimentation, export to ONNX, and run inference through `vc4infer`.

The per-op table behind these ratios is `tests/bench_ops.py` in the
`torch-vc4` checkout; the full forward/gradient parity tables are in its
`RESULTS.md`.

---

## 8. Where to go next

* [API.md](API.md) - `vc4infer` and `torch_vc4` reference, errors, op table.
* [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - every failure mode we hit.
* [ONNX.md](ONNX.md) - the executor, supported nodes, measured models, the
  device byte invariant.
* [NUMERICS.md](NUMERICS.md) - the QPU FPU truncates, denormals flush, and the
  other rules that make a reference differ in the last bits.
* [BUILD.md](BUILD.md) - build parallelism on 1 GB, ccache, `/tmp`.
* [VC4INFER.md](VC4INFER.md) - the packaged product and its CLI.
* [DAEMON.md](DAEMON.md) / [VC4D.md](VC4D.md) - the privileged helper.
* [VC4MON.md](VC4MON.md) - read-only telemetry without `/dev/mem`.
