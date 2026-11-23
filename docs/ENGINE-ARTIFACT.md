# The engine artifact: `vc4infer build` and the `.vc4x` file

An engine artifact is the model after the optimiser, in the form a load can use
without planning or packing: the optimised graph, the weights already in the
arena layout the kernels consume, the arena plan with fixed offsets, the kernel
blobs the plan launches, and the keys that make the file refuse to run anywhere
it was not built for.  It is the same idea as a TensorRT engine: build once on
the target, load many times.

```
vc4infer build model.onnx -o model.vc4x     # build on the board
vc4infer verify model.vc4x                  # every section, every key
vc4infer bench model.vc4x                   # timed load and runs
vc4infer run   model.vc4x --input bus.jpg   # unchanged call path
```

`Engine("model.vc4x")` is the loader.  It is selected by the file's magic
(`VC4XART1`), not by the extension, and `Engine(path, artifact=False)` — the
CLI's `--no-artifact` — runs the ONNX graph the artifact carries, planning from
it exactly as the engine does for an `.onnx`.  That is the fallback and the A/B
control, not a second implementation of the artifact.

## Format

A single little-endian file.  The header is fixed size; the manifest is JSON
with sorted keys; the payload holds named sections.

| offset | bytes | field |
|---:|---:|---|
| 0 | 8 | magic `VC4XART1` |
| 8 | 4 | format version (u32, currently 1) |
| 12 | 4 | manifest length N (u32) |
| 16 | 8 | payload length M (u64) |
| 24 | 32 | sha256 of the manifest bytes |
| 56 | N | manifest JSON |
| 56+N | M | payload: the sections the manifest's table names |

Each section is `{"name", "offset", "bytes", "sha256"}` with the offset relative
to the payload start.  The header and the manifest are hashed on every load (a
truncated or edited manifest cannot reach the executor).  Payload sections are
hashed by `Artifact.verify()` — `vc4infer verify` — and by
`Artifact.load(verify=True)`; the graph section is additionally hashed by
`graph_bytes()` because it is the object the run is reconstructed from.

### Manifest

| key | contents |
|---|---|
| `format`, `format_version`, `layout_version`, `created` | container identity |
| `keys.generator_digest` | sha256(kgen.py, kops.py, kernel ABI tag), the same digest the blob cache stamps into every `.bin` |
| `keys.abi_tag` | the kernel ABI string (`vc4infer-kernel-abi-2` today), read from the library |
| `keys.engine_digest` | sha256 over the engine Python package, by relative path |
| `keys.board` | `/proc/device-tree/model`, e.g. `Raspberry Pi 3 Model B Rev 1.2` |
| `keys.arch`, `keys.engine_version` | reported, not checked |
| `model` | graph sha256, inline-or-referenced, `.pt` export record, pass list with per-rule records, input and output specs |
| `shapes` | the static shape map `coverage.refine_shapes` computed at build |
| `plan` | `pool_bytes`, `segment_bytes`, `scratch_bytes`, `estimate_bytes`, `floor_bytes`, `high_water_bytes`, the allocation table, and (optional) `frozen`, the execution plan |
| `weights` | packed-panel entries (offset into the weights section, bytes, sha256, panel key, kind, layout, shape) and `dropped` initializers |
| `kernels` | blob names, sizes and sha256, embedded or referenced; the generator digest again |
| `tuning` | `{"autotune": "off", "records": []}` — a field for `Engine.autotune` choices; autotuning is off by default and the loader does not invent records |
| `sections` | the payload table above |

### The arena plan

The plan is a keyed allocation table, not a call order.  At build the executor
labels every arena allocation with the thing that asked for it:

| key | allocation |
|---|---|
| `const:<initializer>#n` | a float initializer upload |
| `panel:<i>#0` | the *i*-th prepacked panel, uploaded at load |
| `in:<input>#n` | a graph input, uploaded at the start of each run |
| `n<idx>:<Op>#n` | the *n*-th buffer a node allocates (outputs, temporaries, GEMM copies) |

The table records `(segment, offset, bytes)` for each.  On load the pool hands
each key its recorded offset instead of bump-allocating, and refuses, by name,
an allocation that is absent, a size that changed, an offset outside its
segment, or two keys whose recorded offsets overlap while both are live.  That
last check is the difference between a frozen plan and an unchecked recording:
a plan whose lifetimes no longer match cannot quietly produce wrong numbers.

Segment bases come from the library's arena allocator, so the plan stores
offsets *inside* a segment and rebases onto whatever base the load gets.  The
engine's per-op scratch (the conv padded image and im2col panel, elementwise
workspaces) is not part of the pool: it is carved above the pool, reset by every
op, and its required size is the recorded `scratch_bytes`; `check_arena()`
refuses a plan whose pool plus scratch does not fit the client arena, before the
first launch.

### The frozen execution plan

The arena plan says where the bytes live; the frozen plan says what the frame
does with them.  Without it an artifact runs the Python executor for the whole
frame and pays the per-node cost the plan removes (0.226 s of CPU per 160 frame
against 0.182 s, measured below).  The build records the call sequence from the
frame whose offsets the arena plan pins and stores it in the artifact:

| key | contents |
|---|---|
| `plan.frozen.version` | the plan format (1); a reader refuses another |
| `plan.frozen.abi_bytes` | `sizeof(vc4_plan_op)` at build; a reader refuses a different record size |
| `plan.frozen.section` | the `plan_ops` section: the records, little-endian, `abi_bytes` each |
| `plan.frozen.steps` | `["c", n]` for a run of *n* records, `["p", i]` for a node on the Python path, `["s", i]` for a node the optimiser accounted for |
| `plan.frozen.nodes` | the node indices each step covers, aligned with `steps` |
| `plan.frozen.desc` | every output name the plan can produce: offset, bytes, shape, layout, pitch |
| `plan.frozen.feed_sig` | input name → shape and element count the recording was made for |
| `plan.frozen.segment_bases`, `segment_bytes` | the recording pool's geometry |
| `plan.frozen.kernels_sha256` | sha256 over the artifact's `(name, sha256, stamp)` kernel table |
| `plan.frozen.launches` | the recorded frame's launch count, for the build's own check |

Offsets are absolute arena addresses, so they are recorded with the build's
segment bases and **rebased** on load by the difference between those and the
segments the load receives.  A buffer never spans a segment, so each offset
maps to exactly one segment.

Refusals at load, each naming what disagrees and leaving the artifact on the
Python path with the reason in `runner.plan_error`:

* a plan format version this engine does not read;
* a record size that differs from the loaded library's `vc4_plan_op_size()`;
* a `plan_ops` section whose length disagrees with the step table, or a step
  kind outside `c`/`p`/`s`;
* a segment count or segment size that differs from the load's pool;
* an offset outside every recorded segment;
* a kernel-table digest that differs from the one the plan was recorded
  against - the records name no kernel, the C entry points choose one from the
  table, so a plan and a table that do not belong together are refused by
  name;
* a feed whose shape or element count differs from the recording (at run time;
  the artifact falls back to Python rather than raising, because its own arena
  plan is still valid).

The recording follows the same safety rule as the ONNX path: a frame that
allocated a buffer outliving it is not a recording, and a recording in which a
per-frame allocation overlaps a panel or constant is refused.  At build time
that refusal fails the build with the reason, rather than writing an artifact
that would replay into another buffer.  Pass C of the build then loads the
artifact exactly as a caller would, replays the frozen plan and checks the
outputs bit-exact and the launch count against the recorded frame; an artifact
whose plan does not reproduce the frame is deleted, not shipped.

`VC4_NO_CPLAN=1` on load leaves the plan out and runs the Python path over the
artifact's pinned arena, which is what this path did before the plan existed.

### Weights

`vc4_conv_nhwc_pack_w` builds the kernels' k-major, tap-major B panel
(`k = tap*Cin + ci`, 64 B rows) from `[Cout][Cin][kh][kw]`.  At build the panels
are read back from the arena and stored byte for byte; at load they are uploaded
into their recorded slots, so the packing kernels never run.  The same section
carries the executor's other plan-time buffers: broadcast expansions of graph
constants, and Gemm operand copies.

A conv's raw weight tensor is *not* uploaded on the artifact path when every
consumer of that initializer is a panel-backed Conv (weight and bias).  The
conv describes itself from the host copy and the kernel reads the panel; on
YOLOv8n at 160 that is 12.4 MiB of the 12.4 MiB of constants, so the arena
holds 12.4 MiB of panels instead of 24.8 MiB of panels plus raw weights.  The
artifact records those names under `weights.dropped`; a stub that raises
`ArtifactPlanError` stands in for the tensor, so a path that suddenly asks for
raw bytes fails loudly instead of reading an unrelated buffer.

### Kernels

The list is every blob the context loaded while the build ran the graph through
the load path (the C context keeps the names; the kernel table itself is an LRU
of 128 code slots and a detector loads more than that).  Blobs are embedded by
default; `--no-embed-kernels` records names and hashes only and the loader then
needs the cache.  Either way the loader writes what it has into the kernel cache
with the blob's own `.stamp` sidecar (the per-kernel digest of the generator
hash and the kernel's argument vector), which is what the cache compares; a
blob without its stamp is present but the cache regenerates it.  Checked with
an empty cache directory: loading YOLOv8n's artifact wrote 98 blobs and their
stamps, and the first run took 1.47 s with no generator invocation.

## What a load skips

| work | ONNX path | artifact path |
|---|---|---|
| pass pipeline | runs the `VC4_OPT` spec | recorded; the embedded graph is already rewritten |
| shape refinement | `coverage.refine_shapes` (5 `onnx.shape_inference` calls on YOLOv8n) | used from `shapes` |
| arena estimate | `estimate_pool_bytes/floor/scratch` | `plan` |
| pool layout | bump + free list, decided per run | keyed offsets pinned from the plan |
| weight packing | one GPU pack launch + host permutation per conv | panels uploaded from the artifact |
| executor rewrite tables | `_find_silu` + `_collect_fused` | restored from `exec` |
| kernel generation | JIT on a cold cache | embedded blobs seeded into the cache |
| the frame's call sequence | the Python executor decides per node | replayed from `plan.frozen`, one C call per run of device operations |

The graph is still parsed (it is protobuf, and the executor walks it) and
constants are still uploaded.  The artifact is a plan, not an interpreter.

## Keys and refusal

`Engine` calls `Artifact.validate()` before the arena is opened, so a rejected
artifact costs nothing and leaves the GPU free.  A mismatch names the key:

```
vc4infer: artifact rejected: generator_digest mismatch (artifact '25ebd6…',
engine '8e0dad…'). the kernel generator changed, so the packed weights and the
blob list may describe different kernels; rebuild the artifact
```

| key | checked against | why it must match |
|---|---|---|
| `generator_digest` | the running library (`vc4_generator_digest`) | a different generator makes different kernels for the same name |
| `abi_tag` | `vc4_kernel_abi_tag()` | uniform layout changes with the ABI |
| `layout_version` | this reader | the file layout |
| `engine_digest` | sha256 of the installed package | the plan and the rewrite tables are the executor's decisions |
| `board` | `/proc/device-tree/model` | the arena plan and packed layouts are measured on one board |
| `arena` | the client arena size | the plan and its scratch must fit |

Unlike the ONNX file, an artifact is hardware-, generator- and engine-specific.
Copy it to another board, upgrade `vc4infer`, or change `kgen.py`, and it
refuses; rebuild it there.  `vc4infer verify` reports every key and hashes every
section without opening the device.

## Correctness

For `yolov8n_160` and the three shipped models the artifact path and the ONNX
path (`VC4_OPT=all`, i.e. the same rewritten graph) produce **bit-identical
outputs**: max abs difference 0 over every output element, no tolerance used.
The steady-state launch count matches exactly:

| model | artifact launches | ONNX launches |
|---|---:|---:|
| modelC | 8 | 8 |
| modelB | 15 | 15 |
| modelA | 26 | 26 |
| yolov8n_160 | 536 | 536 |

The *first* run differs, and that is the point:

| model | artifact first run | ONNX first run | difference |
|---|---:|---:|---:|
| modelC | 8 | 9 | 1 pack |
| modelB | 15 | 18 | 3 packs |
| modelA | 26 | 28 | 2 packs |
| yolov8n_160 | 536 | 600 | 64 packs |

`vc4infer build` runs this check itself as its last step (pass C): it opens the
file it just wrote through `Engine`, runs the same feed, and refuses to leave
the artifact on disk unless the outputs are bit-exact and the launch count
equals the recorded steady-state count.

## Measurements

Board: Raspberry Pi 3B, BCM2837, VideoCore IV, kernel 6.18.50, stock clocks
(core 400 MHz, V3D 300 MHz), managed backend through vc4d, client arena 96 MiB.
`tools/thermal-guard.sh --max 71` wrapped every run; the core clock stayed at
400 MHz for all 40 measured loads, the 10 paired runs and the 5 passes-off
runs, peak 65.5 C, `throttled=0x20000` (set before this work).  Each
load/first/steady row is the median of five *separate*
processes; steady state is the median of five timed runs after the first run,
`VC4_OPT=all` on the ONNX path.  Both paths had a warm kernel cache, so no row
includes JIT.

| model | build | artifact | load ONNX | load artifact | first ONNX | first artifact | steady ONNX | steady artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| modelC | 0.7 s | 0.05 MiB | 45.0 ms | 51.1 ms | 120.2 ms | 108.6 ms | 5.67 ms | 5.76 ms |
| modelB | 0.7 s | 0.12 MiB | 60.9 ms | 51.7 ms | 172.0 ms | 138.5 ms | 15.66 ms | 15.88 ms |
| modelA | 0.9 s | 0.75 MiB | 99.4 ms | 65.3 ms | 198.8 ms | 181.8 ms | 25.77 ms | 26.03 ms |
| yolov8n_160 | 12.5 s | 25.08 MiB | 3209.2 ms | 481.4 ms | 1776.9 ms | 1463.1 ms | 472.87 ms | 468.38 ms |

Time to the first result (load + first run) is the number a service pays at
startup: YOLOv8n 4.99 s → 1.94 s (−61%), modelA 0.298 s → 0.247 s (−17%),
modelB 0.233 s → 0.190 s (−18%), modelC 0.165 s → 0.160 s (−3%).

**The gain is startup, not throughput.**  Steady state is the same 536 launches
of the same kernels at the same shapes.  On YOLOv8n the artifact measured
468.4 ms against 472.9 ms (five-process medians) and 468.3 against 473.0 ms in
a paired run that loaded the ONNX path first — about 1%, consistent between the
two orderings but not a launch-count effect.  The plausible cause is the
12.4 MiB smaller resident arena from dropping the raw conv weights; it is not
claimed as a speed-up.  On the three small models the medians differ by 0.1-2%
in either direction (modelC's artifact steady state is 0.09 ms slower).

The plan the artifact freezes is the one the epilogue pass produces: on this
tree YOLOv8n runs at 468-473 ms / 536 launches with the pass on, against
550.0 ms / 1,003 launches with `VC4_OPT=none` (five-process medians).  0.165 to
0.175 ms of wall time per launch is what those 467 extra launches cost, so a
frozen 536-launch plan is a frozen 536-launch cost; the artifact removes
neither the launches nor their price.

### Where the startup difference comes from

YOLOv8n at 160, measured on the first run:

| component | ONNX path | artifact path |
|---|---:|---:|
| planning (passes, shapes, estimate) at load | 2.72 s | none |
| QPU pack kernels in the first run | 22.0 ms (64 launches) | none |
| host bytes moved in the first run | 12.77 MB | 0.17 MB |
| host permutation inside `vc4_conv_nhwc_pack_w` | ~294 ms of the first run's wall time | none |
| panel upload at load | — | 12.37 MB (part of the 0.49 s load) |

The first-run wall difference is 314 ms for YOLOv8n: 22 ms is QPU pack time,
the rest is the host permute and the upload the packer does.  The artifact pays
a 12.37 MB panel upload at load instead, which is inside the 485 ms.  Total host
bytes moved are therefore similar; what changes is *when* and *how often* —
once per build instead of once per process, and without a GPU round trip per
conv.  The load-time saving is the larger one because `refine_shapes` runs five
`onnx.shape_inference` passes over the graph.

modelC's load is 5 ms *slower* on the artifact path: a five-node graph costs
almost nothing to plan, while the artifact load has fixed costs (engine-package
digest, graph parse, manifest).  The artifact is not a win for a graph that
takes longer to read than to plan; it is a win at detector scale.

## `.pt` input

`vc4infer build model.pt -o model.vc4x` exports through Ultralytics first
(`YOLO(pt).export(format="onnx", imgsz=…, opset=…, simplify=False,
dynamic=False, batch=1, device="cpu")`) and records the size and opset in
`model.export`; the defaults are `imgsz=640`, `opset=17`.  The exporter is a
dependency: this environment has neither torch nor Ultralytics installed, so
the `.pt` path could not be exercised here and is not claimed as measured.  The
ONNX models measured above are the shipped exports (`yolov8n_160.onnx`, imgsz
160, opset 17, per `docs/yolo-matrix.json`).

## Tests

`infer/tests/test_artifact.py` runs without a device: container round trip,
truncated header, bad magic, manifest checksum, short payload, section
corruption, referenced-graph change, every key's named refusal, the pinned
allocator's four refusal cases, the dropped-initializer predicate, and the
restored rewrite tables.  The on-board end-to-end check is `vc4infer build`
pass C, which every build runs.

## Limits

- One board, one generator, one engine version: anything else is a rebuild.
- Autotuning has a field (`tuning`) and no behaviour: records are empty and the
  loader applies none.  Turning it on means the build must measure and record
  the per-shape counts, and the loader must feed them to `vc4_autotune`; neither
  is done here.
- Layout assignment and launch batching (passes 4 and 7 of
  `docs/OPTIMIZER.md`) are not part of the artifact; it freezes what the
  optimiser produces today.
- The plan is a recording of the executor at one version.  A code change that
  reorders allocations changes `engine_digest` and forces a rebuild, which is
  the intended failure mode.

### The frozen plan in the artifact (2026-11-23)

The plan is recorded at build and replayed on load.  Numbers below are from
`tools/cost/artifact_plan.py` on an idle board, one client through the packaged
`vc4d` (1.2.9, protocol 2, 96 MiB), ARM pinned at 1.0 GHz, each cell under
`tools/thermal-guard.sh --max 75`; steady state is the median of eight frames
after the first run, client and daemon CPU from `/proc/<pid>/stat`.  "ONNX"
is the same graph loaded from the `.onnx` file with the plan on; "no plan" is
the same artifact with `VC4_NO_CPLAN=1`.

| model | load s | first s | wall ms | client CPU | daemon CPU | total CPU | launches |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n_96, artifact | 0.58 | 1.43 | 283.6 | 0.1037 | 0.0450 | **0.1487** | 279 |
| yolov8n_96, ONNX plan on | 3.59 | 1.98 | 286.0 | 0.1063 | 0.0450 | 0.1513 | 279 |
| yolov8n_96, artifact no plan | 0.48 | — | 328.1 | 0.1475 | 0.0450 | 0.1925 | 279 |
| yolov8n_128, artifact | 0.50 | 1.34 | 314.4 | 0.1075 | 0.0475 | **0.1550** | 276 |
| yolov8n_128, ONNX plan on | 3.42 | 1.89 | 315.0 | 0.1100 | 0.0487 | 0.1587 | 276 |
| yolov8n_128, artifact no plan | 0.50 | — | 356.9 | 0.1487 | 0.0513 | 0.2000 | 276 |
| yolov8n_160, artifact | 0.52 | 1.57 | 389.0 | 0.1213 | 0.0613 | **0.1825** | 332 |
| yolov8n_160, ONNX plan on | 2.96 | 2.12 | 389.6 | 0.1212 | 0.0612 | 0.1825 | 332 |
| yolov8n_160, artifact no plan | 0.44 | — | 432.6 | 0.1637 | 0.0625 | 0.2262 | 332 |

Both claims hold.  The artifact keeps the startup gain: load 0.44-0.58 s
against 2.96-3.59 s for the ONNX path, and time to first result (load + first
run) 1.84-2.09 s against 5.08-5.57 s.  It gets the plan's CPU reduction: within
the artifact itself the plan takes 0.1925/0.2000/0.2262 s down to
0.1487/0.1550/0.1825 s, **-22.8 %, -22.5 % and -19.3 %**, and the steady state
matches the ONNX path with the plan on within 1.7 % (the artifact cell is
0.1487 against 0.1513 at 96, 0.1550 against 0.1587 at 128, and equal at 160).
Wall time and launch counts are unchanged in every pair.

The saving is smaller than the ONNX path's -30 % because the artifact's Python
path is already cheaper than a plain ONNX Python frame: the panels are
prepacked, the offsets are pinned, and the first frame does not pack.  At 160
the artifact without the plan is 0.2262 s against 0.2875 s for the ONNX graph
without it, so the plan removes 0.044 s of what is left rather than 0.086 s.

Correctness: the artifact's frame is byte-identical to the ONNX path with the
plan on for modelA, modelB, modelC and YOLOv8n at 96/128/160
(`artifact_plan.py --check`), with the same launch count; the artifact with
`VC4_NO_CPLAN=1` is byte-identical too and reports the refusal in `plan_error`.
modelA and modelC have no representable node, so neither path freezes a plan;
the loader says so and both run the Python executor.
