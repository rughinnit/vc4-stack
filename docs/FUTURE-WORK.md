# Future work

This is not a wish list. Each item below is either something with a measured
size that was deliberately not taken, or a boundary with a named cause and the
condition that would remove it. What has been measured is in
[BENCHMARKS.md](BENCHMARKS.md); what is broken is in [BUGS.md](BUGS.md); whether
a workstream is done is in [STATUS.md](STATUS.md).

The host-side account is finished: every lever named in
[CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) has been built and measured, and three of
the last five measured zero. A YOLOv8n-160 frame is 400 ms of which 268 ms is
QPU arithmetic, so further host work cannot move it much and the items below say
so rather than implying otherwise.

## Left in the engine, with a size

| item | size | what it needs first |
|---|---:|---|
| Conservative scratch over-marking | 3 ms per frame, 0.77 % | The poison-and-scan check specified in [CIVAC.md](CIVAC.md) §5: fill each `vc4__scratch_alloc` slice with `0xA5`, scan the request's scratch extent at launch for unmarked bytes that differ from the poison, abort on the first, and run it over the op suite. Without it the obligation moves to 30-odd allocation sites and no test can tell a correct mark from a write another mark happened to cover. |
| The plan-fallback path | 19 to 30 % above the 2026-11-22 window, client-side Python (0.1429 s against 0.092 at 160) | A bisect between the 2026-11-22 tree and `06ed012`, one window per candidate. It is a debt only for graphs the frozen plan cannot record; the shipping path is the plan. |
| Fixed per-launch cost at the smallest tensor | the 96 point is 8.8 % above ncnn at three passive threads, 128 and 160 are 9.8 and 16.1 % below | Fewer launches at 96, which the frame cannot provide: the graph is the same 279 launches at every size, so the cost per launch is what dominates the smallest planes. |
| Depth 3 in the client pipeline | would hide the scratch-window wait that keeps duty at 0.65 | A third window does not fit at 160 (5.76 MiB against 19.4 free beside the pool); it fits at 96. `run_pipelined` is opt-in because it missed its duty gate by 0.5 to 1.0 points. |

## Capacity

| item | measured need | condition that would remove it |
|---|---:|---|
| YOLOv8n-640 | 63.0 MiB pool + 87.9 MiB scratch, 181.0 MiB client, 184 MiB daemon | The pool grows one segment at the first conv, which starves the scratch. A planner that accounts for that growth, or a smaller scratch, would put it inside 150 MiB. |
| YOLOv8s-160 | 149.0 MiB client | It fits at 150 MiB and runs at 1.15 FPS. Accuracy was 1.50 to 3.87 before A21 and A22 were fixed and has not been re-measured since. |
| YOLOv5s-160, YOLOv9s-160 | 83.0 and 65.4 MiB planned | Both ran and both failed accuracy before A21 and A22. **Re-measure before quoting either as refused.** |
| ImageNet backbones | Inception 104 to 135 MiB, DenseNet-121 115, ResNet-50 about 150, AlexNet 196, ZFNet 362, VGG-19 603 | Weights, not activations. [MODELS.md](MODELS.md) carries the exact refusal per graph. Nothing on this board reaches 603 MiB. |

A21 and A22 are why the capacity rows above need re-measuring rather than
re-quoting: both bugs corrupt wide graphs, both were found by running wider
models, and every accuracy figure taken on a wide model before `e6e66b6` is
suspect.

## The one change that is not overhead

The QPU is 67 % of a frame. Nothing left in the host can change that. A trained
student model can:

| graph | measured | note |
|---|---|---|
| 5 conv, 96x96 | **55.2 FPS** | random weights; accuracy unproven |
| 8 conv, 96x96 | **36.3 FPS** | random weights; accuracy unproven |
| YOLOv8n, 96x96 | 3.31 FPS | real COCO weights |

The measured fit is `T = 2.5 + 3.13 * convs` ms at 96, so 30 FPS is about nine
convs and 0.05 GFLOP. [ACCURACY-FPS.md](ACCURACY-FPS.md) has the workstation
recipe; on this board a COCO epoch is 20.5 h for the teacher and 2.46 h for the
student, which is why the training belongs on a workstation and the result
belongs here.

## Kernel work that is closed, with its reason

- **Flat-panel im2col**: exact at one QPU, corrupt at two or more, and the
  padding it removes exists only where `align4(wo) < align16(wo)` — which is
  exactly where the flat body costs 1.57 to 1.77x per tap. The constraint for a
  future attempt is in [KERNEL-EFF.md](KERNEL-EFF.md) §7: one load in flight
  before the store, or a single 16-row load where the block lies inside an
  output row.
- **Direct convolution**: rejected on arithmetic, not effort. The naive form
  issues 16 times the A transactions of the panel.
- **GEMM 8.5 against 13.7 GFLOP/s**: not a code gap. Cycles per 8x64 tile-k are
  identical across runs while wall time varies by a third between samples
  minutes apart, with the clocks unchanged and the throttle latch clear. The
  measurement rules that follow are in [KERNELS.md](KERNELS.md) §9.
- **Autotuner**: the default is already optimal at these shapes; a record may
  now only win by 3 %, and the remaining 0.84x at 512³ is drift between two
  measurements of the same launch, not a bad record.
- **Daemon-side pipelining**: measured zero. The client reaps before sending,
  so the connection carries one request at a time, and a genuinely depth-2 wire
  moves the frame by nothing.

## Field widths, as a discipline

Three defects in a row came from a field narrower than its operand: the packer
divisor (B16), the 16-bit VDW stride on a wide GEMM, and `pack3`'s 13-bit source
pitch (A22). A fourth was a false cap where the guard tested the product instead
of the operand. Two generation-time checks now exist, for the 16384 B client code
slot and the 24-bit `copy2d` operand; the systematic follow-up is to audit every
remaining narrow field the generators write — uniform counts, VDR pitches, slot
counts, block counts — and give each a sentence rather than a run-time surprise
on the next model.

## Upstream products

[`UPSTREAM.md`](UPSTREAM.md) defines the layout: `origin` is our fork,
`upstream` is the project it forks, and `upstream-master` is a read-only
mirror.  [`tools/upstream-status.sh`](../tools/upstream-status.sh) reports the
divergence and exits non-zero when a fork is behind.

| fork | behind upstream | what a merge needs |
|---|---:|---|
| `VC4C` | 902 commits on the branch that carries the compiler work | a real merge project: 25 local commits on an old base, with conflicts in register allocation and VPM staging |
| `VC4CL` | 66 commits | the same shape, smaller: the runtime patches apply to code upstream has since changed |
| `VC4CLStdLib` | 0 | nothing |

Nothing in this stack depends on a fork being current: the packaged `vc4cl` and
`vc4c` build from the branches as they are.  Taking the upstream product is a
deliberate choice with a measured cost, not a background sync.

## Framework and model gaps

[STATUS.md](STATUS.md) holds one row per workstream with its state and the
measurement that would move it: TensorFlow has 18 kernels and no graph,
ONNX Runtime's C provider registers and claims nothing, TVM lowers one composite,
JAX accepts 25 VHLO names. [MODELS.md](MODELS.md) holds the per-graph refusals,
including the int8 path (`QLinearConv`), the 7x7 depthwise blocks and grouped
convolution at four groups.
