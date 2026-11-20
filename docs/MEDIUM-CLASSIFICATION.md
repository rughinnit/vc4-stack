# GPU-accelerated image classification on low-end hardware (lnstadrum, 2021)

| | |
|---|---|
| source | <https://medium.com/analytics-vidhya/towards-gpu-accelerated-image-classification-on-low-end-hardware-ec592e125ad9> |
| author | **lnstadrum** (Beatmup author) |
| published | Analytics Vidhya, **2021-04-20**, 17 min read |
| read for this review | full article, 2026-11-20, via the Wayback snapshot `web.archive.org/web/20241205164632/…` (Medium returns HTTP 403 to a direct fetch) |
| method | **GLSL / OpenGL ES 2.0 fragment shaders**, not QPU assembly |
| code | <https://github.com/lnstadrum/beatmup> (inference), <https://github.com/lnstadrum/imagenet_dogs_classifier> (training) |
| status here | **documented, technique reference** — the GL path is not our path, but its constraints and its numbers are directly useful |

## 1. What it did

Ran a **225,112-parameter** image classifier on a **$10 Pi Zero W**, entirely on the
VideoCore IV through **GLSL fragment shaders**, reaching **72.0 % top-1** on a
120-class dogs-and-cats subset of ILSVRC-2012 (72.28 % after averaging three
checkpoints). Throughput **≈670 M MAC/s** on the Zero.

| device | inference time (385×385, 400 M MAC/image) |
|---|---:|
| **Pi Zero W** (VC4, 250 MHz, 1 GHz single-core ARMv6) | **595 ms** |
| **Pi 3 B+** (same VC4, 300 MHz, 1.5 GHz quad A53) | **328 ms** |
| Jetson Nano / desktop GPUs | "a few ms" (RTX 2070) |
| Galaxy M31 (Mali-G72 MP3) | ~8 fps |
| Huawei P20 Lite (Mali-T830 MP2) | ~5 fps |
| Asus Fonepad 8 / PowerVR G6430 (2014) | ~2.3 fps |

Accuracy on every target stayed within **~0.2 %** of the fp32 reference, except on
the two Pis where the 16-bit fixed-point dense layer is used instead of float
textures.

## 2. The constraints it documents for VC4 via OpenGL ES 2.0

These are the rules that shaped the network, quoted because they are hardware
facts about this GPU, not about GLSL style:

1. **Activations live in RGBA8 textures**: 8 bits per channel, values **0…1
   only**. So activations are quantised *and* range-limited.
2. **≤ 25 texels per output value** sampleable from one 4-channel texture, and
   **≤ 8 textures bound per shader** → at most **32 feature maps** visible to one
   shader. That caps a plain 1×1 convolution over wide tensors.
3. No compute shaders (that needs GLES 3.1, i.e. VideoCore VI / Pi 4+), so no
   float in/out, no arbitrary gather.

## 3. The model that fits

- Main block: **residual + grouped separable convolution + channel shuffle**
  (ShuffleNet/ResNeXt ideas), **not** the MobileNet-v2 inverted residual —
  because pointwise convolutions are the worst case for (2).
- Group size 8 in the downsampling/final convs; shuffle applied by *rebinding
  textures in a different order* — "we get this shuffling literally for free in
  inference: there are no memory copies".
- **Activation: a piecewise-linear approximation of sigmoid**, chosen because it
  maps to 0…1, has a non-zero gradient over −6.5…6.5, and is cheap both ways.
  ReLU clipped to 0…1 did not train; ReLU6 trained but capped accuracy below 70 %.
- **BatchNorm fused into the convolution**, activation applied on every write.
- **Dense layer**: 16-bit fixed point packed across two texture channels
  (fractional + integer, covering −128…128), implemented as a blocked GEMV
  (8-wide chunks) with reduction passes. They tried a bfloat16-like format and
  found it *less* accurate for the reduction sums; fixed-point adds/mults are
  exact inside the range.
- Training: Keras/TF 2.3, Adam, lr 0.01 for 400 epochs then halving to 600,
  batch 64, BN momentum 0.75, T4 ≈30 min/epoch, plus a custom sigmoid-like TF op
  and FastAugment.

## 4. Why this matters here: three independent corroborations

**(a) The CPU is the bottleneck, not the GPU.** The same VC4 runs this network
in 595 ms on a Zero (1 GHz single-core ARMv6) and 328 ms on a Pi 3 B+ (1.5 GHz
quad A53), and they attribute the difference to *"the CPU reactivity is still
important to keep the GPU supplied with the shader programs efficiently"*. Our
own measurement says the same thing from the QPU side: 784 launches, **59 % QPU
duty**, 41 % of wall time in the launch path. Two different graphics stacks, one
conclusion: this board is orchestration-bound.

**(b) Per-op overhead dominates at these sizes.** They conclude that GLSL/OpenGL
overhead is "significant" and that on devices with a decent CPU a bigger network
(e.g. MobileNet) will run *faster on the CPU* — with the Zero as the exception
because its CPU is so weak. That is precisely our finding for YOLOv8n at 160:
the Pi's four A53s beat our QPU engine on that workload.

**(c) Co-design beats general-purpose on this hardware.** Their 670 M MAC/s on a
Zero comes from 8-bit activations, grouped convolutions and a shuffle that costs
nothing. We run an unmodified fp32 YOLOv8n with 784 launches and get ~525–570
M MAC/s on a *faster* board (Pi 3B, 273 M MACs in 478–520 ms). Their network is
smaller and 8-bit, so this is not a like-for-like comparison — but the direction
is unambiguous: **matching the model to the machine is worth more than a faster
kernel.**

## 5. What we can borrow, ranked

1. **Activation/weight storage in 8 bits.** We ruled int8 out as a *speed* lever
   (no packed byte MAC, `docs/int8-RESULTS.md`) but not as a *memory* lever. Their
   72 % accuracy with 8-bit activations is direct evidence that 4× less arena is
   achievable for a co-designed model, and our arena is the binding constraint at
   320×320.
2. **Fuse activation into the convolution epilogue.** They fuse BN into the conv
   and activate on every write. We fold BN (`fold_bn_into_conv`) but the
   SiLU/Relu still costs separate launches — 468 of our 784 launches are
   elementwise passes. This is the same conclusion our launch profile gives,
   reached from the other end of the stack.
3. **Co-design for the op set.** A VC4-shaped network needs ~20 ops (conv,
   grouped conv, shuffle-as-index-remap, BN, a sigmoid-like activation, pool,
   GEMV), not the 152 expressible ONNX ops we are grinding through. For a
   *product*, shipping one co-designed model beats covering the opset.
4. **Channel shuffle as a binding/plan permutation, not a copy.** Our `Concat`/
   `Split` copies (5.2 MB and 1.0 MB per YOLO inference) are exactly what they
   avoid by permuting texture bindings. A plan-level channel remap would remove
   copies we currently do with `copy2d`.

## 6. What does not transfer

- It is **GLSL, not QPU assembly**, so there are no instruction-level findings —
  no VPM/DMA layout, no hazard rules, nothing for `docs/NUMERICS.md`. (For that,
  see `docs/VC4CV.md`, which is QPU assembly by a different author.)
- 8-bit quantisation is a *training-time* choice; it cannot be retrofitted to an
  exported fp32 ONNX model without requantising and losing the verified
  accuracy we report.
- Its performance is bounded by GL overhead and texture I/O, so it does not tell
  us what the QPU could do — only what the GL path did.
- The 0…1 texture range forces the sigmoid-like activation; we are not bound by
  that and should not adopt the activation for its own sake.

## 7. Status

Documented and cited; no code or model reused. Two items are worth scheduling
against measurements, in this order: **fused activation epilogue** (we can
measure it directly on YOLOv8n's 468 elementwise launches) and **plan-level
channel permutation** to remove `Concat`/`Split` copies. The 8-bit storage idea
is a design option for 320×320, not a drop-in.
