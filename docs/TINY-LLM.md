# TinyStories-1M on the VC4 QPUs

TinyStories-1M is a GPT-Neo family model (8 blocks, d_model 64) trained on
TinyStories, and the smallest causal language model with a public ONNX export
that the graph path can be pointed at. This document records
what it took to make the HF export static, what the engine ran, and the point
at which the result stops being the model.

**Result in one paragraph.** The graph executes end to end on the QPUs: 378
nodes, none refused, no CPU fallback, 378/378 on VC4, 1.06 tokens/s with a
16-token window and 0.42 tokens/s with 32. The tokens are not the model's.
The last-position logits differ from ONNX Runtime by up to 23 (mean 3.1) at
every step, all eight greedy steps disagree, and the first transformer block
is already 7e-2 off. Two engine defects are isolated in §5. Until they are
fixed this is a launch-and-cost measurement of a transformer on the QPUs, not
a text-generation result.

## 1. Model and licence

| | |
|---|---|
| repository | `onnx-community/TinyStories-1M-ONNX`, the ONNX export of `roneneldan/TinyStories-1M` |
| export revision | `b873615e9ceb8f2cdb6c14a90641c99052fa4a7f` |
| base revision | `77f1b168e219585646439073245fe87e56b3023e` |
| file | `onnx/model.onnx`, fp32, 23,589,363 bytes |
| sha256 | `593db3ed55fb5afe4bd6549c9c0cc210f34706451cfe4638c707875d570b4eb3` |
| architecture | `GPTNeoForCausalLM`: 8 blocks, 16 heads of 4, d_model 64, vocab 50,257 |
| positions | learned absolute, 2,048; local-attention window 256, so no block sees past 256 |
| activation | `gelu_new` |
| parameters | 3.74 M; 3.35 M of them are the token and position embeddings |
| licence | **none declared** |

Neither the base card nor the export carries a licence field or a LICENSE
file. The base card links only the TinyStories dataset, whose own terms
(CDLA-Sharing-1.0) are not inherited by the weights. There is no stated
licence for the weights, so this document does not assume one.

The fp16, int8 and quantized exports in the same repository were not used: the
engine is fp32 only.

## 2. What the export is, and what the wrapper changed

The fp32 export is a `use_cache=True` graph: 800 nodes, 21 operator types, 18
graph inputs (`input_ids`, `attention_mask`, `position_ids` and eight past
key/value pairs) and 17 outputs (`logits` and eight present pairs). 344 of the
800 nodes are metadata: `Shape`/`Gather`/`Unsqueeze`/`Concat` chains that
rebuild the causal mask and the reshape targets at run time. The engine accepts
fp32 inputs with static shapes only, so the export is not runnable as it
stands.

`infer/tools/onnx/make_tinystories.py` rewrites it. Every step was checked
against the original export under ONNX Runtime, with a zero-length past and an
all-ones mask: the rewritten graph reproduces the original last-position
logits to 2.9e-6 at 16 tokens and 1.9e-6 at 32, with the same argmax.

| change | nodes before | after | why |
|---|---|---|---|
| KV cache removed | 16 `Concat` + 1 `Shape` | 0 | each `present = Concat(past, new)` is bypassed and attention reads `new`; the past length the mask builder reads is the constant 0 |
| present outputs dropped | 16 outputs | 1 output (`logits`) | nothing consumes them in a cache-free run |
| `attention_mask` input dropped | 1 input | 0 | a fixed window has no padding |
| causal mask made constant | 8 `Add` + 53 orphaned nodes | 8 `Where` + one bool initializer | `Shape`/`Range`/`Trilu`/`ConstantOfShape` rebuild the same triangle every step; the bias `Add` after the `Where` adds the same mask twice |
| masked value | `-3.4028235e38` | `-1e9` | see §5: the engine refuses a non-finite Softmax input, and the export's value overflows to `-inf` on the way; `exp()` of either is 0 |
| LayerNorm fused | 17 × 9 = 153 | 17 | the decomposed `Sub(x, mean)` is a trailing-axis broadcast between two runtime tensors, which the elementwise path refuses; the engine has a fused `layernorm_fwd` kernel. Opset raised 14 -> 17 (first version with `LayerNormalization`) |
| shape metadata folded | 308 | initializers | `Reshape` targets are built from `Shape` of an activation, so inference cannot resolve any reshaped tensor; the chains are evaluated once and stored |
| MatMul made 2-D | 49 | 49 + 98 `Reshape` | the engine runs one GEMM per batch index and requires equal batch dimensions; a 3-D activation times a 2-D weight has none on the weight side |
| `wte` transpose folded | 1 | 0 | a 12.9 MB copy recomputed per pass becomes an initializer |
| last position only | 0 | 1 `Slice` | generation needs the newest token's logits; it is also the only shape this engine computes correctly (§5.1) |
| inputs cast inside | 2 | 2 `Cast` | `input_ids` and `position_ids` are fp32 inputs (`[1, N]`), cast to int64 in the graph; ids < 2^24 and positions < 2048 are exact in fp32 |

Final graph: **378 nodes, all static, 0 metadata, 0 refused**; inputs
`input_ids` fp32 `[1, N]` and `position_ids` fp32 `[1, N]`, output `logits`
fp32 `[1, 1, 50257]`; opset 17; 130 of the nodes are `Reshape` views with no
kernel launch.

## 3. Cost

Board: Pi 3B, ARM at 1.0 GHz, video core at the stock 400 MHz, `vc4d` with a
96 MiB arena, thermal guard at 75 C (peak 53 C). Engine time is the mean of
eight greedy steps after the first; the first step is 1.3 s / 3.3 s because it
pays the first upload of the 25.8 MB of embeddings. "Launch" is one vc4d
submit.

| window | nodes | on VC4 | refused | launches / token | ms / token | tokens/s | pool | peak |
|---|---|---|---|---|---|---|---|---|
| 16 | 378 | 378 | 0 | 1371 | 940 (min 866) | 1.06 | 29.8 MB | 26.6 MB |
| 32 | 378 | 378 | 0 | 3464 | 2399 (min 2351) | 0.42 | 33.4 MB | 26.6 MB |

A cache-free run recomputes the whole window per token, so these are also the
costs of one forward pass. 25 % of the wall time at 16 tokens and 20 % at 32
is QPU busy (`gpu_ms` 221 / 564); the rest is launch and host path. The daemon
serves 64 MiB of the 96 MiB arena to this client; the graph pool is 29.8/33.4
MiB including the 25.8 MiB of embedding weights.

Per operator, at 16 tokens (wall ms, launches):

| op | nodes | launches | wall ms |
|---|---|---|---|
| Where (causal mask) | 8 | 72 | 239 |
| MatMul | 65 | 610 | 212 |
| LayerNormalization | 17 | 374 | 146 |
| Reshape (views) | 130 | 0 | 50 |
| Add | 57 | 81 | 44 |
| Transpose | 40 | 40 | 38 |
| Mul | 32 | 64 | 29 |
| Softmax | 8 | 24 | 15 |
| Pow / Tanh (gelu_new) | 8 / 8 | 32 / 48 | 13 / 12 |

At 32 tokens the attention matrices dominate: Softmax 976 ms with 1864
launches (233 per node) and the mask `Where` 906 ms with 200 launches, against
252 ms for all 65 MatMuls. The launch count, not the arithmetic, sets the
cost: the 32-token window has 1.8× the MACs of the 16-token one but 2.5× the
launches.

## 4. Parity against ONNX Runtime

Both sides run the same rewritten graph with the same feeds; only the engine
differs.

| window | max abs logit diff per step | mean | greedy argmax agrees |
|---|---|---|---|
| 16 | 18.2 - 23.4 | 3.1 | 0 / 8 |
| 32 | 17.5 - 23.3 | 3.1 | 0 / 8 |

The divergence is not amplification of fp32 rounding (that is 2.9e-6 on this
graph): the engine is executing something else. Truncating the graph at the
first block's attention residual, so the tensor is a real graph output and
nothing later can recycle its buffer, gives:

```
h.0/attn -> residual (33 nodes)  max 6.96e-02   reference max 0.592
h.0 block output     (51 nodes)  max 5.25e-02
```

12 % relative error in the first block, before any later layer can contribute.

## 5. Two engine defects

### 5.1 A multi-row GEMM is wrong for N > 16384

The LM head is the only 50,257-wide GEMM. Isolated `MatMul(A, B)` with a
random `B` on the board:

| M | N | result |
|---|---|---|
| 1 | 50,257 | exact, 1.1e-6 |
| 16 | 16,384 | exact, 1.4e-6 |
| 16 | 16,385 | wrong, 8.0e-1 |
| 4 | 16,384 | exact, 1.1e-6 |
| 4 | 16,385 | wrong, 7.1e-1 |
| 2 | 16,385 | wrong, 6.9e-1 |
| 8 | 32,768 | wrong, 1.0 |
| 16 | 50,257 | wrong, 0.9 - 3.5 |

Exact for `N <= 16384` at any `M`, and for `M = 1` at any width; wrong as soon
as a multi-row output spans more than 16384 columns. 16384 x 4 bytes is
64 KiB, which is where a 16-bit byte offset or a single N-tile would end. The
wrapper therefore slices the final hidden state to its last row and emits
`[1, 1, 50257]`; the full-position graph is available with `--logits all` but
is not correct.

### 5.2 A live attention operand is replaced by the mask constant

Truncating the same graph at `/transformer/h.0/attn/attention/MatMul_1_output_0`
(softmax x V, `[1,16,16,16] x [1,16,16,4]`) returns values at -3.4e38 where the
reference is in [-0.44, 0.53] — the value of the mask constant, not of any
attention product:

```
MatMul_1 output  got [-3.4028235e38, 0.00]   ref [-0.44, 0.53]
```

The same operator pair and shape is exact in isolation (1.9e-6), a fresh copy
of either operand (`Add(x, 0)`) does not change the result, and under the full
continuation the same node is only ~7e-2 wrong — the outcome depends on the
pool layout. That is buffer reuse or a missing invalidate between a live
tensor and a later allocation, not GEMM arithmetic. Every other node measured
this way (embeddings, LayerNorm, QK^T, softmax) matches to 1e-5; the first
departure is at the value matmul.

Reproduce: take `wrap/ts1m_c16.onnx`, keep only the nodes that are ancestors
of the target tensor, set it as the single graph output, and run
`vc4infer.Engine` and ONNX Runtime on that file. The same procedure with the
target at `/transformer/Add_5_output_0`, `.../q_proj/MatMul_output_0_2d` or
`.../attention/Cast_2_output_0` (softmax) matches to 1e-5; the value matmul is
where it breaks. The run records are `docs/results/tiny-llm-*.json`.

### 5.3 The Softmax refusal that the mask value triggers

With the export's own masked value (`-3.4028235e38`), a 32-token run stops at

```
UnsupportedNode: node '/transformer/h.1/attn/attention/Softmax' (Softmax):
Softmax: a non-finite input is refused
```

The value is finite on entry, so the engine's select/sum path is producing
`-inf` from it before Softmax. Replacing it with `-1e9` removes the refusal at
both sizes and changes no softmax output (`exp(-1e9)` is 0 in fp32, as is
`exp(-3.4e38)`). This is a workaround for a genuine refusal, kept because it
is numerically identical.

## 6. Generated text

Greedy, eight tokens, the same 38-token prompt, window at the last N tokens,
window slid by one per step, positions absolute.

Window 16, engine: ` to to to to to to to to`
Window 32, engine: ` M M M M M M M M`

ONNX Runtime on the same graph and feeds: `\n her the the the play play play`
and ` Oneimiimiimiimiimiimiabel`. Both continuations are poor — a 1 M model
with greedy decoding on a truncated context is not a story generator — but the
ORT ones vary and follow the prompt more closely; the engine's collapse to one
token is the signature of §5.

## 7. What a next step would be

1. **Fix §5.2 before anything else.** A 7e-2 error in block 0 means the
   timings above are the cost of a graph that executes, not of the model
   computing. The reproduction is small (33 nodes, one output) and does not
   need the LLM: a synthetic `MatMul(softmax(...), V)` after an add of a large
   constant in a pool of this size should show it.
2. **Fix §5.1.** With the wide multi-row GEMM correct, the graph can emit all
   positions, which is a prerequisite for a KV cache; and the LM head becomes
   16x more useful work per launch instead of one row.
3. **Then attack the launch count.** Softmax at 32 tokens is 233 launches per
   node and 976 ms. A causal softmax (mask folded into the max/exp, one
   kernel) removes both the 906 ms `Where` and most of that. LayerNorm is 22
   launches for a 16x64 tile; its gamma/beta should be tiled once into a panel,
   not per launch.
4. **Only then add a cache.** A fixed-size cache would turn 940 ms per token
   into roughly one token's work plus the mask, but at 1300 launches per
   forward the win is in launches, not FLOPs — the recompute is 2.3x the MACs
   at 32 tokens but 2.5x the launches at 16. Measure after 1-3.
5. **Do not read §3 as a throughput claim for LLMs.** It is the cost of this
   graph on this engine: 1.06 tokens/s at 16 tokens, with 25 % QPU duty. The
   engine has no fused attention, no batched softmax and no cache path; each is
   worth more than a faster GEMM at this size.

## Reproduce

```sh
cd ~/work/vc4-llm
source ~/work/torch-vc4/.venv/bin/activate
export VC4INFER_CACHE=$HOME/.cache/vc4infer/kcache

# fetch (revision pinned)
mkdir -p ~/scratch/refs/tinystories/onnx && cd ~/scratch/refs/tinystories
curl -sL -o onnx/model.onnx \
  https://huggingface.co/onnx-community/TinyStories-1M-ONNX/resolve/\
b873615e9ceb8f2cdb6c14a90641c99052fa4a7f/onnx/model.onnx
for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  curl -sL -o $f https://huggingface.co/onnx-community/TinyStories-1M-ONNX/\
resolve/b873615e9ceb8f2cdb6c14a90641c99052fa4a7f/$f
done

python infer/tools/onnx/make_tinystories.py --src onnx/model.onnx \
  --ctx 16 --out wrap/ts1m_c16.onnx --report wrap/ts1m_c16.json
python infer/tools/onnx/tinystories_decode.py --model wrap/ts1m_c16.onnx \
  --ctx 16 --tokenizer . --tokens 8 --json dec16.json
```

The decode tool needs `onnxruntime` for the reference (`--no-ort` skips it);
the wrapper needs only `onnx` and `numpy`. Run one client at a time through
`vc4d` and keep the ARM at 1.0 GHz: this graph is 1.4-3.5 s per token at 96 MiB
and the thermal guard at 75 C was never approached (peak 53 C).
