# TinyStories-1M on the VC4 QPUs

TinyStories-1M is a GPT-Neo family model (8 blocks, d_model 64) trained on
TinyStories, and the smallest causal language model with a public ONNX export
that the graph path can be pointed at. This document records
what it took to make the HF export static, what the engine ran, and the point
at which the result stops being the model.

**Result in one paragraph.** The graph executes end to end on the QPUs: 378
nodes, none refused, no CPU fallback, 378/378 on VC4, 0.42-1.06 tokens/s
depending on the window. The tokens are the model's: the last-position logits
match ONNX Runtime to 7.7e-4 at every step, all eight greedy steps agree, and
the continuation is coherent text. Two engine defects stood in the way, both
found with a reduced reproduction and fixed: the 8x64 store could not write a
C row pitch above 16384 floats (the VDW stride is a 16-bit field), and the
pool released a same-size *view* of a tensor while the tensor itself was still
live, so a later allocation overwrote the attention input. The parity table,
the two reproductions and the generated text are below.

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
| 16 | 378 | 378 | 0 | 1374 | 940 (min 866) | 1.06 | 29.8 MB | 26.6 MB |
| 32 | 378 | 378 | 0 | 3469 | 2399 (min 2351) | 0.42 | 33.4 MB | 26.6 MB |

The launch counts are from the parity re-run after both fixes; the wall times
are the earlier undisturbed measurement of the same graph. The re-run itself
was taken while another CPU-bound process shared the board (1647 ms and
4550 ms per token) and is not comparable.

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
differs. These are the numbers after both defects were fixed.

| window | max abs logit diff per step | mean | greedy argmax agrees |
|---|---|---|---|
| 16 | 4.3e-4 - 7.7e-4 | 5.5e-4 | 8 / 8 |
| 32 | 4.3e-4 - 6.7e-4 | 5.5e-4 | 8 / 8 |

What is left is accumulated fp32 rounding, not a different computation: the
graph rewrite itself differs from the export by 2.9e-6 under ONNX Runtime, the
two engines accumulate the 8 blocks in different orders, and the reduced
probes agree to that order:

| probe (graph truncated at the tensor, run through both) | max abs diff | nodes |
|---|---|---|
| `/transformer/h.0/attn/attention/MatMul_1_output_0` (softmax x V) | 5.96e-07 | 26 |
| same probe, before the liveness fix | 1.22 | 26 |

The other nodes measured this way before the fixes (the embeddings, both
LayerNorms, QK^T, the 2-D matmul inputs and softmax) were within 1e-5; the
value matmul was the first departure, and it is 5.96e-7 now.

## 5. Two engine defects, found and fixed

Both were reproduced on their own before either was touched - a synthetic GEMM
and a five-node graph - and the reproductions are now tests: `test_matmul`
carries the wide shapes and `test_pool_alias.py` the aliasing graph.

### 5.1 The store could not write a C row pitch above 16384 floats

Isolated `MatMul(A, B)` with a random `B` on the board, before the fix:

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

Exact for `N <= 16384` at any `M` and for `M = 1` at any width, wrong as soon
as a multi-row result spans more than 16384 columns. Reproduced at
M=4, N=16385, K=16 with the row pattern the cause predicts: row 0 exact,
rows 1..3 holding zeros or stale values, e.g.
`row 1 col 15791: got 0.0, reference 15.678`; at M=8 the wrong rows are
1, 2, 3, 5, 6 and 7, i.e. every row that is not first in its four-row group.

**Root cause.** The asm10 store moves a 8x64 tile to memory with eight
VPM->DMA stores of four rows each, and the distance between consecutive
16-word rows goes into the VDW stride word, which is 16 bits wide
(`vpmwords.vdw_stride`). The distance for a `[M][N]` result is `(N-16)*4`
bytes: 65,472 at N=16,384 (fits) and 65,648 at N=16,448 (does not). The
generator ORs the value into the store's control word (0xC0010000 | stride),
so the bit above 15 lands in the control field and the stride itself becomes
0x0070: the four-row store advanced 112 bytes and wrote rows 1..3 inside
row 0.
Row 0 was then written correctly by its own column chunks, which is why the
first row of every tile - and therefore every `M = 1` result - stayed exact.

**Fix.** Above 16384 columns the kernel writes a compact layout instead of a
row-major result: one dense 8x64 tile per (row block, column block), rows of
64 floats, so the store stride is `(64-16)*4 = 192` and the field cannot
overflow. `unpack_compact()` in `vc4_gemm.c` scatters the tiles into the
caller's matrix; the tile base is the column-block offset scaled by 8 (the B
read still uses it as the byte offset of its 64 columns) and the row-block
stride is `N*32`. The store instruction count is unchanged (791 words against
788 for the row-major kernel, two of them the scaled C base), and the copy is
one pass over `M*N` floats on the ARM, paid only by shapes this wide.

The asm3 store the conv path uses has the same 16-bit field
(`emit_store16`, and the col2im store in `build_col2im`, which is on the
conv-backward path); the widest channel pitch any shipped inference model
reaches is 640 floats, so nothing there needed to change. A conv whose output
row pitch exceeded 16384 floats would truncate the same way and is not fixed
here.

All six shapes are now exact against the float64 oracle:

| M | N | K | max abs (fp32) |
|---|---|---|---|
| 4 | 16,384 | 16 | 1.5e-05 |
| 4 | 16,385 | 16 | 9.8e-06 |
| 4 | 16,448 | 16 | 9.4e-06 |
| 1 | 50,257 | 16 | 9.0e-06 |
| 8 | 16,448 | 16 | 1.5e-05 |
| 16 | 16,448 | 16 | 1.2e-05 |

### 5.2 A live attention operand was replaced by the mask constant

Truncating the graph at `/transformer/h.0/attn/attention/MatMul_1_output_0`
(softmax x V, `[1,16,16,16] x [1,16,16,4]`) returned values from the mask path
where the reference is in [-0.44, 0.53] (with the export's own masked value:
`got [-3.4028235e38, 0.00], ref [-0.44, 0.53]`). The same operator pair and
shape is exact in isolation (1.9e-6), a fresh copy of either operand
(`Add(x, 0)`) did not change it, and the outcome depended on the pool layout -
buffer reuse over a live tensor, not GEMM arithmetic. Every other node
measured this way (embeddings, LayerNorm, QK^T, softmax) matched to 1e-5; the
first departure was the value matmul.

**Root cause: a view released its base tensor's block.** `Reshape`, `Flatten`,
`Squeeze` and `Cast` do not copy; their `Tensor` carries the producer's `Buf`.
The pool's free list is keyed by `(offset, aligned bytes)` while the runner's
liveness is per ONNX *name*, so a view whose last consumer had passed released
the block by offset and size - and a same-size view has the same key as the
allocation it is a view of. In this graph the first block's `ln_1/Add_1`
output is reshaped to 2-D for q/k/v; the k projection's 2-D input is that
view, its last consumer is the k projection MatMul, and at the end of that
node the block went back to the pool with `ln_1/Add_1` still live. The k
projection's own output took the block, the q and v projections read the
overwritten bytes, and the value matmul returned mask values where attention
weights times V should have been.

Reproduced with five nodes and a 16-byte tensor
(`infer/tests/test_pool_alias.py`): an input, a same-size `Reshape` of it with
two consumers, a constant of the same shape, and a use of the input after the
view's last consumer. Before the fix that test measured `max|diff| 3.18`
against the ONNX reference evaluator and one allocation overlapping the live
input; after it, 1.2e-7 and no overlap. The TinyStories probe named above went
from 1.22 to 5.96e-7.

**Fix.** The runner returns a block to the pool only when no other live tensor
holds the same buffer offset (`_release_buf`), so ownership of the bytes is
tracked per buffer instead of per name. It is applied wherever the runner
releases storage it no longer names: a node's dead inputs, its per-node
garbage, garbage left by a refused dispatch, and the previous run's outputs.
A view that dies before its base is simply not freed; the base frees the block
when its own last consumer passes.

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
window slid by one per step, positions absolute. ONNX Runtime on the same
graph and feeds picks the same eight tokens step by step (parity is 8/8), so
the two continuations below are the reference's as well.

Window 16: `\n\nOne day, the little girl`
Window 32: ` One day, Lily's mom told her`

Full text at 16 tokens:

> Once upon a time there was a little girl named Lily. She lived with her
> mother and father in a small house near the woods. Every day she went
> outside to play with her friends.
>
> One day, the little girl

Before the fixes the same prompt produced ` to to to to to to to to` and
` M M M M M M M M`. The collapse to one token was the two defects, not the
model: a 1 M-parameter GPT-Neo with greedy decoding is a weak story generator,
but it continues the prompt.

## 7. What a next step would be

1. **Attack the launch count.** Softmax at 32 tokens is 233 launches per node
   and 976 ms. A causal softmax (mask folded into the max/exp, one kernel)
   removes both the 906 ms `Where` and most of that. LayerNorm is 22 launches
   for a 16x64 tile; its gamma/beta should be tiled once into a panel, not per
   launch.
2. **Only then add a cache.** A fixed-size cache would turn 940 ms per token
   into roughly one token's work plus the mask, but at 1300 launches per
   forward the win is in launches, not FLOPs - the recompute is 2.3x the MACs
   at 32 tokens but 2.5x the launches at 16. Measure after 1.
3. **Take the wide-matmul copy out if it shows up.** The compact C layout
   costs one ARM pass over `M*N` floats per matmul above 16384 columns: 0.2 MB
   for the last-position LM head, 3.2 MB if the graph emits all 16 positions.
   That is about a millisecond of a 940 ms token, but a store that could write
   four rows with a 32-bit stride, or a consumer that reads the tile layout
   directly, would remove it.
4. **Do not read §3 as a throughput claim for LLMs.** It is the cost of this
   graph on this engine: 1.06 tokens/s at 16 tokens, with 25 % QPU duty. The
   engine has no fused attention, no batched softmax and no cache path; each is
   worth more than a faster GEMM at this size.

## Reproduce

```sh
cd ~/work/vc4-stack
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

# from the checkout: the fetch above leaves the reader in the download
# directory, and both scripts are relative to the repository root
mkdir -p ~/scratch/refs/tinystories/wrap
python infer/tools/onnx/make_tinystories.py \
  --src ~/scratch/refs/tinystories/onnx/model.onnx \
  --ctx 16 --out ~/scratch/refs/tinystories/wrap/ts1m_c16.onnx \
  --report ~/scratch/refs/tinystories/wrap/ts1m_c16.json
python infer/tools/onnx/tinystories_decode.py \
  --model ~/scratch/refs/tinystories/wrap/ts1m_c16.onnx --ctx 16 \
  --tokenizer ~/scratch/refs/tinystories --tokens 8 --json dec16.json
# add --no-ort when onnxruntime is not installed, as on this board's system
# python; the tokens and the timings are still measured, the parity is not

# the two reproductions, as tests
infer/build/test_matmul 5            # 4x16384, 4x16385, 8/16x16448 included
python3 infer/tests/test_pool_alias.py
```

The decode tool needs `onnxruntime` for the reference (`--no-ort` skips it);
the wrapper needs only `onnx` and `numpy`. Run one client at a time through
`vc4d` and keep the ARM at 1.0 GHz: this graph is 1.4-3.5 s per token at 96 MiB
and the thermal guard at 75 C was never approached (peak 53 C).
