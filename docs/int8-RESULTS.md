# int8 dot products on the VideoCore IV QPU — what exists and what it measures

Board: Raspberry Pi 3B (BCM2837), VideoCore IV V3D, Debian 13 arm64, kernel
6.18.50+rpi-v8.  Stock clocks (clock service untouched; `vcgencmd` reports V3D
300 MHz throughout, but see the clock caveat in §4).  Every GPU run: `flock -w
2400 /tmp/vc4-gpu.lock`, `timeout`, `read_v3d_status` afterwards (`STATE: LIVE`,
SRQCS clean, `get_throttled=0x0`, no wedge).  Work tree `/home/raspi/work/int8`
(asm3's proven launch harness, encoder, packer and VPM store reused; asm3 was not
modified).

**Headline: int8 buys nothing on the MAC side of this GPU.**  The OpenCL
`cl_arm_integer_dot_product_*` extensions are advertised but implemented as scalar
C in the standard library; and the ISA has no packed byte multiply — `v8muld` is a
*normalised* multiply `round(a*b/255)` (measured), so the only exact integer
multiply is `mul24`, one product per lane per instruction, exactly like `fmul`.
Measured int8 and fp32 MAC rates agree to within 0.5 %.

---

## 1. What int8 support actually exists

### 1.1 Advertised (VC4CL)

`VC4CL/src/vc4cl_config.h:145-147` hard-codes four extension strings into
`CL_DEVICE_EXTENSIONS`:

    cl_arm_integer_dot_product_int8            (3, 0)
    cl_arm_integer_dot_product_accumulate_int8 (3, 0)
    cl_arm_integer_dot_product_accumulate_int16(3, 0)
    ..._accumulate_saturate_int8               (3, 0)

### 1.2 Implemented (VC4CLStdLib `include/_extensions.h:114-170`)

Pure C, no intrinsics, no packed path:

```c
int  arm_dot(char4 a, char4 b)   { int4  t = vc4cl_mul24(a,b,SIGNED);   return t.s0+t.s1+t.s2+t.s3; }
uint arm_dot(uchar4 a, uchar4 b) { uint4 t = vc4cl_mul24(a,b,UNSIGNED); return t.s0+t.s1+t.s2+t.s3; }
```

`char4` is not a packed 4-byte vector in VC4C's model — it is four 32-bit values,
so `arm_dot` is 4 multiplies + 3 adds per 4-element dot product (0.5 MAC per
multiply).  A `uchar4` probe compiles to `or.zextByte0To32` index setup, one
`mul24` per component, `add`s, and a rotate+replicate reduction: ~2 instructions
per MAC, no vectorisation.

### 1.3 VC4C does compile char/uchar kernels — and generates poor code

| kernel | k-loop body | instructions per lane-MAC |
|---|---|---|
| `uchar` GEMM (`acc += (int)A[..]*(int)B[..]`) | **49 instructions** | 49 (1 MAC/lane/iteration) |
| compiled fp32 `v16` (accepted stack) | 453 words / 16 MACs | **8.33** |

The `uchar` loop re-widens every byte (`shr` + `and 255`) because the loaded value
lands in `r4`, whose unpack unit is colour-only; it also cannot fold the index
arithmetic.  `char` (signed) multiplies are emulated with **three** `mul24` + adds
(visible in the emitted asm) because `mul24` is unsigned.  No `v8adds`/`v8subs`/
`v8muld` appears in generated arithmetic code; the only `v8*` op VC4C emits is
`v8min`, used as a vector rotate.

### 1.4 What the ISA can actually do

| op | meaning | exact for integers? |
|---|---|---|
| `mul24` | `(a & 0xffffff) * (b & 0xffffff) mod 2^32`, **unsigned** | yes — 1 product per lane per instruction |
| `v8muld` | per byte `round(a*b/255)` (normalised [0,1] blend multiply) | **no** (measured) |
| `v8adds`/`v8subs` | per byte saturating add/sub | only while no saturation |
| `v8min`/`v8max` | per byte min/max | yes, but no arithmetic value |
| `unpack` 8/10/12/14 | zero-extended byte 0..3 of a **file-A** operand | free (same cycle count) |
| `unpack` 2/4 | sign-extended halfword | free |

A product of two int8 needs 16 bits; `v8muld` yields 8.  **No exact 4-byte dot
product exists on this ISA**: every packing scheme either truncates (`v8muld`) or
needs 16-bit products that no instruction produces.

## 2. ISA probes (`probe_isa.py`, `probe2..7.py`, `probe_tmu*.py`)

| probe | result |
|---|---|
| `mul24(-1,-1)` | `0xFFFE0001` -> operands are **unsigned 24-bit** |
| `mul24(0x1000000, 1)` | `0` -> operand truncated to 24 bits |
| `mul24(-3,5)` | `0x00FFFFFD` (16777213*5 mod 2^32): signed int8 needs a +128 bias |
| `mul24(255,255)` | `0x0000FE01` = 65025, exact |
| `v8muld(0x80808080,0x80808080)` | `0x40404040` (128*128/255 = 64) ✓ |
| `v8muld(0x000000FF,0x000000FF)` | `0x000000FF` (255*255/255) ✓ |
| `v8muld(0x03020100,0x40302010)` | `0x01000000` (3*64/255 = 0.75 -> 1; a truncating multiply would give 0xC0) |
| `v8adds(0xF0..,0x20..)` | `0xFFFFFFFF` saturating |
| `v8subs(0x10..,0x20..)` | `0x00000000` saturating |
| unpack 8a/8b/8c/8d of `0x11223344` | `44 / 33 / 22 / 11`, zero-extended |
| unpack on a file-B operand | not applied (only file A and r4 can unpack) |
| write->read latency | accumulators r0-r3: 1 instruction; A/B register file: **2** |

Two hardware facts worth recording beyond this task:

* **The TMU request queue is effectively 2 deep.**  With 3 or 4 requests issued on
  one channel before popping, only the first and the last materialise.  Measured
  with all-at-once, four separate address registers, interleaved, and 4/20/40-word
  issue spacing — always `11111111 44444444 00 00` for four distinct blocks.
  asm3's "3 in flight" is 2 outstanding + 1 issued per iteration.
* `mul24 acc2, r5, r4` is exact when either operand is 1, and exact for constant
  data up to 255, but a full GEMM built this way is a few LSB low on some
  broadcasts of large A values — see §6.

## 3. ALU-only microbenchmarks, 12 QPUs (`bench_alu.py`, `bench_alu.json`)

Deep unrolled loop, register operands only, no memory: 32 packed words per
iteration + counter + branch = 34 slots => 94.1 % of the issue peak.  All four
variants measured **4.00 cycles per word** (perfect issue, one instruction per
4 clocks).

| variant | ops per word | this run (~150 MHz) | first campaign (300.12 MHz) |
|---|---|---|---|
| fp32 `fmul`+`fadd` | 2 | 6.778 GMAC/s = **13.56 GOPS/s** | 13.56 GMAC/s = **27.12 GOPS/s** |
| int32 `mul24`+`add` (the int8 MAC) | 2 | 6.778 GMAC/s = 13.56 GOPS/s | 13.56 GMAC/s = 27.12 GOPS/s |
| `mul24`+`add`, `unpack 8a` on every word | 2 | identical -> **unpack is free** | identical |
| `v8muld`+`v8adds` | 8 byte-ops | 27.11 GMAC/s = **54.22 G byte-ops/s** | 54.23 GMAC/s = 108.5 G byte-ops/s |

Structural peak = 96 ops per MHz (12 QPUs x 16 lanes x 2 ops / 4 clocks) =
28.8 GOPS at 300 MHz, 14.4 GOPS at 150 MHz.  The exact int8 MAC ceiling therefore
equals the fp32 MAC ceiling; the byte SIMD gives 4x the op rate but cannot compute
an exact dot product (§1.4).

## 4. int8 dot-product GEMM (verified, exact)

Since `mul24` and `fmul` have identical throughput, an exact integer dot product
can be computed with either.  The verified path uses fp32 with integer-valued
inputs: every product (<= 65025) and every partial sum (< 256*255*255 = 1.7e7) is
exactly representable in fp32 (< 2^24), so each `fmul`/`fadd` is exact — no
rounding, no denormals.  Verified against an **int64 CPU reference** (not a
tolerance), plus the packed panel verified bit-exactly.

Kernel: asm3's hand-written 16x16-tile QPU GEMM (12 QPUs, 16 lanes = 16 columns,
16 accumulator rows, k-major packed A panel, VPM->DMA stores).

```sh
python3 run8f.py 128 256 --reps 5      # int8 (exact), pack + GEMM, integer check
python3 run_fp32.py 5                  # fp32 reference, same kernel/session
```

| N | path | pack ms | GEMM ms | total ms | GMAC/s | GOPS/s | GFLOP/s | cyc/k | mismatches |
|---|---|---|---|---|---|---|---|---|---|
| 128 | int8 (exact) | 0.063 | 0.823 | 0.886 | **2.55** | **5.10** | - | 150.0 | **0/16384  (5/5)** |
| 128 | fp32, same session | 0.063 | 0.823 | 0.886 | 2.55 | - | **5.10** | 150.0 | 0/16384  (5/5) |
| 256 | int8 (exact) | 0.356 | 6.170 | 6.526 | **2.72** | **5.44** | - | 161.3 | **0/65536  (5/5)** |
| 256 | fp32, same session | 0.355 | 6.176 | 6.531 | 2.72 | - | **5.43** | 161.3 | 0/65536  (5/5) |

`GOPS/s` counts 2 ops per MAC (multiply + add); `GFLOP/s` counts 2 flops per MAC.
So the two columns are the same physical rate: **int8 MAC/s = fp32 MAC/s** —
2.55 vs 2.55 GMAC/s at N=128, 2.72 vs 2.72 at N=256 (0.1 % spread).  Panel
(`pack`) verification: 0/16384 and 0/65536, bit-exact, in every run; 10/10 runs
clean; `STATE: LIVE` after every launch.

### Clock caveat — read this with the absolute numbers

`vcgencmd measure_clock v3d` reports 300 MHz throughout, but the *effective* clock
fluctuates: the ALU-only loop measured `clk 300.12` in one campaign and exactly
`clk 150.0` later in the same session, with **identical cycle counts** (4.00
cyc/word both times).  The GEMMs above ran at the ~150 MHz state.  Independent
check: asm3's own `gemm2_128_d3_v1.bin` through its own `qrun_gemm` in this run
gives the identical 1 382 640 cycles / 150.0 cyc/k but 0.747 ms instead of the
recorded 0.464 ms.  At the 300 MHz state the same kernels are ~2x faster (int8
~10.2/10.9 GOPS/s, fp32 ~10.2/10.9 GFLOP/s), which matches asm3's recorded
7.97/9.48 GFLOP/s within session variation.  Clocks were not modified (out of
scope); the int8:fp32 ratio is unaffected.

## 5. Why the packed int8 GEMM (4 k per word) was not finished

Built and measured (`gemm_i8.py`): A panel with 4 k-bytes per word, one request per
4 k, byte index selected by `unpack 8a..8d`, B one 32-bit word per element.
150 words per 4-k iteration (1024 MACs -> 0.147 words/MAC vs fp32's 0.152).  It
timed 11.07 GOPS/s at N=128 (same session) but the output was **not exact**, and
three independent blockers explain why the design cannot be completed cheaply:

1. **TMU queue depth 2** (§2): the four B requests per iteration cannot be in
   flight together.  Interleaving pops with issues forces the row body to be
   k-major, which re-broadcasts A per k and removes the gain entirely
   (0.21 words/MAC, worse than fp32).
2. **One unpackable operand per instruction** (file A only): the packed side and
   the lane-uniform side cannot both be bytes; one of them must be 4 B/value.
3. `mul24` is unsigned, so signed int8 needs the usual +128 bias with row/column
   sums (exact, but extra work).

## 6. Residual defect found in the `mul24` GEMM (documented, not claimed)

`gemm8b.py` = the same kernel with `fmul->mul24` and `fadd->add` (byte-identical
instruction stream apart from those opcodes; same 150.0 cyc/k).  It is wrong by a
few LSB on some elements: `A=254 -> 250`, `252 -> 249`, `247 -> 241` — only for
values above ~0xF0, only when the 16 lanes hold *different* values, and never with
constant data (A = const up to 255 exact; A=1 with random B exact; random A with
B=identity wrong in 10/256 places, each strictly below the true value).  Since the
instruction stream is identical to the working fp32 kernel except for the two
opcodes, this is a hardware/encoding interaction on the
rotate+replicate -> `mul24` operand path, not a scheduling or hazard bug.  It is
not worked around and no number is claimed from it.

## 7. Honest expectation for YOLO-style convolutions

Measured: exact int8 MAC rate = fp32 MAC rate (2.55-2.72 GMAC/s hand-written,
6.78-13.56 GMAC/s ALU-only ceiling depending on the clock state, 3.4-4.7 GMAC/s
compiled); the byte-SIMD units run 4x faster but compute `round(a*b/255)`;
compiled int8 is a 6x per-MAC regression; and the hand-written GEMM issues one
instruction every 4 clocks with zero stalls (100 % of its own issue floor), so
shrinking memory traffic cannot speed it up.

| path | YOLO conv (im2col GEMM) expectation |
|---|---|
| VC4CL compiled int8 (`char`/`uchar` kernel) | **~0.17x** of compiled fp32 (49 vs 8.33 instr/MAC) — a loss |
| hand-written int8, exact (this report) | **1.00x** of hand-written fp32 (2.55 vs 2.55 GMAC/s; 0.1 %) |
| packed int8 (4x smaller weights/activations) | 1.00x compute, 1.6x less traffic, but blocked by the 2-deep TMU queue and the one-unpackable-operand rule |
| best case if a conv were purely load-bound | up to ~2x, bounded by the TMU queue depth |

On a CPU or a GPU with VNNI/DP4A the same 2-ops-per-MAC accounting buys 4-8x.
On VideoCore IV the multiplier array has no byte-lane mode, so int8 quantisation
is a *memory-footprint* optimisation only — and for the hand-written path even
that is blocked by the ISA's operand/unpack constraints.

## 8. Artifacts (`/home/raspi/work/int8`)

| file | what |
|---|---|
| `qrun_i8.c` | asm3 harness + u8/word/float fill modes, CPU panel, exact C save |
| `qenc.py`, `gen.py` | QPU encoder (adds the `unpack` field), program builder |
| `pr.py` | serialised runner + health gate + verified VPM store |
| `probe_isa.py`, `probe2..7.py`, `probe_tmu*.py` | ISA / register-latency / TMU probes |
| `bench_alu.py`, `bench_alu.json` | ALU-only ceilings |
| `gemm2.py`, `pack.py`, `vpmwords.py` | asm3 GEMM + packer (unchanged, reused) |
| `gemm8b.py`, `run8b.py` | `mul24` GEMM (defective, §6 — not claimed) |
| `gemm_i8.py`, `run8.py` | packed 4-k int8 GEMM (blocked, §5 — not claimed) |
| `run8f.py`, `run_fp32.py` | verified int8 and fp32 runs of §4 |
| `check_i8.c`, `check_panel.c`, `check_gemm.c` | int64 / bit-exact verifiers |
| `gemmf8_runs.jsonl`, `gemmfp32_runs.jsonl` | raw per-rep rows |

```sh
cd /home/raspi/work/int8
gcc -O2 -o qrun_i8 qrun_i8.c && gcc -O2 -o check_i8 check_i8.c
gcc -O2 -o check_panel check_panel.c && gcc -O2 -o check_gemm check_gemm.c -lm
python3 probe_isa.py                      # ISA semantics table
python3 bench_alu.py fp32 i32 u8u v8      # ALU-only ceilings
python3 run8f.py 128 256 --reps 5         # exact int8 GEMM, verified
python3 run_fp32.py 5                     # fp32 reference, same session
```
