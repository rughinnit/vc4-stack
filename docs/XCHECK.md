# XCHECK — independent cross-validation of the hand-written QPU encoder

Date: 2026-11-19, plus the canonicalisation follow-up the same day (§7).
Host: Raspberry Pi 3B, Debian 13 arm64, kernel `6.18.50+rpt-rpi-v8`, stock
clocks (300/400 MHz; nothing was changed). The initial cross-check ran no QPU
kernel and modified no source; its only GPU interaction was the read-only
`DRM_IOCTL_VC4_GET_HANG_STATE` probe in §5, which returned "no hang recorded".
The follow-up fixed the two encoder defects and re-ran the verified GEMM
(N=256, 12 QPUs, one `flock`).

Goal: take the instruction words our kernels actually emit, decode them with
disassemblers/assemblers written by other people, and check the claims in
`NOTES10.md` / `RESULTS10.md` that depend on instruction encoding.

## 0. TL;DR

* Two independent disassemblers were built (Anholt's `vc4_qpu_disasm`,
  hermanhermitage's `qpudis`) plus a third (`vc4dis` from maazl/vc4asm).
* **All 5,329 words** from `gemm6_128`, `pack10`, two `col2im` kernels, two
  `pool` kernels and an elementwise kernel (5,632 word-instances counting the
  separately-run 303-word loop-body slice) decode with **zero unknown or
  illegal opcodes**. The only non-blank flags are the 60 rotate words, which
  each tool renders differently because the rotate encoding is genuinely
  ambiguous in the published ISA (§2.2).
* A field-by-field check of all 653 `gemm6_128` words against the bit map in
  `vc4_qpu_defines.h` gives **0 mismatches** (617 ALU, 32 LDI, 4 branches).
* Reverse direction: 33 representative instructions assembled with
  maazl/vc4asm agree **33/33 bit-exactly**, including **all 16 rotate forms**
  the kernel emits (0 = plain move, 1..15 = small immediates 49..63) in the
  exact `r5rep; rot` word gemm6 uses (§3). qpuasm independently produces the
  same bytes as vc4asm for those words.
* The two encoder defects are **fixed and re-verified** (follow-up, §7): the
  rotate MUX is now the vc4asm form (byte-exact) and `pack`/`unpack`/PM use the
  header widths (4/3/1 bits at 52/57/56). N=256 on hardware: `Cmm=0 Pmm=0`,
  2.829-2.837 ms, 11.83-11.86 GFLOP/s, 398.5 cyc/k - unchanged.
* Claim (a) is **half right**: 49..63 are rotate 1..15, but 48 is *rotate by
  r5*, not a constant and not "not a rotate". The old `gemm6.py` comment
  saying "48 is 1.0f" was wrong (1.0f is 32) and has been corrected. Claim (b)
  is **confirmed**.
* `vc4_dump_hang_state` builds and works on this kernel (§5).

## 1. What was cloned and built

| repo | commit | role |
|---|---|---|
| `anholt/vc4-gpu-tools` | `5de069e` | `include/vc4_qpu_defines.h` (authoritative enums/bit map), `tools/vc4_qpu_disasm.c` |
| `hermanhermitage/videocoreiv-qpu` | `26f5336` | `qpu-sniff/qpudis.c` disassembler, `qpu-tutorial/qpuasm.js` assembler |
| `maazl/vc4asm` | `80ab48d` | `vc4asm` / `vc4dis` (third assembler + disassembler, strongest constraint checker) |
| `hermanhermitage/videocoreiv` | `51be3e2` | `videocoreiv.arch`, `tinyasm` — **scalar VPU only**, see §3.3 |
| `itszor/vc4-toolchain` | `2f57f09` (binutils submodule `708acc8`) | GNU binutils port — **scalar VPU only**, see §3.3 |

All under `/home/raspi/work/xcheck`. Build commands:

```sh
# Anholt disassembler (single C file + headers)
gcc -O2 -I vc4-gpu-tools/include -c vc4-gpu-tools/tools/vc4_qpu_disasm.c -o vc4_qpu_disasm.o
gcc -O2 -I vc4-gpu-tools/include tools/xcheck/anholt_main.c vc4_qpu_disasm.o -o vc4_qpu_disasm

# hermanhermitage disassembler + numeric field dump (FIELDS=1)
gcc -O2 -std=c99 -c videocoreiv-qpu/qpu-sniff/qpudis.c -o qpudis.o
gcc -O2 -std=c99 -I videocoreiv-qpu/qpu-sniff tools/xcheck/videocoreiv_main.c qpudis.o -o qpu_disasm_vc4

# maazl vc4asm / vc4dis
cd vc4asm && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
# Note: aarch64 linking of libvc4asm.so fails (-fPIC); the vc4asm and vc4dis
#       executables build fine. Use the executables directly.
```

The two small driver programs are in `tools/xcheck/` (`anholt_main.c`,
`videocoreiv_main.c`). They read the same raw format our harness writes:
little-endian `uint64_t` words, no header (`qenc.encode_to_bin`).

## 2. Forward: our words through independent disassemblers

Blobs checked (raw words, byte counts / 8):

| blob | words | unknown `???` | `addopN`/`mulopN` | `err?` | rotate words |
|---|---:|---:|---:|---:|---:|
| `asm10/gemm6_128.bin` (full kernel) | 653 | 0 | 0 | 0 | 60 |
| `gemm6` loop body, words 116..418 | 303 | 0 | 0 | 0 | 45 |
| `asm10/pack10_128_k16_single_vp1_us.bin` | 67 | 0 | 0 | 0 | 0 |
| `engine/kcache/col2im_16_144_3_3_16_8.bin` | 1196 | 0 | 0 | 0 | 0 |
| `engine/kcache/col2im_16_144_3_3_32_18.bin` | 1196 | 0 | 0 | 0 | 0 |
| `engine/kcache/pool_16_16_16_c16_fwd.bin` | 337 | 0 | 0 | 0 | 0 |
| `engine/kcache/pool_16_16_16_c16_bwd.bin` | 560 | 0 | 0 | 0 | 0 |
| `engine/kcache/ew_1_q12.bin` (elementwise) | 1320 | 0 | 0 | 0 | 0 |

"Unknown" means Anholt printed `???` for an add/mul opcode name, or
videocoreiv printed one of its `addopN`/`mulopN` placeholders, or `vc4dis -V`
emitted a warning. All three counts are zero everywhere. `vc4dis -V` also
produced no warnings on any blob.

Notes on the body count: the reference states "302 words". `gemm6.py` prints
`loop body 303 words`; that is words 116..418 inclusive and includes the
one-word `prologue` nop at 116. Words 117..418 are 302. Both were checked.

### 2.1 Field-level check (gemm6, all 653 words)

I decoded every word with the bit positions from
`vc4-gpu-tools/include/vc4_qpu_defines.h` (`QPU_*_SHIFT`/`QPU_*_MASK`) and
compared every field against the intent recorded in `gemm6.py`'s word list:

```
sig, unpack, pm, pack, cond_add, cond_mul, sf, ws, waddr_add, waddr_mul,
op_mul, op_add, raddr_a, raddr_b, add_a, add_b, mul_a, mul_b
+ branch: cond, relative, register, 5-bit reg, signed offset
+ ldi:    mode, ws, waddr_add, waddr_mul, 32-bit value
```

Result: **617 ALU + 32 LDI + 4 branches, 0 mismatches.** In particular the
`ws` bit and the A/B bank selection for both ALU slots (`QPU_WS` semantics:
add writes bank B when ws=1, mul writes bank A when ws=1) match Anholt's
disassembler and qenc's intent on every word.

### 2.2 The rotate words are ambiguous in every published source

All 45 body rotates are `v8min rN, r0, <rotate code>` with `sig=small_imm`,
`raddr_b=49..63`. The three tools say:

| tool | output for word 117 (`v8min r1, r0`, rotate 15, canonical blob) |
|---|---|
| Anholt `vc4_qpu_disasm` | `sig_small_imm mov r5, r1 ; mov r1, r0` |
| videocoreiv `qpudis` | `mov r5rep, r1; mov r1, r0 >> 15` |
| maazl `vc4dis` | `mov r5rep, r1; mov r1, r0<<1` |

* In the first pass the word still had B mux REGB; Anholt's disassembler only
  models small immediates 0..47 (`vc4_qpu_disasm.c:305-317`), so it printed
  `<bad imm 63>`. That was **not an unknown opcode**, and 1.0f is code 32, not
  48. With the canonical ACC0 mux it prints a plain `mov`.
* `err?` in the first pass was qpudis failing to name a small-immediate B
  operand (`qpudis.c:133-150`); it decoded the rotation correctly from
  `mula=r0` and the B address field. It no longer prints `err?`.
* `-1` in the first pass was the *constant alias* of rotate code 63 (maazl
  addendum, "Small immediate values": 48..63 alias -16..-1). `vc4dis` now
  prints the rotate directly; the alias is why the encoding is ambiguous.

All three agree on the rotate amount (15). This is the ambiguity to remember:
the rotate amount lives in the B-address field (48+r5 / 49..63), not in a
separate field, and the same code doubles as a constant in other contexts.
Scanners that treat `48..63` as "bad" will produce false positives on this
kernel; `tools/qpu_disasm.py` classifies them as rotates instead.

## 3. Reverse: assemble fixed instructions and compare bytes

### 3.1 vc4asm (maazl) — 33/33 bit-exact

Fixed list covers every shape the kernels emit: A/B register reads, special
reads (`unif`, `qpu_num`), special writes (`vpm`, `vw_setup`, `vw_addr`,
`t0s`), small immediates, LDI, `ldtmu1`, `thrend`, one dual add+mul word,
`r5rep`/`r5quad`, the rotator, and a relative branch.

```
OK    or ra1, unif, unif
OK    or ra0, qpu_num, qpu_num
OK    or vpm, rb0, rb0
OK    add vw_setup, r5, ra0
OK    add vw_addr, ra4, rb1
OK    shl r0, ra0, 2
OK    add r0, unif, r0
OK    add rb0, r4, 1
OK    ldi vw_setup, 0xc0010000
OK    nop
OK    nop.load_tmu1
OK    nop.end
OK    fadd ra0,ra0,r2; fmul r2,r5,rb16
OK    rep ws=1 (r5rep)
OK    rep ws=0 (r5quad)
OK    mov t0s, r3
OK    branch +2 words
OK    r5rep + rot0 .. r5rep + rot15     (all 16 forms, exact bytes)
bit-exact 33/33 cases
```

The branch test (qenc `branch(+16, BR_ALL_C_SET)` vs `brr.allc -, r:tgt` with
the target 2 words past the delay slots) is exactly `00000010 f08809e7` in
both.

Reproduce: `python3 tools/xcheck/compare_vc4asm.py` (needs `--qenc-dir`, and
`--vc4asm` if not at the path above).

### 3.2 The rotate diff: what it is and who is right

The only difference was bits 0..2 of the low word, i.e. the MUL ALU's
B-operand MUX:

* qenc (before): `mul_b = REGB (7)`, `raddr_b = 48+n`, `sig = small_imm`
* vc4asm:        `mul_b = ACC0 (0)`, `raddr_b = 48+n`, `sig = small_imm`

qpuasm (`v8min r1, r0 >> 15, r0 >> 15`) emits **the same bytes as vc4asm**
(`809ff000 d00049e1`), so the two independent assemblers agreed with each
other and disagreed with qenc in those 3 bits.

**Resolved 2026-11-19 - canonical now.** `qenc.rot_inst` emits the vc4asm form
(`mul_b=acc(src)`, mux ACC0); every rotate word in the asm10 blobs is now
byte-identical to vc4asm. The 3 bits are genuinely don't-care on this silicon:
an A/B of the *same* `gemm6_256` blob with only those bits flipped (REGB 7 ->
ACC0 0, 75 words) gives `Cmm=0 Pmm=0`, 398.5 cyc/k and 2.45 ms for **both**
forms; flipping them back reproduces the pre-change blob byte-for-byte (§3.4).

One attempt at this canonicalisation first appeared to fail with
`Cmm=65533/65536`. That was **not** the MUX: the new rotate descriptor carried
a stray `cond_add=COND_NEVER`, and the kernel's mul/add dict merge let it
override the `r5rep` add slot, so the broadcast never executed. The descriptor
now sets no add-slot condition at all, and gemm6's own word checker catches
that class of merge error. Recording this because the first (wrong) conclusion
was "REGB is required by hardware" - it is not.

The second, latent difference from the headers is **fixed**: `qenc.alu` wrote
`unpack` as 4 bits at 56 and `pack` as 5 bits at 52..56 (overlapping each other
and PM). It now follows `vc4_qpu_defines.h`: `unpack` 3 bits at 57..59, `pm`
bit 56, `pack` 4 bits at 52..55. `tools/test_qenc_fields.py` checks every
pack/unpack/PM combination against those header positions. No emitted word
changed, because every kernel uses pack=unpack=0.

### 3.3 binutils / tinyasm are not QPU tools

`itszor/vc4-toolchain`'s binutils port targets the VideoCore **scalar VPU**,
not the QPU. Its opcode table is 32-bit (`fadd$alu32cond`, `add`, `ld`, `b`,
...), and `vc4-elf-as` happily assembles `fadd r0, r1, r2` into a 4-byte
`0f 02 c8 00` object (ELF32, "Broadcom VideoCore III processor"). That is a
different ISA from the 64-bit QPU word we encode; using it as a QPU oracle
would be wrong. `hermanhermitage/videocoreiv`'s `tinyasm` and
`videocoreiv.arch` are likewise the 32-bit scalar unit. They are therefore
**not** used for the byte comparison; vc4asm and qpuasm are the QPU
assemblers.

Build status, for the record: binutils configured for `--target=vc4-elf`;
`gas/as-new` built, but the top-level `make` fails (a) on the missing
`makeinfo` for the docs and (b) on the 2012-era K&R sources under GCC 14
(`ldgram.c: return type defaults to 'int'`). Not pursued because it is the
wrong ISA for this check.

### 3.4 Hardware A/B of the rotate MUX (same blob, 3 bits flipped)

With the encoder fix already in place, take `asm10/gemm6_256.bin`, clear the
low 3 bits of its 75 rotate words (REGB 7 -> ACC0 0), and keep every other bit
in the file identical. Run both under one `flock`:

```
python3 g6.py gemm6_256.bin          12 256 mux_acc0   # canonical (ACC0)
python3 g6.py /tmp/gemm6_256_regb.bin 12 256 mux_regb  # old form (REGB)
```

Both: `Cmm=0 Pmm=0`, 398.5 cyc/k, gemm 2.451 ms, 13.69 GFLOP/s gemm /
11.83-11.86 GFLOP/s total. The reverse flip reproduces the pre-change files
exactly (`sha256 5cec565d…`, `9ca67b05…`, `2dddbb26…` for 128/256/64), so the
only difference between the previously verified blobs and the canonical ones
is those 3 MUX bits.

Caveat for anyone repeating this: the probe kernels in `asm10/rot*.py` are
older and have a rot->rep distance of 1 word (the QPU needs >= 2 to read the
written accumulator), so they do **not** measure the rotator; use the GEMM A/B
above instead. One session of mixed probes also left a QPU stuck
(`TIMEOUT queue=0 err=0 done=0`); a reboot cleared it, and the numbers above
are from a fresh boot with `STATE: LIVE`, `get_throttled=0x0`.

## 4. Claims (a)–(e)

### (a) Rotate amounts are 49..63; 48 is not a rotate — **partly wrong**

Confirmed: small-immediate codes **49..63 are rotate right by 1..15**.

Correction: **48 is rotate by r5** (a dynamic rotate), not a constant and not
"not a rotate". Sources:

* qpuasm's immediate table (`qpuasm.js:100-110`): code 48 = `">> r5"`, then
  `">> 1"` .. `">> 15"` for 49..63.
* `vc4dis` (`Disassembler.cpp:94-115`): `rot = SImmd - 48`; `rot==0` prints
  `>>r5`; 49..63 print `>>1`..`>>15`.
* maazl addendum, "Small immediate values": the rotation codes 48..63 also
  alias the constants **-16..-1** when used as constants.

Consequences:
* rotate by 0 cannot be encoded as a small immediate; it must be a plain
  accumulator-to-accumulator move. `gemm6.rot_inst(0)` already does that
  (`mul_a=acc, mul_b=acc`), so the *code* is right.
* The comment in `gemm6.py` — "small immediate 48 is 1.0f, not 'rotate by
  0'" — **was wrong on both counts**: 1.0f is code 32, and 48 is
  rotate-by-r5. Fixed on 2026-11-19 in `asm10/gemm6.py`, `gemm4.py`,
  `gemm5.py`, `patch_rot0.py` and `engine/tools/kgen.py`; the emitted words
  were right (rotate 0 is already the plain move). The full map is now in the
  encoder docstring and here: **48 = rotate by r5 (dynamic), 49..63 = rotate
  1..15, rotate 0 = plain move, 1.0f = 32, 48..63 also alias the constants
  -16..-1.**
* The task's phrasing "v8min immediate 48 is not a rotate" should read
  "immediate 48 is not a *static* rotate; it is rotate-by-r5".

### (b) `rep` with ws=1 spans 16 lanes, ws=0 is per-quad — **confirmed**

Writing waddr 37 (`r5`) has two names depending on the write bank:

* qpuasm (`qpuasm.js:47,59`): `banka_w[37] = "r5quad"`,
  `bankb_w[37] = "r5rep"`.
* vc4asm (`src/Parser.tables.cpp:103-104`): `r5quad -> {37, R_WRA}`,
  `r5rep -> {37, R_WRB}`. `R_WRB` is selected by ws=1.
* maazl addendum, "Writing to r5": "r5 can be written from the elements
  0, 4, 8 and 12 for each slice or only from element 0, depending on whether
  you write to regfile A or regfile B. vc4asm supports the register names
  r5quad and r5rep."

`QPU_WS` selects the bank. Our `rep_inst(ws=1)` writes 37 through the B bank
(`r5rep`) and therefore broadcasts element 0 to all 16 lanes; `ws=0` would be
`r5quad` (elements 0,4,8,12, one per quad). Confirmed.

This also explains why the kernels need a *full* 16-lane rotate before the
broadcast: `r5rep` takes element 0, so each row must be rotated to element 0
across the whole vector, not just within its quad.

### (c) The ALU word packs one add + one mul — **confirmed**

The 64-bit ALU word has independent fields `OP_ADD` (5 bits, 24..28) and
`OP_MUL` (3 bits, 29..31), with separate destinations, conditions and operand
muxes (`vc4_qpu_defines.h:250-272`). Both disassemblers print two operations
for the dual words (`fadd ... ; fmul ...`), and vc4asm assembles
`fadd ra0, ra0, r2; fmul r2, r5, rb16` to the same one word as qenc
(`21010caf 10024022`). Confirmed.

### (d) TMU is 2 outstanding per channel — **not a disassembly fact**

The TMU queue depth is not encoded in an instruction word, so no disassembler
can confirm or refute it. It is an empirical result from
`docs/int8-RESULTS.md` ("The TMU request queue is effectively 2 deep",
measured with 4 distinct blocks and several issue spacings: only the first and
last materialise). The maazl addendum independently notes the TFREQ FIFO is
"supposedly" 8 slots but "unreliable to use more than 4", which is consistent
with our conservative 2-deep effective limit but is not a proof. Status:
**measurement stands, disassembly-neutral**.

### (e) The FPU truncates — **hardware, not disassembly**

`docs/NUMERICS.md` records the silicon measurement (`1.0 + 0.75 ulp -> 1.0`,
`-1.0 + -0.75 ulp -> -1.0`, products round down). Disassembly can only confirm
that the kernel uses `fadd`/`fmul` (it does); rounding mode is not in the
instruction word. Status: **out of scope for this cross-check; measurement
stands.**

## 5. Hang-state diagnostic for future wedges

`vc4_dump_hang_state.c` **builds and works** on this kernel:

```sh
gcc -O2 -I vc4-gpu-tools/include -I vc4-gpu-tools/include/drm \
    -I/usr/include/libdrm vc4-gpu-tools/tools/vc4_dump_hang_state.c \
    -o vc4_dump_hang_state -ldrm
gcc -O2 -I vc4-gpu-tools/include -I vc4-gpu-tools/include/drm \
    -I/usr/include/libdrm vc4-gpu-tools/tools/vc4_dump_parse.c \
    vc4-gpu-tools/tools/vc4_dump_parse_cl.c \
    vc4-gpu-tools/tools/vc4_qpu_disasm.c -o vc4_dump_parse
```

On 2026-11-19 with no wedged queue:

```
$ sudo ./vc4_dump_hang_state /tmp/hang.bin
No hang state recorded
```

So the `DRM_IOCTL_VC4_GET_HANG_STATE` path exists on 6.18.50+rpt-rpi-v8 and
returns `ENOENT` when there is nothing to report. It must run as root (the tool
checks `geteuid()`, and a non-root run prints "Root permission is required").

Workflow after a wedge (do this **before** any reset/reboot, which would
discard the state):

1. Do not reset the GPU and do not unload the module.
2. `sudo ./vc4_dump_hang_state hang-$(date +%s).bin` — writes the kernel's
   `drm_vc4_get_hang_state`, the BO state table and the captured BO contents.
3. `./vc4_dump_parse hang-*.bin > hang.txt` — parses the dump, prints the
   per-QPU state and disassembles the captured QPU code with Anholt's
   disassembler.
4. Keep `hang-*.bin`; the dump is the only record of the queue/PC state.

Caveat: `vc4_dump_parse` expects the driver's dump format, not a raw kernel
blob. To inspect a raw `.bin` from `engine/kcache/` or `qenc`, use
`tools/qpu_disasm.py` instead.

## 6. Regression tool

`tools/qpu_disasm.py` is self-contained (no qenc import, no third-party
packages). It decodes with the bit map from `vc4_qpu_defines.h` and flags:

* undefined add opcodes (9..11, 25..29) and any mul opcode outside 0..7;
* illegal MUL pack values when `PM` is set;
* `unpack` on a B-file MUL operand;
* non-zero LDI mode (anything other than `LOAD_IMM_32`);
* absolute branches;
* rotate words where the rotated source is not an accumulator.

It prints a listing and a summary, and exits 1 on any unknown/illegal word
(so it can be wired into a pre-commit or acceptance script).

```sh
tools/qpu_disasm.py asm10/gemm6_128.bin                # whole kernel
tools/qpu_disasm.py gemm6_128.bin --first 116 --count 303
tools/qpu_disasm.py col2im.bin --no-listing            # summary only
tools/qpu_disasm.py blob.bin --json                    # machine-readable
tools/qpu_disasm.py blob.bin --allow-unknown           # report but exit 0
```

Example (loop body):

```
 116: 009e7000 100009e7  nop
 117: 959ff240 d0025961  or r5rep, r1, r1; v8min r1, r0   <rotate-right 15>   [sig small_imm]
 118: 354276ee 10024e22  or t0s, r3, r3; fmul r2, r5
 119: 21010caf 10024022  fadd ra0, ra0, r2; fmul r2, r5
 120: 21440eae 10025022  fadd rb0, rb0, r2; fmul r2, r5
```

All current blobs: `0 unknown/illegal`.

Limitation: codes 48..63 in a MUL slot are classified as rotate codes by the
ISA definition. If a kernel intentionally uses them as the constants -16..-1 in
a MUL slot, the listing will call them rotates; the tool does not treat that as
an error, so it will not fail such a kernel. No current kernel does this.

## 7. Canonical now (follow-up, 2026-11-19)

Applied to `asm10/qenc.py` (and the identical `asm3/qenc.py` the engine uses)
and re-verified on the same board: fresh boot, stock 300 MHz,
`get_throttled=0x0`, one `flock -w 900 /tmp/vc4-gpu.lock` per sequence.

| change | effect on emitted words |
|---|---|
| rotate MUX -> vc4asm form (`mul_b=acc(src)`, ACC0), single `qenc.rot_inst` | the 60/75/75 rotate words change by 3 bits; k-loop size unchanged (303 words) |
| `pack` 4 bits @52, `PM` bit 56, `unpack` 3 bits @57, header constants | none (all kernels use pack=unpack=0) |
| 48 / 1.0f comment corrected | none |
| `gemm6.rot_inst` and `kgen.mul_rot` now call `qenc.rot_inst` | no drift between kernel trees |

Evidence:

* `tools/xcheck/compare_vc4asm.py`: **33/33 bit-exact** (19 original cases +
  all 16 rotate forms in the real `r5rep; rot` word).
* `tools/test_qenc_fields.py`: 256/256 pack/unpack/PM combinations and the
  field constants match `vc4_qpu_defines.h`.
* `N=128: loop body 303 words` (653 total; 754 for N=64/256) - unchanged.
* `tools/qpu_disasm.py`: 653/754 words, 60/75 rotates, **0
  unknown/illegal**. Anholt `vc4_qpu_disasm`: 0 `???`; videocoreiv `qpudis`:
  0 `addopN`/`mulopN`/`err?`; `vc4dis -V`: 0 warnings.
* hardware, N=256, 12 QPUs, canonical blob, 3 launches: `Cmm=0 Pmm=0`,
  pack 0.3855-0.3861 ms + gemm 2.4438-2.4510 ms = **2.829-2.837 ms**,
  **11.83-11.86 GFLOP/s total** (gemm 13.69-13.73), **398.5 cyc/k**,
  `SRQCS=0x000c0c00`, `STATE: LIVE`. Identical to the pre-change binary.
* the pre-change binaries are reproduced by flipping those 3 bits back
  (sha256 `5cec565d…` / `9ca67b05…` / `2dddbb26…`), so no other bit moved.

## 8. Bottom line

* Encoder and emitted words are **sound and canonical**: zero unknown opcodes,
  all fields match the authoritative map (pack/unpack/PM included), and
  33/33 reverse-assembled cases are bit-identical, rotate forms included.
* The rotate MUX was the only byte-level disagreement; canonicalisation is
  applied and the result is still element-exact at the same speed on hardware
  (§7).
* Claims (a) and (b) are now traceable to three independent sources each.
  (a) is corrected above; (b) is confirmed.
* (c) is confirmed; (d) and (e) are hardware measurements that are outside the
  scope of an instruction-word check.
