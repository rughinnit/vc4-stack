# XCHECK — independent cross-validation of the hand-written QPU encoder

Date: 2026-11-19, plus the canonicalisation follow-up (§7) and the
fourth-encoder check (§9) the same day.
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
* A **fourth, independent encoder** was added the same day: Idein/py-videocore's
  Python `encoding.py` + `assembler.py` (§9). 64 cases / 66 words: 55 are
  **bit-identical** to qenc's bytes, 11 differ only in fields of *idle* ALU
  slots (the destination/condition of a `nop`, `thrend` or `ldtmu1` add slot,
  and the unused MUL slot of an LDI), and vc4asm takes qenc's side on every one
  of those (63/63 of the cases it can express). **All 27,112 words of every
  shipped blob re-encode byte-for-byte** through py-videocore's field layout.
  **No qenc change was needed, so no new GPU run was made** (§9.7).

## 1. What was cloned and built

| repo | commit | role |
|---|---|---|
| `anholt/vc4-gpu-tools` | `5de069e` | `include/vc4_qpu_defines.h` (authoritative enums/bit map), `tools/vc4_qpu_disasm.c` |
| `hermanhermitage/videocoreiv-qpu` | `26f5336` | `qpu-sniff/qpudis.c` disassembler, `qpu-tutorial/qpuasm.js` assembler |
| `maazl/vc4asm` | `80ab48d` | `vc4asm` / `vc4dis` (third assembler + disassembler, strongest constraint checker) |
| `hermanhermitage/videocoreiv` | `51be3e2` | `videocoreiv.arch`, `tinyasm` — **scalar VPU only**, see §3.3 |
| `itszor/vc4-toolchain` | `2f57f09` (binutils submodule `708acc8`) | GNU binutils port — **scalar VPU only**, see §3.3 |
| `Idein/py-videocore` | `43fbb6b` (v1.0.0) | **independent Python encoder** (`videocore/encoding.py`) + its own assembler DSL (`videocore/assembler.py`), see §9 |

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
| `infer/kcache/col2im_16_144_3_3_16_8.bin` | 1196 | 0 | 0 | 0 | 0 |
| `infer/kcache/col2im_16_144_3_3_32_18.bin` | 1196 | 0 | 0 | 0 | 0 |
| `infer/kcache/pool_16_16_16_c16_fwd.bin` | 337 | 0 | 0 | 0 | 0 |
| `infer/kcache/pool_16_16_16_c16_bwd.bin` | 560 | 0 | 0 | 0 | 0 |
| `infer/kcache/ew_1_q12.bin` (elementwise) | 1320 | 0 | 0 | 0 | 0 |

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
* Idein/py-videocore's rotator encoder, added 2026-11-19 as a fourth source
  (§9.3): `rotate == r5 -> raddr_b = 48`, otherwise `48 + rotate%16`, with the
  same codes checked against the `-16..-1` constants, and 1.0 at 32.

Consequences:
* rotate by 0 cannot be encoded as a small immediate; it must be a plain
  accumulator-to-accumulator move. `gemm6.rot_inst(0)` already does that
  (`mul_a=acc, mul_b=acc`), so the *code* is right.
* The comment in `gemm6.py` — "small immediate 48 is 1.0f, not 'rotate by
  0'" — **was wrong on both counts**: 1.0f is code 32, and 48 is
  rotate-by-r5. Fixed on 2026-11-19 in `asm10/gemm6.py`, `gemm4.py`,
  `gemm5.py`, `patch_rot0.py` and `infer/tools/kgen.py`; the emitted words
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
blob. To inspect a raw `.bin` from `infer/kcache/` or `qenc`, use
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

Two comparison tools drive *foreign* assemblers on the same operation and
compare bytes:

```sh
# third assembler: maazl/vc4asm (§3)
python3 tools/xcheck/compare_vc4asm.py \
    --qenc-dir kernels/wider-tile/asm10 \
    --vc4asm /home/raspi/work/xcheck/vc4asm/build/vc4asm

# fourth encoder: Idein/py-videocore (§9), also re-encodes every shipped word
python3 tools/xcheck/compare_pyvc.py \
    --qenc-dir kernels/wider-tile/asm10 \
    --pyvc-dir /home/raspi/work/xcheck/py-videocore \
    --vc4asm   /home/raspi/work/xcheck/vc4asm/build/vc4asm
```

Both exit non-zero on an unexpected difference, so they can gate a commit.
`compare_pyvc.py` needs numpy (py-videocore's `assembler.py` imports it).

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
* A fourth, independent encoder (py-videocore) agrees with qenc on every
  instruction it can express, and re-encodes every word of every shipped blob
  byte-for-byte (§9). The two encoders disagree only about which condition and
  destination to put in an *idle* ALU slot, and there qenc has vc4asm's
  support. No encoder defect was found; no byte changed; the hardware-verified
  blobs are still the ones described in §7 (§9.7).
## 9. Fourth encoder: Idein/py-videocore (added 2026-11-19)

Sections 2 and 3 check our words with *disassemblers* and with one other
*assembler* (vc4asm). This section adds a fourth oracle of the strongest kind
available without a second ISA document: a second, independently written
**encoder**. Idein/py-videocore's `videocore/encoding.py` states the 64-bit
word as ctypes bit-field structures (`AluInsn`, `LoadInsn`, `BranchInsn`,
`SemaInsn`) and `videocore/assembler.py` is its own Python-DSL assembler that
fills them. Two encoders fed the same operation and producing the same bytes
is a much tighter test than a decoder agreeing on a mnemonic, because every
field - including the ones nobody prints - has to land in the same place.

### 9.1 Clone and Python 3.13 compatibility

| repo | commit | version | role |
|---|---|---|---|
| `Idein/py-videocore` (MIT) | `43fbb6b` | 1.0.0 | `videocore/encoding.py` = independent encoder; `videocore/assembler.py` = its DSL front end |

Cloned to `/home/raspi/work/xcheck/py-videocore` (shallow). The board runs
Python 3.13.5.

* `import videocore.encoding` **works unmodified** - there is no py2-ism in the
  encoding module.
* `import videocore.assembler` failed with
  `ModuleNotFoundError: No module named 'numpy'`. numpy is a declared
  requirement (`requirements.txt`) and is used in exactly one place, an
  `isinstance(val, (list, tuple, numpy.ndarray))` test when encoding
  per-element immediates. Fixed by installing the distro package
  (`sudo apt-get install -y python3-numpy`, 2.2.4).
* **No file in the py-videocore clone was modified** - `git status` there is
  clean. The other two requirements (`rpi-vcsm ~= 3.0.0`, `ioctl-opt ~= 1.2`)
  are imported only by `videocore/driver.py`, which is not used here.
* Everything in this section is CPU-only: no driver, no QPU kernel, no clock
  or kernel change.

So the encoder runs as-is on 3.13 given its declared numpy dependency. Only
`encoding.py`'s bit-field geometry and `assembler.py`'s field filling are
exercised; the driver/mailbox/v3d parts are irrelevant here.

### 9.2 Instruction-level comparison

`tools/xcheck/compare_pyvc.py` builds the same operation three ways wherever
possible:

* **ours**: `qenc` (`kernels/wider-tile/asm10/qenc.py`), the encoder under test;
* **py-videocore**: a `@qpu` function assembled by
  `videocore.assembler.assemble()`, i.e. its encoder;
* **vc4asm**: a maazl/vc4asm spelling of the same instruction, as tie breaker
  (the third opinion already used in §3).

Bytes below are `low32 high32`, the order `qenc.listing()` and the rest of this
document use (the `.bin` files store `low32` first, little-endian). The
`gemm6` column is the index in `asm10/gemm6_128.bin` where the *identical*
word occurs, found by scanning the blob - so the case is tied to bytes the
kernel really executes. `-` means no shipped kernel emits that exact shape
(a synthetic case, still compared three ways).

Instruction selection follows the shapes the kernels actually use: dual-issue
add+mul words, the rotator in both the standalone and the merged `r5rep` form,
both rep forms, pack/unpack/PM, TMU address writes, VPM/DMA setup words, and a
counted loop with a relative branch.

| # | instruction | gemm6 word | ours (lo hi) | py-videocore (lo hi) | verdict |
|---:|---|---:|---|---|---|
| 1 | `nop` (null destination) | 9 | `009e7000 100009e7` | `009e7000 10020827` | diff (9.4) |
| 2 | `or ra25, unif, unif` | 1 | `15827d80 10020667` | `15827d80 10020667` | match |
| 3 | `or ra27, qpu_num, qpu_num` | 8 | `159e6fc0 100206e7` | `159e6fc0 100206e7` | match |
| 4 | `mov rb18, r4` (ws=1) | - | `159e7900 100214a7` | `159e7900 100214a7` | match |
| 5 | `add.setf ra25, ra24, rb22` | 20 | `0c616dc0 10022667` | `0c616dc0 10022667` | match |
| 6 | `shl ra27, ra27, 2` (small imm 2) | 10 | `116c2dc0 d00206e7` | `116c2dc0 d00206e7` | match |
| 7 | `fadd r1, r0, 1.0f` (small imm 32) | - | `019e01c0 d0020867` | `019e01c0 d0020867` | match |
| 8 | `fmul r2, r5, ra16` (MUL slot only) | 303 | `2042702e 100049e2` | `2042702e 100049e2` | match |
| 9 | `fadd ra0, ra0, r2` (ADD slot only) | - | `01027c80 10020027` | `01027c80 10020027` | match |
| 10 | `fadd ra0,ra0,r2; fmul r2,r5,rb16` (dual, mul->add dependent) | 119 | `21010caf 10024022` | `21010caf 10024022` | match |
| 11 | `fadd rb0,rb0,r1; fmul r3,r4,ra18` (dual, independent) | - | `21480e66 10025023` | `21480e66 10025023` | match |
| 12 | `ldi r1, 4` | - | `00000004 e0020867` | `00000004 e0024867` | diff (9.4) |
| 13 | `ldi rb20, 0x200` | 11 | `00000200 e0021527` | `00000200 e0025527` | diff (9.4) |
| 14 | `rot 1` (MUL slot only) | - | `809f1000 d00049e1` | `809f1000 d00049e1` | match |
| 15 | `rot 2` | - | `809f2000 d00049e1` | `809f2000 d00049e1` | match |
| 16 | `rot 3` | - | `809f3000 d00049e1` | `809f3000 d00049e1` | match |
| 17 | `rot 4` | - | `809f4000 d00049e1` | `809f4000 d00049e1` | match |
| 18 | `rot 5` | - | `809f5000 d00049e1` | `809f5000 d00049e1` | match |
| 19 | `rot 6` | - | `809f6000 d00049e1` | `809f6000 d00049e1` | match |
| 20 | `rot 7` | - | `809f7000 d00049e1` | `809f7000 d00049e1` | match |
| 21 | `rot 8` | - | `809f8000 d00049e1` | `809f8000 d00049e1` | match |
| 22 | `rot 9` | - | `809f9000 d00049e1` | `809f9000 d00049e1` | match |
| 23 | `rot 10` | - | `809fa000 d00049e1` | `809fa000 d00049e1` | match |
| 24 | `rot 11` | - | `809fb000 d00049e1` | `809fb000 d00049e1` | match |
| 25 | `rot 12` | - | `809fc000 d00049e1` | `809fc000 d00049e1` | match |
| 26 | `rot 13` | - | `809fd000 d00049e1` | `809fd000 d00049e1` | match |
| 27 | `rot 14` | - | `809fe000 d00049e1` | `809fe000 d00049e1` | match |
| 28 | `rot 15` | - | `809ff000 d00049e1` | `809ff000 d00049e1` | match |
| 29 | `rot by r5` (dynamic) | - | `809f0000 d00049e1` | `809f0000 d00049e1` | match |
| 30 | `move r1, r0` (rot 0, no rotator code) | 115 | `809e7000 100049e1` | `809e7000 100049e1` | match |
| 31 | `mov r5rep, r1` (ws=1) | 207 | `159e7240 10021967` | `159e7240 10021967` | match |
| 32 | `mov r5quad, r1` (ws=0) | - | `159e7240 10020967` | `159e7240 10020967` | match |
| 33 | `mov r5rep, r1; v8min r1, r0` rot 0 | - | `959e7240 10025961` | `959e7240 10025961` | match |
| 34 | `mov r5rep, r1; v8min r1, r0` rot 1 | 201 | `959f1240 d0025961` | `959f1240 d0025961` | match |
| 35 | `mov r5rep, r1; v8min r1, r0` rot 2 | 195 | `959f2240 d0025961` | `959f2240 d0025961` | match |
| 36 | `mov r5rep, r1; v8min r1, r0` rot 3 | 189 | `959f3240 d0025961` | `959f3240 d0025961` | match |
| 37 | `mov r5rep, r1; v8min r1, r0` rot 4 | 183 | `959f4240 d0025961` | `959f4240 d0025961` | match |
| 38 | `mov r5rep, r1; v8min r1, r0` rot 5 | 177 | `959f5240 d0025961` | `959f5240 d0025961` | match |
| 39 | `mov r5rep, r1; v8min r1, r0` rot 6 | 171 | `959f6240 d0025961` | `959f6240 d0025961` | match |
| 40 | `mov r5rep, r1; v8min r1, r0` rot 7 | 165 | `959f7240 d0025961` | `959f7240 d0025961` | match |
| 41 | `mov r5rep, r1; v8min r1, r0` rot 8 | 159 | `959f8240 d0025961` | `959f8240 d0025961` | match |
| 42 | `mov r5rep, r1; v8min r1, r0` rot 9 | 153 | `959f9240 d0025961` | `959f9240 d0025961` | match |
| 43 | `mov r5rep, r1; v8min r1, r0` rot 10 | 147 | `959fa240 d0025961` | `959fa240 d0025961` | match |
| 44 | `mov r5rep, r1; v8min r1, r0` rot 11 | 141 | `959fb240 d0025961` | `959fb240 d0025961` | match |
| 45 | `mov r5rep, r1; v8min r1, r0` rot 12 | 135 | `959fc240 d0025961` | `959fc240 d0025961` | match |
| 46 | `mov r5rep, r1; v8min r1, r0` rot 13 | 129 | `959fd240 d0025961` | `959fd240 d0025961` | match |
| 47 | `mov r5rep, r1; v8min r1, r0` rot 14 | 123 | `959fe240 d0025961` | `959fe240 d0025961` | match |
| 48 | `mov r5rep, r1; v8min r1, r0` rot 15 | 117 | `959ff240 d0025961` | `959ff240 d0025961` | match |
| 49 | `mov r5rep, r1; v8min r1, r0` rot r5 | - | `959f0240 d0025961` | `959f0240 d0025961` | match |
| 50 | `iadd.pack16a ra1, ra0, rb0` | - | `0c000dc0 10122067` | `0c000dc0 10122067` | match |
| 51 | `mov r0, ra0.unpack16a` | - | `15027d80 12020827` | `15027d80 12020827` | match |
| 52 | `mov r0, r4.unpack8a` (unpack + PM) | - | `159e7900 19020827` | `159e7900 19020827` | match |
| 53 | `fmul.pack8a ra1, r0, r1` (ws=1) | - | `209e7001 104059c1` | `209e7001 104059c1` | match |
| 54 | `mov t0s, r3` | 86 | `159e76c0 10020e27` | `159e76c0 10020e27` | match |
| 55 | `add t1s, r3, rb26` | 89 | `0c9da7c0 10020f27` | `0c9da7c0 10020f27` | match |
| 56 | `or t0s, r3, r3; fmul r2, r5, ra16` + `ldtmu0` | - | `354276ee a0024e22` | `354276ee a0024e22` | match |
| 57 | `ldi vw_setup, 0xc0010000` (VDW stride) | - | `c0010000 e0021c67` | `c0010000 e0025c67` | diff (9.4) |
| 58 | `ldi vr_setup, 0x90000200` (VDR stride 512) | - | `90000200 e0020c67` | `90000200 e0024c67` | diff (9.4) |
| 59 | `ldi vw_setup, 0x1a34` (generic write, row 52 stride 1) | - | `00001a34 e0021c67` | `00001a34 e0025c67` | diff (9.4) |
| 60 | `ldi vr_setup, 0x83001010` (VDR basic 16x16, vpitch 1, Y=1) | - | `83001010 e0020c67` | `83001010 e0024c67` | diff (9.4) |
| 61 | `ldi vw_setup, 0x82104000` (VDW basic depth 16 units 4) | - | `82104000 e0021c67` | `82104000 e0025c67` | diff (9.4) |
| 62 | loop word 0: `ldi r1, 4` | - | `00000004 e0020867` | `00000004 e0024867` | diff (9.4) |
| 63 | loop word 1: `add.setf r1, r1, -1` (imm 31) | - | `0c9df3c0 d0022867` | `0c9df3c0 d0022867` | match |
| 64 | loop word 2: `brr.anynz` to word 1 (imm -40) | - | `ffffffd8 f03809e7` | `ffffffd8 f03809e7` | match |
| 65 | `thrend` | 649 | `009e7000 300009e7` | `159e7000 30020827` | diff (9.4) |
| 66 | `nop.load_tmu1` (signal-only word) | 103 | `009e7000 b00009e7` | `009e7000 b0020827` | diff (9.4) |

Totals: 64 cases / 66 words. **55 words bit-exact**, 11 words differ only in
the fields covered in 9.4, 0 unexplained. All 63 cases that have a vc4asm
spelling are also assembled by vc4asm, and vc4asm is **bit-identical to qenc in
all 63**; for the 11 differing words it is qenc that vc4asm reproduces.

Three spelling notes that cost time and are worth recording:

* vc4asm's `mov r1, r0` is an ADD-slot `or` (`159e7000 10020867`), not a
  MUL-slot move. To compare "rotate by 0" you must write `v8min r1, r0, r0`
  (`809e7000 100049e1`), which is what py-videocore's `rotate(...,0)` and
  `qenc.rot_inst(0)` both emit; otherwise it looks like a 6-field diff that
  is really two different instructions.
* vc4asm rejects `mov r1, r0 >> r5` (error A32, "Cannot rotate ALU target
  right by r5") and spells the dynamic rotate `mov r1, r0 << r5`. Both map to
  the same B small immediate 48 (`809f0000`), so it is a syntax quirk, not a
  disagreement. All three encoders put 48 there.
* py-videocore's `nop()` takes its destination from
  `_ADD_DEFAULT_ARGS[nop] = [r0, r0, r0]` (encoding.py:301), so its default
  nop writes `r0`; that is the whole of diff family (1) below.

### 9.3 py-videocore independently confirms the rotate map of claim (a)

Its rotator encoder is an explicit table, not an inference:

```python
# assembler.py:308-314, 318-333
# 'r5 rotate' represents -1.
# 'n-upward rotate' represents n-16.
# +-----------+--------+
# | small imm | rotate |
# +-----------+--------+
# |    -16    |   r5   |
# |     -n    |   -n   |
# +-----------+--------+
...
if rotate == REGISTERS['r5']:
    raddr_b = 48
else:
    raddr_b = 48 + rotate%16
```

So: **48 = rotate by r5**, **49..63 = rotate 1..15**, and those same codes
alias the constants **-16..-1** (its `_SMALL_IMM` lookup is what it checks the
rotate code against). Its small-immediate table also puts **1.0 at 32**
(`2.0**0` is entry 32). That is the fourth independent source for claim (a) in
§4 - qpuasm's table, `vc4dis`, the maazl addendum, and now py-videocore's
encoder - and it agrees with the corrected text.

### 9.4 The 11 differing words: idle-slot defaults, not an ISA disagreement

Both diff families are fields of an ALU slot that is **not doing anything**;
the opcode, operands, rotate code, pack/unpack/PM and signal are identical.

**(1) The idle ADD slot of a plain `nop`, of `thrend`, and of `ldtmu1`**
(3 words). py-videocore's DSL:

```
ours   009e7000 100009e7   nop   (waddr_add=39/null, cond_add=never)
pyvc   009e7000 10020827   nop   (waddr_add=32/r0,   cond_add=always)
```

`waddr_add` 39 -> 32 and `cond_add` 0 -> 1. For `thrend` py-videocore's `exit()`
uses `mov(r0, r0, sig='thread end')`, which additionally makes the slot an
`or` (`op_add` 0 -> 21); for its `ldtmu1` the same default nop is used. Cause
is `_ADD_DEFAULT_ARGS[nop] = [r0, r0, r0]` plus `AddEmitter`'s
`cond='always'` default. Note the write address stays inside the ALU (the
accumulator `r0`), never a register file, and the signal is what makes these
words do their job.

**(2) The idle MUL slot of an LDI word** (8 words: rows 12, 13, 57..62).
py-videocore's `LoadEmitter._emit` does

```python
# assembler.py:423
cond_add = cond_mul = enc._COND[kwargs.get('cond', 'always')]
```

so `cond_mul` is 1 where qenc and vc4asm leave 0. `waddr_mul` is 39 (null) in
both, so nothing is written by that slot.

**Which encoding is canonical?** Ours, on the evidence available:

* vc4asm assembles our bytes **exactly** for all 11 words (and for the other
  55), so the third assembler takes qenc's side on every one. Example:
  `ldi rb20, 0x200` -> `e0021527` (ours, vc4asm) vs `e0025527` (py-videocore).
* The authoritative bit map (`vc4_qpu_defines.h:238-248`) puts exactly these
  fields where qenc and py-videocore both put them: `COND_ADD` 49..51,
  `COND_MUL` 46..48, `WADDR_ADD` 38..43, `WADDR_MUL` 32..37 (and
  `test_qenc_fields.py` passes 256/256 in this tree). What is in dispute is
  only *which value* to put in a disabled slot, not where the bits live.
* py-videocore's own encoder produces **our** bytes when it is given our field
  values (next subsection), so this is a front-end default, not a defect in
  its encoding geometry either.
* For an idle slot, "condition never" is the conservative value: the slot is
  then inert whatever its destination is, instead of relying on that
  destination being null (qenc writes null as well). qenc and vc4asm already
  agree on it, so **no change was made to qenc**.

One more, smaller difference that the table does not show because the case was
built to match vc4asm: a *standalone* rotate word built with
`qenc._ops(**qenc.rot_inst(15))` gets `cond_add=always` (from `_ops`' default),
i.e. `809ff000 d00249e1`, while vc4asm and py-videocore emit
`809ff000 d00049e1` (`cond_add=never`) for a MUL-only word. The add slot is
`op=nop` with `waddr=null` in both, so the bit only gates a null write. It is
deliberately **not** "fixed": `rot_inst` must not carry an add-slot condition,
because when the rotate is merged with the `r5rep` add slot that condition
would override the broadcast - exactly the §3.2 failure. No shipped kernel
emits a standalone rotate (all 60/75/75 are the merged `r5rep` form, rows
33..49, where all three encoders agree byte-for-byte), so nothing is affected.

### 9.5 Word level: every shipped word re-encodes bit-for-bit

`compare_pyvc.py` also takes every 64-bit word of every blob in the tree apart
with qenc's bit map, fills py-videocore's `AluInsn` / `LoadInsn` /
`BranchInsn` with those field values, and compares `insn.to_bytes()` with the
original. This is the field-geometry test: it does not use the py-videocore DSL
at all, so its defaults cannot mask anything. No shipped word is a semaphore
instruction, so the three word classes cover everything in the tree.

| blobs | words |
|---|---:|
| `asm10/gemm6_64.bin`, `gemm6_128.bin`, `gemm6_256.bin` | 2,161 |
| `asm10/pack10_{64,128,256}_k16_single_vp1_us.bin` | 201 |
| `infer/kcache/col2im_*.bin` (3 variants) | 3,588 |
| `infer/kcache/pool_*_fwd/bwd.bin` | 897 |
| `infer/kcache/ew_*.bin` (26 variants) | 20,265 |
| **total, 37 blobs** | **27,112 words, 0 mismatches** |

(More than the 5,329 words of §2 because that count covered one build of each
kernel; this covers every parameter variant in `infer/kcache` as well.)

Every field position is exercised: `sig`, `unpack` 3 bits @57, `pm` 1 bit @56,
`pack` 4 bits @52, both conditions, `sf`, `ws`, both write addresses, both
opcodes, both read addresses, all four operand muxes (ALU), the LDI mode bits
and immediate, and the branch's link/raddr/reg/rel/cond/offset fields. (For
pack/unpack/PM the shipped value is always 0; the non-zero values are the rows
50..53 cases below.) So the corrected geometry of §3.2 matches py-videocore's
independently written `_fields_` list as well as the header.

The non-zero pack/unpack/PM values that no shipped kernel uses are covered by
rows 50..53 instead, and those are bit-identical three ways (qenc, py-videocore,
vc4asm), including the `unpack`+`PM` combination on `mov r0, r4.unpack8a`
(`19020827 159e7900`) and the MUL-ALU pack `fmul.pack8a`
(`104059c1 209e7001`).

### 9.6 Where the VPM words agree

The five VPM/DMA control words (rows 57..61) are bit-identical apart from the
inert `cond_mul`, and the encodings come from two independent implementations:
our `asm10/vpmwords.py` (Table 32/34/36 field layout) and py-videocore's
`setup_vpm_write` / `setup_dma_load_stride` / `setup_dma_store_stride` /
`setup_dma_load` / `setup_dma_store`. Both produce the same
`0xc0010000`, `0x90000200`, `0x1a34`, `0x83001010` and `0x82104000` constants.
One trap when reproducing: py-videocore's `setup_dma_load` takes `Y` and `X`
separately (`Y<<4|X`) while `vpmwords.vdr_basic` takes the row address with the
column always 0, so `Y=1` is `vdr_basic(1)`, not `vdr_basic(16)`.

### 9.7 No encoder change, so no new GPU run - and the blobs are provably unchanged

py-videocore exposed **no defect in qenc**: 55/66 case words and all 27,112
shipped words are bit-identical, and the 11 word-level differences are
py-videocore DSL defaults in idle slots that vc4asm resolves in qenc's favour.
Nothing in `qenc.py`, `vpmwords.py`, `gemm6.py` or any kernel was modified by
this section, so there is no new behaviour to verify on hardware and no QPU
run was made (also: GPU runs need explicit approval and must be serialised).

The previously reported hardware result still applies verbatim, because the
blobs are provably the ones that were run in §7. Their sha256:

```
c1a65dea7d3fb7ccde7afd794b6c1760d44618939ec3759a79ab966dd3f46784  gemm6_64.bin
b7f7adf3706e1f2c6d62e536250bc9d0b146f20d745a8b9e4447c904ec2342f0  gemm6_128.bin
eef3bae2ca862d064ecd9876b4ce37ce0758dc072d8200328dcd304d0d24897d  gemm6_256.bin
```

and clearing the 3 rotate-MUX bits back to the old REGB form (set `mul_b=7` on
the 60/75/75 rotate words with a code in 49..63) reproduces the pre-change
hashes recorded in §3.4/§7 exactly:

```
5cec565d76a1c88f15931c1e5cfa8911bc27d2c7889d02f0fbc7179854960b9d  (128, was 5cec565d…)
9ca67b054b5c0d11033b7d9f7ba68442ed8729d104fbbe5fc0911917908a3c2c  (256, was 9ca67b05…)
2dddbb26eed6fcb15aa3f51b6bc339287a0c700a43c3aece64c165ceeb2085b3  (64,  was 2dddbb26…)
```

So no bit other than those 3 moved since the verified hardware runs, and the
recorded result (`Cmm=0 Pmm=0`, 2.829-2.837 ms, 11.83-11.86 GFLOP/s,
398.5 cyc/k) is still the result for these exact files. Anyone who changes an
encoder line should re-run §7's procedure instead of relying on this paragraph.

### 9.8 Reproduce

```sh
ssh raspberrypi
git clone --depth 1 https://github.com/Idein/py-videocore \
    /home/raspi/work/xcheck/py-videocore
sudo apt-get install -y python3-numpy        # assembler.py imports it
cd /home/raspi/work/vc4-stack
python3 tools/xcheck/compare_pyvc.py \
    --qenc-dir kernels/wider-tile/asm10 \
    --pyvc-dir /home/raspi/work/xcheck/py-videocore \
    --vc4asm   /home/raspi/work/xcheck/vc4asm/build/vc4asm
echo $?    # 0 = no unexpected difference
```

The script prints the whole table above, the per-field diff for every
non-matching word, the gemm6 word index of every case that occurs in the
kernel, and the per-blob word-level re-encode counts.
