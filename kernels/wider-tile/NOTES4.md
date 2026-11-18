# asm4 — widening the QPU GEMM tile to 4 columns per lane (8x64): work log

Work tree `/home/raspi/work/asm4` (copy of `asm3`; `asm3` not modified).
Board: Pi 3B, stock clocks throughout (V3D 300 MHz / core 400 MHz, verified before
and after every session with `vcgencmd measure_clock`, `get_throttled=0x0`).
Every GPU job ran under `flock -w 2400 /tmp/vc4-gpu.lock` + `timeout`, with
`read_v3d_status` before/after.  `STATE: LIVE` in every check.

**Status: PARTIAL.** The new 8-row packed A panel is written and verified
bit-exactly (N=128 and N=256).  The 8x64 GEMM kernel (4 columns per lane) is
built and runs, but is **not correct yet**: the store block writes stale VPM
contents and the tile loop executes only one tile.  No speed numbers can be
reported for it.  Root-cause work and everything needed to continue are below.

---

## 1. Why widen the tile

asm3's kernel keeps 16 lanes = 16 C columns, so the A value `A[r][k]` has to be
made lane-uniform for every one of the 16 rows of every k:

    rotate (v8min + small immediate)  +  replicate (write to the rep port)

That is 2 broadcast ops per row per k, and each broadcast feeds exactly **one**
MAC (`fmul`+`fadd`).  Measured: the loop spends 39 words per k = 32 words of
MAC (16 fmul + 16 fadd) + 2 words/row of broadcast + 5 head + 1 branch, i.e.
6.6 MACs per issued word, and it is at 86-92 % of its issue floor.

If a lane owns C adjacent columns (j = 0..3, one 16-column block each), one
broadcast feeds 4 MACs and the same 2 broadcast ops cover 16 MAC ops.

## 2. Tile and data layout chosen

    tile = 8 rows x 64 columns
    16 lanes = the 16 columns of one 16-wide block; 4 accumulators per lane
    => 32 accumulators: j=0,1 in the A file (addr 2r+j), j=2,3 in the B file

Register budget (the reason for the odd/even split): one instruction word can
read **one** A-file register and **one** B-file register (plus up to four
accumulator-mux operands).  So a MAC word with a per-lane B vector in one file
must keep its C accumulator in the other file.  All 32 accumulators cannot live
in one file (32 registers total, and ~12 more are needed for addresses and
constants), hence j=0,1 in the A file + B0,B1 in the B file and j=2,3 in the B
file + B2,B3 in the A file.  Consequence: an fmul whose B vector is in the A
file can only share its word with a fadd into a B-file accumulator, and vice
versa.  That is what fixes the row schedule below.

A panel (new, see §3):

    PANEL8[(rb*N + k)*8 + r] = A[rb*8 + r][k]

so one 64 B TMU request delivers **two** k columns at once
(lane r = A[8rb+r][k], lane 8+r = A[8rb+r][k+1]); the k loop therefore runs one
iteration per k **pair** with a single A request, and the two halves differ only
in compile-time rotation amounts (lane r vs lane 8+r), so one code path serves
both.

B stays row-major, 4 requests per k, one per 16-column block, lane L =
B[k][j*16+L].  Register sets P (k even) and Q (k odd) double-buffer the pops so
that fetches for the next k never clobber a live vector.

## 3. The packer: 8-row panel (DONE, verified)

`pack8.py`, driven by `qrun_gemm` with `QRUN_PACK_ROWS=8` (harness change, see
§5).  Per 8-row strip of A and per 8-k chunk it is still only two VPM engines:

1. **VDR** (VPM DMA load): RowLength=8 words, NumberRows=8, memory pitch N*4 →
   VPM rows 16q..16q+7 hold A[8rb+r][k0..k0+7].
2. **VDW vertical**: Depth=8, Units=8 → 8 consecutive **32 B** vectors, vector j
   = column j of the 8 loaded rows = PANEL8[rb][k0+j][0..7], i.e. 256 B written
   contiguously.  In-block dst step 256 B, row-block step 32*N (u3).

Verified with `check_panel8` (bit-exact against A, 8-row map):

| N | pack QPUs | panel mismatches | pack time |
|---|-----------|------------------|-----------|
| 128 | 4 | **0/16384** | 0.087 ms |
| 256 | 4 | **0/65536** | 0.46 ms |

Pack cost is higher than asm3's (0.063 / 0.356 ms) because an 8-k strip is a
quarter of the old 16-row/16-k tile, i.e. 4x as many DMA round trips for the
same bytes.  Known fix, not implemented: load 16 k per VDR (RowLength=16) and
issue two VDWs with the setup word's Column field set to 0 and 8 → 3 DMA trips
per 16 k instead of 4.

### 3.1 VDW field semantics (measured, contradicts the spec table as usually read)

`probe_vdw8c.py` (ramp-filled VPM, three vertical stores) established, on this
silicon:

* the **vector width is the `Depth` field** (words per output vector) and the
  **number of vectors is the `Units` field** — the opposite of what the
  Broadcom table wording suggests for vertical mode;
* the **destination step between vectors is `max(64, ...)` bytes**: with
  `Depth=Units=0` (`=16`) asm3 gets contiguous 64 B vectors, but with
  `Depth=8` the vectors are still 64 B apart even though only 32 B are written
  (the panel then has 32 B of data + 32 B of padding per k).  `Depth=Units=8`
  gives exactly contiguous 32 B vectors, which is why the packer uses 8-k
  strips;
* the RAM-side stride field is the *gap* after each vector, as in the
  horizontal store path.

Two probe bugs of mine are worth recording for anyone repeating this: a
constant loaded with `ldi` into a *B-file* register needs `ws=1`
(`ldi(0x42C80000, 19, ws=1)`); without it the value lands in the A file and the
probe silently reads a garbage B register.

## 4. The kernel: gemm4.py

### 4.1 Word schedule (7 words per row per k)

    w0: rot(r+1) | <A-file copy>        rot -> r1, lane 0 = row r+1 of this half
    w1: fmul(j1) | rep                  r1 lane 0 -> r5 on all 16 lanes
    w2: fmul(j0) | fadd(j1)
    w3: <B-file copy> | fadd(j0)
    w4: fmul(j3) | <issue / copy>
    w5: fmul(j2) | fadd(j3)             TMU1 pops ride here
    w6: <free> | fadd(j2)

Single mul temp r2, consumed exactly one word after it is written, so every
fadd completes inside its row.  `gap` = 2 words between `rep` and the first
fmul that reads r5 (the same spacing asm3 uses; the accumulator rule would
allow 1, untested here).

Free slots per row (16 rows per k pair): w1.mul (ws=1 → A-file copies),
w3.mul (ws=0 → B-file copies), w4.add (raddr free → issues/bumps) and w0.add
(raddr_a + mux only → A-file copies and the raddr_a-only ops).  Issues and
address bumps are placed where the *raddr* fields allow it: a word that reads
`unif`/a TMU address register needs `raddr_a`, an offset constant needs
`raddr_b`, and a word can only provide one of each — that constraint, not the
slot count, decided the final placement.

Pipeline: per k pair, TMU1 gets 8 requests (4 for k+1 in half 0, 4 for the next
k0 in half 1) and 8 pops, all inside one iteration, so the FIFO never holds
more than 4 and nothing is left in flight at the loop exit (asm3's drain is
wrong here and was removed).  A is 1 request/iteration, popped in the same
iteration.  BADR lives in spare accumulator r3 (as a per-lane value), so the B
issues need only `raddr_b`; AADR/KCNT live in the A file.

Word/instruction budget (computed, not yet measured):

| item | words per k pair | per k |
|---|---|---|
| 7 words/row x 16 rows | 112 | 56 |
| counter + AADR bump (utility words) | 2 | 1 |
| branch + 3 delay slots | 4 | 2 |
| **total** | **118** | **59** |

1024 MAC per QPU per k pair / 118 words = 8.7 MACs per word vs asm3's 6.6 →
predicted **1.32x** (≈12.6 GFLOP/s at N=256 with the pack included, ≈11 GFLOP/s
at N=128, versus 9.48 / 7.97 today).  Note the 7-word row is forced by the
A-file/B-file operand constraint (§2): the ideal 5-word row is not encodable.

Store: 8 chunks per tile, 4 rows x 16 words each, the asm3-proven sequence
(generic VPM write setup `0x1A00 + 4*qpu`, VDW `UNITS=4 rows, DEPTH=16 words`,
stride `(N-16)*4`, one row base `4*qpu*128`, `vpw_wait` + 4 nops).  The 16-field
encoding rule (16 == 0) is honoured everywhere and all VPM control words are
generated through `vpmwords.py`.

### 4.2 The two open bugs

**(a) The store block writes stale VPM contents.**  With `DBGSTORE=first`
(store four `0x3F80000x` constants right at the top of the program, DMA to the
C base) the destination memory receives the *previous* VPM content (A-panel
data left by the packer), not the just-written rows.  The same store block in
`test_store4.py` (a 200-word kernel) does land in C at the right addresses, and
asm3's `rows4.bin` / `rows4_s192.bin` still work in this same harness and arena,
so the arena, the DMA and the destination addresses are fine.  Evidence:

  * `test_store4.py` 8-chunk store → C rows 0..5 all contain acc(0,0) = the
    lane ramp, rows 6,7 are 0 (i.e. one VPM row is what the DMA stored for
    several destination rows);
  * a real GEMM run writes nothing to C at all (full 2 MB arena dump: C region
    untouched, A/B/panel intact);
  * the k loop itself runs (64 iterations = its full count, ~7800 instructions).

Prime suspects, in the order I would check them:
1. the per-chunk *VPM source row* the DMA reads vs. the rows the generic write
   setup wrote (i.e. the VPM base-row interaction with `V3D_VPMBASE=16` when the
   same VPM rows are reused 8 times per tile);
2. `ldi(value, R5, ws=1)` (write port 37 = rep) being used to materialise the
   VPM setup constant — proven in asm3, but it is now also the register that
   carries the broadcast during the MAC loop;
3. whether the 4 VPM writes of a chunk need spacing before the DMA trigger when
   the write port has just been used by the MAC loop.

**FIXED (this is why only one tile ran):** `T += TSTR` was in the **k-loop**
branch's delay slots, so it executed on every one of the 64 k iterations and the
tile index jumped to 64*TSTR after the first tile.  It now sits in the
*tile-loop* branch's delay slots (one increment per tile, like asm3).  After the
fix a launch still ends after one tile's worth of work for a different reason
(below), but the tile index is no longer corrupted.

**(b) The tile loop executes exactly one tile.**  `exec_cycles` is ~31k (one
tile's k loop), `instr_hits` ~7800, then the program ends.  The tile test itself
is asm3's (`add.setf S1, T, -ntiles`, `BR_ALL_N_SET`), `T += TSTR` is in the
k-loop branch's delay slot, and `TSTR = u6 = nqpu`.  Next things to check:
the uniform read order (u0..u6 are consumed by seven `or rX, unif, unif`
instructions; if one is consumed elsewhere everything shifts by one), and
whether the k-loop branch's delay slot 2 really executes on the *last* iteration.

### 4.3 What is already known-good in the kernel

* `gemm4.py` builds for N=128 and N=256, `Asm.check()` enforces the asm3 hazard
  rules (no issue+pop word, no pop+immediate word, one issue per word) and the
  builder refuses to double-use a slot;
* every instruction encodes (the Asm layer raises on A-file/B-file conflicts);
* the B/A pipelines are balanced (8 in / 8 out per k pair, 1 in / 1 out for A)
  and the issue→pop distance is ≥ 8 words everywhere;
* the prologue/epilogue use the same TMU/VPM idioms as asm3.

## 5. Harness changes (asm4/qrun_gemm.c)

* `QRUN_PACK_ROWS=8` — the pack-launch uniform setup now derives the row-block
  size from this instead of hardcoding 16 rows (`u1 = i*per*rows*4*N`,
  `u3 = rows*4*N`), so the same harness runs the asm3 16-row packer and the new
  8-row one.
* `QRUN_CPUPANEL8=1` — ARM reference pack with the 8-row map (used to validate
  the checker).
* `QRUN_DUMP=off,len` + `QRUN_DUMP_FILE` — dump raw arena bytes after an L2
  writeback (this is how I proved that nothing is written to the C region).

New tools: `pack8.py` (packer), `check_panel8.c` (8-row panel checker,
bit-exact), `gemm4.py` (kernel generator, `DBGSTORE=1|first` for the debug
store, `AUDIT=1` for the slot/op dump, `LIST=1` for a disassembly),
`run4.py` (pack+GEMM in one process, flock + health check + element-wise
verification + `runs4.jsonl`), `test_pack4.py`, `test_store4.py`,
`run_store4.py`, `probe_vdw8b.py`/`probe_vdw8c.py` (+ runners).

## 6. Where to continue

1. Fix (a) with a minimal kernel: 8 chunks x (4 VPM writes + one VPM DMA) over
   the same 4 VPM rows, one 0x3F80000x constant per chunk, and dump the arena
   after each chunk (`QRUN_DUMP`).  That isolates "reuse the same VPM rows" from
   everything else.  If reuse is the problem, either use 8 VPM rows per QPU
   (needs <= 8 QPUs) or write all 8 rows of a tile with one setup (rows.py does
   exactly that for 4).
2. Then fix (b): print `exec_cycles` per tile with a 1-tile / 2-tile N to see
   whether the loop exits or the branch is wrong.
3. Then re-check the element-wise result at 1 QPU, then 12 QPUs, 5 consecutive
   runs at N=128 and N=256 with the packer included (the acceptance protocol),
   and finally measure.  The measured floor is the issue floor: 118 words per
   k pair at 4 cycles per word = 472 cycles per k pair, i.e. 236 cycles per k
   for 512 MACs per QPU.
