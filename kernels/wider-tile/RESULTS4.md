# asm4 — results so far (stock clocks, 300/400 MHz, untouched)

Work tree `/home/raspi/work/asm4`.  Board `STATE: LIVE` before and after every
run, `get_throttled=0x0`, every GPU job under `flock -w 2400 /tmp/vc4-gpu.lock`
+ `timeout`.  `asm3` was not modified.

## Delivered and verified

**8-row packed A panel** (`pack8.py`, harness option `QRUN_PACK_ROWS=8`):

    PANEL8[(rb*N + k)*8 + r] = A[rb*8 + r][k]

bit-exact against A on silicon (`check_panel8`), 4 packing QPUs:

| N | panel check | pack time | pack QPUs |
|---|-------------|-----------|-----------|
| 128 | **0 / 16384 mismatches** | 0.087 ms | 4 |
| 256 | **0 / 65536 mismatches** | 0.460 ms | 4 |

This is the enabler for a 64-column tile: one 64 B TMU request delivers two k
columns of an 8-row block, so the 4-column kernel needs a single A request per
k pair.

Also delivered: the measured **vertical-VDW field semantics** (`Depth` = words
per output vector, `Units` = number of vectors, destination step
`max(64, ...)` bytes — see NOTES4.md §3.1), `check_panel8.c`,
`QRUN_PACK_ROWS` / `QRUN_CPUPANEL8` / `QRUN_DUMP` in `qrun_gemm.c`, the
`gemm4.py` kernel generator with the hazard-checking assembler layer, `run4.py`
(pack+GEMM in one process, flock, health check, element-wise verification,
`runs4.jsonl`), `test_pack4.py`, `test_store4.py`, `run_store4.py`, and the VPM
DMA probes.

## Not delivered: the 8x64 GEMM kernel is not correct yet

`gemm4.py` builds for N=128/256 (366 instructions) and runs, but:

* the VPM store block writes its data to the **A panel** instead of the C
  matrix: the destination it uses is `panel_base + rowblk*32N`, so the register
  holding the "C base" uniform actually contains u4 (the panel base) — the
  uniform read order in this kernel is shifted (see NOTES4.md §4.2);
* `T += TSTR` was in the k-loop branch's delay slots (64 increments per tile) —
  found and fixed; the tile index is correct now;
* the tile loop executes exactly one tile per launch (≈7.8 k instructions,
  exec_cycles ≈ 31 k), then the program ends.

Therefore there is **no speed number to report** and the acceptance bars
(0.527 ms / 3.539 ms) are not addressed by measurement in this work tree.
asm3's numbers remain the accepted result.

Isolation status of the store bug (the most useful pointer for continuing):

* `rows4.bin` (asm3's proven single 4-row store) and `rows4_s192.bin` still
  work in this harness/arena;
* an 8-chunk copy of the *same* store sequence (`test_store4.py`) does land in
  C, but the four destination rows of a chunk receive registers that are
  **shifted by one row** relative to the four `or vpm` writes that preceded
  them (e.g. rows 0-3 get acc(3,1), acc(4,1), acc(5,1), acc(6,1) instead of
  acc(4,1)..acc(7,1)).  That is a VPM-write → DMA-trigger ordering/lag effect
  when the same four VPM rows are re-used for every chunk, not an address
  problem (the destination address is correct and the stride is correct);
* a first attempt to read the `test_store4` numbers was confounded by my own
  bug: `ldi(1000, rb30, ws=1)` loads the integer bit pattern (a denormal float
  ≈ 0), not 1000.0 — all "missing" constants in the first run were that.

## Design and predicted performance (for when it works)

8 rows x 64 columns, 4 accumulators per lane (j=0..3), 32 accumulators
(j=0,1 A file, j=2,3 B file), A from the 8-row panel (2 k per request), B row
major (4 requests per k, P/Q double buffered), 7 words per row per k, store in
8 chunks of 4 rows with the asm3-proven sequence.

Computed budget: **118 words per k pair** (112 MAC words + 2 utility + branch +
3 delay slots) = 236 cycles per k at 4 cycles/word for 512 MACs per QPU,
i.e. **8.7 MACs per issued word vs asm3's 6.6 → 1.32x**, predicted
≈12.6 GFLOP/s at N=256 (pack included) and ≈11 GFLOP/s at N=128, against the
acceptance bars 9.48 / 7.97.  The 7-word row is forced by the A-file/B-file
operand constraint (a word reads at most one register from each file), not by
the broadcast count: the theoretical 5-word row is not encodable with this
accumulator split.

Ceiling: the loop would be at 1024 MAC per 236 cycles = 4.34 MAC/cycle/QPU →
12 QPUs at 300 MHz = 2·12·4.34·3e8 = 31.2 GFLOP/s issue-bound; the 16.42
synthetic ceiling would be reached at ~52 % of that, i.e. the kernel would still
be ~2x off the pure-ALU ceiling because 2 of the 14 ops in every row are the
broadcast (rot + rep) and the fadd/fmul pairing costs one more word per row.
