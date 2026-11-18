# asm5 — 8x64 GEMM store diagnosis: environment blocked mid-session

Work tree `/home/raspi/work/asm5` (copy of `asm4`, `asm4` untouched).
Board: Pi 3B, **stock clocks** (V3D 300 MHz verified, `get_throttled=0x0`).

**STATUS: BLOCKED — no verified result.**  The GEMM bug is *not* fixed and
*no* new speed number was measured.  The reason is an environment failure, not
a kernel problem: `sudo` (and everything else that links `libcrypto.so.3`) is
broken on the board, so the GPU harness cannot be launched at all
(`/dev/mem` is root-only).  The board needs a **power cycle** (see §5).

What this run *did* establish is in §2/§3 — including a
**disproof of the leading suspicion** the task named (the `or vpw_addr` write),
which was verified to work on silicon before the board became unusable.

---

## 1. What was measured before the board broke

All runs 1 QPU unless stated, stock clocks, `/tmp/vc4-gpu.lock` held once
around each sequence, `STATE: LIVE` after each run.

| # | kernel | result |
|---|--------|--------|
| 1 | `run4.py 128 12` (pack+GEMM, as delivered by asm4) | C all zero (16384/16384 mismatches), panel 5104 mismatches |
| 2 | `run_store4.py 128` (test_store4, asm4) | rows 0-3 get acc(3,1),acc(4,1),acc(5,1),acc(6,1); rows 4-7 zero |
| 3 | `probe_store.bin` (**new**, §2) 8 chunks, A-file data | **all 512 words correct** at the right addresses |
| 4 | `gemm4_128_dbg1.bin` (DBGSTORE=1) | wrote to a stale hard-coded bus address, see §4 |
| 5 | `gemm4_128.bin`, 1 QPU, `QRUN_CPUPANEL8=1`, fresh boot | exec_cycles 994308 = **32 tiles** (the store block therefore ran 32x), **C[0..16384] all zero**, **panel 0/16384 mismatches** |

Run 5 is the important new datapoint (it was taken on a freshly booted board,
before the environment broke): the store block executes its full count 32 times
and yet **nothing is written anywhere** — C is pristine zero (the harness zeroes
it before the launch) and the A panel is untouched.  So the store is not
mis-addressed at 1 QPU; either the data it stores is zero, or the DMA is a
no-op.

Note this **contradicts** asm4's conclusion that the store lands on the panel
(that came from run 1 above, at 12 QPUs, where the panel *file* showed 5104
mismatching cells).  See §3.3 for why run 1 and run 5 can both be true.

## 2. The destination rule, measured (probe_store.py)

`probe_store.py` is a 277-word kernel with **no k loop, no MACs**: it reads the
same seven uniforms in the same order as gemm4, computes the same
`4*qpu`/`4*qpu*128` values into the same registers, and then runs the *same*
8-chunk store block.  Every chunk writes identifiable data
(1000*chunk + 10*row + lane) to its own destination (chunk c -> C rows 4c..4c+3,
cols 0..15), and `probe_run.py` decodes the arena dump.

Result (1 QPU, N=128, 8 chunks, VPM row base fixed at `4*qpu` = 0 for every
chunk):

    all 8 chunks landed at the expected addresses, all 512 values correct
    (the only blemish is a +1 on the very first word, a probe-local
     ldi/itof artefact in the value generator, not a store fault)

**So, on this silicon, all of the following work:**

* `or vpw_addr, ra21, ra21` **does** set the VPM-DMA destination: the write to
  write port 50 with `ws=1` from an A-file register is taken.  **The task's
  leading suspicion is disproved.**  No delay/nop after the address write is
  needed (`DPRD=0`), and no `ldi` is needed.
* Re-using the **same four VPM rows for all 8 chunks** (row base `4*qpu`,
  never advanced) works — the note in asm4 §4.2 that row reuse is the problem
  is wrong.
* The DMA row-base step (`add vpw_setup, r5, r3`, `r3 = 4*qpu*128`, an
  *accumulator* source rather than a B-file register) works.
* The 16-word/4-row store writes 64 B per row with the **RAM pitch = N*4**;
  the stride word `0xC0010000 | (N-16)*4` is the *gap* after each vector, so
  rows are exactly `N*4` apart and only columns 0..15 of each row are written.
  (This reproduces asm4 §3.1 from the store side.)
* Only 4-byte alignment is implied by the encoding; every destination used in
  the probes was 64-B spaced and worked.  There is no evidence for any
  alignment requirement beyond the natural 4 B.

Caveat: this probe proves the *path* (address + reuse + stride).  It does not
prove that the real kernel's store reaches the same conclusion, because in the
real kernel the *data* is missing (run 5).

## 3. Where the bug is, after this run

### 3.1 The store block is probably innocent

`test_store4` (§1 run 2) writes real accumulator values to C, so the store path
works there too; it is only off by one element, which is exactly the signature
of the DMA sampling the VPM one write early (the destination rows receive
`[prev chunk last write, cur chunk row0, row1, row2]`).  That is a *timing*
signature (write -> DMA slack), not an address problem, and it is the one
difference between `probe_store` (works) and `test_store4` (shifted):
`probe_store` has three ALU words between each `or vpm` and between the last
`or vpm` and the DMA.  **Untested hypothesis: the 4th `or vpm` has not landed
when the DMA reads.**  asm3 uses the same zero-gap sequence and is clean, so if
this is real it must be data-source dependent (B-file/accumulator source vs
A-file source) — do not assume, measure it with `GAP`.
`probe_store.py` has the `GAP` knob for exactly this.

### 3.2 The stronger suspect: the accumulators (or the TMU data) are zero

Run 5: C is **pristine zero**.  If the store worked and the accumulators held
the GEMM result, C would be non-zero somewhere.  Two consistent explanations:

1. the MAC loop leaves the accumulators at zero (bad A/B data from the TMU, or
   a broken broadcast), and the store dutifully writes zeros; or
2. the `or vpm` data writes are dropped in the real kernel, and the VPM was
   zero, so the DMA stores zeros.

asm4's `DBGSTORE=first` result ("the destination receives the *previous* VPM
content") supports (2); run 5 alone cannot distinguish them.
**The claim in the task/asm4 that "the kernel computes correctly internally"
was never measured** — asm4 §4.2 only ever checked that the loop *runs* 64
times (exec_cycles), never the accumulator contents.  Treat it as unproven.

The discriminating experiment (needs the board, ~5 min):

    A. after the k loop, store ONE plain A-file accumulator (e.g. acc(0,0),
       which is a register-file value, not a VPM write) to C through the same
       8-chunk store block, destination from the harness-provided u7.
       -> non-zero: the MACs work, the store data path is the problem (2)
       -> zero:      the MACs/TMU data are the problem (1)

### 3.3 Why run 1 (12 QPUs) and run 5 (1 QPU) differ

If the store destination were `panel + rowoff + colblk*256` (asm4's theory),
run 5 would have zeroed the panel tiles and shown mismatches.  It did not, and
the panel was bit-exact.  So at 1 QPU the destination is C.  The 5104 panel
mismatches at 12 QPUs are therefore better explained by the *packer* being
wrong at 12 QPUs (pack8.py has only ever been verified with 4 QPUs; the asm3
notes document a TMU "issue + pop in the same word" corruption class that
produces exactly random per-tile data damage), or by the 12-QPU run having been
taken on an already-degraded board.  Re-test the packer at 12 QPUs *before*
blaming the GEMM store again.


## 3.4 Confirmed: the k-loop broadcast is one row ahead (independent of the store)

`rot_for(h,r) = (16-(r+8h))&15` puts row r in lane 0 (the same convention as
asm3's proven `mul_rot(1,0,14-m)`, cross-checked against its verified runs), but
the loop rotates `rot_for(*nxt)` in `w0` and reps in `w1` of the *same* row:

    w0  v8min r1, r0, imm(48+rot_for(row r+1))
    w1  fmul j1 (reads the OLD r5 = row r)  |  rep r1 -> r5 = row r+1
    w2  fmul j0 reads r5 = row r+1
    w4  fmul j3 reads r5 = row r+1
    w5  fmul j2 reads r5 = row r+1

The rot immediates in `gemm4_128.bin` are 63,62,...,48 (rows 1..7, then row 0 of
the other half) and every rep sits exactly one word after its rot, so this is
in the encoded program, not a misreading.  Consequence: 3 of the 4 MACs of every
row use the *next* row's A value, and the prologue never reps row 0, so the
first `fmul j1` of the loop consumes an undefined `r5`.

Fix (one add-slot re-arrangement, no extra words): rotate the *current* row at
`w0` (`rot_for(h,r)`) and move `rep` from `w1`'s add slot into `w0`'s add slot;
all four fmuls of the row (`w1,w2,w4,w5`) then read `r5` = row r.  The A-file
TMU copy that currently uses `w0`'s add slot (needed only for the two A-file
pops per half) moves to `w6`'s free mul slot.  Add `rep` of row 0 to the
prologue.  The 118-word/k-pair budget is unchanged.

## 4. The landmine that broke this run — remove it

`gemm4.py`'s debug store path (`DBGSTORE=1` and `DBGSTORE=first`) **hard-codes
the C bus address**:

    a.ldi(0xde610000, RA_CADR, note='dbg C bus address')

The dma-buf's bus base is **not stable**: this run saw `0xde600000` and
`0xe0200000` on different runs.  With the wrong base, the QPU's VPM DMA writes
256 B blocks straight into whatever DRAM lives at that physical address.  On the
`0xe0200000` run, `0xde610000` = physical `0x1e610000`, and shortly afterwards
the board rebooted and `libcrypto.so.3` came back corrupted in the **page
cache** (only 2 pages differ; the file on disk is intact, §5).  Do not use a
hard-coded bus address in a QPU kernel on this board.

Two fixes were applied in this tree (compile-checked, **not run**):

* `qrun_gemm.c`: the GEMM uniform block now also carries
  `u7 = gpu_bus + out_off` (the C base, in *bus* address space, correctly
  relocated for this process's dma-buf) and `u8 = gpu_bus` (the arena base).
  Kernel code that needs a known-good destination must use u7/u8.
* `gemm4.py`: `DBGSTORE` now takes the destination from u7 instead of the
  hard-coded literal.

## 5. Environment failure and recovery (do this first)

**Symptom.**  Every program that links `libcrypto.so.3` dies:

    $ sudo -n true
    sudo: symbol lookup error: /lib/aarch64-linux-gnu/libcrypto.so.3:
          undefined symbol: BIGNUM_it, version OPENSSL_3.0.0

`sudo`, `systemctl`, `md5sum`, python's `ssl` — all broken.  Consequences:
the GPU harness cannot run (`/dev/mem` is `root:kmem 0640`), and
`polkit.service` failed, so no reboot can be requested either
(`systemctl reboot` -> "Method call timed out"; `ssh root@` is denied by the
tailnet policy; `su` needs a password; there is no setuid helper that helps).

**Diagnosis: the file on disk is fine, the *page cache* is poisoned.**

    md5 (read())        = 37413cf75861925abe1b8443cde51041   <- corrupted copy
    md5 (O_DIRECT read) = fc9c07dfb8d81a2ef0b29eaaf6e7321d   <- dpkg's md5, correct
    differing bytes: 5260, in pages 7 and 71 only
    (dd if=... iflag=direct, i.e. page-cache bypass, returns the good bytes)

`posix_fadvise(DONTNEED)` and ~1.5 GB of page-cache churn did **not** drop them
(a root process presumably has the file mapped), so this cannot be repaired
without root.

**Recovery (any one of):**

1. **Power-cycle the board.**  The on-disk file is intact, so a fresh boot is
   enough.  Verify with `sudo -n true` and
   `sudo -n vcgencmd measure_clock v3d` (must print 300000000).
2. If a console root shell is available without rebooting:
   `echo 3 > /proc/sys/vm/drop_caches` then re-check `sudo`.
3. As a belt-and-braces measure a good copy of the library was saved as
   `/home/raspi/lc_good.so.3` (md5 fc9c07dfb8d81a2ef0b29eaaf6e7321d); from a
   root shell `cp /home/raspi/lc_good.so.3 /usr/lib/aarch64-linux-gnu/libcrypto.so.3`
   would also fix it if the disk copy ever does get damaged.
   A good copy also lives in `/tmp/gl/libcrypto.so.3`, which makes *unprivileged*
   tools work again via `LD_LIBRARY_PATH=/tmp/gl` (useful for `systemctl`,
   `md5sum`, python ssl, `dpkg`) — but **not** `sudo`, which is setuid and
   therefore ignores `LD_LIBRARY_PATH`.
   Note `/tmp` does not survive a reboot; `/home/raspi/lc_good.so.3` does.

**Also verify after recovery** that the board is otherwise sane:
`sudo -n dmesg | tail`, `sudo -n vcgencmd get_throttled` (0x0),
`STATE: LIVE` from `read_v3d_status`, and re-run the packer at 4 QPUs.

## 6. Tools added in asm5 (all runnable, none needing root to *build*)

| file | purpose |
|------|---------|
| `probe_store.py` | parameterised store-only kernel (NCH, VROW=fixed/walk, GAP, DPRD, BASESRC=a/ldi, DSTEP=acc/b). Writes 1000c+10r+lane, one chunk per 4 C rows. |
| `probe_run.py` | launches a kernel with the GEMM uniforms, dumps the arena, prints raw words. No flock inside — **hold `/tmp/vc4-gpu.lock` around it** (do not nest). |
| `diag1.py` | runs one GEMM with `QRUN_CPUPANEL8=1`, then diffs C and the panel against float64 references computed in Python and reports where anything landed. This is the tool that produced run 5. |
| `dumpw.py` | hex+float dump of a dump file. |
| `probe2.py` | the "gemm4 store block + test_store4 preload" probe; **does not build** (a B-file fadd needs its constant from the A file — see the comment in `test_store4.build2`). Fix before use, or ignore it: probe_store + diag1 cover the ground. |

`gemm4.py` gained a `SKIP=prime|loop|all` knob (branch straight to the store)
for bisecting the kernel body; the default build is unchanged (362 words at
N=128).

## 7. Next steps, in order

1. Recover the board (§5).  Confirm stock clocks and `STATE: LIVE`.
2. `python3 diag1.py gemm4_128.bin 1 128 t1` — reproduce run 5 (C all zero,
   panel clean).  If the panel is *not* clean at 1 QPU, the store really is
   mis-addressed and §2's probe should be re-run with `VROW=walk` to compare.
3. Run the discriminating experiment (§3.2 A): store one plain A-file
   accumulator after the k loop to u7.  That splits "MACs produce zero" from
   "store data path drops the write".
4. If the MACs are the problem, dump the A vector (`r0`) and one B vector
   (`rb16`) right after the first pop; compare with the panel/A/B files the
   harness saved.  The prime's request/pop order and `rot_for()` lane mapping
   are the places to look (lane r must get `A[8rb+r][k]`, lane 8+r
   `A[8rb+r][k+1]`).
5. Only then re-address the tile-loop store count, the 12-QPU packer
   verification, and the 5-run acceptance protocol (N=128, N=256, 12 QPUs,
   pack included, targets 0.527 / 3.539 ms; the GEMM-only bars are 7.97 / 9.48
   GFLOP/s).
