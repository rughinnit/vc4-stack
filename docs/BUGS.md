# Bugs and fixes — the register

Every defect found while building this stack, what caused it, what fixed it,
and how the fix was verified. The rule this file exists to enforce: **a bug is
only "fixed" when a test or a measurement fails before the fix and passes
after** — otherwise it is "gated" (it refuses) or "open".

**Status legend**

| status | meaning |
|---|---|
| **fixed** | reproduced, fixed, and re-verified (test or measurement named) |
| **gated** | not fixed; the entry point refuses with a reason, so it cannot misbehave silently |
| **open** | known, not fixed, and reachable — with a workaround |
| **avoided** | a process/infrastructure failure that was fixed by changing how we work, not code |

Severity: **S1** silent wrong answer · **S2** crash/wedge/hang · **S3** wrong but
loud · **S4** build/package/test defect · **S5** measurement artifact.

**Measurement conditions.** Rows dated before **2026-11-20 (fan removed)** were
taken without temperature logging, so a long one may have drifted into a
different thermal regime (`docs/BENCHMARKS.md` rule 4). The B12 row and the
re-run acceptance below it carry start/peak temperatures; anything older should
be re-taken under `tools/thermal-guard.sh` before being quoted as a measurement.

---

## A. Silent wrong answers (the dangerous class)

| # | symptom | root cause | fix | verification | commit |
|---|---|---|---|---|---|
| A1 | Conv output-channel blocks silently missing, data-dependently | `vc4_conv_nhwc_pack_w` built the weight panel on the CPU but never flushed it through `vc4d`, so the GPU read stale arena bytes | added `vc4__mark_dirty` before the pack kernel and after the bias fill that lands post-launch | YOLOv8n went from 24–36 elements wrong to 0/44,100 | `236f132` (track) |
| A2 | SPPF max-pool output off by 0.11 | ONNX pads MaxPool with **−inf**; the NHWC kernel padded with **0**, so padded cells won the max where they should have lost | the −inf border is materialised on the host/QPU instead of 0 | 5×5 SPPF cases 0 mismatch; whole-model 0/44,100 | `236f132` |
| A3 | Elementwise ops mixed pixels when operands had equal shapes | two operands with equal *shapes* but different NHWC *pitches* were treated as identical layouts | the runner compares pitches, not just shapes, before a zero-copy op | elementwise suite + modelA/B/C | `236f132` |
| A4 | `conv2d` could not be called from Python at all | the ctypes `argtypes` for `vc4_conv2d` had the wrong **count and order** | corrected binding and exposed `maxpooln`, `softmax_fwd`, `vc4_get_stats` | `vc4infer.ort` and every Python conv test | `dab6cc6` |
| A5 | A pip-installed engine built a **d=1** kernel for a dilated conv (`--dil 2` ignored) | `kgen.main()` parsed options generically and ignored unknown ones, so a stale generator silently produced a different kernel | unknown options now fail with **rc=2** and are named | `--dil 2 --cstride 64` → rc 2; default-arg blobs bit-identical (sha256) | `d5f3c69`, `00a863d` |
| A6 | Grouped conv wrote a wrong C pitch from a pip install | the two tracked `kgen.py` copies had diverged **in opposite directions**: `tools/` had `cstride`/`dilation`, the wheel copy had `nvec` but had lost them | 3-way merge into one canonical generator; `tools/check_kgen_sync.py` fails the build on drift | both copies byte-identical; grouped conv 0 mismatch | `d5f3c69` |
| A7 | Transcendentals returned input-independent values | **B8 collision**: the harness writes the `b` operand into B8 before every vector body, and B8 was also in the constant pool, so `ONE`/`ee` overwrote `x` | B8 excluded from both pools; B21 freed for the group counter | 24 kernel ops, 0 mismatches / 105,264 elements; `Exp` nv16 exact | `e0ea110` |
| A8 | `Round` at nvec=4 returned a fixed pattern | the body overwrote the store's staging registers (`VPMQ7`, `CSTEP`) | rebuild them from `A25` immediately before `_store4` | `Round` bit-exact at nv4; nv8 path for the slot limit | `e0ea110` |
| A9 | 4-vector store wrote nothing / raised VPM range errors | VPM **write** row passed `q<<9`; it must be `4q` (write) and `q<<9` (DMA read) | corrected the two values | one-QPU 4-vector stores exact | `e0ea110` |
| A10 | Every long unary body was wrong, at any spacing | emitter staged one operand with an **add-ALU `mov`** and the multiply read it the **next word**; add-ALU→mul-ALU needs 3 words | `E.mul`/`mul24` pad three words; rule recorded | mul→mul exact at 0–5 nops; the failing case exact after padding | `e0ea110` |
| A11 | Whole-model output drifted run to run (24 → 22,700 elements over) | the client's arena mapping is **cacheable**; the DFL softmax read stale lines from a recycled pool offset | invalidate before every arena→host copy (read side was the one that mattered), centralised in the access helpers | **20/20 identical runs**, 0 over tolerance | `158fcae` |
| A12 | `Resize` nearest output off by up to 1.97 | every nearest mode computed `floor((s+0.5)·scale−0.5)`; ONNX's default is `round_prefer_floor` | all four `nearest_mode` values + coordinate modes implemented | `--only Resize` → **0 mismatch, 0 refused** | `564c04c` |
| A13 | Harmonics of the same class: `Tanh`/`Softplus` off at large `|x|` | truncating FPU rounds `1−eps` to `1−2^-24`, so a composed `Gelu` drifted at ±1e10 | `Erf` selects exactly ±1 for `|x|≥6` | composed ops 0 mismatch on the 1e10 edge cases | `e0ea110` |
| A14 | `torch_vc4` leaked **every** tensor | `c10::DataPtr` was built with a null context; c10 only calls the deleter when the context is non-null, so `arena_free` never ran (died after 1,531 allocations) | context = the data pointer; `alloc_info()` + a 48 MB churn test | churn test passes | `4146054` |

## B. Crashes, wedges and hangs

| # | symptom | root cause | fix | verification | commit |
|---|---|---|---|---|---|
| B1 | A small immediate in the **A operand slot wedges the QPU** | only the b-slot accepts a small immediate; the `floor` `0 − n` encoding put one in A | use the `ZERO` register; recorded as an encoder rule | long bodies run; no wedge since | `e0ea110` |
| B2 | Stale kernel blob reused forever after a generator change, wedging the board | `vc4__kernel` keyed the cache on the **name only** | sidecar `<blob>.bin.stamp` = sha256(generator + kops + ABI tag + argv); mismatch ⇒ regenerate | `stamp_test.sh`: unstamped regenerates, valid reuses, corrupt regenerates | `addef42` |
| B3 | Denormals behaved inconsistently | **the FPU flushes denormals to zero** (`1.4e-45` survives `Abs`, dies in `LeakyRelu`) — not documented anywhere | measured and recorded as a numeric rule; range reductions keep away from them | denormal probe | `0c80588` |
| B4 | `vc4_reduce`, `vc4_cumsum` kill the daemon context on **any** shape; `vc4_select3(WHERE)` returns every element wrong | TMU address taken from an accumulator; pop order does not match `build_ew`; fixed 256-output group walks past the caller's buffer | **gated**: `VC4_ERR_UNSUPPORTED` with the defect named, override only via `VC4_ALLOW_UNVERIFIED_KERNELS=1` | `test_gate`: `ALL UNVERIFIED KERNELS REFUSE` | `d15610d` |
| B5 | `vc4_gather` / `vc4_blit` corrupt memory / copy nothing | same TMU-from-accumulator defect; the 4-vector store wrote no output | **gated**, same contract | `test_gate` | `d15610d` |
| B6 | `vc4-query` **segfaulted** on every run | packaged binary built against an older `vc4_op_info` ABI and linked `libvc4infer` 1.2.3 | rebuilt from the current headers; version injected at build time | `vc4-query` rc=0, 86 entries, prints `1.0.2` | `b09b286` |
| B7 | `vc4d` could not start at all after a purge | unit had **unquoted** `Environment=VC4D_ARGS=--socket …`, split on whitespace into invalid assignments; it only ever worked because `/etc/default/vc4d` overrode it | quote the value; `EnvironmentFile=-`; `systemd-analyze verify` in the acceptance | fresh install + conffile deleted ⇒ `vc4d` active | `vc4d 1.2.1` |
| B8 | `pool` kernel could not be regenerated | launched with a `--c` option the generator rejects | pass the accepted argument | suite green after a kops edit | `f1b28b4` |
| B9 | `vc4d` stayed **dead** after one GPU timeout (was G1) | `gpu_abort()` set `dead=1` and nothing ever cleared it, so every later launch failed with `-12 daemon is dead (GPU timeout earlier)` until `systemctl restart vc4d`, although SRQCS read back idle and ERRSTAT was back to `0x1000 VCD-idle` | after a timeout the daemon resets the V3D context (queue, VPM base, L2/SLC, error latch, counters) under the GPU lock and **verifies it with a launch of its own 4-instruction probe program** in the reserved last code slot; only a probe that completes with no new ERRSTAT bits re-enables launches, otherwise the daemon exits 75 and systemd (`Restart=always`) starts a clean one | `test_daemon_recover`: 3× (a ~5 ms GEMM on a 1 ms deadline → `dead=0` → a new client's GEMM completes; reset+probe verified in 0.0 ms). Before, the same sequence left `dead=1` and the next GEMM failed `-12 daemon is dead (GPU timeout earlier)` after 10 s | `fix/daemon-g1-g11` |
| B10 | the daemon did not reclaim a **departed** client's arena region (was G9) | a region was released only when the *socket* closed, so any client whose socket outlived the process that allocated it (fork, inherited fd) pinned its arena forever; measured: one 8 MiB region survived its owner and the next client was denied `op=ALLOC: arena exhausted` with nothing running. The free list also had a hard cap (`MAX_FREEBLK`) that silently dropped a block when full | liveness is anchored on the peer pid from `SO_PEERCRED`: `kill(pid,0)==ESRCH` (or `EPERM` = alive) is checked every poll iteration, on `HELLO` and before refusing a connection; the heap is now a page bitmap, so a freed page cannot be lost | `test_daemon_arena`: 24 short-lived clients (every third without FREE) → `arena_used=0`; orphaned 8 MiB in a 16 MiB arena reclaimed → next client gets 3×4 MiB; before: `0 × 4 MiB after 5.1 s` | `fix/daemon-g1-g11` |
| B11 | a `direct` client and a running `vc4d` drove the QPUs at the same time (was G11) | the direct path never took `/tmp/vc4-gpu.lock`, which the daemon holds per launch, so the two wrote SRQCS/SRQPC/VPMBASE over each other. Measured with 400 direct launches against a 360-launch daemon run: 399/400 direct launches failed `GPU timeout (context is dead)` and the daemon marked itself dead | the direct path takes the same lock for the same window (kick → completion), bounded at 30 s with a message that names the lock and says to remove any `flock` wrapper; `VC4INFER_NO_GPU_LOCK=1` / `VC4INFER_GPU_LOCK_MS` for the rare case that needs it; the engine never holds it across client work, and `tests/util.h` no longer takes it at all | same 400/360 run after: **0 failures, 0 wrong results, 0 timeouts on either side**. `test_direct_lock`: a direct launch waits out a 600 ms hold (557 ms) and a stuck holder fails in 252 ms naming the lock; 16 interleaved launches clean | `fix/daemon-g1-g11` |
| B12 | a plain C client was handed an arena of exactly `VC4_DATA_BASE` (3 MiB) and could not allocate one byte: `vc4_init()` returned ok, `vc4_arena_bytes() == vc4_arena_used() == 3145728`, and every `vc4_alloc` failed with `arena exhausted: need 262144 B, have 0 B` | `infer/Makefile`'s object rule had **no header prerequisites**, so adding `lock_fd` to `struct vc4_ctx` (G11) did not rebuild `vc4_dev_remote.o`. The stale object wrote `arena_bytes`/`arena_top`/`scratch` 8 bytes below where the fresh objects read them (i.e. into `gpu_phys`/`arena_bytes`/`arena_top`): the client saw the *old* `arena_top` (3 MiB) as its arena size, the *old* `scratch` (3 MiB) as its top, and 0 as its scratch. Reproduced exactly by relinking a pre-G11 `vc4_dev_remote.o` into the current archive | every object now depends on `src/*.h`, `include/*.h`, `../training/src/*.h` and the daemon's client/proto headers, and `libvc4client.a` rebuilds when its sources change. The client also validates the grant (`off`/`len` inside the HELLO arena, `len > VC4_DATA_BASE`, `len >= wanted`) and releases it, and `vc4_init()` re-reads the context it got and refuses a mixed build by name | the mixed archive now stops at `vc4_init` with `vc4infer: vc4d context inconsistent after init: arena 3145728 B … stale .o files`; a clean build reports a 13 MiB region and `vc4_alloc(256 KiB)` ok; `REPS=1 ./run_tests.sh` green (63-65 C, peak under the 72 C guard, `throttled=0x0`) | `main` (this fix) |

## C. Wrong but loud (refusals that were wrong)

| # | symptom | root cause | fix | commit |
|---|---|---|---|---|
| C1 | `Conv` grouped through the ONNX runner raised `copy2d_x: invalid argument` | group slice passed an invalid pitch/offset to `vc4_copy2d` (the C API path was fine, which is why `test_nhwc` passed) | fixed in the NCHW→NHWC grouped path | `564c04c` |
| C2 | `Softmax` on axis 1 with >16 classes refused | the kernel is last-axis, ≤16 | last-axis restriction documented; composed path for the head; `LogSoftmax`/any-width still **open** | — |
| C3 | `DepthToSpace` refused | an output row alternates between `blocksize` channels, so no constant lane stride works, and `copy2d` writes 16 *contiguous* words; the kernel that could do it (`vc4_gather`) is gated | **open**, documented with the exact reason | — |
| C4 | `graph pool exhausted ... of 8388608 B` on a model that fits (was G10) | `Runner` sized the pool as `max(arena - VC4_POOL_RESERVE_MB, 8 MiB)`: with a small client arena the result was always the **8 MiB floor**, and the run died mid-graph with a message that reads like "model too big" (the model needed ~13 MiB of a 16 MiB free arena) | the pool is sized from the graph's own peak (float weights + ONNX liveness of activations + cached conv panels), clamped to what the arena can give beside the engine's per-op scratch (conv padded image + im2col panel, elementwise tails); a short estimate grows by another segment, and a model that cannot fit is refused at plan time with need, reserve and arena | `test_pool_sizing.py`: 8 MiB arena → refused with `needs at least 12.5 MiB ... arena is 8.0 MiB`; 24 MiB arena → pool 19.06 MiB (was exactly 8388608 B) | `fix/daemon-g1-g11` |

## D. Test and measurement defects (they lied about the code)

| # | symptom | root cause | fix | verification |
|---|---|---|---|---|
| D1 | `suite` red: `test_dispatch` "should be dispatchable" | the test demanded **every** registry entry be dispatchable, while the table correctly lists unimplemented ops with reasons — and later demanded `layernorm` be unsupported after it was implemented | invariant rewritten: every enum appears in the table, the helpers agree with the row, unsupported rows must carry a reason, and a named list of still-unimplemented ops must exist | `PASS dispatch` |
| D2 | `suite` red: "evidence client exited before it could be inspected" | the probe ran `test_matmul 1` (~4 ms) then slept 0.5 s; with a warm cache the client was gone | 300 reps + poll for the daemon socket instead of sleeping | `SUITE_EXIT=0` |
| D3 | `sum_bwd axis1` failed intermittently | the check wrote 30 outputs into a 10-element buffer (heap corruption) | exact broadcast check | `rel_L2 0` |
| D4 | `maxpool 3x3s2p1 dx` failed | **the check was wrong**: with `h=2e-2` the finite difference flips the argmax when a window's top two values are within `h`; the op matched an fp64 scatter reference on all 42 elements | the check skips near-ties and **reports the count** | `rel_L2 9.1e-07, over_tol 0` |
| D5 | `div_bwd` "passed" a bad check | validated by a difference of `1/y`, ill-conditioned by `1/y²` | closed-form check | `9.3e-8` |
| D6 | `mm 256³` reported "vc4 faster than CPU" | the CPU figure (6.17 ms) was taken while a concurrent job was running; the real value is 3.632 ms | protocol added: pinned threads, quiet board, median of 5 after 2 warm-ups | CPU 3.632 vs vc4 5.090 ms |
| D7 | "The engine's NHWC conv hangs the board" | a name-keyed blob cache was being rebuilt concurrently; the call verifies | claim withdrawn, call re-verified (`1.9e-05`), native path enabled | torch-vc4 `COMPLETENESS.md` |
| D8 | "same binary, half the speed" (twice) | the firmware's DVFS cannot see QPU work through `vc4d` and drops the core **400 → 250 MHz**; the second time, the boot config had no `core_freq` at all | sampled during load; pinned `core_freq=400` + `core_freq_min=400` (stock, no voltage change) | 400 MHz in 5/5 samples under load; 3.73 → 8.47 GFLOP/s |
| D9 | 250 MHz looked like a droop during an 18 s "slow" run | **250 MHz is also the idle clock**; a dead GPU process had left the board idle | sample only while work is in flight; trust cycle counts over one clock reading | `docs/CLOCK-VARIANCE.md` |
| D10 | `kgen._opt()` recursed into itself | a bulk rename replaced `opts.get(` inside `_opt`'s own body | fixed by the $1 work | softmax/probe regeneration works |

## E. Packaging and provenance

| # | symptom | root cause | fix | commit |
|---|---|---|---|---|
| E1 | Clean install could not JIT any kernel | the shipped `.so` baked **`engine/tools/kgen.py`**, a build-tree path the reorg deleted | resolution: `$VC4INFER_KGEN` → baked path *if it exists* → `/usr/lib/vc4infer/kgen/kgen.py`; `make KGEN_DEFAULT=…` | `00a863d` |
| E2 | A system client run from `/` could not JIT | cache defaulted to `./kcache` relative to the **caller's cwd**, and nothing created it | `$VC4INFER_CACHE` → `$XDG_CACHE_HOME`/`~/.cache/vc4infer/kcache` → `/tmp`, created recursively | `00a863d` |
| E3 | The installed package was **stale**: 35 ops, no unary family, while the tree had 51 | packages were built from an older commit and nothing checked | `infer/packaging/check_release.sh` compares `coverage.OPS` installed vs tree and **fails the build** on mismatch | `check_release.sh` |
| E4 | `torch_vc4` broken after the repo reorg: every JIT `status=2` | `setup.py` still pointed at `vc4-stack/engine` | default changed to `infer/`; rebuilt | `6496a83` |
| E5 | `vc4-toolkit-fast` missed the `vc4infer/ort/` subpackage and left empty dirs after purge | packaging list not updated with the new module; purge removed files but not directories | file lists updated; postrm cleans | `1901db0` |
| E6 | `vc4infer` reported library version `0.1.0` next to a 1.2.2 package | `VC4INFER_VERSION_STRING` never bumped | tracks the release | `380a2e9` |
| E7 | `vc4-query` printed a hardcoded `1.0.0` | version string in the source | passed in at build time (`-DVC4_QUERY_VERSION`), so binary and package cannot disagree | `b09b286` |
| E8 | `infer/kcache/` churned and blocked `git pull --rebase` twice | 239 blobs were **tracked** despite being in `.gitignore` | untracked; files stay on disk | `87fcd46` |
| E9 | `/tmp` worktrees and caches consumed **452 MB of RAM** | `/tmp` is tmpfs, and tmpfs cannot be reclaimed to disk; the Pi has 905 MB | `systemctl mask tmp.mount`; `~/scratch/{tmp,refs,evidence,logs}`; `TMPDIR` exported | `5fb662d` |

## F. Process failures that cost real time (not code bugs)

| # | failure | what happened | rule adopted |
|---|---|---|---|
| F1 | `flock /tmp/vc4-gpu.lock` around a client | the daemon takes the same lock per launch, so the client never got it: `GPU lock wait timed out`, launches denied, a concurrent job's training run blocked for 4.5 minutes. **Deadlocked the board three times.** | never hold that lock; the daemon serialises; check `ps` and wait |
| F2 | A resurrected probe on a dead daemon | `LAUNCH: daemon is dead (GPU timeout earlier)` — every later test failed for the wrong reason | check `systemctl is-active vc4d` before blaming code; `systemctl restart vc4d` |
| F3 | Two daemons on one V3D | a private `vc4d-tv` alongside the installed one produced a spurious `V3D_ERR_STAT=0x1000` | one daemon; stale sockets removed |
| F4 | Purge during a concurrent test | a suite run was invalidated mid-flight by `apt-get purge` | announce purges; sequence the board |
| F5 | Concurrent builds editing the same tree | a "failure" (`sum_bwd`) was a stale binary; another was a half-built archive | re-run every gate on the current tree before reporting it |
| F6 | A stray QPU DMA corrupted the ARM page cache of `libcrypto`/`libm` | silent memory corruption from an early experiment; disk intact, power cycle recovered | all GPU memory access goes through `vc4d`; no raw addresses in clients |

## G. Open bugs, with workarounds

| # | bug | impact | workaround | where |
|---|---|---|---|---|
| G2 | **CMA fragmentation silently halves the arena** | after churn, 320×320 fails with `graph pool exhausted` | reboot; the halving is reported, not prevented | `docs/TROUBLESHOOTING.md` |
| G3 | **128 program slots** per daemon session | long sessions cannot launch | restart the daemon; no LRU eviction | `docs/TROUBLESHOOTING.md` |
| G4 | **Uniform/address collision**: any float ≤ −2.0 cannot cross `vc4d` (bit pattern ≥ `0xC0000000` looks like a bus address) | `Pow(x,−2)`, `Clip(−3,…)`, `LeakyRelu(−2)` impossible in managed mode | `--backend direct`; a per-word value flag in the protocol is designed but **not implemented** | `docs/BACKENDS.md`, `vc4_guard.c` |
| G5 | nv8 store path unexplained (`2^92` scale) | `Exp/Log/Round` limited to the nv4 path | works on nv4 | `docs/ONNX.md` §12 |
| G6 | `Gelu`-erf **backward** impossible | 4,222 words against a 2,048-instruction slot | forward only | `docs/TRAINING.md` |
| G7 | 5 kernels gated (B4, B5) | `DepthToSpace` refused; some ops use expensive compositions | refuse loudly | above |
| G8 | ~74 of 152 expressible ONNX ops still refused | models using them fail by name | `vc4infer nodes` before running | `docs/COMPLETENESS.md` |

G1, G9, G10 and G11 left this table on 2026-11-20: they are fixed and
re-verified as **B9** (daemon no longer dies for good after a timeout),
**B10** (the arena is reclaimed from a departed client), **B11** (direct mode
and `vc4d` share the GPU lock) and **C4** (the graph pool is sized from the
model, not from an 8 MiB floor).  Each entry names the test that fails before
the fix and passes after it.

