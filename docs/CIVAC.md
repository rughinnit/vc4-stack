# The client's cache maintenance: what is dirty, and what it costs

Status: **measured. The volume is close to irreducible by scheduling; the
zero-fill of the conv pad was 73 % of the bytes it wrote, and that part is
fixed.**  This page is the accounting behind [DAEMON-PIPE.md](DAEMON-PIPE.md)
§6.2, which attributed 70-95 ms per frame to `dc civac` at 160.  That figure
was wrong: a direct per-frame measurement puts the dirty-range cleaning at
**6.7 ms**, the pad's own zero-fill at **14.4 ms**, and the invalidation before
a CPU write at **0.9 ms**.  The correction matters, because it moves the target
from the cache maintenance to the pad.

## 1. How the client cleans

The dma-buf mapping is cacheable.  Two operations maintain it:

* **invalidate before a CPU write** (`vc4__invalidate_off`): CIVAC over the
  range the CPU is about to overwrite, so that no stale line of the client's is
  written back later over what the QPU has put there.  Counted as
  `civac_off`.
* **clean before a launch** (`remote_clean` at the end of each request): CIVAC
  over the byte intervals the CPU has written since the last clean, from a
  merge-on-insert dirty list (`remote_mark`/`remote_mark_nb`).  Counted as
  `clean`.

`DC IVAC` is undefined at EL0 on arm64, so both are CIVAC.  CIVAC is safe
because every line is cleaned before a launch: a later CIVAC finds clean or
invalid lines and cannot push a stale copy over a result.  Nothing in this
branch changes that rule.

## 2. The accounting

Instrumented in `infer/src/vc4_dev_remote.c` (`VC4INFER_PROF=1`, exposed
through `vc4_prof_ptr()`).  Per frame, one client, YOLOv8n, one frame after
three warm-up frames, packaged daemon on `/run/vc4d.sock`, ARM 1.0 GHz,
47-49 C:

| imgsz | bytes marked dirty | calls | cleaned | unique pages | clean ms | inval. ms | pad zero ms | pad copy ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 36.0 MB | 633 | 35.3 MB | 2.5 MB | 3.91 | 0.37 | 11.74 | 2.21 |
| 128 | 41.8 MB | 709 | 41.6 MB | 4.1 MB | 4.51 | 0.51 | 12.94 | 3.54 |
| 160 | 63.4 MB | 982 | 61.4 MB | 6.8 MB | 6.65 | 0.88 | 14.39 | 5.73 |

* **Marked ≈ cleaned** at every size: the dirty list is not over-cleaning.
  What is written is what is cleaned, once each.
* **The hull collapse never fires at YOLO shapes**: `ndirty` peaks at 4 and
  `mk_hull` is 0.  It *does* fire for a batch request with many entries (a
  64-entry batch marks a uniform block and its scratch per entry, so the list
  reaches its old limit of 24 inside one request); there it collapsed to a
  range spanning the arena - measured max 22 MB - and cleaned it.  The limit is
  now 160, which is a batch's worth of entries, and the suite shows 0
  collapses and a max range of 2.6 MB instead of 22 MB.
* **Unique pages are ~10× smaller than the cleaned volume** (6.8 MB against
  61.4 MB at 160).  That is not repeated cleaning: it is the *same buffers being
  rewritten*.  The conv pad scratch is 5.8 MB at 160 and every conv node pads
  into it again, so the same 5.8 MB of pages is written ~10 times per frame and
  must be cleaned once per write.  The redundancy is in the writing, not in the
  cleaning.

## 3. What makes the bytes dirty

Per frame at 160, by producer, from the tags added to the mark path:

| producer | MB/frame | what it is |
|---|---:|---|
| launch path, scratch high-water | 59.9 | every request marks `[window_base, scratch_cursor)` |
| ops (`vc4__mark_dirty`) | 3.0 | weight panels, in-place elementwise, transposes |
| uniform shadow | 0.5 | 64 B x nqpu per launch |
| **total** | **63.4** | |

and the CPU-write volume inside that:

| writer | MB/frame | ms/frame |
|---|---:|---:|
| conv pad: zero-fill | 23.6 | 14.39 |
| conv pad: interior copy | 6.3 | 5.73 |
| weight panels, layout copies, elementwise | 3.5 | (inside the op times) |

So of the 63.4 MB marked, 23.6 MB is the pad's zero-fill, 6.3 MB the pad's own
copy, ~3.5 MB other CPU writes - and **~30 MB is the scratch high-water mark
covering bytes the CPU never wrote** (GPU scratch: GEMM panels, im2col
outputs).  Those 30 MB are cleaned too, at ~3 ms per frame, for nothing.

The pad is 73 % zeros: at 160 one pad writes 5.8 MB of which the border is
4.2 MB.  The im2col kernel reads the whole padded image, so the bytes have to
exist - but they do not have to be *stored* per conv node.

## 4. What was changed

**Border-only zeroing in `nhwc_pad_image`** (`infer/src/vc4_nhwc.c`).  The pad
used to `memset` the whole padded image (all images, plus the slack the
im2col over-read needs) and then copy the interior over it.  It now zeroes only
the strips the copy does not write: the top and bottom rows, the left and right
columns of the middle rows, and the slack.  Same bytes afterwards, because the
interior is copied over the same addresses.

| imgsz | pad zero-fill base | after | wall, single frame base | after |
|---:|---:|---:|---:|---:|
| 96 | 19.4 MB / 11.74 ms | 17.1 MB / 10.40 ms | 316.5 ms | 313.1 ms |
| 128 | 21.3 MB / 12.94 ms | 17.2 MB / 10.38 ms | 361.0 ms | 358.3 ms |
| 160 | 23.6 MB / 14.39 ms | 17.3 MB / 10.58 ms | 465.5 ms | 460.1 ms |

Steady state, 20 frames after 3 warm-up, private daemon (protocol 4), one
client, same binary:

| imgsz | path | wall base | wall after | FPS base | FPS after | client CPU base | after |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | sync | 292.3 | 292.3 | 3.42 | 3.42 | 0.110 | 0.110 |
| 96 | pipelined | 287.9 | 287.4 | 3.47 | 3.48 | 0.120 | 0.120 |
| 128 | sync | 329.0 | 327.9 | 3.04 | 3.05 | 0.120 | 0.110 |
| 128 | pipelined | 320.8 | 319.6 | 3.12 | 3.13 | 0.120 | 0.120 |
| 160 | sync | 414.1 | 411.4 | 2.41 | 2.43 | 0.130 | 0.130 |
| 160 | pipelined | 404.2 | 400.9 | 2.47 | 2.49 | 0.140 | 0.130 |

**The delta is 2.7-3.3 ms at 160, 1.2-2.1 ms at 128 and 0.4-0.5 ms at 96**, i.e.
0.8 % of the frame at 160 and 2.47 to 2.49 FPS.  The single-frame numbers move
by more (5.4 ms) because a cold frame pays the pad more often relative to its
own QPU time.

**Dirty-list limit 24 → 256.**  No effect on YOLO (its lists peak at 4), but a
batch request with many entries used to collapse the list to the hull of the
arena and clean up to 22 MB of it; a 64-entry batch marks 129 intervals, the
collapse count in the suite is now 0, and no cleaned range is one nobody
wrote.

## 5. What is left, with sizes

* **The pad itself: 20.1 ms/frame at 160** (14.4 zero + 5.7 copy), and 73 % of
  what it stores is zeros.  A border-aware im2col that never materialises the
  padded image removes most of that; it is a kernel change (the im2col program
  has to handle the edge rows itself) and it is the largest single client item
  after the execution plan.  This branch does not attempt it.
* **The scratch over-mark: ~3 ms/frame at 160.**  30 MB of scratch is marked
  and cleaned although the CPU never wrote it.  The safe form is to let each op
  mark what it writes and drop the high-water mark, which turns a conservative
  invariant into a per-op obligation; the byte-identity gate over six models
  would catch a missed one, and it is the next step rather than this one.
* **Nothing else.**  Marked ≈ cleaned, the list is precise at YOLO shapes, and
  the unique footprint is small only because the pad buffer is rewritten - which
  the cleaning cannot avoid while the writing happens.

## 6. Reproduce

```sh
cd ~/work/vc4-civac
VC4INFER_DAEMON=1 VC4INFER_PROF=1 python3 - <<'PY'   # one frame, all counters
# see scratch/probe.py in the window log; it prints the row of section 2
PY
VC4INFER_DAEMON=1 VC4INFER_PROF=1 \
  python3 tools/cost/breakdown.py --size 160 --runs 6   # per-frame averages
VC4INFER_DAEMON=1 tools/daemonpipe_check.py --suite     # the byte gate
```

`tools/cost/breakdown.py` divides the C counters by its run count and, since
this branch, reads the dirty accounting too; its `pad_zero_s`/`clean_s` fields
read as 0 because it re-reads the struct after a reset - use the one-frame
probe above for those two, and its byte columns are correct.
