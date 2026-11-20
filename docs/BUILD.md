# Building on the 1 GB Pi: jobs, ccache and what is actually safe

Board: Raspberry Pi 3B, 4 cores, 905 MiB usable RAM, 2 GiB swapfile on SD.
All numbers below were taken on one boot, stock clocks (core 400 MHz, V3D
300 MHz, `throttled=0x0`), with no foreign compile running, using
`tools/build-fast.sh`'s sampler (it counts only the build's own process tree,
so an unrelated job on the shared board cannot inflate the peak-RSS figure).

## Short version

| build | safe jobs | why | speed-up |
|---|---|---|---|
| engine `libvc4infer` (plain C, 9 TUs + 11 test links) | **-j4** | 171 MiB peak tree RSS, 0 swap | 1.39x vs -j2 |
| VC4C compiler (C++ + LLVM, 101 TUs) | **-j2, -j1 if shared** | one TU is ~400 MiB; -j4 swaps 800 MiB *and is slower* | ccache is the real win |

`-j4` is a CPU-count choice; on this board the binding constraint is RAM, and
for VC4C `-j4` is actively counter-productive.  With ccache warm, a rebuild
after touching a source file drops from 14 s to 11 s (engine) and from 114 s to
8.5 s (VC4C).

## Engine: `-j4` vs `-j2` (same boot, quiet board, warm page cache)

| case | wall | peak tree RSS | swap growth |
|---|---|---|---|
| clean, -j4 | **15.1 s** | 171 MiB | 0 |
| clean, -j2 | 21.0 s | 96 MiB | 0 |
| clean, -j4 (first build after boot, cold page cache) | 21.5-33.2 s | 171 MiB | 0 |
| incremental (one TU touched), -j4 | 14.0 s | 170 MiB | 0 |
| incremental, -j2 | 16.1 s | 111 MiB | 0 |

The first build after a reboot reads the whole tree from the SD card and is
~40 % slower than the same build once the page cache is warm; always compare
like with like.  `MinAvailable` never fell below 570 MiB and no OOM occurred.
A C translation unit costs ~45 MiB here, so `-j4` has ~4x headroom.

## VC4C: the compiler is a memory hog

| case (4 heavy TUs touched) | wall | peak tree RSS | swap growth | min MemAvailable |
|---|---|---|---|---|
| -j2, no ccache | **113.9 s** | 609 MiB | 319 MiB | 89 MiB |
| -j4, no ccache | 132.4 s | 673 MiB | **800 MiB** | 54 MiB |
| -j4, ccache populate | 123.4 s | 707 MiB | 600 MiB | 47 MiB |
| -j4, ccache warm | 10.0 s | 159 MiB | 0 | 521 MiB |
| -j2, ccache warm | **8.5 s** | 159 MiB | 0 | 522 MiB |

Single TU cost, measured at -j1:

| TU | wall | peak RSS |
|---|---|---|
| `src/optimization/Memory.cpp` | 139.2 s | 398 MiB |
| `src/Compiler.cpp` | 34.4 s | 364 MiB |

That is the whole story: four 400 MiB compilers do not fit in 905 MiB.  `-j4`
on VC4C runs out of RAM, pushes 800 MiB into swap and ends up **16 % slower**
than `-j2`; the extra jobs only add page-fault churn.  Nothing was OOM-killed
in these runs, but MemAvailable reached 47-54 MiB, which is the territory where
the OOM killer starts picking victims on a shared board.

The same applies to any other LLVM-weight build on this machine.  During a
sustained mixed-load run, a concurrent job's `pip install -e .` of `torch_vc4`
(one `cc1plus` at 490 MiB) pushed swap to 731 MiB and MemAvailable to 60 MiB
while the engine was being rebuilt at -j4.  Two big C++ builds never belong on
this board at the same time.

## ccache

Installed with `sudo apt install -y ccache` (4.11.2), default cache dir
`/home/raspi/.cache/ccache`, default limit 5 GiB.

* engine, configured by `CC="ccache gcc"` (it is a plain Makefile, so there is
  no CMake launcher variable to set): clean -j4 15.1 s -> 10.7 s warm;
  incremental 14.0 s -> 10.9 s.  Hit rate after a few builds: 93 %.
* VC4C, configured in the CMake cache with
  `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache`:
  four heavy TUs 113.9 s -> 8.5 s warm (-j2).  The remaining time is the link.
* **Cache size after the first builds: 8.3 MB.**  The objects are small (the
  whole VC4C build is a few MB), so the 5 GiB default is pointless on a board
  with an SD-card root.  A 1 GiB limit is more than enough:
  `ccache -M 1G`.

`ccache` never changes the recorded binaries: the 4-TU rebuild with ccache warm
produced the same libraries, and the VC4CL/engine measurements in
`CLOCK-VARIANCE.md` were taken before ccache was installed.

## The compiler is single-threaded per kernel

`vc4c` compiles one kernel on one core.  `src/Compiler.cpp` and
`src/optimization/Optimizer.cpp` use a `ThreadPool` over the *list of kernels*
in a module (`ThreadPool::scheduleAll<Method*>("CodeGenerator", kernels, ...)`),
so a `.cl` file that defines several kernels is compiled in parallel, but a
single-kernel file gets no speed-up from `-j4` or from more cores.

Measured, `vc4c --bin` on the benchmark kernels, quiet board:

| kernel | wall | user+sys |
|---|---|---|
| `m19/src/r16g_256.cl` (one kernel) | 3.75 s | 3.8 s |
| `m19/src/r16g_128.cl` (one kernel) | 3.74 s | 3.8 s |

Wall time equals CPU time to within 2 %, i.e. 100 % of one core.  This is why
build parallelism helps *builds* (many TUs, many kernels) but does nothing for
the latency of compiling one kernel, which is what VC4CL pays at first launch.

## `tools/build-fast.sh`

```
build-fast.sh [DIR] [--clean] [-jN] [-- <extra build args>]
```

* picks `-j$(nproc)` when the memory fits, otherwise the largest job count that
  fits in `MemAvailable` minus a 200 MiB reserve;
* assumes 350 MiB per job for a C++/CMake project and 60 MiB for a pure C one
  (`BUILD_FAST_PER_JOB` overrides);
* sets the ccache launchers when ccache is installed (`BUILD_FAST_NO_CCACHE=1`
  to opt out), including re-configuring an existing CMake cache once;
* prints the chosen jobs, the memory reading, elapsed time, swap delta and the
  ccache hit/miss counts.

Examples:

```sh
tools/build-fast.sh /home/raspi/work/engine-daemon --clean
BUILD_FAST_TARGET=VC4CC tools/build-fast.sh /home/raspi/work/combined/build
```

Example output:

```
build-fast: /home/raspi/work/engine-daemon
build-fast: mode=make jobs=4/4 mem_avail=650MiB reserve=200MiB mem_per_job=60MiB
build-fast: ccache=/usr/bin/ccache (CC/CXX launcher)
build-fast: rc=0 elapsed=10.7s mem_avail=605MiB swap_used=0MiB_delta
build-fast: ccache hits=50/54(92.59%) misses=4/54(7.41%)
```

## Recommendation

* Engine: `-j4` with ccache.  Safe, 1.4x faster than `-j2`, 1.4x faster again
  with a warm cache.
* VC4C: `ccache` mandatory, then `-j2` on a quiet board, `-j1` if anything else
  heavy is running.  Never `-j4`.
* Shared board: check `MemAvailable` and `pgrep cc1plus` before starting a big
  build; two LLVM-weight builds will swap no matter what `-j` says.

## Scratch space: `/tmp` was RAM (`2026-11-20`)

Raspberry Pi OS ships `tmpfs` for `/tmp` (452 MB of the board's 905 MB), so every
worktree, kernel cache, venv or build log placed there was **RAM**, not SD card.
On a 1 GB board that is not a detail: it is half the memory, and tmpfs pages
cannot be reclaimed to disk, so the kernel has no choice but to evict page cache
and swap.

`tmp.mount` is now masked and `/tmp` is a directory on the root filesystem:

```sh
sudo systemctl mask tmp.mount     # symlink to /dev/null, survives reboot
sudo reboot
findmnt /tmp                      # prints nothing: /tmp is on / now
df -hT /tmp                       # /dev/mmcblk0p2  ext4  58G
```

Use `$HOME/scratch/{tmp,refs,evidence,logs}` for anything large, and export
`TMPDIR` (see `~/scratch/env.sh`); `mktemp -d` then follows it.  Lock files and
sockets (`/tmp/vc4-gpu.lock`, `/run/vc4d.sock`) are fine either way.

**Measured after the change** (YOLOv8n 160, stock clocks, `vc4infer util`):

| | before | after |
|---|---:|---:|
| wall / FPS | 539-543 ms / 1.84 | **478.7 ms / 2.09** |
| GPU (pack+kernel) | 321 ms | **259 ms** |
| QPU duty | 59 % | 54 % |
| MemAvailable at run time | ~189 MB free | **647 MB free** |

That is a before/after observation, not a controlled A/B - the board cannot be
returned to the old memory layout without re-enabling the tmpfs - but the
mechanism is ordinary memory pressure, and the run-to-run variance across 3 reps
was small (478.7 ms mean).

Still tmpfs and deliberately so: `/run` (200 MB, sockets/state) and `/dev/shm`
(500 MB, POSIX shared memory).  Do not put model data or build output there
either.
