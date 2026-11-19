# vc4mon — read-only V3D register view (vc4top without root)

Source: `kernel/vc4mon.c` in the vc4top repo, installed to
`/usr/share/doc/vc4top/kernel` by the 1.0.3 package. Loaded at boot via
`/etc/modules-load.d/vc4mon.conf`. Out-of-tree module (taints the kernel; rebuild after a
kernel upgrade).

## Interface

`/sys/kernel/vc4mon/` — attributes mode 0444, no store path:
- `ident`  — IDENT0/1/2
- `regs`   — 78 lines, `%05x %08x`, all documented V3D registers incl. PCTR/PCTRS
- `perf`   — PCTR0..15 / PCTRS0..15, observation only

The module finds `brcm,vc4-v3d` through the device tree, resolves the resource with
`of_address_to_resource()` and `ioremap`s it. It does NOT register a platform driver, does NOT
`request_mem_region`, and does NOT bind/unbind `3fc00000.v3d`. Load performs zero MMIO: no
counter enable/clear, no reset, no clock/power change.

## Why ioread32 and not memcpy_fromio

`memcpy_fromio()` is wrong for this block: `lib/iomap.c` implements a 4-byte copy as four
`readb()` calls, and V3D returns valid data only for 32-bit accesses. Measured: a 4-byte
`memcpy_fromio()` returned only the low byte (`0x02443356` -> `0x00000056`). Bulk
`memcpy_fromio()` would issue 64-bit loads, riskier still on a 32-bit register block.
The module uses `ioread32()` per register, which is what `drm/vc4` itself does.

## vc4top behaviour

- root: `/dev/mem` path, unchanged.
- unprivileged: reads `vc4mon`; never opens `/dev/mem`; the register write helper is a no-op.
- Performance counters stay root-only, because sampling them writes PCTRE/PCTRC.

## Verification

- 78/78 registers identical to `drm/vc4`'s own `/sys/kernel/debug/dri/0/v3d_regs`, back-to-back.
- Root vs unprivileged `--once`: 20/20 QPU fields matched across pairs; the only differing field
  was ERRSTAT bit 12 (live VCD-idle), which also differs between two root runs.
- `strace`: only `openat("/sys/kernel/vc4mon/regs", O_RDONLY)`; 20 `/proc/<pid>/fd` samples of a
  running `--poll` had zero `/dev/mem`.
- No side effects: unprivileged `--frame` left PCTRE at 0x80000003.
- PTY: 0 scrolls at 80x24, 140x40, register view and help; `rows-1` newlines per frame.
- `rmmod` removes the directory and unmaps; `modprobe` restores it; reboot auto-load at 9.5 s,
  before `vc4_v3d` binds.

## Package

vc4top 1.0.3, `vc4top_1.0.3_arm64.deb`, sha256 `d6548b0c...fcd866`, signed apt repo at
https://rughinnit.github.io/vc4top, installed on the Pi.
