# vc4-smi

The `nvidia-smi` analogue: one shot, text, scriptable.

```
$ vc4-smi
vc4-smi 1.0.0  --  VideoCore IV (V3D)  2026-11-20 08:02:41 WIB
GPU  12 QPUs / 3 slices   backend vc4d

STATE   UTIL   CORE         V3D          TEMP    THROTTLE   CMA USED     QPU
ok      0%     400.000 MHz  300.000 MHz  46.7 °C 0x0 (none) 200/256 MiB  idle

util:        0% (0% busy over a 200 ms window)
errstat:     0x00001000
undervoltage: 0 (ok)
clients:     none

GPU is healthy: no throttle, no undervoltage, no queue or ERRSTAT errors.
```

`--json` prints one object for monitoring. `NO_COLOR` is irrelevant to the
output (it is always plain text, kept that way on purpose); the wrapper sets it
for `vc4top` internally.

## What it reads

- Everything textual from `vc4top --once` with `NO_COLOR=1` (temperature,
  clocks, throttle, CMA, ERRSTAT, QPU count, scheduler queue).
- The VCD-idle bit of `ERRSTAT` in `/sys/kernel/vc4mon/regs`, sampled ten
  times over 200 ms, for the utilisation column.
- `/proc/*/fd` to list the processes attached to `/run/vc4d.sock`
  (only own processes are visible to a non-root caller, which is the normal
  case: that is who is using the GPU).
- It never opens `/dev/mem` and never writes a register.

## Exit status

0 when the GPU is healthy; 1 when a current throttle bit is set, the firmware
reports undervoltage, the scheduler queue reports an error, or `ERRSTAT` has
bits other than VCD-idle.  Historical throttle bits (16-19) are reported in
the THROTTLE column but do not by themselves fail the check.

This is an observation tool: it changes no clock, no governor and no voltage.
