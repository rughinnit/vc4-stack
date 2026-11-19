# Board performance variance: check the core clock before comparing runs

The SoC core clock (which drives L2 and the memory path) is not fixed at 400 MHz on
this board. Under sustained 12-QPU load it was observed to drop to 250 MHz while
V3D stayed at 300 MHz and get_throttled reported 0x0. The effect is a ~1.6-1.7x
wall-time increase in memory-bound kernels with *identical* cycle counts - exactly
the discrepancy seen between runs recorded at different times.

Rule: every comparison must record `vcgencmd measure_clock v3d` and
`vcgencmd measure_clock core`, plus the cycle counters, not just wall time.
If core reads 250, idle until it returns to 400 before measuring.
