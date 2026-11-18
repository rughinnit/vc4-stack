#!/usr/bin/env python3
"""asm4 driver: pack (8-row panel) + 8x64-tile GEMM in ONE process, verified.

  python3 run4.py N nqpu [--reps K] [--pack-qpus P] [--no-pack] [--noverify]

Each rep: launch the packer, then the GEMM, element-verify the packed panel
bit-exactly against A and every C element against the double-precision oracle.
Every launch is serialised with flock and health-checked afterwards.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ASM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASM)
import gemm4
import pack8

QRUN = os.path.join(ASM, "qrun_gemm")
STATUS = "/home/raspi/vc4cl/tools/read_v3d_status"
CLK = 300e6


def health():
    r = subprocess.run(["sudo", "-n", STATUS], capture_output=True, text=True)
    m = re.search(r"SRQCS\s*=\s*(0x[0-9a-f]+)", r.stdout)
    return "STATE: LIVE" in r.stdout, (m.group(1) if m else "?")


def run_pair(gpath, nqpu, ppath, pnq, N, a, b, c, panel, tmo=240):
    env = ["QRUN_TMO_MS=60000", "QRUN_GEMM=%d" % N, "QRUN_SAVE_A=" + a,
           "QRUN_SAVE_B=" + b, "QRUN_SAVE_C=" + c]
    if ppath:
        env += ["QRUN_PACK_BIN=" + ppath, "QRUN_PACK_NQPU=%d" % pnq,
                "QRUN_PACK_ROWS=8", "QRUN_SAVE_PANEL=" + panel]
    cmd = ["flock", "-w", "2400", "/tmp/vc4-gpu.lock", "timeout", "-k", "3", str(tmo),
           "sudo", "-n", "env"] + env + [QRUN, gpath, str(nqpu), "1", "0", "0x10000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    live, srq = health()
    if r.returncode != 0 or "TIMEOUT" in out or not live:
        print("ABORT rc=%d live=%s SRQCS=%s" % (r.returncode, live, srq))
        print("\n".join(l for l in out.splitlines() if not l.startswith(("OUT", "AFTER"))))
        sys.exit(2)
    mp = re.search(r"PACK wall_ns=(\d+)", out)
    mg = re.search(r"DONE wall_ns=(\d+) exec_cycles=(\d+)", out)
    if not mg:
        print(out)
        sys.exit(2)
    return (int(mp.group(1)) if mp else 0), int(mg.group(1)), int(mg.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int)
    ap.add_argument("nqpu", type=int)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--pack-qpus", type=int, default=4)
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--noverify", action="store_true")
    args = ap.parse_args()
    N, nqpu = args.N, args.nqpu
    a, b, c = ["/tmp/r4_%d_%s.bin" % (N, x) for x in "abc"]
    panel = "/tmp/r4_%d_panel.bin" % N

    gpath = os.path.join(ASM, "gemm4_%d.bin" % N)
    gemm4.build(N).write(gpath)
    ppath = None
    if not args.no_pack:
        ppath = os.path.join(ASM, "pack8_%d_vp1.bin" % N)
        pack8.build(N).write(ppath)

    nrb = N // 8
    ncb = N // 64
    ntiles = nrb * ncb
    iters = ((ntiles + nqpu - 1) // nqpu) * (N // 2)
    rows = []
    for rep in range(args.reps):
        pns, gns, cyc = run_pair(gpath, nqpu, ppath, args.pack_qpus, N, a, b, c, panel)
        pack_ms, gemm_ms = pns / 1e6, gns / 1e6
        tot = pack_ms + gemm_ms
        gflop = 2.0 * N ** 3 / (gemm_ms * 1e-3) / 1e9
        gflop_tot = 2.0 * N ** 3 / (tot * 1e-3) / 1e9
        mm = pm = -1
        if not args.noverify:
            chk = subprocess.run([os.path.join(ASM, "check_gemm"), a, b, c, str(N)],
                                 capture_output=True, text=True)
            mm = int(re.search(r"mismatches=(\d+)/", chk.stdout).group(1))
            if mm:
                print(chk.stdout.strip()[:2000])
            if ppath:
                chk = subprocess.run([os.path.join(ASM, "check_panel8"), a, panel,
                                      str(N), "8"], capture_output=True, text=True)
                pm = int(re.search(r"mismatches=(\d+)/", chk.stdout).group(1))
        row = dict(rep=rep, N=N, nqpu=nqpu, pack_qpus=args.pack_qpus,
                   pack_ms=pack_ms, gemm_ms=gemm_ms, total_ms=tot,
                   gflops=gflop, gflops_total=gflop_tot,
                   cyc_k=cyc / nqpu / iters, wall_cyc_k=gemm_ms * 1e-3 * CLK / iters,
                   mismatches=mm, panel_mismatches=pm)
        rows.append(row)
        print("rep%d N=%d q=%d pq=%d | pack %7.4f + gemm %7.4f = %7.4f ms"
              " | gemm %6.3f GFLOP/s total %6.3f | cyc/k %6.1f/%6.1f | Cmm=%d Pmm=%d"
              % (rep, N, nqpu, args.pack_qpus, pack_ms, gemm_ms, tot, gflop,
                 gflop_tot, row["cyc_k"], row["wall_cyc_k"], mm, pm))
    clean = sum(1 for r in rows if r["mismatches"] == 0 and r["panel_mismatches"] in (0, -1))
    print("SUMMARY N=%d q=%d reps=%d clean=%d/%d pack_med=%.4f gemm_best=%.4f"
          " total_best=%.4f gflops_total_best=%.3f gflops_gemm_best=%.3f"
          % (N, nqpu, args.reps, clean, args.reps,
             sorted(r["pack_ms"] for r in rows)[len(rows) // 2],
             min(r["gemm_ms"] for r in rows), min(r["total_ms"] for r in rows),
             max(r["gflops_total"] for r in rows), max(r["gflops"] for r in rows)))
    with open(os.path.join(ASM, "runs4.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 0 if clean == args.reps else 1


if __name__ == "__main__":
    sys.exit(main())
