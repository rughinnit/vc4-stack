#!/usr/bin/env python3
"""asm6 driver: one launch, full element verification.  NO flock inside -- the
caller must hold /tmp/vc4-gpu.lock around the whole sequence (single lock
level).

  python3 g6.py <bin> <nqpu> <N> [tag] [--nopack|--cpu-panel] [--noverify]

Prints one line: pack_ms gemm_ms total_ms gflops_gemm gflops_total cyc/k and
the check_gemm / check_panel8 mismatch counts.
"""
import os
import re
import subprocess
import sys

ASM = os.path.dirname(os.path.abspath(__file__))
QRUN = os.path.join(ASM, "qrun_gemm")
STATUS = "/home/raspi/vc4cl/tools/read_v3d_status"
CLK = 300e6


def health():
    r = subprocess.run(["sudo", "-n", STATUS], capture_output=True, text=True)
    m = re.search(r"SRQCS\s*=\s*(0x[0-9a-f]+)", r.stdout)
    return "STATE: LIVE" in r.stdout, (m.group(1) if m else "?")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    binp, nqpu, N = args[0], int(args[1]), int(args[2])
    tag = args[3] if len(args) > 3 else "g6"
    a = "/tmp/%s_a.bin" % tag
    b = "/tmp/%s_b.bin" % tag
    c = "/tmp/%s_c.bin" % tag
    p = "/tmp/%s_p.bin" % tag
    env = ["QRUN_TMO_MS=60000", "QRUN_GEMM=%d" % N,
           "QRUN_SAVE_A=" + a, "QRUN_SAVE_B=" + b, "QRUN_SAVE_C=" + c]
    if "--cpu-panel" in flags:
        env += ["QRUN_CPUPANEL8=1", "QRUN_SAVE_PANEL=" + p]
        ppath = None
    elif "--nopack" in flags:
        ppath = None
    else:
        ppath = os.environ.get("PACK_BIN") or os.path.join(ASM, "pack8_%d_vp1.bin" % N)
        env += ["QRUN_PACK_BIN=" + ppath,
                "QRUN_PACK_NQPU=%d" % int(os.environ.get("PACK_QPUS", "4")),
                "QRUN_PACK_ROWS=8", "QRUN_SAVE_PANEL=" + p]
    keep = [k + "=" + v for k, v in os.environ.items()
            if k.startswith(("QRUN_", "SKIP", "KPN", "DUMPREG", "DBGSTORE"))]
    for kv in keep:
        if kv.split("=")[0] in [e.split("=")[0] for e in env]:
            continue
        env.append(kv)
    env = [e for e in env if not e.startswith(("QRUN_PACK_BIN=",))or ppath]
    cmd = ["timeout", "-k", "3", "300", "sudo", "-n", "env"] + env + \
          [QRUN, binp, str(nqpu), "1", "0", "0x10000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    live, srq = health()
    body = "\n".join(l for l in out.splitlines()
                     if not l.startswith(("OUT", "AFTER", "arena:")))
    mp = re.search(r"PACK wall_ns=(\d+)", out)
    mg = re.search(r"DONE wall_ns=(\d+) exec_cycles=(\d+) instr_hits=(\d+) v3d_errors=(0x[0-9a-f]+)", out)
    if r.returncode != 0 or not mg or "TIMEOUT" in out or not live:
        print(body)
        print("ABORT rc=%d live=%s SRQCS=%s" % (r.returncode, live, srq))
        return 2
    pack_ns = int(mp.group(1)) if mp else 0
    gemm_ns, cyc, hits, errs = int(mg.group(1)), int(mg.group(2)), int(mg.group(3)), mg.group(4)
    nrb, ncb = N // 8, N // 64
    ntiles = nrb * ncb
    qiters = ((ntiles + nqpu - 1) // nqpu) * min(N // 2, int(os.environ.get("KPN", N // 2)))
    gfl = 2.0 * N ** 3 / (gemm_ns * 1e-9) / 1e9
    tot = gemm_ns + pack_ns
    gflt = 2.0 * N ** 3 / (tot * 1e-9) / 1e9
    mm = pm = -1
    nz = -1
    pmatch = None
    if "--noverify" not in flags:
        chk = subprocess.run([os.path.join(ASM, "check_gemm"), a, b, c, str(N)],
                             capture_output=True, text=True)
        mm = int(re.search(r"mismatches=(\d+)/", chk.stdout).group(1))
        nz = int(re.search(r"nonzero=(\d+)", chk.stdout).group(1))
        pmatch = re.search(r"maxabs=(\S+)", chk.stdout)
        if mm:
            print(chk.stdout.strip()[:600])
    if ppath or "--cpu-panel" in flags:
        chk = subprocess.run([os.path.join(ASM, "check_panel8"), a, p, str(N), "8"],
                             capture_output=True, text=True)
        pm = int(re.search(r"mismatches=(\d+)/", chk.stdout).group(1))
    print("N=%d q=%d | pack %8.4f + gemm %8.4f = %8.4f ms | gemm %6.3f tot %6.3f "
          "GFLOP/s | cyc/k %.1f | Cmm=%d Pmm=%d nonzero=%d maxabs=%s | cycles=%d "
          "hits=%d errs=%s %s"
          % (N, nqpu, pack_ns / 1e6, gemm_ns / 1e6, tot / 1e6, gfl, gflt,
             cyc / nqpu / max(qiters, 1), mm, pm, nz,
             pmatch.group(1) if pmatch else "-", cyc, hits, errs, srq))
    return 0 if (mm == 0 and pm in (0, -1)) else 1


if __name__ == "__main__":
    sys.exit(main())
