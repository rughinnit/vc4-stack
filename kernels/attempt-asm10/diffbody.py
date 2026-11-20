import os
import sys
import struct

# These comparison scripts were written against the asm9/asm3 generator trees,
# which are not part of this repository.  Point VC4_ASM_DIR at a checkout that
# has them (tools/, asm3/, asm9/, asm9_gemm6.py) to run the comparison.
BASE = os.environ.get("VC4_ASM_DIR")
if not BASE:
    sys.exit("set VC4_ASM_DIR to the asm tree this script compares against")
for _sub in ("", "tools", "asm3", "asm9"):
    sys.path.insert(0, os.path.join(BASE, _sub) if _sub else BASE)
import kgen
mine = kgen.build_gemm10(64, 64, 64, 1)
import importlib.util
spec = importlib.util.spec_from_file_location("gemm6", os.path.join(BASE, "asm9_gemm6.py"))
g6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g6)
ref = g6.build(64)
mw = mine.words(); rw = ref.words()
ml = mine.labels['kloop']; rl = ref.labels['kloop']
print("my kloop at", ml, "gemm6 kloop at", rl, "len", len(mw), len(rw))
# gemm6 body: kloop .. its branch + 3 delay slots; mine likewise
def body(words, start):
    i = start
    while True:
        w = words[i]
        if ((w >> 60) & 0xF) == 15:      # SIG_BRANCH
            return words[start+1:i+4]    # skip prologue nop, plus 3 delays
        i += 1
mb = body(mw, ml); rb = body(rw, rl)
print("my body words", len(mb), "gemm6 body", len(rb))
n = min(len(mb), len(rb)); bad = 0
for i in range(n):
    if mb[i] != rb[i]:
        bad += 1
        if bad <= 8:
            print("diff at body word %d:" % i)
            print("   mine  ", kgen.disasm(mb[i]))
            print("   gemm6 ", kgen.disasm(rb[i]))
print("total differing words:", bad, "of", n)
