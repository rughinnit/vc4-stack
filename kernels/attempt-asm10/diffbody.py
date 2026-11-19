import sys, os, struct
sys.path.insert(0, '/opt/vc4-asm/tools')
sys.path.insert(0, '/opt/vc4-asm/asm3')
sys.path.insert(0, '/opt/vc4-asm/asm9')
import kgen
mine = kgen.build_gemm10(64, 64, 64, 1)
import importlib.util
spec = importlib.util.spec_from_file_location("gemm6", "/opt/vc4-asm/asm9_gemm6.py")
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
