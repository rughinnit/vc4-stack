import sys
import kgen
import importlib.util
spec = importlib.util.spec_from_file_location("gemm6", os.path.join(BASE, "asm9_gemm6.py"))
g6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g6)
N = 64
a = g6.build(N)
w = a.w
idx = next(i for i, d in enumerate(w) if d.get('note','').startswith('prime r0'))
print("prime r0 copy at", idx)
probe = [
    dict(add=dict(add_op=21, add_out=0, ws=0, add_a=(0,39), add_b=(0,39)), sig=1, ws=0, mul=None, note='PROBE A0=acc0'),
    dict(add=None, mul=None, sig=1, ws=0, note=''),
    dict(add=None, mul=None, sig=1, ws=0, note=''),
    dict(branch=15, target='store', sig=1, ws=0, mul=None, add=None, note='PROBE jump'),
    dict(add=None, mul=None, sig=1, ws=0, note=''),
    dict(add=None, mul=None, sig=1, ws=0, note=''),
    dict(add=None, mul=None, sig=1, ws=0, note=''),
]
w[idx+1:idx+1] = probe
a.labels = {k: (v + 7 if v > idx else v) for k, v in a.labels.items()}
a.write(os.path.join(os.environ.get('TMPDIR', '/tmp'), 'g6probe_64.bin'))
print("wrote", len(w), "words")
