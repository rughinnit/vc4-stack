#!/usr/bin/env python3
"""Decode the MLIR bytecode that jaxlib hands to PJRT_Client_Compile.

Prototype for the C reader in pjrt/src.  Reads the vhlo module enough to
print ops, operand indices, result types and the few attributes the QPU
executor needs.
"""
import sys, struct

# ---------------------------------------------------------------- primitives
class R:
    def __init__(self, buf, pos=0):
        self.b = buf
        self.p = pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def bytes(self, n):
        v = self.b[self.p:self.p+n]; self.p += n; return v
    def varint(self):
        b = self.byte()
        if b & 1:
            return b >> 1
        if b == 0:
            v = struct.unpack('<Q', self.bytes(8))[0]
            return v
        tz = 0
        while not (b >> tz) & 1:
            tz += 1
        raw = bytes([b]) + self.bytes(tz)
        v = struct.unpack('<Q', raw + b'\0' * (8 - len(raw)))[0]
        return v >> (tz + 1)
    def svarint(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)
    def vflag(self):
        v = self.varint()
        return v >> 1, bool(v & 1)
    def empty(self):
        return self.p >= len(self.b)

MAGIC = b'ML\xefR'

class Module:
    def __init__(self, data):
        self.data = data
        r = R(data)
        assert r.bytes(4) == MAGIC, "bad magic"
        self.version = r.varint()
        end = data.index(b'\0', r.p)
        self.producer = data[r.p:end].decode()
        r.p = end + 1
        self.sections = {}
        base = 0
        while not r.empty():
            idbyte = r.byte()
            sid = idbyte & 0x7f
            length = r.varint()
            if idbyte & 0x80:
                align = r.varint()
                while (r.p - base) % align:
                    r.p += 1
            self.sections[sid] = r.bytes(length)
        self.strings = self._strings()
        self.dialects = []
        self.opnames = []
        self._dialects()
        self.attr_data, self.attr_meta, self.types_data, self.type_meta = \
            self._attrtype()
        self.attrs = [None] * len(self.attr_data)
        self.types = [None] * len(self.types_data)
        self.iresolving = set()
        self.props = self._properties()
        self.opt_attr_cache = {}

    # ---- sections
    def _str(self, r):
        idx = r.varint()
        return self.strings[idx]
    def _strings(self):
        r = R(self.sections[0])
        n = r.varint()
        out = [None] * n
        end = len(self.sections[0])
        for i in range(n - 1, -1, -1):
            size = r.varint()
            start = end - size
            out[i] = self.sections[0][start:end-1].decode()
            end = start
        return out
    def _dialects(self):
        r = R(self.sections[1])
        n = r.varint()
        for _ in range(n):
            idx, hasver = r.vflag()
            self.dialects.append(self.strings[idx])
            if hasver:
                idb = r.byte(); sid = idb & 0x7f
                ln = r.varint()
                if idb & 0x80:
                    al = r.varint()
                    # embedded section alignment, ignore for small versions
                    while r.p % al:
                        r.p += 1
                r.bytes(ln)
        if self.version >= 4:
            self.num_ops = r.varint()
        while not r.empty():
            didx = r.varint()
            count = r.varint()
            for _ in range(count):
                idx, registered = r.vflag()
                self.opnames.append((self.dialects[didx], self.strings[idx], registered))
    def _attrtype(self):
        r = R(self.sections[3])
        na = r.varint(); nt = r.varint()
        metas = []
        for total in (na, nt):
            i = 0
            while i < total:
                didx = r.varint()
                cnt = r.varint()
                for _ in range(cnt):
                    size, custom = r.vflag()
                    metas.append((self.dialects[didx], size, custom))
                    i += 1
        sec = self.sections[2]
        off = 0
        chunks = []
        for dialect, size, custom in metas:
            chunks.append((dialect, sec[off:off+size], custom))
            off += size
        assert len(chunks) == na + nt
        return chunks[:na], metas[:na], chunks[na:], metas[na:]

    def _properties(self):
        raw = self.sections.get(8, b"")
        out = []
        if not raw:
            return out
        r = R(raw)
        n = r.varint()
        for _ in range(n):
            sz = r.varint()
            out.append(r.bytes(sz))
        return out

    # ---- resolution
    def attr(self, i):
        if i >= len(self.attrs):
            raise IndexError(f"attr {i} >= {len(self.attrs)}")
        if self.attrs[i] is None:
            dialect, data, custom = self.attr_data[i]
            r = R(data)
            if custom:
                if dialect == 'builtin':
                    self.attrs[i] = ('builtin', self.builtin_attr(r))
                elif dialect == 'vhlo':
                    self.attrs[i] = ('vhlo', self.vhlo_attr(r))
                else:
                    raise ValueError(f"attr dialect {dialect}")
            else:
                # textual assembly entry
                self.attrs[i] = ('asm', data)
        return self.attrs[i]
    def type(self, i):
        if i >= len(self.types):
            raise IndexError(f"type {i} >= {len(self.types)}")
        if self.types[i] is None:
            dialect, data, custom = self.types_data[i]
            r = R(data)
            if custom:
                if dialect == 'builtin':
                    self.types[i] = ('builtin', self.builtin_type(r))
                elif dialect == 'vhlo':
                    self.types[i] = ('vhlo', self.vhlo_type(r))
                else:
                    raise ValueError(f"type dialect {dialect}")
            else:
                self.types[i] = ('asm', data)
        return self.types[i]

    def tidx(self, r): return self.type(r.varint())
    def aidx(self, r): return self.attr(r.varint())

    def opt_attr(self, r):
        idx, present = r.vflag()
        return self.attr(idx) if present else None
    def decode_props(self, op, pid):
        r = R(self.props[pid])
        d = {}
        if op == 'builtin.module':
            d['sym_name'] = self.opt_attr(r)
            d['sym_visibility'] = self.opt_attr(r)
        elif op == 'vhlo.func_v1':
            for k in ('arg_attrs', 'function_type', 'res_attrs', 'sym_name',
                      'sym_visibility'):
                d[k] = self.aidx(r)
        elif op == 'vhlo.constant_v1':
            d['value'] = self.aidx(r)
        elif op == 'vhlo.iota_v1':
            d['iota_dimension'] = self.aidx(r)
        elif op == 'vhlo.broadcast_in_dim_v1':
            d['broadcast_dimensions'] = self.aidx(r)
        elif op == 'vhlo.reduce_v1':
            d['dimensions'] = self.aidx(r)
        elif op == 'vhlo.transpose_v1':
            d['permutation'] = self.aidx(r)
        elif op == 'vhlo.slice_v1':
            for k in ('start_indices', 'limit_indices', 'strides'):
                d[k] = self.aidx(r)
        elif op == 'vhlo.dot_general_v2':
            for k in ('accumulation_type', 'allow_imprecise_accumulation',
                      'lhs_batching_dimensions', 'lhs_component_count',
                      'lhs_contracting_dimensions', 'lhs_precision_type',
                      'num_primitive_operations', 'precision_config',
                      'rhs_batching_dimensions', 'rhs_component_count',
                      'rhs_contracting_dimensions', 'rhs_precision_type'):
                d[k] = self.aidx(r)
        else:
            d['__raw__'] = self.props[pid].hex()
        if r.p != len(r.b):
            d['__trailing__'] = r.b[r.p:].hex()
        return d

    # ---- builtin attributes
    def _apint(self, r, width):
        if width <= 8:
            return r.byte()
        if width <= 64:
            return r.svarint()
        n = r.varint()
        return [r.svarint() for _ in range(n)]
    def builtin_attr(self, r):
        code = r.varint()
        if code == 0:  # ArrayAttr
            n = r.varint(); return ('array', [self.aidx(r) for _ in range(n)])
        if code == 1:  # DictionaryAttr
            n = r.varint()
            return ('dict', [(self.aidx(r), self.aidx(r)) for _ in range(n)])
        if code == 2:  # StringAttr
            return ('str', self._str(r))
        if code == 3:  # StringAttrWithType
            return ('strt', self._str(r), self.tidx(r))
        if code == 4:  # FlatSymbolRefAttr
            return ('symref', self.aidx(r))
        if code == 5:  # SymbolRefAttr
            root = self.aidx(r); n = r.varint()
            return ('symref2', root, [self.aidx(r) for _ in range(n)])
        if code == 6:  # TypeAttr
            return ('typeattr', self.tidx(r))
        if code == 7:  # UnitAttr
            return ('unit',)
        if code == 8:  # IntegerAttr
            t = self.tidx(r)
            w = t[1]['width'] if t[0] == 'builtin' and t[1]['kind'] == 'int' else 64
            return ('int', t, self._apint(r, w))
        if code == 9:  # FloatAttr
            t = self.tidx(r)
            w = t[1].get('bits', 32) if t[0] == 'builtin' else 32
            return ('float', t, self._apint(r, w))
        if code == 10:  # CallSiteLoc
            return ('callsite', self.aidx(r), self.aidx(r))
        if code == 11:  # FileLineColLoc
            fn = self.aidx(r); return ('fileloc', fn, r.varint(), r.varint())
        if code == 12:  # FusedLoc
            n = r.varint(); return ('fused', [self.aidx(r) for _ in range(n)])
        if code == 15:  # UnknownLoc
            return ('unknownloc',)
        if code == 17:  # DenseArrayAttr
            et = self.tidx(r); size = r.varint(); blob = r.bytes(r.varint())
            return ('densearray', et, size, blob)
        if code == 18:  # DenseTypedElementsAttr
            t = self.tidx(r); blob = r.bytes(r.varint())
            return ('dense', t, blob)
        if code == 22:  # FileLineColRange
            fn = self.aidx(r); n = r.varint(); data = [r.varint() for _ in range(n)]
            return ('filerange', fn, data)
        return ('builtin?', code, r.b[r.p:r.p+16].hex())

    # ---- builtin types
    def builtin_type(self, r):
        code = r.varint()
        if code == 0:  # IntegerType
            v = r.varint()
            return {'kind': 'int', 'width': v >> 2, 'signed': v & 3}
        if code == 1: return {'kind': 'index'}
        if code == 2:  # FunctionType
            ni = r.varint(); ins = [self.tidx(r) for _ in range(ni)]
            no = r.varint(); outs = [self.tidx(r) for _ in range(no)]
            return {'kind': 'func', 'in': ins, 'out': outs}
        if code == 3: return {'kind': 'bf16'}
        if code == 4: return {'kind': 'f16', 'bits': 16}
        if code == 5: return {'kind': 'f32', 'bits': 32}
        if code == 6: return {'kind': 'f64', 'bits': 64}
        if code == 12: return {'kind': 'none'}
        if code == 13:  # RankedTensorType
            n = r.varint(); shape = [r.svarint() for _ in range(n)]
            return {'kind': 'tensor', 'shape': shape, 'elem': self.tidx(r)}
        if code == 15:  # TupleType
            n = r.varint(); return {'kind': 'tuple', 'elems': [self.tidx(r) for _ in range(n)]}
        return {'kind': 'builtin?', 'code': code}

    # ---- vhlo attributes / types
    def vhlo_attr(self, r):
        code = r.varint()
        if code == 1:  # ArrayV1
            n = r.varint(); return ('array', [self.aidx(r) for _ in range(n)])
        if code == 2:
            return ('bool', r.varint())
        if code == 3:
            return ('cmpdir', r.varint())
        if code == 4:
            return ('cmptype', r.varint())
        if code == 6:  # DictionaryV1
            n = r.varint()
            return ('dict', [(self.aidx(r), self.aidx(r)) for _ in range(n)])
        if code == 8:  # FloatV1
            t = self.tidx(r); return ('float', t, self._apint(r, t[1].get('bits', 32)))
        if code == 9:  # IntegerV1
            t = self.tidx(r); return ('int', t, self._apint(r, t[1].get('width', 32)))
        if code == 14:
            return ('str', self._str(r))
        if code == 15:  # TensorV1
            t = self.tidx(r); return ('tensorblob', t, r.bytes(r.varint()))
        if code == 17:
            return ('typeattr', self.tidx(r))
        return ('vhlo?', code, r.b[r.p:r.p+16].hex())

    def vhlo_type(self, r):
        code = r.varint()
        simple = {0: 'bool', 2: 'bf16', 3: 'f16', 4: 'f32', 5: 'f64',
                  9: 'index', 11: 'i8', 12: 'i16', 13: 'i32', 14: 'i64',
                  15: 'u4', 16: 'u8', 17: 'u16', 18: 'u32', 19: 'u64',
                  22: 'token', 31: 'i2', 32: 'u2'}
        if code in simple:
            return {'kind': simple[code], 'bits': 32 if code == 4 else (64 if code == 5 else 0)}
        if code == 8:  # FunctionV1
            ni = r.varint(); ins = [self.tidx(r) for _ in range(ni)]
            no = r.varint(); outs = [self.tidx(r) for _ in range(no)]
            return {'kind': 'func', 'in': ins, 'out': outs}
        if code == 20:  # RankedTensorV1
            n = r.varint(); shape = [r.svarint() for _ in range(n)]
            return {'kind': 'tensor', 'shape': shape, 'elem': self.tidx(r)}
        if code == 23:  # TupleV1
            n = r.varint(); return {'kind': 'tuple', 'elems': [self.tidx(r) for _ in range(n)]}
        return {'kind': 'vhlo?', 'code': code}

# ------------------------------------------------------------------ IR parse
class IR:
    def __init__(self, mod):
        self.m = mod
        self.values = []   # list of (op_id, result_no) or ('arg', n)
        self.ops = []
        self.out = []
        r = R(mod.sections[4])
        self.r = r
        # top level block header
        numops, hasargs = r.vflag()
        self.blockargs(numops, hasargs)
        for _ in range(numops):
            self.op(r)
    def blockargs(self, numops, hasargs):
        self.numops = numops
        if hasargs:
            n = self.r.varint()
            for _ in range(n):
                tidx, hasloc = self.r.vflag()
                t = self.m.type(tidx)
                if hasloc: self.m.attr(self.r.varint())
                self.values.append(('arg', len(self.values)))
        if numops >= 0 and self.r.p < len(self.r.b):
            # block-level use-list orders: one byte flag (only if hasArgs)
            if hasargs:
                hasul = self.r.byte()
                if hasul:
                    raise NotImplementedError("block use-list order")
    def op(self, r):
        name = self.m.opnames[r.varint()]
        mask = r.byte()
        loc = self.m.attr(r.varint())
        attrs = None
        if mask & 1:
            attrs = self.m.attr(r.varint())
        props = None
        if mask & 0x40:
            pid = r.varint()
            props = self.m.decode_props(f"{name[0]}.{name[1]}", pid)
        nres = 0
        rtypes = []
        if mask & 2:
            nres = r.varint()
            rtypes = [self.m.type(r.varint()) for _ in range(nres)]
        nop = 0; operands = []
        if mask & 4:
            nop = r.varint()
            operands = [r.varint() for _ in range(nop)]
        if mask & 8:
            n = r.varint(); r.bytes(0)
            for _ in range(n): r.varint()
        if mask & 0x20:
            n = 1
            if nres > 1: n = r.varint()
            for _ in range(n):
                if nres > 1: r.varint()
                nv, _flag = r.vflag()
                for _ in range(nv): r.varint()
        nregions = 0; isolated = False
        if mask & 0x10:
            nregions, isolated = r.vflag()
        oid = len(self.ops)
        self.ops.append((name, rtypes, operands, props, attrs))
        self.out.append((name, rtypes, operands, props))
        for k in range(nres):
            self.values.append((oid, k))
        for _ in range(nregions):
            self.region(isolated)

    def region(self, isolated=False):
        rd = self.r
        if isolated:
            idb = rd.byte()
            sid = idb & 0x7f
            ln = rd.varint()
            if idb & 0x80:
                al = rd.varint()
                while rd.p % al:
                    rd.p += 1
            body = rd.bytes(ln)
            assert sid == 4, ("isolated region section", sid)
            rd = R(body)
        nb = rd.varint()
        nv = rd.varint() if nb else 0
        if nb == 0:
            return
        saved_vals, saved_r = self.values, self.r
        self.values = []
        self.r = rd

        def block():
            numops, hasargs = self.r.vflag()
            if hasargs:
                n = self.r.varint()
                for _ in range(n):
                    tidx, hasloc = self.r.vflag()
                    self.m.type(tidx)
                    if hasloc:
                        self.m.attr(self.r.varint())
                    self.values.append(('arg', len(self.values)))
                hasul = self.r.byte()
                if hasul:
                    nv2, _ = self.r.vflag()
                    for _ in range(nv2):
                        self.r.varint()
                        k, _ = self.r.vflag()
                        for _ in range(k):
                            self.r.varint()
            for _ in range(numops):
                self.op(self.r)

        for _ in range(nb):
            block()
        self.values, self.r = saved_vals, saved_r

def main(path):
    data = open(path, 'rb').read()
    m = Module(data)
    print(f"version={m.version} producer={m.producer} ops={len(m.opnames)} "
          f"attrs={len(m.attr_data)} types={len(m.types_data)}")
    print("dialects:", m.dialects)
    print("opnames:", [x[1] for x in m.opnames])
    print("strings:", m.strings)
    ir = IR(m)
    for i, (name, rt, ops, props) in enumerate(ir.out):
        print(f"{i:3d} {name[0]}.{name[1]:22s} res={dump(rt)} ops={ops} props={props}")

def dump(t):
    out = []
    for e in t:
        d = e[1]
        if d.get('kind') == 'tensor':
            out.append(f"f32{tuple(d['shape'])}" if d['elem'][1].get('kind') in ('f32', 'f32') else str(d))
        else:
            out.append(str(d))
    return out

if __name__ == '__main__':
    main(sys.argv[1])
