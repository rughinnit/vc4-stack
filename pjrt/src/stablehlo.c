/*
 * A reader for the subset of MLIR bytecode that jaxlib hands to
 * PJRT_Client_Compile, and a lowering of the VHLO/StableHLO subset into a
 * linear instruction list over arena tensors.
 *
 * jaxlib serialises the module as a StableHLO portable artifact: MLIR
 * bytecode version 6 carrying the `vhlo` (versioned StableHLO) dialect.
 * There is no text form and no MLIR library on this board, so the reader
 * below implements exactly the encodings the emitter produces:
 *
 *   - top-level sections, string table, dialect and op-name tables
 *   - the attribute/type tables (builtin and vhlo encodings)
 *   - the operation mask, locations (skipped), properties, results,
 *     operands and nested regions
 *   - operation properties, whose field order is the ODS order and
 *     therefore alphabetical for the VHLO ops used here
 *
 * Anything it does not understand is reported with the operation name; it
 * never guesses.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stablehlo.h"

/* --------------------------------------------------------------- byte reader */

typedef struct {
    const uint8_t *b;
    size_t n, p;
} rd;

static void seterr(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int rv(rd *r, uint64_t *out) {
    if (r->p >= r->n) return -1;
    uint8_t b = r->b[r->p++];
    if (b & 1) { *out = b >> 1; return 0; }
    if (b == 0) {
        if (r->n - r->p < 8) return -1;
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= (uint64_t)r->b[r->p + i] << (8 * i);
        r->p += 8;
        *out = v;
        return 0;
    }
    unsigned tz = 0;
    while (!((b >> tz) & 1)) tz++;
    if (tz > 7 || r->n - r->p < tz) return -1;
    uint64_t v = b;
    for (unsigned i = 0; i < tz; i++)
        v |= (uint64_t)r->b[r->p + i] << (8 * (i + 1));
    r->p += tz;
    *out = v >> (tz + 1);
    return 0;
}

static int rs(rd *r, int64_t *out) {
    uint64_t v;
    if (rv(r, &v)) return -1;
    *out = (int64_t)((v >> 1) ^ (uint64_t)-(int64_t)(v & 1));
    return 0;
}

static int rvf(rd *r, uint64_t *val, int *flag) {
    uint64_t v;
    if (rv(r, &v)) return -1;
    *val = v >> 1;
    *flag = (int)(v & 1);
    return 0;
}

static int rbyte(rd *r, uint8_t *out) {
    if (r->p >= r->n) return -1;
    *out = r->b[r->p++];
    return 0;
}

static int rbytes(rd *r, size_t len, const uint8_t **out) {
    if (r->n - r->p < len) return -1;
    if (out) *out = r->b + r->p;
    r->p += len;
    return 0;
}

static int align_to(rd *r, unsigned align) {
    while (r->p % align) {
        if (r->p >= r->n) return -1;
        r->p++;
    }
    return 0;
}

/* ------------------------------------------------------------------- model */

enum {
    DT_F32 = 0,
    DT_I1,
    DT_I64,
    DT_I32,
    DT_OTHER,
};

enum {
    NK_TENSOR = 1,
    NK_FUNC,
    NK_TUPLE,
    NK_SCALAR,
    NK_NONE,
};

typedef struct {
    int kind;               /* NK_* */
    int rank;
    int64_t dims[VC4SL_MAXRANK];
    int dtype;              /* DT_* for tensors and scalars */
    int *ins, nins;         /* function types */
    int *outs, nouts;
} ctype;

enum {
    AK_ARRAY = 1,
    AK_DICT,
    AK_STRING,
    AK_TYPE,
    AK_UNIT,
    AK_INT,
    AK_BOOL,
    AK_TENSOR,              /* shaped blob, e.g. a constant or an i64 list */
    AK_OPAQUE,
};

typedef struct {
    int kind;
    int64_t ival;
    char *sval;
    int type_idx;
    int dtype;
    int rank;
    int64_t dims[VC4SL_MAXRANK];
    const uint8_t *blob;
    size_t blob_len;
    int *items;             /* AK_ARRAY: attribute indices */
    int nitems;
    struct { int k, v; } *pairs;  /* AK_DICT */
    int npairs;
} cattr;

typedef struct {
    int dialect;
    int registered;
    const char *name;
} copname;

typedef struct {
    int dialect;
    const uint8_t *data;
    size_t len;
    int custom;
} centry;

typedef struct {
    int dialect;
    const char *name;
} cdialect;

struct sl_parser;

typedef struct snode snode;

typedef struct {
    int nargs;
    int *arg_types;
    snode *ops;
    int nops;
} sregion;

struct snode {
    int opname;
    int nres;
    int res_types[8];
    int noperands;
    int operands[16];
    int nregions;
    sregion *regions[4];
    cattr props[16];
    int nprops;
};

typedef struct sl_parser {
    const uint8_t *data;
    size_t len;
    char *err;
    size_t errlen;

    uint64_t version;
    unsigned char *sections[9];
    size_t section_len[9];

    char **strings;
    int nstrings;

    cdialect *dialects;
    int ndialects;

    copname *opnames;
    int nopnames;

    centry *attrs;
    int nattrs;
    centry *types;
    int ntypes;

    cattr *attr_cache;
    unsigned char *attr_state;
    ctype *type_cache;
    unsigned char *type_state;

    const uint8_t *props_data;
    size_t props_len;
    struct { const uint8_t *p; size_t n; } *prop_blobs;
    int nprop_blobs;

    /* string table parsed from section 0 */
    unsigned char *strings_owned;
} sl_parser;

/* ------------------------------------------------------------ string table */

static int parse_strings(sl_parser *s) {
    rd r = {s->sections[0], s->section_len[0], 0};
    uint64_t n;
    if (rv(&r, &n) || n > 100000) {
        seterr(s->err, s->errlen, "bad string table");
        return -1;
    }
    s->strings = calloc(n ? n : 1, sizeof(char *));
    if (!s->strings) return -1;
    s->nstrings = (int)n;
    size_t end = s->section_len[0];
    for (int i = (int)n - 1; i >= 0; i--) {
        uint64_t sz;
        if (rv(&r, &sz) || sz == 0 || sz > end) {
            seterr(s->err, s->errlen, "bad string entry");
            return -1;
        }
        size_t start = end - sz;
        char *str = malloc(sz);
        if (!str) return -1;
        memcpy(str, s->sections[0] + start, sz - 1);
        str[sz - 1] = 0;
        s->strings[i] = str;
        end = start;
    }
    return 0;
}

static int get_string(sl_parser *s, uint64_t idx, const char **out) {
    if (idx >= (uint64_t)s->nstrings) return -1;
    *out = s->strings[idx];
    return 0;
}

/* ------------------------------------------------------ dialect and op names */

static int parse_dialects(sl_parser *s) {
    rd r = {s->sections[1], s->section_len[1], 0};
    uint64_t n;
    if (rv(&r, &n) || n > 512) return -1;
    s->dialects = calloc(n ? n : 1, sizeof(cdialect));
    s->ndialects = (int)n;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t idx;
        int has_version;
        if (rvf(&r, &idx, &has_version)) return -1;
        if (get_string(s, idx, &s->dialects[i].name)) return -1;
        if (has_version) {
            uint8_t idb;
            uint64_t len;
            if (rbyte(&r, &idb) || rv(&r, &len)) return -1;
            if (idb & 0x80) {
                uint64_t al;
                if (rv(&r, &al) || !al) return -1;
                if (align_to(&r, (unsigned)al)) return -1;
            }
            const uint8_t *d;
            if (rbytes(&r, (size_t)len, &d)) return -1;
        }
    }
    uint64_t num_ops = 0;
    if (s->version >= 4 && rv(&r, &num_ops)) return -1;
    s->opnames = calloc(num_ops ? (size_t)num_ops : 1, sizeof(copname));
    if (!s->opnames) return -1;
    while (r.p < r.n) {
        uint64_t didx, cnt;
        if (rv(&r, &didx) || rv(&r, &cnt) || didx >= (uint64_t)s->ndialects)
            return -1;
        for (uint64_t i = 0; i < cnt; i++) {
            uint64_t sidx;
            int registered = 0;
            if (s->version >= 5) {
                if (rvf(&r, &sidx, &registered)) return -1;
            } else if (rv(&r, &sidx)) {
                return -1;
            }
            const char *name;
            if (get_string(s, sidx, &name)) return -1;
            if (s->nopnames >= (int)num_ops) return -1;
            s->opnames[s->nopnames].dialect = (int)didx;
            s->opnames[s->nopnames].name = name;
            s->opnames[s->nopnames].registered = registered;
            s->nopnames++;
        }
    }
    return 0;
}

/* ----------------------------------------------------- attribute/type tables */

static int parse_attr_type_tables(sl_parser *s) {
    rd r = {s->sections[3], s->section_len[3], 0};
    uint64_t na, nt;
    if (rv(&r, &na) || rv(&r, &nt)) return -1;
    if (na > 100000 || nt > 100000) return -1;
    centry *entries = calloc((size_t)(na + nt), sizeof(centry));
    if (!entries) return -1;
    size_t total = 0;
    for (int pass = 0; pass < 2; pass++) {
        uint64_t count = pass ? nt : na;
        uint64_t i = 0;
        while (i < count) {
            uint64_t didx, cnt;
            if (rv(&r, &didx) || rv(&r, &cnt) || didx >= (uint64_t)s->ndialects)
                goto fail;
            for (uint64_t k = 0; k < cnt; k++) {
                uint64_t size;
                int custom = 0;
                if (rvf(&r, &size, &custom)) goto fail;
                if (i >= count) goto fail;
                entries[total].dialect = (int)didx;
                entries[total].len = (size_t)size;
                entries[total].custom = custom;
                total++;
                i++;
            }
        }
    }
    size_t off = 0;
    for (size_t i = 0; i < total; i++) {
        if (off + entries[i].len > s->section_len[2]) goto fail;
        entries[i].data = s->sections[2] + off;
        off += entries[i].len;
    }
    s->attrs = entries;
    s->nattrs = (int)na;
    s->types = entries + na;
    s->ntypes = (int)nt;
    s->attr_cache = calloc(na ? (size_t)na : 1, sizeof(cattr));
    s->attr_state = calloc(na ? (size_t)na : 1, 1);
    s->type_cache = calloc(nt ? (size_t)nt : 1, sizeof(ctype));
    s->type_state = calloc(nt ? (size_t)nt : 1, 1);
    if (!s->attr_cache || !s->attr_state || !s->type_cache || !s->type_state)
        goto fail;
    return 0;
fail:
    free(entries);
    seterr(s->err, s->errlen, "malformed attribute/type table");
    return -1;
}

static int parse_props(sl_parser *s) {
    if (!s->section_len[8]) return 0;
    rd r = {s->sections[8], s->section_len[8], 0};
    uint64_t count;
    if (rv(&r, &count) || count > 100000) return -1;
    const uint8_t *buf;
    size_t avail = r.n - r.p;
    buf = r.b + r.p;
    rd or = {buf, avail, 0};
    s->prop_blobs = calloc(count ? (size_t)count : 1, sizeof(*s->prop_blobs));
    if (!s->prop_blobs) return -1;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t sz;
        const uint8_t *d;
        if (rv(&or, &sz) || rbytes(&or, (size_t)sz, &d)) return -1;
        s->prop_blobs[i].p = d;
        s->prop_blobs[i].n = (size_t)sz;
    }
    s->nprop_blobs = (int)count;
    return 0;
}

/* --------------------------------------------------------------- type decode */

static int decode_type(sl_parser *s, int idx, ctype *out);

static int scalar_dtype_from_builtin(uint64_t code, unsigned width) {
    switch (code) {
    case 0: /* IntegerType */
        if (width == 64) return DT_I64;
        if (width == 32) return DT_I32;
        return DT_OTHER;
    case 3: case 4: case 5: case 6:
        return DT_OTHER;
    default:
        return DT_OTHER;
    }
}

static int decode_builtin_type(sl_parser *s, rd *r, ctype *out) {
    uint64_t code;
    if (rv(r, &code)) return -1;
    memset(out, 0, sizeof(*out));
    switch (code) {
    case 0: {
        uint64_t v;
        if (rv(r, &v)) return -1;
        out->kind = NK_SCALAR;
        out->dtype = scalar_dtype_from_builtin(0, (unsigned)(v >> 2));
        return 0;
    }
    case 1:
        out->kind = NK_SCALAR;
        out->dtype = DT_I64;
        return 0;
    case 2: {
        uint64_t ni, no;
        if (rv(r, &ni) || ni > 16) return -1;
        out->kind = NK_FUNC;
        out->ins = calloc(ni ? (size_t)ni : 1, sizeof(int));
        out->nins = (int)ni;
        for (uint64_t i = 0; i < ni; i++)
            if (rv(r, (uint64_t *)&out->ins[i])) return -1;
        if (rv(r, &no) || no > 16) return -1;
        out->outs = calloc(no ? (size_t)no : 1, sizeof(int));
        out->nouts = (int)no;
        for (uint64_t i = 0; i < no; i++)
            if (rv(r, (uint64_t *)&out->outs[i])) return -1;
        return 0;
    }
    case 3: case 4: case 5: case 6:
        out->kind = NK_SCALAR;
        out->dtype = DT_OTHER;
        return 0;
    case 12:
        out->kind = NK_NONE;
        return 0;
    case 13: {
        uint64_t n;
        if (rv(r, &n) || n > VC4SL_MAXRANK) return -1;
        for (uint64_t i = 0; i < n; i++)
            if (rs(r, &out->dims[i])) return -1;
        out->rank = (int)n;
        uint64_t elem;
        if (rv(r, &elem) || elem >= (uint64_t)s->ntypes) return -1;
        ctype et;
        if (decode_type(s, (int)elem, &et)) return -1;
        out->kind = NK_TENSOR;
        out->dtype = et.dtype;
        return 0;
    }
    case 15: {
        uint64_t n;
        if (rv(r, &n) || n > 16) return -1;
        out->kind = NK_TUPLE;
        out->ins = calloc(n ? (size_t)n : 1, sizeof(int));
        out->nins = (int)n;
        for (uint64_t i = 0; i < n; i++)
            if (rv(r, (uint64_t *)&out->ins[i])) return -1;
        return 0;
    }
    default:
        out->kind = NK_NONE;
        return 0;
    }
}

static int vhlo_dtype(uint64_t code, unsigned *bits) {
    *bits = 0;
    switch (code) {
    case 2: *bits = 16; return DT_OTHER;          /* bf16 */
    case 3: *bits = 16; return DT_OTHER;          /* f16 */
    case 4: *bits = 32; return DT_F32;
    case 5: *bits = 64; return DT_OTHER;
    case 9: return DT_I64;                        /* index */
    case 11: *bits = 8; return DT_OTHER;
    case 12: *bits = 16; return DT_OTHER;
    case 13: *bits = 32; return DT_I32;
    case 14: *bits = 64; return DT_I64;
    case 16: *bits = 8; return DT_OTHER;
    case 17: *bits = 16; return DT_OTHER;
    case 18: *bits = 32; return DT_I32;
    case 19: *bits = 64; return DT_I64;
    case 31: case 32: case 10: case 15: return DT_OTHER;
    case 33: return DT_OTHER;                     /* none */
    default: return DT_OTHER;
    }
}

static int decode_vhlo_type(sl_parser *s, rd *r, ctype *out) {
    uint64_t code;
    if (rv(r, &code)) return -1;
    memset(out, 0, sizeof(*out));
    if (code == 0) { /* BooleanV1 */
        out->kind = NK_SCALAR;
        out->dtype = DT_I1;
        return 0;
    }
    if (code == 8) { /* FunctionV1 */
        uint64_t ni, no;
        if (rv(r, &ni) || ni > 16) return -1;
        out->kind = NK_FUNC;
        out->ins = calloc(ni ? (size_t)ni : 1, sizeof(int));
        out->nins = (int)ni;
        for (uint64_t i = 0; i < ni; i++) {
            uint64_t t;
            if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
            out->ins[i] = (int)t;
        }
        if (rv(r, &no) || no > 16) return -1;
        out->outs = calloc(no ? (size_t)no : 1, sizeof(int));
        out->nouts = (int)no;
        for (uint64_t i = 0; i < no; i++) {
            uint64_t t;
            if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
            out->outs[i] = (int)t;
        }
        return 0;
    }
    if (code == 20) { /* RankedTensorV1 */
        uint64_t n;
        if (rv(r, &n) || n > VC4SL_MAXRANK) return -1;
        for (uint64_t i = 0; i < n; i++)
            if (rs(r, &out->dims[i])) return -1;
        out->rank = (int)n;
        uint64_t elem;
        if (rv(r, &elem) || elem >= (uint64_t)s->ntypes) return -1;
        ctype et;
        if (decode_type(s, (int)elem, &et)) return -1;
        out->kind = NK_TENSOR;
        out->dtype = et.dtype;
        return 0;
    }
    if (code == 23) { /* TupleV1 */
        uint64_t n;
        if (rv(r, &n) || n > 16) return -1;
        out->kind = NK_TUPLE;
        out->ins = calloc(n ? (size_t)n : 1, sizeof(int));
        out->nins = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t t;
            if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
            out->ins[i] = (int)t;
        }
        return 0;
    }
    if (code == 33) {
        out->kind = NK_NONE;
        return 0;
    }
    unsigned bits;
    out->kind = NK_SCALAR;
    out->dtype = vhlo_dtype(code, &bits);
    return 0;
}

static int decode_type(sl_parser *s, int idx, ctype *out) {
    if (idx < 0 || idx >= s->ntypes) return -1;
    if (s->type_state[idx] == 2) {
        *out = s->type_cache[idx];
        return 0;
    }
    if (s->type_state[idx] == 1) return -1;
    s->type_state[idx] = 1;
    centry *e = &s->types[idx];
    rd r = {e->data, e->len, 0};
    int rc;
    if (!strcmp(s->dialects[e->dialect].name, "builtin"))
        rc = decode_builtin_type(s, &r, out);
    else if (!strcmp(s->dialects[e->dialect].name, "vhlo"))
        rc = decode_vhlo_type(s, &r, out);
    else
        rc = -1;
    if (rc) {
        s->type_state[idx] = 0;
        return -1;
    }
    s->type_cache[idx] = *out;
    s->type_state[idx] = 2;
    return 0;
}

/* ---------------------------------------------------------- attribute decode */

static int decode_attr(sl_parser *s, int idx, cattr *out);

static int read_string_attr(sl_parser *s, rd *r, char **out) {
    uint64_t idx;
    if (rv(r, &idx)) return -1;
    const char *str;
    if (get_string(s, idx, &str)) return -1;
    *out = strdup(str);
    return *out ? 0 : -1;
}

static int apint_dtype_width(ctype *t) {
    if (t->kind == NK_SCALAR && t->dtype == DT_I64) return 64;
    if (t->kind == NK_SCALAR && t->dtype == DT_I32) return 32;
    if (t->kind == NK_SCALAR && t->dtype == DT_I1) return 1;
    return 64;
}

static int decode_builtin_attr(sl_parser *s, rd *r, cattr *out) {
    uint64_t code;
    if (rv(r, &code)) return -1;
    memset(out, 0, sizeof(*out));
    switch (code) {
    case 0: { /* ArrayAttr */
        uint64_t n;
        if (rv(r, &n) || n > 1000000) return -1;
        out->kind = AK_ARRAY;
        out->items = calloc(n ? (size_t)n : 1, sizeof(int));
        out->nitems = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t v;
            if (rv(r, &v) || v >= (uint64_t)s->nattrs) return -1;
            out->items[i] = (int)v;
        }
        return 0;
    }
    case 1: { /* DictionaryAttr */
        uint64_t n;
        if (rv(r, &n) || n > 100000) return -1;
        out->kind = AK_DICT;
        out->pairs = calloc(n ? (size_t)n : 1, sizeof(*out->pairs));
        out->npairs = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t k, v;
            if (rv(r, &k) || rv(r, &v) || k >= (uint64_t)s->nattrs ||
                v >= (uint64_t)s->nattrs)
                return -1;
            out->pairs[i].k = (int)k;
            out->pairs[i].v = (int)v;
        }
        return 0;
    }
    case 2: /* StringAttr */
        out->kind = AK_STRING;
        return read_string_attr(s, r, &out->sval);
    case 3: { /* StringAttrWithType */
        uint64_t t;
        out->kind = AK_STRING;
        if (read_string_attr(s, r, &out->sval) || rv(r, &t)) return -1;
        out->type_idx = (int)t;
        return 0;
    }
    case 4: { /* FlatSymbolRefAttr */
        uint64_t a;
        if (rv(r, &a) || a >= (uint64_t)s->nattrs) return -1;
        cattr sub;
        if (decode_attr(s, (int)a, &sub)) return -1;
        if (sub.kind != AK_STRING) return -1;
        out->kind = AK_STRING;
        out->sval = sub.sval;
        return 0;
    }
    case 5: { /* SymbolRefAttr (root only) */
        uint64_t a, n;
        if (rv(r, &a) || a >= (uint64_t)s->nattrs) return -1;
        cattr sub;
        if (decode_attr(s, (int)a, &sub)) return -1;
        if (sub.kind != AK_STRING) return -1;
        if (rv(r, &n) || n > 16) return -1;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t x;
            if (rv(r, &x)) return -1;
        }
        out->kind = AK_STRING;
        out->sval = sub.sval;
        return 0;
    }
    case 6: { /* TypeAttr */
        uint64_t t;
        out->kind = AK_TYPE;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        out->type_idx = (int)t;
        return 0;
    }
    case 7: /* UnitAttr */
        out->kind = AK_UNIT;
        return 0;
    case 8: { /* IntegerAttr */
        uint64_t t;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        ctype ty;
        if (decode_type(s, (int)t, &ty)) return -1;
        int width = apint_dtype_width(&ty);
        if (width <= 8) {
            uint8_t b;
            if (rbyte(r, &b)) return -1;
            out->ival = b;
        } else if (width <= 64) {
            int64_t sv;
            if (rs(r, &sv)) return -1;
            out->ival = sv;
        } else {
            uint64_t n;
            if (rv(r, &n) || n > 16) return -1;
            for (uint64_t i = 0; i < n; i++) {
                int64_t w;
                if (rs(r, &w)) return -1;
                if (i == 0) out->ival = w;
            }
        }
        out->kind = AK_INT;
        return 0;
    }
    case 9: { /* FloatAttr */
        uint64_t t;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        int64_t bits;
        if (rs(r, &bits)) return -1;
        out->kind = AK_INT; /* unused; kept as raw bits */
        out->ival = bits;
        return 0;
    }
    case 11: case 12: case 14: case 22: /* locations */
        out->kind = AK_OPAQUE;
        return 0;
    case 17: { /* DenseArrayAttr */
        uint64_t t;
        out->kind = AK_TENSOR;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        ctype ty;
        if (decode_type(s, (int)t, &ty)) return -1;
        out->dtype = ty.dtype;
        uint64_t n, sz;
        if (rv(r, &n) || rv(r, &sz)) return -1;
        const uint8_t *d;
        if (rbytes(r, (size_t)sz, &d)) return -1;
        out->rank = 1;
        out->dims[0] = (int64_t)n;
        out->blob = d;
        out->blob_len = (size_t)sz;
        return 0;
    }
    case 18: { /* DenseTypedElementsAttr */
        uint64_t t;
        out->kind = AK_TENSOR;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        ctype ty;
        if (decode_type(s, (int)t, &ty) || ty.kind != NK_TENSOR) return -1;
        out->dtype = ty.dtype;
        out->rank = ty.rank;
        memcpy(out->dims, ty.dims, sizeof(ty.dims));
        uint64_t sz;
        if (rv(r, &sz)) return -1;
        const uint8_t *d;
        if (rbytes(r, (size_t)sz, &d)) return -1;
        out->blob = d;
        out->blob_len = (size_t)sz;
        return 0;
    }
    default:
        out->kind = AK_OPAQUE;
        return 0;
    }
}

static int decode_vhlo_attr(sl_parser *s, rd *r, cattr *out) {
    uint64_t code;
    if (rv(r, &code)) return -1;
    memset(out, 0, sizeof(*out));
    switch (code) {
    case 1: { /* ArrayV1 */
        uint64_t n;
        if (rv(r, &n) || n > 100000) return -1;
        out->kind = AK_ARRAY;
        out->items = calloc(n ? (size_t)n : 1, sizeof(int));
        out->nitems = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t v;
            if (rv(r, &v) || v >= (uint64_t)s->nattrs) return -1;
            out->items[i] = (int)v;
        }
        return 0;
    }
    case 2: { /* BooleanV1 */
        uint64_t v;
        if (rv(r, &v)) return -1;
        out->kind = AK_BOOL;
        out->ival = (int64_t)v;
        return 0;
    }
    case 3: case 4: case 5: case 7: case 11: case 12: case 13: /* enums */
    case 16: {
        uint64_t v;
        if (rv(r, &v)) return -1;
        out->kind = AK_INT;
        out->ival = (int64_t)v;
        return 0;
    }
    case 6: { /* DictionaryV1 */
        uint64_t n;
        if (rv(r, &n) || n > 100000) return -1;
        out->kind = AK_DICT;
        out->pairs = calloc(n ? (size_t)n : 1, sizeof(*out->pairs));
        out->npairs = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t k, v;
            if (rv(r, &k) || rv(r, &v) || k >= (uint64_t)s->nattrs ||
                v >= (uint64_t)s->nattrs)
                return -1;
            out->pairs[i].k = (int)k;
            out->pairs[i].v = (int)v;
        }
        return 0;
    }
    case 8: { /* FloatV1 */
        uint64_t t;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        int64_t bits;
        if (rs(r, &bits)) return -1;
        out->kind = AK_INT;
        out->ival = bits;
        return 0;
    }
    case 9: { /* IntegerV1 */
        uint64_t t;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        ctype ty;
        if (decode_type(s, (int)t, &ty)) return -1;
        int width = apint_dtype_width(&ty);
        if (width <= 8) {
            uint8_t b;
            if (rbyte(r, &b)) return -1;
            out->ival = b;
        } else {
            int64_t sv;
            if (rs(r, &sv)) return -1;
            out->ival = sv;
        }
        out->kind = AK_INT;
        return 0;
    }
    case 14: /* StringV1 */
        out->kind = AK_STRING;
        return read_string_attr(s, r, &out->sval);
    case 15: { /* TensorV1 */
        uint64_t t;
        out->kind = AK_TENSOR;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        ctype ty;
        if (decode_type(s, (int)t, &ty) || ty.kind != NK_TENSOR) return -1;
        out->dtype = ty.dtype;
        out->rank = ty.rank;
        memcpy(out->dims, ty.dims, sizeof(ty.dims));
        uint64_t sz;
        if (rv(r, &sz)) return -1;
        const uint8_t *d;
        if (rbytes(r, (size_t)sz, &d)) return -1;
        out->blob = d;
        out->blob_len = (size_t)sz;
        return 0;
    }
    case 17: { /* TypeV1 */
        uint64_t t;
        out->kind = AK_TYPE;
        if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
        out->type_idx = (int)t;
        return 0;
    }
    default:
        out->kind = AK_OPAQUE;
        return 0;
    }
}

static int decode_attr(sl_parser *s, int idx, cattr *out) {
    if (idx < 0 || idx >= s->nattrs) return -1;
    if (s->attr_state[idx] == 2) {
        *out = s->attr_cache[idx];
        return 0;
    }
    if (s->attr_state[idx] == 1) return -1;
    s->attr_state[idx] = 1;
    centry *e = &s->attrs[idx];
    rd r = {e->data, e->len, 0};
    int rc;
    if (!strcmp(s->dialects[e->dialect].name, "builtin"))
        rc = decode_builtin_attr(s, &r, out);
    else if (!strcmp(s->dialects[e->dialect].name, "vhlo"))
        rc = decode_vhlo_attr(s, &r, out);
    else
        rc = -1;
    if (rc) {
        s->attr_state[idx] = 0;
        return -1;
    }
    s->attr_cache[idx] = *out;
    s->attr_state[idx] = 2;
    return 0;
}

/* ---------------------------------------------------------- IR and regions */

static int parse_region(sl_parser *s, rd *outer, int isolated, sregion *reg);

static int parse_block_header(sl_parser *s, rd *r, sregion *reg, int *num_ops) {
    uint64_t nops;
    int has_args = 0;
    if (rvf(r, &nops, &has_args)) return -1;
    if (nops > 100000) return -1;
    *num_ops = (int)nops;
    if (!has_args) return 0;
    uint64_t nargs;
    if (rv(r, &nargs) || nargs > 64) return -1;
    reg->nargs = (int)nargs;
    reg->arg_types = calloc(nargs ? (size_t)nargs : 1, sizeof(int));
    if (!reg->arg_types) return -1;
    for (uint64_t i = 0; i < nargs; i++) {
        uint64_t tidx;
        int has_loc = 0;
        if (rvf(r, &tidx, &has_loc)) return -1;
        if (tidx >= (uint64_t)s->ntypes) return -1;
        ctype t;
        if (decode_type(s, (int)tidx, &t)) return -1;
        reg->arg_types[i] = (int)tidx;
        if (has_loc) {
            uint64_t loc;
            if (rv(r, &loc)) return -1;
        }
        if (!s->type_state[tidx]) return -1;
    }
    if (s->version >= 3) {
        uint8_t has_ul;
        if (rbyte(r, &has_ul)) return -1;
        if (has_ul) {
            uint64_t nv;
            if (rv(r, &nv) || nv > 64) return -1;
            for (uint64_t i = 0; i < nv; i++) {
                uint64_t idx, count;
                int flag;
                if (rv(r, &idx) || rvf(r, &count, &flag)) return -1;
                for (uint64_t k = 0; k < count; k++) {
                    uint64_t x;
                    if (rv(r, &x)) return -1;
                }
            }
        }
    }
    return 0;
}

static int parse_op(sl_parser *s, rd *r, snode *node);

static int parse_region(sl_parser *s, rd *outer, int isolated, sregion *reg) {
    rd r = *outer;
    if (isolated) {
        uint8_t idb;
        uint64_t len;
        if (rbyte(&r, &idb) || rv(&r, &len) || (idb & 0x7f) != 4) return -1;
        if (idb & 0x80) {
            uint64_t al;
            if (rv(&r, &al) || !al || align_to(&r, (unsigned)al)) return -1;
        }
        const uint8_t *body;
        if (rbytes(&r, (size_t)len, &body)) return -1;
        *outer = r;
        r.b = body;
        r.n = (size_t)len;
        r.p = 0;
    }
    uint64_t nblocks, nvalues = 0;
    if (rv(&r, &nblocks) || nblocks > 64) return -1;
    if (nblocks && rv(&r, &nvalues)) return -1;
    (void)nvalues;
    memset(reg, 0, sizeof(*reg));
    for (uint64_t b = 0; b < nblocks; b++) {
        int nops = 0;
        if (parse_block_header(s, &r, reg, &nops)) return -1;
        if (b == 0) {
            reg->ops = calloc(nops ? (size_t)nops : 1, sizeof(snode));
            if (!reg->ops) return -1;
            reg->nops = nops;
        } else if (nops) {
            return -1; /* multiple blocks are outside the subset */
        }
        for (int i = 0; i < nops; i++) {
            /* Only block 0 is kept; later blocks are rejected above. */
            if (b != 0) return -1;
            if (parse_op(s, &r, &reg->ops[i])) return -1;
        }
    }
    if (!isolated) *outer = r;
    else {
        /* outer already advanced past the nested section */
    }
    return 0;
}

static int parse_props_for(sl_parser *s, snode *node, uint64_t pid,
                           const char *dialect, const char *name);

static int parse_op(sl_parser *s, rd *r, snode *node) {
    memset(node, 0, sizeof(*node));
    uint64_t opidx;
    if (getenv("VC4SL_DEBUG")) fprintf(stderr, "op at %zu\n", r->p);
    if (rv(r, &opidx) || opidx >= (uint64_t)s->nopnames) return -1;
    node->opname = (int)opidx;
    uint8_t mask;
    if (rbyte(r, &mask)) return -1;
    if (getenv("VC4SL_DEBUG"))
        fprintf(stderr, "  %s.%s mask=0x%02x\n",
                s->dialects[s->opnames[opidx].dialect].name,
                s->opnames[opidx].name, mask);
    uint64_t loc;
    if (rv(r, &loc)) return -1; /* location index: not needed */
    if (mask & 0x01) {
        uint64_t attrs;
        if (rv(r, &attrs)) return -1;
        /* The attribute dictionary is discardable metadata; not used. */
    }
    if (mask & 0x40) {
        uint64_t pid;
        if (rv(r, &pid)) return -1;
        const copname *on = &s->opnames[node->opname];
        const char *dn = s->dialects[on->dialect].name;
        /* Sharding annotations (sdy.mesh) arrive when more than one backend
           is registered.  They are not part of the executed subset and the
           properties are never needed, so the index is consumed and the
           blob ignored. */
        if (strcmp(dn, "vhlo") && !(strcmp(dn, "builtin") &&
                                    !strcmp(on->name, "module")))
            goto props_done;
        if (parse_props_for(s, node, pid, dn, on->name)) {
            if (!s->err[0])
                seterr(s->err, s->errlen, "cannot read properties of %s.%s",
                       s->dialects[on->dialect].name, on->name);
            return -1;
        }
    }
props_done:
    if (mask & 0x02) {
        uint64_t n;
        if (rv(r, &n) || n > 8) return -1;
        node->nres = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t t;
            if (rv(r, &t) || t >= (uint64_t)s->ntypes) return -1;
            node->res_types[i] = (int)t;
        }
    }
    if (mask & 0x04) {
        uint64_t n;
        if (rv(r, &n) || n > 16) return -1;
        node->noperands = (int)n;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t v;
            if (rv(r, &v)) return -1;
            node->operands[i] = (int)v;
        }
    }
    if (mask & 0x08) {
        uint64_t n;
        if (rv(r, &n) || n > 16) return -1;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t v;
            if (rv(r, &v)) return -1;
        }
    }
    if ((mask & 0x20) && s->version >= 3) {
        uint64_t nvals = 1;
        if (node->nres > 1 && rv(r, &nvals)) return -1;
        for (uint64_t i = 0; i < nvals; i++) {
            uint64_t ridx = 0, count;
            int flag;
            if (node->nres > 1 && rv(r, &ridx)) return -1;
            if (rvf(r, &count, &flag)) return -1;
            for (uint64_t k = 0; k < count; k++) {
                uint64_t x;
                if (rv(r, &x)) return -1;
            }
        }
    }
    if (mask & 0x10) {
        uint64_t nregions;
        int isolated = 0;
        if (rvf(r, &nregions, &isolated) || nregions > 4) return -1;
        node->nregions = (int)nregions;
        for (uint64_t i = 0; i < nregions; i++) {
            sregion *reg = calloc(1, sizeof(*reg));
            if (!reg) return -1;
            if (parse_region(s, r, isolated, reg)) {
                if (!s->err[0])
                    seterr(s->err, s->errlen,
                           "cannot parse a region of %s.%s",
                           s->dialects[s->opnames[node->opname].dialect].name,
                           s->opnames[node->opname].name);
                return -1;
            }
            node->regions[i] = reg;
        }
    }
    return 0;
}

/* ---------------------------------------------------------- op properties */

static const char *op_full_name(sl_parser *s, snode *n) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "%s.%s", s->dialects[s->opnames[n->opname].dialect].name,
             s->opnames[n->opname].name);
    return buf;
}

/* Read an attribute reference. */
static int prop_attr(sl_parser *s, rd *r, cattr *out) {
    uint64_t idx;
    if (rv(r, &idx) || idx >= (uint64_t)s->nattrs) {
        if (getenv("VC4SL_DEBUG")) fprintf(stderr, "prop index read failed\n");
        return -1;
    }
    if (decode_attr(s, (int)idx, out)) {
        if (getenv("VC4SL_DEBUG"))
            fprintf(stderr, "attr %llu failed (dialect %s, %zu B)\n",
                    (unsigned long long)idx,
                    s->dialects[s->attrs[idx].dialect].name,
                    s->attrs[idx].len);
        return -1;
    }
    return 0;
}

/* Read an optional attribute reference. */
static int prop_opt_attr(sl_parser *s, rd *r, cattr *out, int *present) {
    uint64_t idx;
    int flag;
    if (rvf(r, &idx, &flag)) return -1;
    *present = flag;
    if (!flag) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    if (idx >= (uint64_t)s->nattrs) return -1;
    return decode_attr(s, (int)idx, out);
}

static int add_prop(snode *node, const char *name, const cattr *v) {
    if (node->nprops >= 16) return -1;
    node->props[node->nprops] = *v;
    /* the name is recovered from the position; nothing else needs it */
    (void)name;
    node->nprops++;
    return 0;
}

static int parse_props_for(sl_parser *s, snode *node, uint64_t pid,
                           const char *dialect, const char *name) {
    if (pid >= (uint64_t)s->nprop_blobs) return -1;
    rd r = {s->prop_blobs[pid].p, s->prop_blobs[pid].n, 0};
    if (!strcmp(dialect, "builtin") && !strcmp(name, "module")) {
        cattr a;
        int present;
        if (prop_opt_attr(s, &r, &a, &present) ||
            add_prop(node, "sym_name", &a))
            return -1;
        if (prop_opt_attr(s, &r, &a, &present) ||
            add_prop(node, "sym_visibility", &a))
            return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "func_v1")) {
        static const char *names[] = {"arg_attrs", "function_type", "res_attrs",
                                      "sym_name", "sym_visibility"};
        for (int i = 0; i < 5; i++) {
            cattr a;
            if (prop_attr(s, &r, &a) || add_prop(node, names[i], &a)) return -1;
        }
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "constant_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "value", &a)) return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "iota_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "iota_dimension", &a))
            return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "broadcast_in_dim_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "broadcast_dimensions", &a))
            return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "reduce_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "dimensions", &a)) return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "transpose_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "permutation", &a)) return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "slice_v1")) {
        static const char *names[] = {"start_indices", "limit_indices",
                                      "strides"};
        for (int i = 0; i < 3; i++) {
            cattr a;
            if (prop_attr(s, &r, &a) || add_prop(node, names[i], &a)) return -1;
        }
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "call_v1")) {
        cattr a;
        if (prop_attr(s, &r, &a) || add_prop(node, "callee", &a)) return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "compare_v1")) {
        cattr a;
        /* ODS order: compare_type, comparison_direction */
        if (prop_attr(s, &r, &a) || add_prop(node, "compare_type", &a))
            return -1;
        if (prop_attr(s, &r, &a) || add_prop(node, "comparison_direction", &a))
            return -1;
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "dot_general_v2")) {
        /* ODS emits properties in name order. */
        static const char *names[] = {
            "accumulation_type", "allow_imprecise_accumulation",
            "lhs_batching_dimensions", "lhs_component_count",
            "lhs_contracting_dimensions", "lhs_precision_type",
            "num_primitive_operations", "precision_config",
            "rhs_batching_dimensions", "rhs_component_count",
            "rhs_contracting_dimensions", "rhs_precision_type"};
        for (int i = 0; i < 12; i++) {
            cattr a;
            if (prop_attr(s, &r, &a) || add_prop(node, names[i], &a)) return -1;
        }
        return 0;
    }
    if (!strcmp(dialect, "vhlo") && !strcmp(name, "dot_general_v1")) {
        static const char *names[] = {"lhs_batching_dimensions",
                                      "rhs_batching_dimensions",
                                      "lhs_contracting_dimensions",
                                      "rhs_contracting_dimensions",
                                      "precision_config"};
        for (int i = 0; i < 5; i++) {
            cattr a;
            if (prop_attr(s, &r, &a) || add_prop(node, names[i], &a)) return -1;
        }
        return 0;
    }
    /* An op outside the executed subset: its properties are not needed.
       Lowering refuses it by name. */
    return 0;
}

/* --------------------------------------------------------------- public API */

static int parse_top(sl_parser *s) {
    rd r = {s->data, s->len, 0};
    const uint8_t *magic;
    if (rbytes(&r, 4, &magic) || memcmp(magic, "ML\xefR", 4))
        return -1;
    if (rv(&r, &s->version)) return -1;
    size_t z = r.p;
    while (z < r.n && r.b[z]) z++;
    if (z >= r.n) return -1;
    r.p = z + 1;
    while (r.p < r.n) {
        uint8_t idb;
        uint64_t len;
        if (rbyte(&r, &idb) || rv(&r, &len)) return -1;
        unsigned sid = idb & 0x7f;
        if (sid >= 9) return -1;
        if (idb & 0x80) {
            uint64_t al;
            if (rv(&r, &al) || !al || (al & (al - 1)) || align_to(&r, (unsigned)al))
                return -1;
        }
        const uint8_t *d;
        if (rbytes(&r, (size_t)len, &d)) return -1;
        s->sections[sid] = (unsigned char *)d;
        s->section_len[sid] = (size_t)len;
    }
    if (!s->sections[0] || !s->sections[1] || !s->sections[2] ||
        !s->sections[3] || !s->sections[4])
        return -1;
    if (parse_strings(s) || parse_dialects(s) || parse_attr_type_tables(s) ||
        parse_props(s))
        return -1;
    return 0;
}

/* Find the entry function: public "main" if present, else the only one. */
static snode *find_main(sl_parser *s, snode *module, snode **funcs, int *nfuncs) {
    *nfuncs = 0;
    snode *chosen = NULL;
    if (!module || module->nregions < 1 || !module->regions[0]) return NULL;
    sregion *reg = module->regions[0];
    for (int i = 0; i < reg->nops; i++) {
        snode *n = &reg->ops[i];
        const copname *on = &s->opnames[n->opname];
        if (strcmp(on->name, "func_v1")) continue;
        if (*nfuncs < 32) funcs[(*nfuncs)++] = n;
        const cattr *nm = n->nprops > 3 ? &n->props[3] : NULL;
        if (nm && nm->kind == AK_STRING && !strcmp(nm->sval, "main"))
            chosen = n;
    }
    return chosen ? chosen : (*nfuncs == 1 ? funcs[0] : NULL);
}

static const cattr *prop(const snode *n, int idx) {
    return idx < n->nprops ? &n->props[idx] : NULL;
}

static int vc4sl_compile(sl_parser *s, snode *entry, snode **funcs, int nfuncs,
                         vc4sl_program *p, char *err, size_t errlen);

int vc4sl_parse(const uint8_t *data, size_t len, vc4sl_program *prog,
                char *err, size_t errlen) {
    memset(prog, 0, sizeof(*prog));
    sl_parser *s = calloc(1, sizeof(*s));
    if (!s) return -1;
    s->data = data;
    s->len = len;
    s->err = err;
    s->errlen = errlen;
    if (parse_top(s)) {
        seterr(err, errlen, "cannot read MLIR bytecode (version %" PRIu64 ")",
               s->version);
        return -1;
    }
    /* Parse the nested regions iteratively: the module's own region holds the
     * functions. */
    rd r = {s->sections[4], s->section_len[4], 0};
    uint64_t nops;
    int has_args;
    if (rvf(&r, &nops, &has_args)) {
        seterr(err, errlen, "bad IR section");
        return -1;
    }
    if (has_args) {
        uint64_t nargs;
        if (rv(&r, &nargs)) return -1;
        for (uint64_t i = 0; i < nargs; i++) {
            uint64_t t;
            if (rv(&r, &t)) return -1;
        }
        uint8_t ul;
        if (rbyte(&r, &ul)) return -1;
        if (ul) return -1;
    }
    snode *top = calloc(nops ? (size_t)nops : 1, sizeof(snode));
    for (uint64_t i = 0; i < nops; i++)
        if (parse_op(s, &r, &top[i])) {
            if (!err[0])
                seterr(err, errlen, "cannot parse operation %" PRIu64, i);
            return -1;
        }
    snode *module = nops ? &top[0] : NULL;
    snode *funcs[32];
    int nfuncs = 0;
    snode *entry = find_main(s, module, funcs, &nfuncs);
    if (!entry) {
        seterr(err, errlen, "no entry function in module");
        return -1;
    }
    if (vc4sl_compile(s, entry, funcs, nfuncs, prog, err, errlen)) {
        if (err && errlen && !err[0])
            seterr(err, errlen, "program is outside the executed subset");
        return -1;
    }
    /* keep no parse state: everything the program needs is copied out */
    return 0;
}

/* ==========================================================================
 * Lowering
 *
 * The module is a single-block function calling other single-block private
 * functions; calls are inlined here.  Every SSA value that survives into an
 * instruction gets one arena tensor slot.  Reshape and f32->f32 convert are
 * aliases because the storage is contiguous row-major either way.
 * ========================================================================== */

typedef struct {
    sl_parser *s;
    vc4sl_program *p;
    snode *funcs[32];
    int nfuncs;
    char *err;
    size_t errlen;
    int depth;
} comp;

static int cfail(comp *c, const char *fmt, ...) {
    if (c->err && c->errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(c->err, c->errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static size_t shape_elems(const vc4sl_shape *s) {
    size_t n = 1;
    for (int i = 0; i < s->rank; i++) {
        if (s->dims[i] < 0) return 0;
        n *= (size_t)s->dims[i];
    }
    return n;
}

static int type_shape(comp *c, int tidx, vc4sl_shape *out) {
    ctype t;
    if (decode_type(c->s, tidx, &t)) return -1;
    if (t.kind != NK_TENSOR) return -1;
    memset(out, 0, sizeof(*out));
    out->rank = t.rank;
    memcpy(out->dims, t.dims, sizeof(t.dims));
    out->dtype = (vc4sl_dtype)t.dtype;
    return 0;
}

static int new_tensor(comp *c, vc4sl_shape sh, const char *what) {
    if (c->p->num_tensors >= VC4SL_MAXTENSORS)
        return cfail(c, "too many tensors for the %s", what);
    int slot = c->p->num_tensors++;
    c->p->tensors[slot].shape = sh;
    c->p->tensors[slot].alias_of = -1;
    return slot;
}

static int phys(vc4sl_program *p, int slot) {
    int guard = 0;
    while (slot >= 0 && p->tensors[slot].alias_of >= 0 && guard++ < 32)
        slot = p->tensors[slot].alias_of;
    return slot;
}

static int new_table(comp *c, int n, vc4sl_insn *insn, int32_t **out) {
    if (c->p->num_tables >= VC4SL_MAXINSNS)
        return cfail(c, "too many layout tables");
    int t = c->p->num_tables++;
    c->p->tables[t].idx = calloc(n ? (size_t)n : 1, sizeof(int32_t));
    if (!c->p->tables[t].idx) return cfail(c, "out of memory for a layout table");
    c->p->tables[t].n = n;
    vc4sl_shape sh = {1, {n}, VC4SL_DT_I32};
    int slot = new_tensor(c, sh, "table");
    if (slot < 0) return -1;
    insn->table = t;
    insn->table_slot = slot;
    *out = c->p->tables[t].idx;
    return 0;
}

static int emit(comp *c, vc4sl_insn in) {
    if (c->p->num_insns >= VC4SL_MAXINSNS)
        return cfail(c, "too many instructions");
    c->p->insns[c->p->num_insns++] = in;
    return 0;
}

static int prop_i64(comp *c, const snode *n, int which, int64_t *out, int *count) {
    if (which >= n->nprops || n->props[which].kind != AK_TENSOR)
        return cfail(c, "%s: missing integer list attribute",
                     op_full_name(c->s, (snode *)n));
    const cattr *a = &n->props[which];
    if (a->blob_len % 8) return cfail(c, "bad integer list");
    int k = (int)(a->blob_len / 8);
    if (k > *count)
        return cfail(c, "integer list is longer than %d entries", *count);
    for (int i = 0; i < k; i++) memcpy(&out[i], a->blob + 8 * i, 8);
    *count = k;
    return 0;
}

static int cmp_code(int64_t dir) {
    switch (dir) {
    case 0: return 4; /* EQ */
    case 2: return 1; /* GE */
    case 3: return 0; /* GT */
    case 4: return 3; /* LE */
    case 5: return 2; /* LT */
    default: return -1;
    }
}

static int compile_region(comp *c, sregion *reg, const int *args, int nargs,
                          int *outs, int *nouts, int depth, const char *fname);

static int emit_simple(comp *c, vc4sl_opcode op, int a, int b, int dst,
                       vc4sl_shape sh, const char *name) {
    vc4sl_insn in;
    memset(&in, 0, sizeof(in));
    in.op = op;
    in.a = a;
    in.b = b;
    in.c = 0;
    in.dst = dst;
    in.shape = sh;
    in.table = -1;
    in.table_slot = -1;
    in.name = name;
    return emit(c, in);
}

static snode *find_func(comp *c, const char *name) {
    for (int i = 0; i < c->nfuncs; i++) {
        snode *f = c->funcs[i];
        const cattr *n = prop(f, 3); /* sym_name */
        if (n && n->kind == AK_STRING && !strcmp(n->sval, name)) return f;
    }
    return NULL;
}

static int const_is_zero(comp *c, sregion *reg, int value_idx) {
    /* Is the value produced by a constant op whose payload is all zero? */
    int seen = reg->nargs;
    if (getenv("VC4SL_DEBUG"))
        fprintf(stderr, "zero? value=%d nops=%d nargs=%d\n", value_idx,
                reg->nops, reg->nargs);
    if (value_idx < reg->nargs) return 0; /* a parameter */
    for (int i = 0; i < reg->nops; i++) {
        snode *n = &reg->ops[i];
        if (getenv("VC4SL_DEBUG"))
            fprintf(stderr, "   op%d %s nres=%d seen=%d\n", i,
                    c->s->opnames[n->opname].name, n->nres, seen);
        if (value_idx >= seen && value_idx < seen + n->nres) {
            const char *nm = c->s->opnames[n->opname].name;
            if (strcmp(nm, "constant_v1")) return 0;
            const cattr *v = n->nprops > 0 ? &n->props[0] : NULL;
            if (getenv("VC4SL_DEBUG"))
                fprintf(stderr, "  def %s nprops=%d kind=%d len=%zu\n", nm,
                        n->nprops, v ? v->kind : -1, v ? v->blob_len : 0);
            if (!v || v->kind != AK_TENSOR) return 0;
            for (size_t k = 0; k < v->blob_len; k++)
                if (v->blob[k]) return 0;
            return 1;
        }
        seen += n->nres;
    }
    return 0;
}

static int compile_region(comp *c, sregion *reg, const int *args, int nargs,
                          int *outs, int *nouts, int depth, const char *fname) {
    if (depth > 8) return cfail(c, "call nesting too deep");
    int vals[128];
    int nvals = 0;
    if (nargs > 128) return cfail(c, "too many block arguments");
    for (int i = 0; i < nargs; i++) vals[nvals++] = args[i];

    for (int oi = 0; oi < reg->nops; oi++) {
        snode *n = &reg->ops[oi];
        const copname *on = &c->s->opnames[n->opname];
        const char *name = on->name;
        int rslots[8] = {0};
        int nres = n->nres;

        if (n->noperands > 128) return cfail(c, "too many operands");

        /* Verify every operand index is already defined. */
        for (int i = 0; i < n->noperands; i++)
            if (n->operands[i] < 0 || n->operands[i] >= nvals)
                return cfail(c, "%s: forward operand reference", name);

        /* --- aliases and calls first (they decide their own result slots) */
        int aliased = 0;
        if (!strcmp(name, "reshape_v1")) {
            if (nres != 1 || n->noperands != 1)
                return cfail(c, "reshape_v1: unexpected arity");
            vc4sl_shape sh;
            if (type_shape(c, n->res_types[0], &sh))
                return cfail(c, "reshape_v1: unsupported result type");
            rslots[0] = new_tensor(c, sh, "reshape");
            if (rslots[0] < 0) return -1;
            c->p->tensors[rslots[0]].alias_of = vals[n->operands[0]];
            aliased = 1;
        } else if (!strcmp(name, "convert_v1")) {
            if (nres != 1 || n->noperands != 1)
                return cfail(c, "convert_v1: unexpected arity");
            vc4sl_shape in, out;
            if (type_shape(c, n->res_types[0], &out) || out.dtype != VC4SL_DT_F32)
                return cfail(c, "convert_v1: only f32->f32 is supported");
            in = c->p->tensors[vals[n->operands[0]]].shape;
            if (in.dtype != VC4SL_DT_F32)
                return cfail(c, "convert_v1: only f32->f32 is supported");
            rslots[0] = new_tensor(c, out, "convert");
            if (rslots[0] < 0) return -1;
            c->p->tensors[rslots[0]].alias_of = vals[n->operands[0]];
            aliased = 1;
        } else if (!strcmp(name, "call_v1")) {
            if (n->nprops < 1 || n->props[0].kind != AK_STRING)
                return cfail(c, "call_v1: malformed callee");
            snode *callee = find_func(c, n->props[0].sval);
            if (!callee || callee->nregions < 1)
                return cfail(c, "call_v1: callee @%s not found",
                             n->props[0].sval);
            if (callee->regions[0]->nargs != n->noperands)
                return cfail(c, "call_v1: arity mismatch calling @%s",
                             n->props[0].sval);
            int cargs[16];
            for (int i = 0; i < n->noperands; i++)
                cargs[i] = vals[n->operands[i]];
            int couts[8];
            int ncout = 0;
            if (compile_region(c, callee->regions[0], cargs, n->noperands,
                               couts, &ncout, depth + 1, n->props[0].sval))
                return -1;
            if (ncout != nres)
                return cfail(c, "call_v1: @%s returns %d, call expects %d",
                             n->props[0].sval, ncout, nres);
            for (int i = 0; i < nres; i++) rslots[i] = couts[i];
            aliased = 1;
        } else {
            for (int i = 0; i < nres; i++) {
                vc4sl_shape sh;
                if (type_shape(c, n->res_types[i], &sh))
                    return cfail(c, "%s: unsupported result type", name);
                if (sh.dtype != VC4SL_DT_F32 && sh.dtype != VC4SL_DT_I1)
                    return cfail(c, "%s: result dtype is not fp32", name);
                rslots[i] = new_tensor(c, sh, name);
                if (rslots[i] < 0) return -1;
            }
        }

        if (aliased) goto push_results;

        /* --- the operations themselves */
        vc4sl_insn in;
        memset(&in, 0, sizeof(in));
        in.table = -1;
        in.name = name;

        if (!strcmp(name, "add_v1") || !strcmp(name, "subtract_v1") ||
            !strcmp(name, "multiply_v1")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            in.op = !strcmp(name, "add_v1")      ? VC4SL_ADD
                    : !strcmp(name, "subtract_v1") ? VC4SL_SUB
                                                    : VC4SL_MUL;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.b = phys(c->p, vals[n->operands[1]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (c->p->tensors[in.a].shape.dtype != VC4SL_DT_F32)
                return cfail(c, "%s: operands must be fp32", name);
            if (shape_elems(&c->p->tensors[in.a].shape) !=
                shape_elems(&c->p->tensors[in.b].shape))
                return cfail(c, "%s: operand shapes differ (no broadcast)",
                             name);
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "maximum_v1")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "maximum_v1: unexpected arity");
            int a = phys(c->p, vals[n->operands[0]]);
            int b = phys(c->p, vals[n->operands[1]]);
            vc4sl_shape sh = c->p->tensors[rslots[0]].shape;
            int pred = new_tensor(c, sh, "maximum");
            if (pred < 0) return -1;
            /* cond = a >= b; out = where(cond, a, b).  Both kernels are
               verified on silicon after the VPM store fix. */
            if (emit_simple(c, VC4SL_COMPARE, a, b, pred, sh, name))
                return -1;
            vc4sl_insn in2;
            memset(&in2, 0, sizeof(in2));
            in2.op = VC4SL_SELECT;
            in2.a = pred;
            in2.b = a;
            in2.c = b;
            in2.dst = rslots[0];
            in2.shape = sh;
            in2.table = -1;
            in2.name = name;
            if (emit(c, in2)) return -1;
        } else if (!strcmp(name, "negate_v1")) {
            if (n->noperands != 1 || nres != 1)
                return cfail(c, "negate_v1: unexpected arity");
            in.op = VC4SL_NEGATE;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "relu_v1")) {
            in.op = VC4SL_RELU;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "compare_v1")) {
            if (n->noperands != 2 || nres != 1 || n->nprops < 2)
                return cfail(c, "compare_v1: malformed");
            int code = cmp_code(n->props[1].ival);
            if (code < 0)
                return cfail(c, "compare_v1: comparison direction %" PRId64
                                " is outside the executed subset",
                             n->props[1].ival);
            in.op = VC4SL_COMPARE;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.b = phys(c->p, vals[n->operands[1]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            in.aux[0] = code;
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "select_v1")) {
            if (n->noperands != 3 || nres != 1)
                return cfail(c, "select_v1: unexpected arity");
            vc4sl_insn in2;
            memset(&in2, 0, sizeof(in2));
            in2.op = VC4SL_SELECT;
            in2.a = phys(c->p, vals[n->operands[0]]);
            in2.b = phys(c->p, vals[n->operands[1]]);
            in2.c = phys(c->p, vals[n->operands[2]]);
            in2.dst = rslots[0];
            in2.shape = c->p->tensors[in2.dst].shape;
            in2.table = -1;
            in2.name = name;
            if (emit(c, in2)) return -1;
        } else if (!strcmp(name, "broadcast_in_dim_v1")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 1)
                return cfail(c, "broadcast_in_dim_v1: malformed");
            int64_t dims[VC4SL_MAXRANK];
            int nd = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, dims, &nd))
                return cfail(c, "broadcast_in_dim_v1: bad dimension list");
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dst = c->p->tensors[rslots[0]].shape;
            if (nd != src.rank)
                return cfail(c, "broadcast_in_dim_v1: rank mismatch");
            size_t sstride[VC4SL_MAXRANK];
            size_t acc = 1;
            for (int i = src.rank - 1; i >= 0; i--) {
                sstride[i] = acc;
                acc *= (size_t)src.dims[i];
            }
            for (int i = 0; i < nd; i++)
                if (dims[i] < 0 || dims[i] >= dst.rank)
                    return cfail(c, "broadcast_in_dim_v1: bad dimension");
            size_t nout = shape_elems(&dst);
            int32_t *tab = NULL;
            in.op = VC4SL_GATHER;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = dst;
            if (new_table(c, (int)nout, &in, &tab)) return -1;
            size_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < nout; o++) {
                size_t si = 0;
                for (int j = 0; j < src.rank; j++) {
                    size_t c = src.dims[j] == 1 ? 0 : idx[dims[j]];
                    si += c * sstride[j];
                }
                tab[o] = (int32_t)si;
                for (int d = dst.rank; d-- > 0;) {
                    if (++idx[d] < (size_t)dst.dims[d]) break;
                    idx[d] = 0;
                }
            }
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "transpose_v1")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 1)
                return cfail(c, "transpose_v1: malformed");
            int64_t perm[VC4SL_MAXRANK];
            int np = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, perm, &np))
                return cfail(c, "transpose_v1: bad permutation");
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dst = c->p->tensors[rslots[0]].shape;
            if (np != src.rank || src.rank != dst.rank)
                return cfail(c, "transpose_v1: rank mismatch");
            size_t sstride[VC4SL_MAXRANK];
            size_t acc = 1;
            for (int i = src.rank - 1; i >= 0; i--) {
                sstride[i] = acc;
                acc *= (size_t)src.dims[i];
            }
            size_t nout = shape_elems(&dst);
            int32_t *tab = NULL;
            in.op = VC4SL_GATHER;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = dst;
            if (new_table(c, (int)nout, &in, &tab)) return -1;
            size_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < nout; o++) {
                size_t si = 0;
                for (int j = 0; j < src.rank; j++)
                    si += idx[perm[j]] * sstride[j];
                tab[o] = (int32_t)si;
                for (int d = dst.rank; d-- > 0;) {
                    if (++idx[d] < (size_t)dst.dims[d]) break;
                    idx[d] = 0;
                }
            }
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "dot_general_v1") ||
                   !strcmp(name, "dot_general_v2") ||
                   !strcmp(name, "dot_general_v3")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            int base = !strcmp(name, "dot_general_v1") ? 0 : 0;
            /* v2+ property order: batch(2), lhs_cc, lhs_contract, ..., rhs(2), ... */
            int li_batch = !strcmp(name, "dot_general_v1") ? 0
                           : !strcmp(name, "dot_general_v2") ? 2 : 2;
            int li_lcont = !strcmp(name, "dot_general_v1") ? 2
                           : !strcmp(name, "dot_general_v2") ? 4 : 4;
            int li_rbatch = !strcmp(name, "dot_general_v1") ? 1
                            : !strcmp(name, "dot_general_v2") ? 8 : 8;
            int li_rcont = !strcmp(name, "dot_general_v1") ? 3
                           : !strcmp(name, "dot_general_v2") ? 10 : 10;
            (void)base;
            int64_t lb[2] = {0}, lc[2] = {0}, rb[2] = {0}, rc[2] = {0};
            int nlb = 2, nlc = 2, nrb = 2, nrc = 2;
            if (prop_i64(c, n, li_lcont, lc, &nlc) ||
                prop_i64(c, n, li_rcont, rc, &nrc) ||
                prop_i64(c, n, li_batch, lb, &nlb) ||
                prop_i64(c, n, li_rbatch, rb, &nrb))
                return cfail(c, "%s: malformed dimension lists", name);
            if (nlb || nrb)
                return cfail(c, "%s: batching dimensions are outside the "
                                "executed subset", name);
            if (nlc != 1 || nrc != 1)
                return cfail(c, "%s: exactly one contracting dimension per "
                                "operand is required", name);
            int lhs = vals[n->operands[0]], rhs = vals[n->operands[1]];
            vc4sl_shape ls = c->p->tensors[lhs].shape;
            vc4sl_shape rs = c->p->tensors[rhs].shape;
            if (ls.rank != 2 || rs.rank != 2)
                return cfail(c, "%s: only rank-2 dot_general is executed", name);
            int lcont = (int)lc[0], rcont = (int)rc[0];
            if (lc[0] < 0 || lc[0] > 1 || rc[0] < 0 || rc[0] > 1)
                return cfail(c, "%s: bad contracting dimension", name);
            int M = (int)ls.dims[1 - lcont];
            int K = (int)ls.dims[lcont];
            int N = (int)rs.dims[1 - rcont];
            if ((int)rs.dims[rcont] != K)
                return cfail(c, "%s: contracting lengths differ", name);
            int left = lhs;
            if (lcont == 0) {
                /* materialise the transpose on the QPU before the GEMM */
                vc4sl_shape ts = {2, {K, M}, VC4SL_DT_F32};
                int tmp = new_tensor(c, ts, "dot_general_v1 lhs transpose");
                if (tmp < 0) return -1;
                vc4sl_insn t;
                memset(&t, 0, sizeof(t));
                t.op = VC4SL_GATHER;
                t.name = name;
                t.a = phys(c->p, lhs);
                t.dst = tmp;
                t.shape = ts;
                t.table = -1;
                int32_t *tab = NULL;
                if (new_table(c, M * K, &t, &tab)) return -1;
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++)
                        tab[i * K + k] = k * M + i;
                if (emit(c, t)) return -1;
                left = tmp;
            }
            int right = rhs;
            if (rcont == 1) {
                /* vc4_matmul_bt requires all dims % 16; transpose B on the
                   host (bytes only) and use the general GEMM instead. */
                vc4sl_shape bts = {2, {K, N}, VC4SL_DT_F32};
                int tmpb = new_tensor(c, bts, "dot_general rhs transpose");
                if (tmpb < 0) return -1;
                vc4sl_insn t;
                memset(&t, 0, sizeof(t));
                t.op = VC4SL_GATHER;
                t.name = name;
                t.a = phys(c->p, rhs);
                t.dst = tmpb;
                t.shape = bts;
                t.table = -1;
                int32_t *tab = NULL;
                if (new_table(c, N * K, &t, &tab)) return -1;
                for (int k = 0; k < K; k++)
                    for (int n = 0; n < N; n++)
                        tab[k * N + n] = n * K + k;
                if (emit(c, t)) return -1;
                right = tmpb;
            }
            in.op = VC4SL_DOT;
            in.a = phys(c->p, left);
            in.b = phys(c->p, right);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            in.aux[0] = M;
            in.aux[1] = N;
            in.aux[2] = K;
            if (emit(c, in)) return -1;
        } else if (!strcmp(name, "reduce_v1")) {
            if (n->noperands < 2 || nres != 1 || n->nprops < 1)
                return cfail(c, "reduce_v1: malformed");
            if (!const_is_zero(c, reg, n->operands[1]))
                return cfail(c, "reduce_v1: only a zero initial value is "
                                "supported");
            int64_t axes[VC4SL_MAXRANK];
            int na = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, axes, &na))
                return cfail(c, "reduce_v1: bad dimension list");
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            int isred[VC4SL_MAXRANK] = {0};
            for (int i = 0; i < na; i++) {
                if (axes[i] < 0 || axes[i] >= src.rank)
                    return cfail(c, "reduce_v1: bad dimension");
                isred[axes[i]] = 1;
            }
            size_t outer = 1, red = 1;
            int seen_reduced = 0;
            for (int i = 0; i < src.rank; i++) {
                if (isred[i]) {
                    seen_reduced = 1;
                    red *= (size_t)src.dims[i];
                } else {
                    if (seen_reduced)
                        return cfail(c, "reduce_v1: only a trailing "
                                        "contiguous reduction is executed");
                    outer *= (size_t)src.dims[i];
                }
            }
            vc4sl_insn in2;
            memset(&in2, 0, sizeof(in2));
            in2.op = VC4SL_REDUCE_SUM;
            in2.a = phys(c->p, vals[n->operands[0]]);
            in2.dst = rslots[0];
            in2.shape = c->p->tensors[in2.dst].shape;
            in2.aux[0] = (int64_t)outer;
            in2.aux[1] = (int64_t)red;
            in2.table = -1;
            in2.name = name;
            if (emit(c, in2)) return -1;
        } else if (!strcmp(name, "constant_v1")) {
            if (nres != 1 || n->nprops < 1)
                return cfail(c, "constant_v1: malformed");
            const cattr *v = &n->props[0];
            if (v->kind != AK_TENSOR)
                return cfail(c, "constant_v1: unsupported value attribute");
            vc4sl_tensor *t = &c->p->tensors[rslots[0]];
            size_t want = shape_elems(&t->shape) * 4;
            t->is_constant = 1;
            t->const_len = v->blob_len;
            t->const_data = malloc(v->blob_len ? v->blob_len : 1);
            if (!t->const_data) return cfail(c, "out of memory");
            memcpy(t->const_data, v->blob, v->blob_len);
            if (v->blob_len != want)
                return cfail(c, "constant_v1: payload is %zu B for %zu B of "
                                "fp32", v->blob_len, want);
        } else if (!strcmp(name, "return_v1")) {
            if (n->noperands > 8)
                return cfail(c, "return_v1: too many values");
            for (int i = 0; i < n->noperands; i++)
                outs[i] = vals[n->operands[i]];
            *nouts = n->noperands;
            return 0;
        } else {
            return cfail(c, "%s.%s is outside the executed subset",
                         c->s->dialects[on->dialect].name, on->name);
        }

push_results:
        for (int i = 0; i < nres; i++) vals[nvals++] = rslots[i];
    }
    return cfail(c, "function @%s ends without return_v1", fname);
}

static int vc4sl_compile(sl_parser *s, snode *entry, snode **funcs, int nfuncs,
                         vc4sl_program *p, char *err, size_t errlen) {
    comp c;
    memset(&c, 0, sizeof(c));
    c.s = s;
    c.p = p;
    c.err = err;
    c.errlen = errlen;
    memcpy(c.funcs, funcs, sizeof(c.funcs));
    c.nfuncs = nfuncs;
    if (!entry->nregions || !entry->regions[0])
        return cfail(&c, "entry function has no body");
    sregion *reg = entry->regions[0];
    if (reg->nargs > 16) return cfail(&c, "too many parameters");
    int args[16];
    for (int i = 0; i < reg->nargs; i++) {
        vc4sl_shape sh;
        if (type_shape(&c, reg->arg_types[i], &sh))
            return cfail(&c, "parameter %d has an unsupported type", i);
        if (sh.dtype != VC4SL_DT_F32)
            return cfail(&c, "parameter %d is not fp32", i);
        int slot = new_tensor(&c, sh, "parameter");
        if (slot < 0) return -1;
        args[i] = slot;
        p->param_slots[p->num_params++] = slot;
    }
    int outs[8] = {0};
    int nout = 0;
    if (compile_region(&c, reg, args, reg->nargs, outs, &nout, 0, "main"))
        return -1;
    if (nout > 16) return cfail(&c, "too many outputs");
    for (int i = 0; i < nout; i++) {
        if (p->tensors[outs[i]].shape.dtype != VC4SL_DT_F32)
            return cfail(&c, "output %d is not fp32", i);
        p->output_shapes[p->num_outputs] = p->tensors[outs[i]].shape;
        p->output_slots[p->num_outputs++] = phys(p, outs[i]);
    }
    return 0;
}

void vc4sl_free(vc4sl_program *prog) {
    for (int i = 0; i < prog->num_tensors; i++) free(prog->tensors[i].const_data);
    for (int i = 0; i < prog->num_tables; i++) free(prog->tables[i].idx);
    prog->num_tensors = prog->num_tables = prog->num_insns = 0;
}

size_t vc4sl_num_elements(const vc4sl_program *prog, int slot) {
    return shape_elems(&prog->tensors[slot].shape);
}
