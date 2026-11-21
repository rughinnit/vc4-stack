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
#include <math.h>
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
    cattr props[24];
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
    if (node->nprops >= 24) return -1;
    node->props[node->nprops] = *v;
    /* the name is recovered from the position; nothing else needs it */
    (void)name;
    node->nprops++;
    return 0;
}

/* VHLO operation names carry the version that was serialized ("slice_v1",
 * "gather_v2").  Split the suffix so the property table can be chosen by op
 * and version instead of by exact spelling. */
static void vhlo_split(const char *name, char *base, size_t n, int *ver) {
    size_t len = strlen(name);
    size_t cut = len;
    int v = 0;
    if (len > 3 && name[len - 3] == '_' && name[len - 2] == 'v' &&
        name[len - 1] >= '0' && name[len - 1] <= '9') {
        v = name[len - 1] - '0';
        cut = len - 3;
    }
    if (cut >= n) cut = n - 1;
    memcpy(base, name, cut);
    base[cut] = 0;
    *ver = v;
}

/* MLIR writes an operation's properties as its attributes in name order.
 * Gather and convolution declare their dimension numbers as individual
 * attributes, so those lists are long but flat. */
static const char *const PROP_FUNC[] = {
    "arg_attrs", "function_type", "res_attrs", "sym_name", "sym_visibility"};
static const char *const PROP_SLICE[] = {
    "limit_indices", "start_indices", "strides"};
static const char *const PROP_PAD[] = {
    "edge_padding_high", "edge_padding_low", "interior_padding"};
static const char *const PROP_GATHER_V1[] = {
    "collapsed_slice_dims", "index_vector_dim", "indices_are_sorted",
    "offset_dims", "slice_sizes", "start_index_map"};
static const char *const PROP_GATHER_V2[] = {
    "collapsed_slice_dims", "index_vector_dim", "indices_are_sorted",
    "offset_dims", "operand_batching_dims", "slice_sizes", "start_index_map",
    "start_indices_batching_dims"};
static const char *const PROP_CONV[] = {
    "batch_group_count", "feature_group_count", "input_batch_dimension",
    "input_feature_dimension", "input_spatial_dimensions",
    "kernel_input_feature_dimension", "kernel_output_feature_dimension",
    "kernel_spatial_dimensions", "lhs_dilation", "output_batch_dimension",
    "output_feature_dimension", "output_spatial_dimensions", "padding",
    "precision_config", "rhs_dilation", "window_reversal", "window_strides"};
static const char *const PROP_DOT_V1[] = {
    "lhs_batching_dimensions", "lhs_contracting_dimensions",
    "precision_config", "rhs_batching_dimensions",
    "rhs_contracting_dimensions"};
static const char *const PROP_DOT_V2[] = {
    "accumulation_type", "allow_imprecise_accumulation",
    "lhs_batching_dimensions", "lhs_component_count",
    "lhs_contracting_dimensions", "lhs_precision_type",
    "num_primitive_operations", "precision_config",
    "rhs_batching_dimensions", "rhs_component_count",
    "rhs_contracting_dimensions", "rhs_precision_type"};

static int prop_list(sl_parser *s, rd *r, snode *node, const char *const *names,
                     int n) {
    for (int i = 0; i < n; i++) {
        cattr a;
        if (prop_attr(s, r, &a) || add_prop(node, names[i], &a)) return -1;
    }
    return 0;
}

static int parse_props_for(sl_parser *s, snode *node, uint64_t pid,
                           const char *dialect, const char *name) {
    if (pid >= (uint64_t)s->nprop_blobs) return -1;
    rd r = {s->prop_blobs[pid].p, s->prop_blobs[pid].n, 0};
    char base[32];
    int ver;
    vhlo_split(name, base, sizeof(base), &ver);
    if (!strcmp(dialect, "builtin") && !strcmp(base, "module")) {
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
    if (strcmp(dialect, "vhlo")) return 0;

    if (!strcmp(base, "func") && ver == 1)
        return prop_list(s, &r, node, PROP_FUNC, 5);
    if (!strcmp(base, "constant") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) || add_prop(node, "value", &a) ? -1 : 0;
    }
    if (!strcmp(base, "iota") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) ||
                       add_prop(node, "iota_dimension", &a)
                   ? -1 : 0;
    }
    if (!strcmp(base, "broadcast_in_dim") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) ||
                       add_prop(node, "broadcast_dimensions", &a)
                   ? -1 : 0;
    }
    if (!strcmp(base, "reduce") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) ||
                       add_prop(node, "dimensions", &a)
                   ? -1 : 0;
    }
    if (!strcmp(base, "transpose") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) ||
                       add_prop(node, "permutation", &a)
                   ? -1 : 0;
    }
    if (!strcmp(base, "reverse") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) ||
                       add_prop(node, "dimensions", &a)
                   ? -1 : 0;
    }
    if (!strcmp(base, "call") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) || add_prop(node, "callee", &a) ? -1 : 0;
    }
    if (!strcmp(base, "slice") && ver == 1)
        return prop_list(s, &r, node, PROP_SLICE, 3);
    if (!strcmp(base, "pad") && ver == 1)
        return prop_list(s, &r, node, PROP_PAD, 3);
    if (!strcmp(base, "compare") && ver == 1) {
        cattr a;
        /* compare_type, comparison_direction */
        if (prop_attr(s, &r, &a) || add_prop(node, "compare_type", &a))
            return -1;
        if (prop_attr(s, &r, &a) || add_prop(node, "comparison_direction", &a))
            return -1;
        return 0;
    }
    if (!strcmp(base, "concatenate") && ver == 1) {
        cattr a;
        return prop_attr(s, &r, &a) || add_prop(node, "dimension", &a) ? -1 : 0;
    }
    if (!strcmp(base, "gather")) {
        if (ver == 1) return prop_list(s, &r, node, PROP_GATHER_V1, 6);
        if (ver == 2) return prop_list(s, &r, node, PROP_GATHER_V2, 8);
    }
    if (!strcmp(base, "convolution") && ver == 1)
        return prop_list(s, &r, node, PROP_CONV, 17);
    if (!strcmp(base, "dot_general")) {
        if (ver == 1) return prop_list(s, &r, node, PROP_DOT_V1, 5);
        if (ver == 2 || ver == 3)
            return prop_list(s, &r, node, PROP_DOT_V2, 12);
    }
    if (!strcmp(base, "func") || !strcmp(base, "slice") ||
        !strcmp(base, "pad") || !strcmp(base, "gather") ||
        !strcmp(base, "convolution") || !strcmp(base, "dot_general") ||
        !strcmp(base, "reduce") || !strcmp(base, "reverse") ||
        !strcmp(base, "iota") || !strcmp(base, "broadcast_in_dim") ||
        !strcmp(base, "transpose") || !strcmp(base, "concatenate")) {
        seterr(s->err, s->errlen, "unsupported version %s", name);
        return -1;
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

/* ==========================================================================
 * Lowering
 *
 * The module is a single-block function calling other single-block private
 * functions; calls are inlined here.  Every SSA value that survives into an
 * instruction gets one arena tensor slot.  Reshape and f32->f32 convert are
 * aliases because the storage is contiguous row-major either way.
 *
 * Integer values (indices, predicates, iota ramps) never reach a kernel.  An
 * expression whose leaves are all constant_v1 is folded here, at compile
 * time, and its values build the int32 gather tables.  The fold reads no
 * argument and no kernel result, so it is not arithmetic on device data.
 * ========================================================================== */

typedef struct {
    int ok;
    int rank;
    int64_t dims[VC4SL_MAXRANK];
    int64_t *v;             /* row-major element values */
} foldv;

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

static void shape_strides(const vc4sl_shape *s, size_t *st) {
    size_t acc = 1;
    for (int i = s->rank - 1; i >= 0; i--) {
        st[i] = acc;
        acc *= (size_t)s->dims[i];
    }
}

/* Row-major coordinate increment; returns 0 when it wrapped. */
static int next_coord(int rank, const int64_t *dims, int64_t *idx) {
    for (int d = rank - 1; d >= 0; d--) {
        if (++idx[d] < dims[d]) return 1;
        idx[d] = 0;
    }
    return 0;
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

/* One vc4_gather over a compile-time index table.  The caller fills `*tab`
 * with the source element index of every output element; a negative entry
 * writes `fill`. */
static int emit_table(comp *c, vc4sl_insn *in, int src, int dst,
                      vc4sl_shape sh, const char *name, float fill,
                      int32_t **tab) {
    memset(in, 0, sizeof(*in));
    in->op = VC4SL_GATHER;
    in->name = name;
    in->a = phys(c->p, src);
    in->dst = dst;
    in->shape = sh;
    in->table = -1;
    in->fill = fill;
    if (new_table(c, (int)shape_elems(&sh), in, tab)) return -1;
    return emit(c, *in);
}

/* The same gather when the destination already has its slot. */
static int emit_table_to(comp *c, int src, int dst, vc4sl_shape sh,
                         const char *name, float fill, int32_t **tab) {
    vc4sl_insn in;
    return emit_table(c, &in, src, dst, sh, name, fill, tab);
}

static int prop_i64(comp *c, const snode *n, int which, int64_t *out,
                    int *count) {
    if (which >= n->nprops || n->props[which].kind != AK_TENSOR)
        return cfail(c, "%s: missing integer list attribute",
                     op_full_name(c->s, (snode *)n));
    const cattr *a = &n->props[which];
    if (a->blob_len % 8) return cfail(c, "bad integer list");
    int ents = 1;
    for (int i = 0; i < a->rank; i++) ents *= (int)a->dims[i];
    if (ents < 0) ents = 0;
    int k = (int)(a->blob_len / 8);
    if (k == 0 && ents == 0) {
        *count = 0;
        return 0;
    }
    /* MLIR stores a splat (every element equal, e.g. strides [1,1] or an
       all-zero padding list) as a single element. */
    if (k == 1 && ents > 1) {
        if (ents > *count)
            return cfail(c, "integer list is longer than %d entries", *count);
        int64_t v;
        memcpy(&v, a->blob, 8);
        for (int i = 0; i < ents; i++) out[i] = v;
        *count = ents;
        return 0;
    }
    if (k != ents)
        return cfail(c, "integer list holds %d of %d entries", k, ents);
    if (k > *count)
        return cfail(c, "integer list is longer than %d entries", *count);
    for (int i = 0; i < k; i++) memcpy(&out[i], a->blob + 8 * i, 8);
    *count = k;
    return 0;
}

/* A scalar integer property (IntegerV1 or an encoded enum). */
static int prop_int(comp *c, const snode *n, int which, int64_t *out,
                    const char *name) {
    if (which >= n->nprops ||
        (n->props[which].kind != AK_INT && n->props[which].kind != AK_BOOL))
        return cfail(c, "%s: missing integer attribute %d", name, which);
    *out = n->props[which].ival;
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

/* The defining operation of an SSA value inside one region, or NULL for a
 * block argument. */
static snode *def_of(sregion *reg, int value_idx) {
    int seen = reg->nargs;
    if (value_idx < reg->nargs) return NULL;
    for (int i = 0; i < reg->nops; i++) {
        snode *n = &reg->ops[i];
        if (value_idx >= seen && value_idx < seen + n->nres) return n;
        seen += n->nres;
    }
    return NULL;
}

/* ------------------------------------------------- compile-time int folding */

__attribute__((unused)) static void fold_drop(foldv *f) {
    free(f->v);
    memset(f, 0, sizeof(*f));
}

/* f->dims must already be set; allocates the element vector. */
static int fold_alloc(comp *c, foldv *f) {
    size_t n = 1;
    for (int i = 0; i < f->rank; i++) {
        if (f->dims[i] < 0) return cfail(c, "fold: negative extent");
        n *= (size_t)f->dims[i];
    }
    f->v = calloc(n ? n : 1, sizeof(int64_t));
    if (!f->v) return cfail(c, "out of memory folding an index value");
    f->ok = 1;
    return 0;
}

static size_t fold_elems(const foldv *f) {
    size_t n = 1;
    for (int i = 0; i < f->rank; i++) n *= (size_t)f->dims[i];
    return n;
}

/* Read a constant_v1 payload as integers.  Returns 0 when the payload is an
 * integer or boolean tensor this folder understands. */
static int fold_constant(comp *c, snode *n, foldv *f) {
    vc4sl_shape sh;
    if (n->nprops < 1 || n->props[0].kind != AK_TENSOR) return 0;
    if (type_shape(c, n->res_types[0], &sh)) return 0;
    f->rank = sh.rank;
    memcpy(f->dims, sh.dims, sizeof(sh.dims));
    const cattr *a = &n->props[0];
    size_t nel = shape_elems(&sh);
    if (sh.dtype == VC4SL_DT_I32 &&
        (a->blob_len == nel * 4 || (nel > 1 && a->blob_len == 4))) {
        if (fold_alloc(c, f)) return -1;
        for (size_t i = 0; i < nel; i++) {
            int32_t w;
            memcpy(&w, a->blob + 4 * (a->blob_len == 4 ? 0 : i), 4);
            f->v[i] = w;
        }
        return 1;
    }
    if (sh.dtype == VC4SL_DT_I64 &&
        (a->blob_len == nel * 8 || (nel > 1 && a->blob_len == 8))) {
        if (fold_alloc(c, f)) return -1;
        for (size_t i = 0; i < nel; i++)
            memcpy(&f->v[i], a->blob + 8 * (a->blob_len == 8 ? 0 : i), 8);
        return 1;
    }
    if (sh.dtype == VC4SL_DT_I1 &&
        (a->blob_len == (nel + 7) / 8 || (nel > 1 && a->blob_len == 1))) {
        if (fold_alloc(c, f)) return -1;
        for (size_t i = 0; i < nel; i++)
            f->v[i] = a->blob_len == 1 ? (a->blob[0] & 1)
                                       : (a->blob[i / 8] >> (i % 8)) & 1;
        return 1;
    }
    return 0;
}

/* Fold one operation whose operands are all compile-time integer values.
 * Returns 1 when the result was folded, 0 when the op is not foldable, and
 * -1 on error. */
static int fold_op(comp *c, sregion *reg, snode *n, const char *name,
                   int ver, const foldv *fold, const foldv *const *argfold,
                   foldv *dst) {
    (void)ver;
    char base[32];
    int bver;
    vhlo_split(name, base, sizeof(base), &bver);
    if (n->nres != 1) return 0;

    if (!strcmp(base, "constant")) return fold_constant(c, n, dst);

    if (!strcmp(base, "iota")) {
        int64_t dim;
        vc4sl_shape sh;
        if (n->nprops < 1 || type_shape(c, n->res_types[0], &sh)) return 0;
        if (sh.dtype != VC4SL_DT_I32 && sh.dtype != VC4SL_DT_I64) return 0;
        if (prop_int(c, n, 0, &dim, name)) return -1;
        if (dim < 0 || dim >= sh.rank)
            return cfail(c, "%s: iota_dimension %lld is outside rank %d",
                         name, (long long)dim, sh.rank);
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        int64_t idx[VC4SL_MAXRANK] = {0};
        for (size_t o = 0; o < shape_elems(&sh); o++) {
            dst->v[o] = idx[dim];
            next_coord(sh.rank, sh.dims, idx);
        }
        return 1;
    }

    if (!strcmp(base, "broadcast_in_dim")) {
        int64_t dims[VC4SL_MAXRANK];
        int nd = VC4SL_MAXRANK;
        vc4sl_shape sh;
        if (n->noperands != 1 || n->nprops < 1) return 0;
        if (!fold[n->operands[0]].ok) return 0;
        if (type_shape(c, n->res_types[0], &sh) || prop_i64(c, n, 0, dims, &nd))
            return 0;
        const foldv *s = &fold[n->operands[0]];
        if (nd != s->rank) return cfail(c, "%s: rank mismatch", name);
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        int64_t idx[VC4SL_MAXRANK] = {0};
        for (size_t o = 0; o < shape_elems(&sh); o++) {
            size_t si = 0;
            for (int j = s->rank - 1; j >= 0; j--) {
                int64_t cd = s->dims[j] == 1 ? 0 : idx[dims[j]];
                si = si * (size_t)s->dims[j] + (size_t)cd;
            }
            dst->v[o] = s->v[si];
            next_coord(sh.rank, sh.dims, idx);
        }
        return 1;
    }

    if (!strcmp(base, "reshape")) {
        vc4sl_shape sh;
        if (n->noperands != 1 || !fold[n->operands[0]].ok) return 0;
        if (type_shape(c, n->res_types[0], &sh)) return 0;
        const foldv *s = &fold[n->operands[0]];
        if (shape_elems(&sh) != fold_elems(s)) return cfail(c, "%s: element count", name);
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        memcpy(dst->v, s->v, fold_elems(s) * sizeof(int64_t));
        return 1;
    }

    if (!strcmp(base, "convert")) {
        if (n->noperands != 1 || !fold[n->operands[0]].ok) return 0;
        const foldv *s = &fold[n->operands[0]];
        dst->rank = s->rank;
        memcpy(dst->dims, s->dims, sizeof(s->dims));
        if (fold_alloc(c, dst)) return -1;
        memcpy(dst->v, s->v, fold_elems(s) * sizeof(int64_t));
        return 1;
    }

    if (!strcmp(base, "negate")) {
        if (n->noperands != 1 || !fold[n->operands[0]].ok) return 0;
        const foldv *s = &fold[n->operands[0]];
        dst->rank = s->rank;
        memcpy(dst->dims, s->dims, sizeof(s->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < fold_elems(s); i++) dst->v[i] = -s->v[i];
        return 1;
    }

    if (!strcmp(base, "reduce")) {
        /* A reduction of a compile-time value (the bounds mask jnp.take
         * builds, for instance) is folded like any other index expression. */
        if (n->noperands != 2 || n->nprops < 1 || n->nregions < 1 ||
            !n->regions[0] || !fold[n->operands[0]].ok ||
            !fold[n->operands[1]].ok)
            return 0;
        const foldv *init = &fold[n->operands[1]];
        if (fold_elems(init) != 1) return 0;
        sregion *body = n->regions[0];
        snode *bop = NULL;
        for (int i = 0; i < body->nops; i++) {
            const char *bn = c->s->opnames[body->ops[i].opname].name;
            char bt[32];
            int bv;
            vhlo_split(bn, bt, sizeof(bt), &bv);
            if (!strcmp(bt, "return")) continue;
            if (bop) return 0;
            bop = &body->ops[i];
        }
        if (!bop) return 0;
        char bb[32];
        int bvv;
        vhlo_split(c->s->opnames[bop->opname].name, bb, sizeof(bb), &bvv);
        int kind;
        if (!strcmp(bb, "add")) kind = 0;
        else if (!strcmp(bb, "maximum")) kind = 1;
        else if (!strcmp(bb, "minimum")) kind = 2;
        else if (!strcmp(bb, "multiply")) kind = 3;
        else if (!strcmp(bb, "and")) kind = 4;
        else if (!strcmp(bb, "or")) kind = 5;
        else return 0;
        int64_t axes[VC4SL_MAXRANK];
        int na = VC4SL_MAXRANK;
        vc4sl_shape sh;
        if (prop_i64(c, n, 0, axes, &na) || type_shape(c, n->res_types[0], &sh))
            return 0;
        const foldv *s0 = &fold[n->operands[0]];
        int isred[VC4SL_MAXRANK] = {0};
        for (int i = 0; i < na; i++) {
            if (axes[i] < 0 || axes[i] >= s0->rank) return 0;
            isred[axes[i]] = 1;
        }
        size_t ostack[VC4SL_MAXRANK];
        int rank = s0->rank;
        size_t acc = 1;
        for (int d = rank - 1; d >= 0; d--) {
            ostack[d] = 0;
            if (isred[d]) continue;
            ostack[d] = acc;
            acc *= (size_t)s0->dims[d];
        }
        if (acc != shape_elems(&sh))
            return cfail(c, "%s: folded reduction has the wrong extent", name);
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < acc; i++) dst->v[i] = init->v[0];
        int64_t idx[VC4SL_MAXRANK] = {0};
        for (size_t i = 0; i < fold_elems(s0); i++) {
            size_t o = 0;
            for (int d = 0; d < rank; d++)
                if (!isred[d]) o += (size_t)idx[d] * ostack[d];
            int64_t v = s0->v[i], *a2 = &dst->v[o];
            switch (kind) {
            case 0: *a2 += v; break;
            case 1: if (v > *a2) *a2 = v; break;
            case 2: if (v < *a2) *a2 = v; break;
            case 3: *a2 *= v; break;
            case 4: *a2 = (*a2 && v); break;
            default: *a2 = (*a2 || v); break;
            }
            next_coord(rank, s0->dims, idx);
        }
        return 1;
    }

    if (!strcmp(base, "and") || !strcmp(base, "or") || !strcmp(base, "xor")) {
        if (n->noperands != 2 || !fold[n->operands[0]].ok ||
            !fold[n->operands[1]].ok)
            return 0;
        const foldv *a = &fold[n->operands[0]], *b = &fold[n->operands[1]];
        if (fold_elems(a) != fold_elems(b))
            return cfail(c, "%s: operand sizes differ", name);
        dst->rank = a->rank;
        memcpy(dst->dims, a->dims, sizeof(a->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < fold_elems(a); i++)
            dst->v[i] = !strcmp(base, "and")   ? (a->v[i] && b->v[i])
                        : !strcmp(base, "or") ? (a->v[i] || b->v[i])
                                              : (a->v[i] != b->v[i]);
        return 1;
    }

    if (!strcmp(base, "not")) {
        if (n->noperands != 1 || !fold[n->operands[0]].ok) return 0;
        const foldv *s = &fold[n->operands[0]];
        dst->rank = s->rank;
        memcpy(dst->dims, s->dims, sizeof(s->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < fold_elems(s); i++) dst->v[i] = !s->v[i];
        return 1;
    }

    if (!strcmp(base, "add") || !strcmp(base, "subtract") ||
        !strcmp(base, "multiply") || !strcmp(base, "maximum") ||
        !strcmp(base, "minimum")) {
        if (n->noperands != 2 || !fold[n->operands[0]].ok ||
            !fold[n->operands[1]].ok)
            return 0;
        const foldv *a = &fold[n->operands[0]], *b = &fold[n->operands[1]];
        if (fold_elems(a) != fold_elems(b))
            return cfail(c, "%s: operand sizes differ", name);
        dst->rank = a->rank;
        memcpy(dst->dims, a->dims, sizeof(a->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < fold_elems(a); i++) {
            int64_t x = a->v[i], y = b->v[i];
            dst->v[i] = !strcmp(base, "add")        ? x + y
                        : !strcmp(base, "subtract") ? x - y
                        : !strcmp(base, "multiply") ? x * y
                        : !strcmp(base, "maximum")  ? (x > y ? x : y)
                                                    : (x < y ? x : y);
        }
        return 1;
    }

    if (!strcmp(base, "compare")) {
        if (n->noperands != 2 || n->nprops < 2 || !fold[n->operands[0]].ok ||
            !fold[n->operands[1]].ok)
            return 0;
        int code = cmp_code(n->props[1].ival);
        if (code < 0) return 0;
        const foldv *a = &fold[n->operands[0]], *b = &fold[n->operands[1]];
        if (fold_elems(a) != fold_elems(b))
            return cfail(c, "%s: operand sizes differ", name);
        dst->rank = a->rank;
        memcpy(dst->dims, a->dims, sizeof(a->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < fold_elems(a); i++) {
            int64_t x = a->v[i], y = b->v[i];
            int r = code == 0 ? x > y : code == 1 ? x >= y
                    : code == 2       ? x < y
                    : code == 3       ? x <= y
                                      : x == y;
            dst->v[i] = r;
        }
        return 1;
    }

    if (!strcmp(base, "select")) {
        if (n->noperands != 3 || !fold[n->operands[0]].ok ||
            !fold[n->operands[1]].ok || !fold[n->operands[2]].ok)
            return 0;
        const foldv *p = &fold[n->operands[0]];
        const foldv *a = &fold[n->operands[1]], *b = &fold[n->operands[2]];
        size_t na = fold_elems(a), np = fold_elems(p);
        if (na != fold_elems(b) || (np != 1 && np != na))
            return cfail(c, "%s: predicate and operand shapes differ", name);
        dst->rank = a->rank;
        memcpy(dst->dims, a->dims, sizeof(a->dims));
        if (fold_alloc(c, dst)) return -1;
        for (size_t i = 0; i < na; i++)
            dst->v[i] = p->v[np == 1 ? 0 : i] ? a->v[i] : b->v[i];
        return 1;
    }

    if (!strcmp(base, "slice")) {
        int64_t start[VC4SL_MAXRANK], limit[VC4SL_MAXRANK], stride[VC4SL_MAXRANK];
        int ns = VC4SL_MAXRANK, nl = VC4SL_MAXRANK, nst = VC4SL_MAXRANK;
        vc4sl_shape sh;
        if (n->noperands != 1 || !fold[n->operands[0]].ok || n->nprops < 3)
            return 0;
        const foldv *s = &fold[n->operands[0]];
        if (prop_i64(c, n, 0, limit, &nl) || prop_i64(c, n, 1, start, &ns) ||
            prop_i64(c, n, 2, stride, &nst) ||
            type_shape(c, n->res_types[0], &sh))
            return 0;
        if (ns != s->rank || nl != s->rank || nst != s->rank)
            return cfail(c, "%s: rank mismatch", name);
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        int64_t idx[VC4SL_MAXRANK] = {0};
        for (size_t o = 0; o < shape_elems(&sh); o++) {
            size_t si = 0;
            for (int d = 0; d < s->rank; d++) {
                int64_t c2 = start[d] + idx[d] * stride[d];
                si = si * (size_t)s->dims[d] + (size_t)c2;
            }
            dst->v[o] = s->v[si];
            next_coord(sh.rank, sh.dims, idx);
        }
        return 1;
    }

    if (!strcmp(base, "reverse")) {
        int64_t dims[VC4SL_MAXRANK];
        int nd = VC4SL_MAXRANK;
        vc4sl_shape sh;
        if (n->noperands != 1 || !fold[n->operands[0]].ok || n->nprops < 1)
            return 0;
        const foldv *s = &fold[n->operands[0]];
        if (prop_i64(c, n, 0, dims, &nd) ||
            type_shape(c, n->res_types[0], &sh))
            return 0;
        dst->rank = s->rank;
        memcpy(dst->dims, s->dims, sizeof(s->dims));
        if (fold_alloc(c, dst)) return -1;
        int rev[VC4SL_MAXRANK] = {0};
        for (int i = 0; i < nd; i++) {
            if (dims[i] < 0 || dims[i] >= s->rank)
                return cfail(c, "%s: bad dimension", name);
            rev[dims[i]] = 1;
        }
        int64_t idx[VC4SL_MAXRANK] = {0};
        for (size_t o = 0; o < fold_elems(s); o++) {
            size_t si = 0;
            for (int d = 0; d < s->rank; d++) {
                int64_t cd = rev[d] ? s->dims[d] - 1 - idx[d] : idx[d];
                si = si * (size_t)s->dims[d] + (size_t)cd;
            }
            dst->v[o] = s->v[si];
            next_coord(s->rank, s->dims, idx);
        }
        return 1;
    }

    if (!strcmp(base, "concatenate")) {
        int64_t dim;
        vc4sl_shape sh;
        if (n->noperands < 1 || n->nprops < 1) return 0;
        if (type_shape(c, n->res_types[0], &sh)) return 0;
        if (prop_int(c, n, 0, &dim, name)) return -1;
        if (dim < 0 || dim >= sh.rank)
            return cfail(c, "%s: concatenate dimension %lld is outside rank %d",
                         name, (long long)dim, sh.rank);
        for (int i = 0; i < n->noperands; i++)
            if (!fold[n->operands[i]].ok) return 0;
        size_t inner = 1;
        for (int d = (int)dim + 1; d < sh.rank; d++)
            inner *= (size_t)sh.dims[d];
        size_t outer = 1;
        for (int d = 0; d < (int)dim; d++) outer *= (size_t)sh.dims[d];
        if (!outer) outer = 1;
        size_t row = (shape_elems(&sh) / outer) * 1;
        dst->rank = sh.rank;
        memcpy(dst->dims, sh.dims, sizeof(sh.dims));
        if (fold_alloc(c, dst)) return -1;
        size_t off = 0;
        for (int i = 0; i < n->noperands; i++) {
            const foldv *s = &fold[n->operands[i]];
            if (s->rank != sh.rank)
                return cfail(c, "%s: operand rank differs", name);
            size_t oi = 1;
            for (int d = (int)dim + 1; d < s->rank; d++)
                oi *= (size_t)s->dims[d];
            if (oi != inner) return cfail(c, "%s: operand shapes differ", name);
            size_t chunk = (size_t)s->dims[dim] * inner;
            for (size_t o = 0; o < outer; o++)
                memcpy(dst->v + o * row + off, s->v + o * chunk,
                       chunk * sizeof(int64_t));
            off += chunk;
        }
        return 1;
    }

    (void)reg;
    (void)argfold;
    (void)bver;
    return 0;
}

/* ------------------------------------------------------ scalar constant read */

/* The scalar float value of an SSA value that is a constant, a convert of an
 * integer constant, or a call argument holding one.  Used for pad's padding
 * value and for a reduction's initial value. */
static int const_float(comp *c, sregion *reg, const foldv *const *argfold,
                       int value_idx, float *out, int depth) {
    if (depth > 2) return -1;
    if (value_idx < reg->nargs) {
        const foldv *a = argfold ? argfold[value_idx] : NULL;
        if (!a || !a->ok || fold_elems(a) != 1) return -1;
        *out = (float)a->v[0];
        return 0;
    }
    snode *n = def_of(reg, value_idx);
    if (!n) return -1;
    char base[32];
    int ver;
    vhlo_split(c->s->opnames[n->opname].name, base, sizeof(base), &ver);
    (void)ver;
    if (!strcmp(base, "constant") && n->nprops >= 1 &&
        n->props[0].kind == AK_TENSOR) {
        vc4sl_shape sh;
        const cattr *a = &n->props[0];
        if (type_shape(c, n->res_types[0], &sh)) return -1;
        if (sh.dtype == VC4SL_DT_F32 && a->blob_len == 4) {
            memcpy(out, a->blob, 4);
            return 0;
        }
        if (sh.dtype == VC4SL_DT_I32 && a->blob_len == 4) {
            int32_t w;
            memcpy(&w, a->blob, 4);
            *out = (float)w;
            return 0;
        }
        if (sh.dtype == VC4SL_DT_I64 && a->blob_len == 8) {
            int64_t w;
            memcpy(&w, a->blob, 8);
            *out = (float)w;
            return 0;
        }
        return -1;
    }
    if (!strcmp(base, "convert") && n->noperands == 1)
        return const_float(c, reg, argfold, n->operands[0], out, depth + 1);
    return -1;
}

static int const_float_is(comp *c, sregion *reg, const foldv *const *argfold,
                          int value_idx, float want) {
    float got;
    if (const_float(c, reg, argfold, value_idx, &got, 0)) return 0;
    if (got == 0.0f && want == 0.0f) return 1; /* +0 and -0 are the same sum */
    uint32_t a, b;
    memcpy(&a, &got, 4);
    memcpy(&b, &want, 4);
    return a == b;
}

/* ------------------------------------------------------------------ lowering */

static int compile_region(comp *c, sregion *reg, const int *args, int nargs,
                          const foldv *const *argfold, int *outs, int *nouts,
                          int depth, const char *fname);

/* Emit the gather that reorders `src` into `sh` following `perm`
 * (destination axis k reads source axis perm[k]). */
static int emit_permute(comp *c, int src, const vc4sl_shape *ssrc, int dst,
                        vc4sl_shape sh, const int *perm, const char *name) {
    size_t sst[VC4SL_MAXRANK];
    shape_strides(ssrc, sst);
    int32_t *tab = NULL;
    if (emit_table_to(c, src, dst, sh, name, 0.0f, &tab)) return -1;
    int64_t idx[VC4SL_MAXRANK] = {0};
    for (size_t o = 0; o < shape_elems(&sh); o++) {
        size_t si = 0;
        for (int k = 0; k < sh.rank; k++)
            si += (size_t)idx[k] * sst[perm[k]];
        tab[o] = (int32_t)si;
        next_coord(sh.rank, sh.dims, idx);
    }
    return 0;
}

/* A folded predicate that an instruction consumes (select) needs real
 * storage: write it as a constant mask of 0.0/1.0, which is what the compare
 * kernel produces. */
static int materialise_mask(comp *c, int slot, const foldv *f, const char *name) {
    slot = phys(c->p, slot); /* an alias has no storage of its own */
    vc4sl_tensor *t = &c->p->tensors[slot];
    if (t->is_constant) return 0;
    size_t nel = shape_elems(&t->shape);
    size_t fe = fold_elems(f);
    if (fe != nel && fe != 1)
        return cfail(c, "%s: predicate holds %zu of %zu elements", name, fe,
                     nel);
    t->is_constant = 1;
    t->const_len = nel * 4;
    t->const_data = malloc(t->const_len ? t->const_len : 1);
    if (!t->const_data) return cfail(c, "out of memory for a predicate");
    float *fp = (float *)t->const_data;
    for (size_t i = 0; i < nel; i++)
        fp[i] = f->v[fe == 1 ? 0 : i] ? 1.0f : 0.0f;
    return 0;
}

static int lower_gather(comp *c, snode *n, const char *name, int ver,
                        const int *vals, const foldv *fold, int dst) {
    const int ivd_i = 1, off_i = 3, coll_i = 0;
    const int simap_i = ver == 1 ? 5 : 6;
    const int ssizes_i = ver == 1 ? 4 : 5;
    const int obatch_i = 4, sibatch_i = 7;
    int64_t ivd = 0;
    int64_t offd[VC4SL_MAXRANK], colld[VC4SL_MAXRANK], simap[VC4SL_MAXRANK];
    int64_t ssizes[VC4SL_MAXRANK], obd[VC4SL_MAXRANK], sibd[VC4SL_MAXRANK];
    int noff = VC4SL_MAXRANK, ncoll = VC4SL_MAXRANK, nsimap = VC4SL_MAXRANK;
    int nss = VC4SL_MAXRANK, nobd = 0, nsibd = 0;

    if (prop_int(c, n, ivd_i, &ivd, name) ||
        prop_i64(c, n, off_i, offd, &noff) ||
        prop_i64(c, n, coll_i, colld, &ncoll) ||
        prop_i64(c, n, simap_i, simap, &nsimap) ||
        prop_i64(c, n, ssizes_i, ssizes, &nss))
        return cfail(c, "%s: malformed dimension attributes", name);
    if (ver == 2) {
        if (prop_i64(c, n, obatch_i, obd, &nobd) ||
            prop_i64(c, n, sibatch_i, sibd, &nsibd))
            return cfail(c, "%s: malformed batching attributes", name);
        if (nobd || nsibd)
            return cfail(c, "%s: operand batching dimensions [%lld] are "
                            "outside the executed subset",
                         name, (long long)(nobd ? obd[0] : sibd[0]));
    }

    vc4sl_shape ds = c->p->tensors[vals[n->operands[0]]].shape;
    vc4sl_shape is = c->p->tensors[vals[n->operands[1]]].shape;
    vc4sl_shape rs = c->p->tensors[dst].shape;
    if (ds.dtype != VC4SL_DT_F32)
        return cfail(c, "%s: the gathered operand must be fp32", name);
    if (nss != ds.rank)
        return cfail(c, "%s: slice_sizes has %d entries for a rank-%d operand",
                     name, nss, ds.rank);
    if (ivd < 0 || ivd > is.rank)
        return cfail(c, "%s: index_vector_dim %lld is outside rank %d",
                     name, (long long)ivd, is.rank);
    int nvec = ivd < is.rank ? (int)is.dims[ivd] : 0;
    if (nsimap != nvec)
        return cfail(c, "%s: start_index_map has %d entries but the index "
                        "vector holds %d", name, nsimap, nvec);

    /* index-space axes: every axis of the indices tensor except the vector */
    int isd[VC4SL_MAXRANK], nisd = 0;
    for (int d = 0; d < is.rank; d++)
        if (d != ivd) isd[nisd++] = d;
    if (nisd + noff != rs.rank)
        return cfail(c, "%s: %d index axes + %d offset axes do not make a "
                        "rank-%d result", name, nisd, noff, rs.rank);

    /* The slice window axes are the non-collapsed data axes, in ascending
     * order; the offset axes describe them in ascending result order. */
    int sliced[VC4SL_MAXRANK], nsliced = 0;
    for (int d = 0; d < ds.rank; d++) {
        if (ssizes[d] < 1 || ssizes[d] > ds.dims[d])
            return cfail(c, "%s: slice size %lld on axis %d of extent %lld",
                         name, (long long)ssizes[d], d, (long long)ds.dims[d]);
        int incoll = 0, insmap = 0;
        for (int k = 0; k < ncoll; k++)
            if (colld[k] == d) incoll = 1;
        for (int k = 0; k < nsimap; k++)
            if (simap[k] == d) insmap = 1;
        if (incoll && (!insmap || ssizes[d] != 1))
            return cfail(c, "%s: axis %d is collapsed but not indexed with a "
                            "slice size of 1", name, d);
        if (ssizes[d] == 1 && insmap && !incoll)
            return cfail(c, "%s: axis %d has a sliced index but is not "
                            "collapsed", name, d);
        if (!insmap && ssizes[d] != ds.dims[d])
            return cfail(c, "%s: axis %d is not indexed and its slice size "
                            "%lld is not the extent %lld", name, d,
                         (long long)ssizes[d], (long long)ds.dims[d]);
        if (ssizes[d] > 1) sliced[nsliced++] = d;
    }
    if (nsliced != noff)
        return cfail(c, "%s: %d sliced axes for %d offset axes", name, nsliced,
                     noff);
    int offpos[VC4SL_MAXRANK];
    for (int k = 0; k < noff; k++) {
        if (offd[k] < 0 || offd[k] >= rs.rank)
            return cfail(c, "%s: offset dimension %lld is outside rank %d",
                         name, (long long)offd[k], rs.rank);
        if (k && offd[k] <= offd[k - 1])
            return cfail(c, "%s: offset dimensions are not ascending", name);
        offpos[k] = (int)offd[k];
    }
    /* result axes that are not offset axes are the index-space axes, in
     * ascending order */
    int idxdim[VC4SL_MAXRANK], nidx = 0;
    for (int r = 0; r < rs.rank; r++) {
        int inoff = 0;
        for (int k = 0; k < noff; k++)
            if (offpos[k] == r) inoff = 1;
        if (!inoff) {
            if (nidx >= nisd) return cfail(c, "%s: too many index axes", name);
            idxdim[nidx++] = r;
        }
    }
    if (nidx != nisd)
        return cfail(c, "%s: %d result axes map to %d index axes", name, nidx,
                     nisd);

    const foldv *iv = &fold[n->operands[1]];
    if (!iv->ok)
        return cfail(c, "%s: the indices are not a compile-time constant, so "
                        "there is no static gather window", name);
    if (fold_elems(iv) != shape_elems(&is))
        return cfail(c, "%s: folded indices hold %zu of %zu elements", name,
                     fold_elems(iv), shape_elems(&is));

    size_t ist[VC4SL_MAXRANK], dst_[VC4SL_MAXRANK];
    shape_strides(&is, ist);
    shape_strides(&ds, dst_);
    size_t nout = shape_elems(&rs);
    int32_t *tab = NULL;
    if (emit_table_to(c, vals[n->operands[0]], dst, rs, name, 0.0f, &tab))
        return -1;
    int64_t rc[VC4SL_MAXRANK] = {0};
    for (size_t o = 0; o < nout; o++) {
        int64_t base = 0;
        for (int k = 0; k < nidx; k++)
            base += rc[idxdim[k]] * (int64_t)ist[isd[k]];
        int64_t src = 0;
        for (int d = 0; d < ds.rank; d++) {
            int64_t pos = 0;
            for (int j = 0; j < nsimap; j++) {
                if (simap[j] != d) continue;
                int64_t lo = 0, hi = ds.dims[d] - ssizes[d];
                int64_t v = iv->v[base + j * (int64_t)ist[ivd]];
                pos = v < lo ? lo : (v > hi ? hi : v);
            }
            for (int k = 0; k < nsliced; k++)
                if (sliced[k] == d) pos += rc[offpos[k]];
            src += pos * (int64_t)dst_[d];
        }
        tab[o] = (int32_t)src;
        next_coord(rs.rank, rs.dims, rc);
    }
    return 0;
}

static int lower_conv(comp *c, snode *n, const char *name, const int *vals,
                      int dst) {
    int64_t bgc = 0, fgc = 0, ibd = 0, ifd = 0, kifd = 0, kofd = 0;
    int64_t obd = 0, ofd = 0;
    int64_t isp[VC4SL_MAXRANK], ksp[VC4SL_MAXRANK], osp[VC4SL_MAXRANK];
    int64_t wst[VC4SL_MAXRANK], ldil[VC4SL_MAXRANK], rdil[VC4SL_MAXRANK];
    int64_t pad[VC4SL_MAXRANK * 2];
    int nisp = VC4SL_MAXRANK, nksp = VC4SL_MAXRANK, nosp = VC4SL_MAXRANK;
    int nwst = VC4SL_MAXRANK, nld = VC4SL_MAXRANK, nrd = VC4SL_MAXRANK;
    int npad = VC4SL_MAXRANK * 2;

    if (n->nprops < 17)
        return cfail(c, "%s: malformed properties", name);
    if (prop_int(c, n, 0, &bgc, name) || prop_int(c, n, 1, &fgc, name) ||
        prop_int(c, n, 2, &ibd, name) || prop_int(c, n, 3, &ifd, name) ||
        prop_i64(c, n, 4, isp, &nisp) || prop_int(c, n, 5, &kifd, name) ||
        prop_int(c, n, 6, &kofd, name) || prop_i64(c, n, 7, ksp, &nksp) ||
        prop_i64(c, n, 8, ldil, &nld) || prop_int(c, n, 9, &obd, name) ||
        prop_int(c, n, 10, &ofd, name) || prop_i64(c, n, 11, osp, &nosp) ||
        prop_i64(c, n, 12, pad, &npad) || prop_i64(c, n, 14, rdil, &nrd) ||
        prop_i64(c, n, 16, wst, &nwst))
        return cfail(c, "%s: malformed properties", name);

    vc4sl_shape ls = c->p->tensors[vals[n->operands[0]]].shape;
    vc4sl_shape rs = c->p->tensors[vals[n->operands[1]]].shape;
    vc4sl_shape os = c->p->tensors[dst].shape;
    if (ls.rank != 4 || rs.rank != 4 || os.rank != 4)
        return cfail(c, "%s: only a 2-D convolution over rank-4 tensors is "
                        "executed (operands are rank %d and %d)",
                     name, ls.rank, rs.rank);
    if (ls.dtype != VC4SL_DT_F32 || rs.dtype != VC4SL_DT_F32)
        return cfail(c, "%s: operands must be fp32", name);
    if (ibd != 0 || ifd != 1 || nisp != 2 || isp[0] != 2 || isp[1] != 3)
        return cfail(c, "%s: input dimension numbers %lld/%lld/[%lld,%lld] are "
                        "not NCHW", name, (long long)ibd, (long long)ifd,
                     (long long)isp[0], (long long)isp[1]);
    if (kofd != 0 || kifd != 1 || nksp != 2 || ksp[0] != 2 || ksp[1] != 3)
        return cfail(c, "%s: kernel dimension numbers %lld/%lld/[%lld,%lld] are "
                        "not OIHW", name, (long long)kofd, (long long)kifd,
                     (long long)ksp[0], (long long)ksp[1]);
    if (obd != 0 || ofd != 1 || nosp != 2 || osp[0] != 2 || osp[1] != 3)
        return cfail(c, "%s: output dimension numbers %lld/%lld/[%lld,%lld] are "
                        "not NCHW", name, (long long)obd, (long long)ofd,
                     (long long)osp[0], (long long)osp[1]);
    if (fgc != 1)
        return cfail(c, "%s: feature_group_count %lld is outside the executed "
                        "subset (1)", name, (long long)fgc);
    if (bgc != 1)
        return cfail(c, "%s: batch_group_count %lld is outside the executed "
                        "subset (1)", name, (long long)bgc);
    if (nwst != 2 || wst[0] != 1 || wst[1] != 1)
        return cfail(c, "%s: window_strides [%lld,%lld] are outside the "
                        "executed subset ([1,1])", name, (long long)wst[0],
                     (long long)wst[1]);
    if (nld != 2 || ldil[0] != 1 || ldil[1] != 1)
        return cfail(c, "%s: lhs_dilation [%lld,%lld] is outside the executed "
                        "subset ([1,1])", name, (long long)ldil[0],
                     (long long)ldil[1]);
    if (nrd != 2 || rdil[0] != 1 || rdil[1] != 1)
        return cfail(c, "%s: rhs_dilation [%lld,%lld] is outside the executed "
                        "subset ([1,1])", name, (long long)rdil[0],
                     (long long)rdil[1]);
    if (npad != 4 && npad != 6)
        return cfail(c, "%s: padding holds %d values for 2 spatial dimensions",
                     name, npad);
    int cols = npad / 2;
    if (pad[0] != pad[1] || pad[0] != pad[cols] || pad[1] != pad[cols + 1])
        return cfail(c, "%s: padding [[%lld,%lld],[%lld,%lld]] is not the same "
                        "on both sides of both spatial axes", name,
                     (long long)pad[0], (long long)pad[1], (long long)pad[cols],
                     (long long)pad[cols + 1]);
    if (cols == 3 && (pad[2] != 0 || pad[cols + 2] != 0))
        return cfail(c, "%s: interior padding [%lld,%lld] is outside the "
                        "executed subset ([0,0])", name, (long long)pad[2],
                     (long long)pad[cols + 2]);
    if (n->nprops > 15 && n->props[15].kind == AK_TENSOR) {
        const cattr *wr = &n->props[15];
        for (size_t i = 0; i < wr->blob_len; i++)
            if (wr->blob[i])
                return cfail(c, "%s: window_reversal is outside the executed "
                                "subset (no reversed windows)", name);
    }

    int N = (int)ls.dims[0], Cin = (int)ls.dims[1];
    int H = (int)ls.dims[2], W = (int)ls.dims[3];
    int Cout = (int)rs.dims[0], Kin = (int)rs.dims[1];
    int KH = (int)rs.dims[2], KW = (int)rs.dims[3];
    int HO = (int)os.dims[2], WO = (int)os.dims[3];
    if (os.dims[0] != N || os.dims[1] != Cout)
        return cfail(c, "%s: result is [%lld,%lld,...] for %d images and %d "
                        "filters", name, (long long)os.dims[0],
                     (long long)os.dims[1], N, Cout);
    if (Kin != Cin)
        return cfail(c, "%s: kernel has %d input channels for %d", name, Kin,
                     Cin);
    if (KH != KW)
        return cfail(c, "%s: kernel %dx%d is not square", name, KH, KW);
    if (!(KH == 1 || KH == 3))
        return cfail(c, "%s: kernel %dx%d is outside the executed subset "
                        "(1x1 and 3x3)", name, KH, KW);
    if (Cin % 16)
        return cfail(c, "%s: %d input channels are not a multiple of 16", name,
                     Cin);
    if (Cout % 16)
        return cfail(c, "%s: %d output channels are not a multiple of 16", name,
                     Cout);
    int pad_v = (int)pad[0];
    if (pad_v < 0)
        return cfail(c, "%s: negative padding %d is outside the executed "
                        "subset", name, pad_v);
    if (KH == 1 && pad_v != 0)
        return cfail(c, "%s: 1x1 convolution with padding %d", name, pad_v);
    int want_ho = (H + 2 * pad_v - KH) + 1;
    int want_wo = (W + 2 * pad_v - KW) + 1;
    if (HO != want_ho || WO != want_wo)
        return cfail(c, "%s: result %dx%d does not match the %dx%d window",
                     name, HO, WO, want_ho, want_wo);
    int wp = (int)((W + 2 * pad_v + 15) / 16) * 16;
    if ((size_t)(H + 2 * pad_v) * (size_t)wp * 4u > 8191u)
        return cfail(c, "%s: padded %dx%d image exceeds the 8191-byte VDR "
                        "pitch", name, H + 2 * pad_v, W + 2 * pad_v);

    vc4sl_insn in;
    memset(&in, 0, sizeof(in));
    in.op = VC4SL_CONV;
    in.name = name;
    in.a = phys(c->p, vals[n->operands[0]]);
    in.b = phys(c->p, vals[n->operands[1]]);
    in.dst = dst;
    in.shape = os;
    in.table = -1;
    in.table_slot = -1;
    in.aux[0] = Cin;
    in.aux[1] = H;
    in.aux[2] = W;
    in.aux[3] = Cout;
    in.aux[4] = KH;
    in.aux[5] = pad_v;
    in.aux[6] = N;
    in.aux[7] = HO;
    in.aux[8] = WO;
    return emit(c, in);
}

static int lower_dot(comp *c, snode *n, const char *name, int ver,
                     const int *vals, int dst) {
    int lb_i, lc_i, rb_i, rc_i;
    if (ver == 1) {
        lb_i = 0;
        lc_i = 1;
        rb_i = 3;
        rc_i = 4;
    } else {
        lb_i = 2;
        lc_i = 4;
        rb_i = 8;
        rc_i = 10;
    }
    int64_t lb[VC4SL_MAXRANK], lc[VC4SL_MAXRANK], rb[VC4SL_MAXRANK];
    int64_t rc[VC4SL_MAXRANK];
    int nlb = VC4SL_MAXRANK, nlc = VC4SL_MAXRANK, nrb = VC4SL_MAXRANK;
    int nrc = VC4SL_MAXRANK;
    if (prop_i64(c, n, lb_i, lb, &nlb) || prop_i64(c, n, lc_i, lc, &nlc) ||
        prop_i64(c, n, rb_i, rb, &nrb) || prop_i64(c, n, rc_i, rc, &nrc))
        return cfail(c, "%s: malformed dimension lists", name);
    if (nlb != nrb)
        return cfail(c, "%s: %d lhs batching dimensions and %d rhs", name, nlb,
                     nrb);
    if (nlc != nrc)
        return cfail(c, "%s: %d lhs contracting dimensions and %d rhs", name,
                     nlc, nrc);

    vc4sl_shape ls = c->p->tensors[vals[n->operands[0]]].shape;
    vc4sl_shape rs = c->p->tensors[vals[n->operands[1]]].shape;
    vc4sl_shape os = c->p->tensors[dst].shape;
    if (ls.dtype != VC4SL_DT_F32 || rs.dtype != VC4SL_DT_F32)
        return cfail(c, "%s: operands must be fp32", name);

    int lseen[VC4SL_MAXRANK] = {0}, rseen[VC4SL_MAXRANK] = {0};
    for (int i = 0; i < nlb; i++) {
        if (lb[i] < 0 || lb[i] >= ls.rank || rb[i] < 0 || rb[i] >= rs.rank)
            return cfail(c, "%s: batching dimension is outside the operands",
                         name);
        if (lseen[lb[i]]++ || rseen[rb[i]]++)
            return cfail(c, "%s: a batching dimension appears twice", name);
        if (ls.dims[lb[i]] != rs.dims[rb[i]])
            return cfail(c, "%s: batching extents differ (%lld and %lld)", name,
                         (long long)ls.dims[lb[i]],
                         (long long)rs.dims[rb[i]]);
    }
    for (int i = 0; i < nlc; i++) {
        if (lc[i] < 0 || lc[i] >= ls.rank || rc[i] < 0 || rc[i] >= rs.rank)
            return cfail(c, "%s: contracting dimension is outside the operands",
                         name);
        if (lseen[lc[i]]++ || rseen[rc[i]]++)
            return cfail(c, "%s: a contracting dimension appears twice", name);
        if (ls.dims[lc[i]] != rs.dims[rc[i]])
            return cfail(c, "%s: contracting extents differ (%lld and %lld)",
                         name, (long long)ls.dims[lc[i]],
                         (long long)rs.dims[rc[i]]);
    }

    /* Canonical orders.  Batches keep the listed order so the pairing between
     * lhs and rhs is preserved; the free axes ascend, which is the order the
     * result uses; the contracting axes keep the listed order for the same
     * pairing reason. */
    int lorder[VC4SL_MAXRANK], rorder[VC4SL_MAXRANK];
    int lpos = 0, rpos = 0;
    for (int i = 0; i < nlb; i++) {
        lorder[lpos++] = (int)lb[i];
        rorder[rpos++] = (int)rb[i];
    }
    for (int d = 0; d < ls.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (lb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (lc[i] == d) used = 1;
        if (!used) lorder[lpos++] = d;
    }
    for (int i = 0; i < nlc; i++) lorder[lpos++] = (int)lc[i];
    for (int i = 0; i < nlc; i++) rorder[rpos++] = (int)rc[i];
    for (int d = 0; d < rs.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (rb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (rc[i] == d) used = 1;
        if (!used) rorder[rpos++] = d;
    }
    if (lpos != ls.rank || rpos != rs.rank)
        return cfail(c, "%s: dimension lists do not cover both operands", name);

    int64_t nbatch = 1, M = 1, K = 1, N = 1;
    for (int i = 0; i < nlb; i++) nbatch *= ls.dims[lb[i]];
    for (int i = 0; i < nlc; i++) K *= ls.dims[lc[i]];
    for (int d = 0; d < ls.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (lb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (lc[i] == d) used = 1;
        if (!used) M *= ls.dims[d];
    }
    for (int d = 0; d < rs.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (rb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (rc[i] == d) used = 1;
        if (!used) N *= rs.dims[d];
    }

    /* result order: batch, lhs free, rhs free */
    int64_t want[VC4SL_MAXRANK];
    int nwant = 0;
    for (int i = 0; i < nlb; i++) want[nwant++] = ls.dims[lb[i]];
    for (int d = 0; d < ls.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (lb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (lc[i] == d) used = 1;
        if (!used) want[nwant++] = ls.dims[d];
    }
    for (int d = 0; d < rs.rank; d++) {
        int used = 0;
        for (int i = 0; i < nlb; i++)
            if (rb[i] == d) used = 1;
        for (int i = 0; i < nlc; i++)
            if (rc[i] == d) used = 1;
        if (!used) want[nwant++] = rs.dims[d];
    }
    if (nwant != os.rank)
        return cfail(c, "%s: result rank %d, expected %d", name, os.rank, nwant);
    for (int i = 0; i < os.rank; i++)
        if (want[i] != os.dims[i])
            return cfail(c, "%s: result shape does not match the free "
                            "dimensions", name);

    /* One index-table transpose per operand when its axes are not already in
     * canonical order; the GEMM then reads contiguous MxK and KxN panels. */
    int lsrc = phys(c->p, vals[n->operands[0]]);
    int rsrc = phys(c->p, vals[n->operands[1]]);
    int identity = 1;
    for (int k = 0; k < ls.rank; k++)
        if (lorder[k] != k) identity = 0;
    if (!identity) {
        vc4sl_shape t;
        memset(&t, 0, sizeof(t));
        t.rank = ls.rank;
        t.dtype = VC4SL_DT_F32;
        for (int k = 0; k < ls.rank; k++) t.dims[k] = ls.dims[lorder[k]];
        int tmp = new_tensor(c, t, "dot_general lhs layout");
        if (tmp < 0) return -1;
        if (emit_permute(c, lsrc, &ls, tmp, t, lorder, name)) return -1;
        lsrc = tmp;
    }
    identity = 1;
    for (int k = 0; k < rs.rank; k++)
        if (rorder[k] != k) identity = 0;
    if (!identity) {
        vc4sl_shape t;
        memset(&t, 0, sizeof(t));
        t.rank = rs.rank;
        t.dtype = VC4SL_DT_F32;
        for (int k = 0; k < rs.rank; k++) t.dims[k] = rs.dims[rorder[k]];
        int tmp = new_tensor(c, t, "dot_general rhs layout");
        if (tmp < 0) return -1;
        if (emit_permute(c, rsrc, &rs, tmp, t, rorder, name)) return -1;
        rsrc = tmp;
    }

    vc4sl_insn in;
    memset(&in, 0, sizeof(in));
    in.op = nbatch > 1 ? VC4SL_DOT_N : VC4SL_DOT;
    in.name = name;
    in.a = lsrc;
    in.b = rsrc;
    in.dst = dst;
    in.shape = os;
    in.table = -1;
    in.table_slot = -1;
    in.aux[0] = M;
    in.aux[1] = N;
    in.aux[2] = K;
    in.aux[3] = nbatch;
    return emit(c, in);
}

static int lower_reduce(comp *c, sregion *reg, snode *n, const char *name,
                        const int *vals, const foldv *const *argfold, int dst) {
    if (n->noperands != 2 || n->nprops < 1 || n->nregions < 1 ||
        !n->regions[0])
        return cfail(c, "%s: malformed", name);
    sregion *body = n->regions[0];
    snode *bop = NULL;
    for (int i = 0; i < body->nops; i++) {
        const char *bn = c->s->opnames[body->ops[i].opname].name;
        char bt[32];
        int bv;
        vhlo_split(bn, bt, sizeof(bt), &bv);
        if (!strcmp(bt, "return")) continue;
        if (bop)
            return cfail(c, "%s: the reduction body is not a single operation",
                         name);
        bop = &body->ops[i];
    }
    if (!bop || bop->nres != 1 || bop->noperands != 2)
        return cfail(c, "%s: the reduction body is not a scalar binary "
                        "operation", name);
    const char *bname = c->s->opnames[bop->opname].name;
    char bbase[32];
    int bver;
    vhlo_split(bname, bbase, sizeof(bbase), &bver);
    int redop;
    float ident;
    const char *ident_text;
    if (!strcmp(bbase, "add")) {
        redop = 0;
        ident = 0.0f;
        ident_text = "0";
    } else if (!strcmp(bbase, "maximum")) {
        redop = 2;
        ident = -INFINITY;
        ident_text = "-inf";
    } else if (!strcmp(bbase, "minimum")) {
        redop = 3;
        ident = INFINITY;
        ident_text = "+inf";
    } else {
        return cfail(c, "%s: the reduction body %s is outside the executed "
                        "subset (add, maximum, minimum)", name, bname);
    }
    if (!const_float_is(c, reg, argfold, n->operands[1], ident))
        return cfail(c, "%s: a %s reduction needs the initial value %s",
                     name, bbase, ident_text);

    int64_t axes[VC4SL_MAXRANK];
    int na = VC4SL_MAXRANK;
    if (prop_i64(c, n, 0, axes, &na))
        return cfail(c, "%s: bad dimension list", name);
    vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
    if (src.dtype != VC4SL_DT_F32)
        return cfail(c, "%s: the reduced operand must be fp32", name);
    int isred[VC4SL_MAXRANK] = {0};
    for (int i = 0; i < na; i++) {
        if (axes[i] < 0 || axes[i] >= src.rank)
            return cfail(c, "%s: dimension %lld is outside rank %d", name,
                         (long long)axes[i], src.rank);
        isred[axes[i]] = 1;
    }
    if (na == 0) { /* an empty axis list is the identity */
        vc4sl_shape osh = c->p->tensors[dst].shape;
        if (osh.rank != src.rank)
            return cfail(c, "%s: an empty reduction cannot change the rank",
                         name);
        for (int d = 0; d < src.rank; d++)
            if (osh.dims[d] != src.dims[d])
                return cfail(c, "%s: an empty reduction cannot change the "
                                "shape", name);
        c->p->tensors[dst].alias_of = vals[n->operands[0]];
        return 0;
    }

    size_t outer = 1, red = 1;
    int trailing = 1, seen_reduced = 0;
    for (int d = 0; d < src.rank; d++) {
        if (isred[d]) {
            seen_reduced = 1;
            red *= (size_t)src.dims[d];
        } else {
            if (seen_reduced) trailing = 0;
            outer *= (size_t)src.dims[d];
        }
    }
    int srcslot = vals[n->operands[0]];
    if (!trailing) {
        /* Move the reduced axes to the end with one index-table gather; the
         * reduced extents then form the trailing contiguous run vc4_reduce
         * walks. */
        int perm[VC4SL_MAXRANK];
        vc4sl_shape t = src;
        int k = 0;
        for (int d = 0; d < src.rank; d++)
            if (!isred[d]) perm[k++] = d;
        for (int d = 0; d < src.rank; d++)
            if (isred[d]) perm[k++] = d;
        for (int d = 0; d < src.rank; d++) t.dims[d] = src.dims[perm[d]];
        int tmp = new_tensor(c, t, "reduce layout");
        if (tmp < 0) return -1;
        if (emit_permute(c, srcslot, &src, tmp, t, perm, name)) return -1;
        srcslot = tmp;
    }

    vc4sl_shape osh = c->p->tensors[dst].shape;
    int kept = 0;
    for (int d = 0; d < src.rank; d++) {
        if (isred[d]) continue;
        if (kept >= osh.rank || osh.dims[kept] != src.dims[d])
            return cfail(c, "%s: the result shape does not match the kept "
                            "axes", name);
        kept++;
    }
    if (kept != osh.rank)
        return cfail(c, "%s: the result shape does not match the kept axes",
                     name);

    vc4sl_insn in;
    memset(&in, 0, sizeof(in));
    in.op = VC4SL_REDUCE;
    in.name = name;
    in.a = phys(c->p, srcslot);
    in.dst = dst;
    in.shape = c->p->tensors[dst].shape;
    in.table = -1;
    in.table_slot = -1;
    in.aux[0] = (int64_t)outer;
    in.aux[1] = (int64_t)red;
    in.aux[2] = redop;
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

static int compile_region_body(comp *c, sregion *reg, const int *args,
                               int nargs, const foldv *const *argfold,
                               int *outs, int *nouts, int depth,
                               const char *fname, foldv *fold) {
    if (depth > 8) return cfail(c, "call nesting too deep");
    int vals[128];
    int nvals = 0;
    if (nargs > 128) return cfail(c, "too many block arguments");
    for (int i = 0; i < nargs; i++) vals[nvals++] = args[i];
    for (int i = 0; i < nargs; i++) {
        if (!argfold || !argfold[i]) continue;
        fold[i] = *argfold[i];
        size_t nel = fold_elems(&fold[i]);
        fold[i].v = malloc(nel ? nel * sizeof(int64_t) : 1);
        if (!fold[i].v) return cfail(c, "out of memory for an index argument");
        memcpy(fold[i].v, argfold[i]->v, nel * sizeof(int64_t));
    }

    for (int oi = 0; oi < reg->nops; oi++) {
        snode *n = &reg->ops[oi];
        const copname *on = &c->s->opnames[n->opname];
        const char *name = on->name;
        int rslots[8] = {0};
        int nres = n->nres;
        int folded = 0;
        char base[32];
        int ver;
        vhlo_split(name, base, sizeof(base), &ver);

        if (n->noperands > 128) return cfail(c, "too many operands");

        /* Verify every operand index is already defined. */
        for (int i = 0; i < n->noperands; i++)
            if (n->operands[i] < 0 || n->operands[i] >= nvals)
                return cfail(c, "%s: forward operand reference", name);

        /* --- aliases and calls first (they decide their own result slots) */
        int aliased = 0;
        if (!strcmp(base, "reshape")) {
            if (nres != 1 || n->noperands != 1)
                return cfail(c, "%s: unexpected arity", name);
            vc4sl_shape sh;
            if (type_shape(c, n->res_types[0], &sh))
                return cfail(c, "%s: unsupported result type", name);
            rslots[0] = new_tensor(c, sh, "reshape");
            if (rslots[0] < 0) return -1;
            c->p->tensors[rslots[0]].alias_of = vals[n->operands[0]];
            aliased = 1;
        } else if (!strcmp(base, "convert")) {
            if (nres != 1 || n->noperands != 1)
                return cfail(c, "%s: unexpected arity", name);
            vc4sl_shape in, out;
            if (type_shape(c, n->res_types[0], &out))
                return cfail(c, "%s: unsupported result type", name);
            in = c->p->tensors[vals[n->operands[0]]].shape;
            if (in.dtype == VC4SL_DT_F32 && out.dtype == VC4SL_DT_F32) {
                rslots[0] = new_tensor(c, out, "convert");
                if (rslots[0] < 0) return -1;
                c->p->tensors[rslots[0]].alias_of = vals[n->operands[0]];
                aliased = 1;
            } else if (in.dtype == VC4SL_DT_F32 || out.dtype != VC4SL_DT_F32) {
                return cfail(c, "%s: only f32->f32 and a constant integer to "
                                "f32 are executed", name);
            } else {
                rslots[0] = new_tensor(c, out, "convert");
                if (rslots[0] < 0) return -1;
            }
            /* an integer constant converted to fp32 becomes a constant
               tensor below, because the executor has no integer storage */
        } else if (!strcmp(base, "call")) {
            if (n->nprops < 1 || n->props[0].kind != AK_STRING)
                return cfail(c, "%s: malformed callee", name);
            snode *callee = find_func(c, n->props[0].sval);
            if (!callee || callee->nregions < 1)
                return cfail(c, "%s: callee @%s not found", name,
                             n->props[0].sval);
            if (callee->regions[0]->nargs != n->noperands)
                return cfail(c, "%s: arity mismatch calling @%s", name,
                             n->props[0].sval);
            int cargs[16];
            const foldv *cargfold[16] = {0};
            for (int i = 0; i < n->noperands; i++) {
                cargs[i] = vals[n->operands[i]];
                cargfold[i] = fold[n->operands[i]].ok ? &fold[n->operands[i]]
                                                      : NULL;
            }
            int couts[8];
            int ncout = 0;
            if (compile_region(c, callee->regions[0], cargs, n->noperands,
                               cargfold, couts, &ncout, depth + 1,
                               n->props[0].sval))
                return -1;
            if (ncout != nres)
                return cfail(c, "%s: @%s returns %d, call expects %d", name,
                             n->props[0].sval, ncout, nres);
            for (int i = 0; i < nres; i++) rslots[i] = couts[i];
            aliased = 1;
        } else {
            for (int i = 0; i < nres; i++) {
                vc4sl_shape sh;
                if (type_shape(c, n->res_types[i], &sh))
                    return cfail(c, "%s: unsupported result type", name);
                if (sh.dtype != VC4SL_DT_F32 && sh.dtype != VC4SL_DT_I1 &&
                    sh.dtype != VC4SL_DT_I32 && sh.dtype != VC4SL_DT_I64)
                    return cfail(c, "%s: result dtype is not fp32", name);
                rslots[i] = new_tensor(c, sh, name);
                if (rslots[i] < 0) return -1;
            }
        }

        if (aliased) goto push_results;

        /* An integer expression whose leaves are constants is evaluated here.
         * Nothing in the vector can depend on an argument, so the fold is
         * compile-time index arithmetic, not a host fallback. */
        if (nres == 1) {
            vc4sl_shape rsh = c->p->tensors[rslots[0]].shape;
            if (!strcmp(base, "convert") && rsh.dtype == VC4SL_DT_F32) {
                const foldv *s = &fold[n->operands[0]];
                if (!s->ok)
                    return cfail(c, "%s: an integer convert that is not a "
                                    "compile-time constant has no fp32 storage "
                                    "in this executor", name);
                if (shape_elems(&rsh) != fold_elems(s))
                    return cfail(c, "%s: element count", name);
                vc4sl_tensor *t = &c->p->tensors[rslots[0]];
                t->is_constant = 1;
                t->const_len = fold_elems(s) * 4;
                t->const_data = malloc(t->const_len ? t->const_len : 1);
                if (!t->const_data) return cfail(c, "out of memory");
                float *fp = (float *)t->const_data;
                for (size_t i = 0; i < fold_elems(s); i++)
                    fp[i] = (float)s->v[i];
                folded = 1;
            } else {
                int rc = fold_op(c, reg, n, name, ver, fold, argfold,
                                 &fold[nvals]);
                if (rc < 0) return -1;
                folded = rc;
            }
            if (folded) goto push_results;
            vc4sl_shape sh = c->p->tensors[rslots[0]].shape;
            if (sh.dtype == VC4SL_DT_I32 || sh.dtype == VC4SL_DT_I64)
                return cfail(c, "%s: an integer result that is not a "
                                "compile-time constant has no fp32 storage in "
                                "this executor", name);
        }

        /* --- the operations themselves */
        vc4sl_insn in;
        memset(&in, 0, sizeof(in));
        in.table = -1;
        in.table_slot = -1;
        in.name = name;

        if (!strcmp(base, "add") || !strcmp(base, "subtract") ||
            !strcmp(base, "multiply")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            in.op = !strcmp(base, "add")        ? VC4SL_ADD
                    : !strcmp(base, "subtract") ? VC4SL_SUB
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
        } else if (!strcmp(base, "maximum") || !strcmp(base, "minimum")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            int a = phys(c->p, vals[n->operands[0]]);
            int b = phys(c->p, vals[n->operands[1]]);
            vc4sl_shape sh = c->p->tensors[rslots[0]].shape;
            int pred = new_tensor(c, sh, "select predicate");
            if (pred < 0) return -1;
            /* maximum: cond = a >= b; minimum: cond = a <= b, then
               out = where(cond, a, b). */
            vc4sl_insn cmp;
            memset(&cmp, 0, sizeof(cmp));
            cmp.op = VC4SL_COMPARE;
            cmp.a = a;
            cmp.b = b;
            cmp.dst = pred;
            cmp.shape = sh;
            cmp.table = -1;
            cmp.table_slot = -1;
            cmp.name = name;
            cmp.aux[0] = !strcmp(base, "maximum") ? 1 : 3; /* GE / LE */
            if (emit(c, cmp)) return -1;
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
        } else if (!strcmp(base, "negate")) {
            if (n->noperands != 1 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            in.op = VC4SL_NEGATE;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (emit(c, in)) return -1;
        } else if (!strcmp(base, "relu")) {
            in.op = VC4SL_RELU;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (emit(c, in)) return -1;
        } else if (!strcmp(base, "compare")) {
            if (n->noperands != 2 || nres != 1 || n->nprops < 2)
                return cfail(c, "%s: malformed", name);
            int code = cmp_code(n->props[1].ival);
            if (code < 0)
                return cfail(c, "%s: comparison direction %lld is outside the "
                                "executed subset", name,
                             (long long)n->props[1].ival);
            in.op = VC4SL_COMPARE;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.b = phys(c->p, vals[n->operands[1]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            in.aux[0] = code;
            if (emit(c, in)) return -1;
        } else if (!strcmp(base, "select")) {
            if (n->noperands != 3 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            if (fold[n->operands[0]].ok) {
                if (materialise_mask(c, vals[n->operands[0]],
                                     &fold[n->operands[0]], name))
                    return -1;
            } else if (c->p->tensors[vals[n->operands[0]]].shape.dtype ==
                       VC4SL_DT_I1) {
                snode *pd = def_of(reg, n->operands[0]);
                const char *pn = pd ? c->s->opnames[pd->opname].name : "?";
                char pb[32];
                int pv;
                vhlo_split(pn, pb, sizeof(pb), &pv);
                if (strcmp(pb, "compare"))
                    return cfail(c, "%s: the predicate is neither a compare nor "
                                    "a compile-time constant (%s)", name, pn);
            }
            in.op = VC4SL_SELECT;
            in.a = phys(c->p, vals[n->operands[0]]);
            in.b = phys(c->p, vals[n->operands[1]]);
            in.c = phys(c->p, vals[n->operands[2]]);
            in.dst = rslots[0];
            in.shape = c->p->tensors[in.dst].shape;
            if (emit(c, in)) return -1;
        } else if (!strcmp(base, "broadcast_in_dim")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 1)
                return cfail(c, "%s: malformed", name);
            int64_t dims[VC4SL_MAXRANK];
            int nd = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, dims, &nd))
                return cfail(c, "%s: bad dimension list", name);
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (nd != src.rank)
                return cfail(c, "%s: rank mismatch", name);
            size_t sstride[VC4SL_MAXRANK];
            shape_strides(&src, sstride);
            for (int i = 0; i < nd; i++)
                if (dims[i] < 0 || dims[i] >= dstsh.rank)
                    return cfail(c, "%s: bad dimension", name);
            size_t nout = shape_elems(&dstsh);
            int32_t *tab = NULL;
            if (emit_table(c, &in, vals[n->operands[0]], rslots[0], dstsh,
                           name, 0.0f, &tab))
                return -1;
            int64_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < nout; o++) {
                size_t si = 0;
                for (int j = 0; j < src.rank; j++) {
                    int64_t cd = src.dims[j] == 1 ? 0 : idx[dims[j]];
                    si += (size_t)cd * sstride[j];
                }
                tab[o] = (int32_t)si;
                next_coord(dstsh.rank, dstsh.dims, idx);
            }
        } else if (!strcmp(base, "transpose")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 1)
                return cfail(c, "%s: malformed", name);
            int64_t perm[VC4SL_MAXRANK];
            int np = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, perm, &np))
                return cfail(c, "%s: bad permutation", name);
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (np != src.rank || src.rank != dstsh.rank)
                return cfail(c, "%s: rank mismatch", name);
            int pm[VC4SL_MAXRANK];
            for (int i = 0; i < np; i++) {
                if (perm[i] < 0 || perm[i] >= src.rank)
                    return cfail(c, "%s: bad dimension", name);
                pm[i] = (int)perm[i];
            }
            if (emit_permute(c, vals[n->operands[0]], &src, rslots[0], dstsh,
                             pm, name))
                return -1;
        } else if (!strcmp(base, "slice")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 3)
                return cfail(c, "%s: malformed", name);
            int64_t start[VC4SL_MAXRANK], limit[VC4SL_MAXRANK];
            int64_t stride[VC4SL_MAXRANK];
            int ns = VC4SL_MAXRANK, nl = VC4SL_MAXRANK, nst = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, limit, &nl) ||
                prop_i64(c, n, 1, start, &ns) ||
                prop_i64(c, n, 2, stride, &nst))
                return cfail(c, "%s: bad index lists", name);
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (ns != src.rank || nl != src.rank || nst != src.rank)
                return cfail(c, "%s: rank-%d slice of a rank-%d operand", name,
                             ns, src.rank);
            size_t sstride[VC4SL_MAXRANK];
            shape_strides(&src, sstride);
            for (int d = 0; d < src.rank; d++) {
                if (stride[d] < 1)
                    return cfail(c, "%s: stride %lld on axis %d is not "
                                    "positive", name, (long long)stride[d], d);
                if (start[d] < 0 || limit[d] > src.dims[d] ||
                    start[d] > limit[d])
                    return cfail(c, "%s: [%lld:%lld] is outside axis %d of "
                                    "extent %lld", name, (long long)start[d],
                                 (long long)limit[d], d,
                                 (long long)src.dims[d]);
                int64_t extent =
                    (limit[d] - start[d] + stride[d] - 1) / stride[d];
                if (extent != dstsh.dims[d])
                    return cfail(c, "%s: axis %d yields %lld elements, the "
                                    "result holds %lld", name, d,
                                 (long long)extent,
                                 (long long)dstsh.dims[d]);
            }
            int32_t *tab = NULL;
            if (emit_table(c, &in, vals[n->operands[0]], rslots[0], dstsh,
                           name, 0.0f, &tab))
                return -1;
            int64_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < shape_elems(&dstsh); o++) {
                size_t si = 0;
                for (int d = 0; d < src.rank; d++)
                    si += (size_t)(start[d] + idx[d] * stride[d]) * sstride[d];
                tab[o] = (int32_t)si;
                next_coord(dstsh.rank, dstsh.dims, idx);
            }
        } else if (!strcmp(base, "reverse")) {
            if (n->noperands != 1 || nres != 1 || n->nprops < 1)
                return cfail(c, "%s: malformed", name);
            int64_t dims[VC4SL_MAXRANK];
            int nd = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, dims, &nd))
                return cfail(c, "%s: bad dimension list", name);
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (dstsh.rank != src.rank)
                return cfail(c, "%s: rank mismatch", name);
            int rev[VC4SL_MAXRANK] = {0};
            for (int i = 0; i < nd; i++) {
                if (dims[i] < 0 || dims[i] >= src.rank)
                    return cfail(c, "%s: bad dimension", name);
                rev[dims[i]] = 1;
            }
            size_t sstride[VC4SL_MAXRANK];
            shape_strides(&src, sstride);
            int32_t *tab = NULL;
            if (emit_table(c, &in, vals[n->operands[0]], rslots[0], dstsh,
                           name, 0.0f, &tab))
                return -1;
            int64_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < shape_elems(&dstsh); o++) {
                size_t si = 0;
                for (int d = 0; d < src.rank; d++) {
                    int64_t cd = rev[d] ? src.dims[d] - 1 - idx[d] : idx[d];
                    si += (size_t)cd * sstride[d];
                }
                tab[o] = (int32_t)si;
                next_coord(dstsh.rank, dstsh.dims, idx);
            }
        } else if (!strcmp(base, "iota")) {
            if (nres != 1 || n->nprops < 1)
                return cfail(c, "%s: malformed", name);
            int64_t dim;
            if (prop_int(c, n, 0, &dim, name))
                return cfail(c, "%s: bad iota dimension", name);
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (dim < 0 || dim >= dstsh.rank)
                return cfail(c, "%s: iota_dimension %lld is outside rank %d",
                             name, (long long)dim, dstsh.rank);
            if (dstsh.dims[dim] > (1 << 24))
                return cfail(c, "%s: %lld values along axis %d exceed the "
                                "2^24 that fp32 represents exactly", name,
                             (long long)dstsh.dims[dim], dim);
            size_t nel = shape_elems(&dstsh);
            vc4sl_tensor *t = &c->p->tensors[rslots[0]];
            t->is_constant = 1;
            t->const_len = nel * 4;
            t->const_data = malloc(t->const_len ? t->const_len : 1);
            if (!t->const_data) return cfail(c, "out of memory");
            int64_t idx[VC4SL_MAXRANK] = {0};
            float *f = (float *)t->const_data;
            for (size_t o = 0; o < nel; o++) {
                f[o] = (float)idx[dim];
                next_coord(dstsh.rank, dstsh.dims, idx);
            }
        } else if (!strcmp(base, "pad")) {
            if (n->noperands != 2 || nres != 1 || n->nprops < 3)
                return cfail(c, "%s: malformed", name);
            int64_t low[VC4SL_MAXRANK], high[VC4SL_MAXRANK];
            int64_t inter[VC4SL_MAXRANK];
            int nlow = VC4SL_MAXRANK, nhigh = VC4SL_MAXRANK;
            int nint = VC4SL_MAXRANK;
            if (prop_i64(c, n, 0, high, &nhigh) ||
                prop_i64(c, n, 1, low, &nlow) ||
                prop_i64(c, n, 2, inter, &nint))
                return cfail(c, "%s: bad padding lists", name);
            vc4sl_shape src = c->p->tensors[vals[n->operands[0]]].shape;
            vc4sl_shape dstsh = c->p->tensors[rslots[0]].shape;
            if (nlow != src.rank || nhigh != src.rank || nint != src.rank)
                return cfail(c, "%s: rank-%d padding of a rank-%d operand",
                             name, nlow, src.rank);
            float fill = 0.0f;
            if (const_float(c, reg, argfold, n->operands[1], &fill, 0))
                return cfail(c, "%s: the padding value is not a compile-time "
                                "constant, so it cannot be a gather fill",
                             name);
            size_t sstride[VC4SL_MAXRANK];
            shape_strides(&src, sstride);
            /* output axis extent and the mapping back into the input */
            for (int d = 0; d < src.rank; d++) {
                if (low[d] < 0 || high[d] < 0 || inter[d] < 0)
                    return cfail(c, "%s: negative padding on axis %d", name, d);
                int64_t extent = low[d] + high[d] + src.dims[d] +
                                 (src.dims[d] - 1) * inter[d];
                if (extent != dstsh.dims[d])
                    return cfail(c, "%s: axis %d pads to %lld elements, the "
                                    "result holds %lld", name, d,
                                 (long long)extent,
                                 (long long)dstsh.dims[d]);
            }
            int32_t *tab = NULL;
            if (emit_table(c, &in, vals[n->operands[0]], rslots[0], dstsh,
                           name, fill, &tab))
                return -1;
            int64_t idx[VC4SL_MAXRANK] = {0};
            for (size_t o = 0; o < shape_elems(&dstsh); o++) {
                size_t si = 0;
                int inside = 1;
                for (int d = 0; d < src.rank; d++) {
                    int64_t t2 = idx[d] - low[d];
                    if (t2 < 0) {
                        inside = 0;
                        break;
                    }
                    int64_t q = t2 / (1 + inter[d]);
                    int64_t r = t2 % (1 + inter[d]);
                    if (r != 0 || q >= src.dims[d]) {
                        inside = 0;
                        break;
                    }
                    si += (size_t)q * sstride[d];
                }
                tab[o] = inside ? (int32_t)si : -1;
                next_coord(dstsh.rank, dstsh.dims, idx);
            }
        } else if (!strcmp(base, "gather")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            if (lower_gather(c, n, name, ver, vals, fold, rslots[0]))
                return -1;
        } else if (!strcmp(base, "dot_general")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            if (lower_dot(c, n, name, ver, vals, rslots[0])) return -1;
        } else if (!strcmp(base, "convolution")) {
            if (n->noperands != 2 || nres != 1)
                return cfail(c, "%s: unexpected arity", name);
            if (lower_conv(c, n, name, vals, rslots[0])) return -1;
        } else if (!strcmp(base, "reduce")) {
            if (lower_reduce(c, reg, n, name, vals, argfold, rslots[0]))
                return -1;
        } else if (!strcmp(base, "constant")) {
            if (nres != 1 || n->nprops < 1)
                return cfail(c, "%s: malformed", name);
            const cattr *v = &n->props[0];
            if (v->kind != AK_TENSOR)
                return cfail(c, "%s: unsupported value attribute", name);
            vc4sl_tensor *t = &c->p->tensors[rslots[0]];
            size_t want = shape_elems(&t->shape) * 4;
            t->is_constant = 1;
            t->const_len = want;
            t->const_data = malloc(want ? want : 1);
            if (!t->const_data) return cfail(c, "out of memory");
            if (v->blob_len == want) {
                memcpy(t->const_data, v->blob, want);
            } else if (v->blob_len == 4) { /* splat */
                for (size_t i = 0; i < want / 4; i++)
                    memcpy(t->const_data + 4 * i, v->blob, 4);
            } else {
                return cfail(c, "%s: payload is %zu B for %zu B of fp32", name,
                             v->blob_len, want);
            }
        } else if (!strcmp(base, "return")) {
            if (n->noperands > 8)
                return cfail(c, "%s: too many values", name);
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

static int compile_region(comp *c, sregion *reg, const int *args, int nargs,
                          const foldv *const *argfold, int *outs, int *nouts,
                          int depth, const char *fname) {
    foldv *fold = calloc(128, sizeof(*fold));
    if (!fold) return cfail(c, "out of memory for the fold table");
    int rc = compile_region_body(c, reg, args, nargs, argfold, outs, nouts,
                                 depth, fname, fold);
    for (int i = 0; i < 128; i++) free(fold[i].v);
    free(fold);
    return rc;
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
    if (compile_region(&c, reg, args, reg->nargs, NULL, outs, &nout, 0, "main"))
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
