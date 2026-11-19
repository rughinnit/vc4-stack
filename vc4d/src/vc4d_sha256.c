/* vc4d_sha256.c - minimal SHA-256 (FIPS 180-4), used for the program
 * allowlist.  Standard implementation; no external dependencies. */
#include "vc4d.h"

#include <stdio.h>
#include <string.h>

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void block(struct sha256 *s, const uint8_t *p)
{
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
               (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        uint32_t a = w[i - 15], b = w[i - 2];
        w[i] = w[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[i - 7] +
               (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
    uint32_t e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                      ((e & f) ^ (~e & g)) + K[i] + w[i];
        uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                      ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

void sha256_init(struct sha256 *s)
{
    static const uint32_t iv[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    memcpy(s->h, iv, sizeof(iv));
    s->bytes = 0;
    s->n = 0;
}

void sha256_update(struct sha256 *s, const void *p, size_t n)
{
    const uint8_t *q = p;
    s->bytes += n;
    while (n) {
        size_t take = 64 - s->n;
        if (take > n)
            take = n;
        memcpy(s->buf + s->n, q, take);
        s->n += take; q += take; n -= take;
        if (s->n == 64) {
            block(s, s->buf);
            s->n = 0;
        }
    }
}

void sha256_final(struct sha256 *s, uint8_t out[32])
{
    uint64_t bits = s->bytes * 8;
    uint8_t pad[72];
    size_t pn = 64 - s->n;
    memset(pad, 0, sizeof(pad));
    pad[0] = 0x80;
    if (pn < 9)
        pn += 64;
    for (int i = 0; i < 8; ++i)
        pad[pn - 1 - i] = (uint8_t)(bits >> (8 * i));
    sha256_update(s, pad, pn);
    for (int i = 0; i < 8; ++i) {
        out[4 * i]     = (uint8_t)(s->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)(s->h[i]);
    }
}

/* parse "hex64 [comment]" lines; returns 1 if the hash is listed */
int sha256_allowlisted(const char *path, const uint8_t hash[32], char *err, size_t errn)
{
    FILE *f = fopen(path, "re");
    if (!f) {
        snprintf(err, errn, "allowlist %s: cannot open", path);
        return -1;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t')
            ++p;
        if (*p == '#' || *p == '\n' || !*p)
            continue;
        int ok = 1;
        for (int i = 0; i < 32; ++i) {
            int hi, lo;
            char a = p[2 * i], b = p[2 * i + 1];
            hi = a >= '0' && a <= '9' ? a - '0' : (a | 32) >= 'a' && (a | 32) <= 'f' ? (a | 32) - 'a' + 10 : -1;
            lo = b >= '0' && b <= '9' ? b - '0' : (b | 32) >= 'a' && (b | 32) <= 'f' ? (b | 32) - 'a' + 10 : -1;
            if (hi < 0 || lo < 0) { ok = 0; break; }
            if ((uint8_t)(hi << 4 | lo) != hash[i]) { ok = 0; break; }
        }
        if (ok) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}
