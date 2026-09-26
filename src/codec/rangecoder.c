/*
 * Exception-list codec: adaptive binary range coder (LZMA-style, 11-bit
 * probabilities, adaptation shift 5).
 *
 * Input: n exceptions, sorted strictly increasing position indices idx[k]
 * (rank of the position in the canonical enumeration order) and the true
 * label lab[k] (0..7). The decoder knows, for every position of the table, a
 * context byte ctx[pos] (e.g. model's first * 5 + second choice); the label of
 * exception k is coded in context ctx[idx[k]].
 *
 * Stream, per exception:
 *   gap g = idx[k] - idx[k-1] (idx[-1] = -1), g >= 1
 *   nb = floor(log2 g) coded with a 6-bit binary tree, context = min(prev nb, 47)
 *   the bits below the leading one: the top min(nb,4) coded adaptively
 *   (context nb, tree position), the rest as raw (p = 1/2) bits
 *   label: 3-bit binary tree, context ctx[idx[k]]
 * n itself is stored by the caller (container header).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PBITS 11
#define PONE (1u << PBITS)
#define SHIFT 5
#define TOP (1u << 24)
#define NB_CTX 48
#define MANT_ADAPT 4

typedef struct {
  uint64_t low;
  uint32_t range;
  uint8_t cache;
  uint64_t cache_size;
  uint8_t *out;
  size_t pos, cap;
  int overflow;
} Enc;

static void shift_low(Enc *e) {
  if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
    uint8_t temp = e->cache;
    do {
      if (e->pos < e->cap) e->out[e->pos] = (uint8_t)(temp + (uint8_t)(e->low >> 32));
      else e->overflow = 1;
      e->pos++;
      temp = 0xFF;
    } while (--e->cache_size != 0);
    e->cache = (uint8_t)(e->low >> 24);
  }
  e->cache_size++;
  e->low = (e->low & 0x00FFFFFFu) << 8;
}

static inline void enc_bit(Enc *e, uint16_t *p, int bit) {
  uint32_t bound = (e->range >> PBITS) * *p;
  if (!bit) { e->range = bound; *p += (PONE - *p) >> SHIFT; }
  else { e->low += bound; e->range -= bound; *p -= *p >> SHIFT; }
  while (e->range < TOP) { e->range <<= 8; shift_low(e); }
}

static inline void enc_direct(Enc *e, int bit) {
  e->range >>= 1;
  if (bit) e->low += e->range;
  while (e->range < TOP) { e->range <<= 8; shift_low(e); }
}

typedef struct {
  uint32_t range, code;
  const uint8_t *in;
  size_t pos, len;
} Dec;

static inline uint8_t next_byte(Dec *d) { return d->pos < d->len ? d->in[d->pos++] : 0; }

static inline int dec_bit(Dec *d, uint16_t *p) {
  uint32_t bound = (d->range >> PBITS) * *p;
  int bit;
  if (d->code < bound) { d->range = bound; *p += (PONE - *p) >> SHIFT; bit = 0; }
  else { d->code -= bound; d->range -= bound; *p -= *p >> SHIFT; bit = 1; }
  while (d->range < TOP) { d->range <<= 8; d->code = (d->code << 8) | next_byte(d); }
  return bit;
}

static inline int dec_direct(Dec *d) {
  d->range >>= 1;
  int bit = 0;
  if (d->code >= d->range) { d->code -= d->range; bit = 1; }
  while (d->range < TOP) { d->range <<= 8; d->code = (d->code << 8) | next_byte(d); }
  return bit;
}

typedef struct {
  uint16_t nb[NB_CTX][64];
  uint16_t mant[64][1 << (MANT_ADAPT + 1)];
  uint16_t lab[256][8];
} Model;

static void model_init(Model *m) {
  uint16_t *p = (uint16_t *)m;
  for (size_t i = 0; i < sizeof(Model) / 2; i++) p[i] = PONE / 2;
}

static inline int ilog2(uint64_t g) { return 63 - __builtin_clzll(g); }

/* returns number of bytes written (may exceed cap: then call again with a
   larger buffer) */
size_t exc_encode(const uint64_t *idx, const uint8_t *lab, const uint8_t *ctx,
                  size_t n, uint8_t *out, size_t cap) {
  Model *m = malloc(sizeof(Model));
  model_init(m);
  Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0, cap, 0};
  int64_t prev = -1;
  int pnb = 0;
  for (size_t k = 0; k < n; k++) {
    uint64_t g = (uint64_t)((int64_t)idx[k] - prev);
    prev = (int64_t)idx[k];
    int nb = ilog2(g);
    uint16_t *t = m->nb[pnb < NB_CTX ? pnb : NB_CTX - 1];
    int node = 1;
    for (int i = 5; i >= 0; i--) { int b = (nb >> i) & 1; enc_bit(&e, &t[node], b); node = node * 2 + b; }
    int na = nb < MANT_ADAPT ? nb : MANT_ADAPT;
    node = 1;
    for (int i = nb - 1; i >= 0; i--) {
      int b = (g >> i) & 1;
      if (nb - 1 - i < na) { enc_bit(&e, &m->mant[nb][node], b); node = node * 2 + b; }
      else enc_direct(&e, b);
    }
    pnb = nb;
    uint16_t *lt = m->lab[ctx[idx[k]]];
    node = 1;
    for (int i = 2; i >= 0; i--) { int b = (lab[k] >> i) & 1; enc_bit(&e, &lt[node], b); node = node * 2 + b; }
  }
  for (int i = 0; i < 5; i++) shift_low(&e);
  free(m);
  return e.pos;
}

/* decodes n exceptions; returns 0 on success */
int exc_decode(const uint8_t *in, size_t len, size_t n, const uint8_t *ctx,
               uint64_t npos, uint64_t *idx, uint8_t *lab) {
  Model *m = malloc(sizeof(Model));
  model_init(m);
  Dec d = {0xFFFFFFFFu, 0, in, 0, len};
  for (int i = 0; i < 5; i++) d.code = (d.code << 8) | next_byte(&d);
  int64_t prev = -1;
  int pnb = 0;
  for (size_t k = 0; k < n; k++) {
    uint16_t *t = m->nb[pnb < NB_CTX ? pnb : NB_CTX - 1];
    int node = 1;
    for (int i = 0; i < 6; i++) node = node * 2 + dec_bit(&d, &t[node]);
    int nb = node - 64;
    int na = nb < MANT_ADAPT ? nb : MANT_ADAPT;
    uint64_t g = 1;
    node = 1;
    for (int i = nb - 1; i >= 0; i--) {
      int b;
      if (nb - 1 - i < na) { b = dec_bit(&d, &m->mant[nb][node]); node = node * 2 + b; }
      else b = dec_direct(&d);
      g = (g << 1) | (uint64_t)b;
    }
    pnb = nb;
    prev += (int64_t)g;
    if (prev < 0 || (uint64_t)prev >= npos) { free(m); return 1; }
    idx[k] = (uint64_t)prev;
    uint16_t *lt = m->lab[ctx[prev]];
    node = 1;
    for (int i = 0; i < 3; i++) node = node * 2 + dec_bit(&d, &lt[node]);
    lab[k] = (uint8_t)(node - 8);
  }
  free(m);
  return 0;
}
