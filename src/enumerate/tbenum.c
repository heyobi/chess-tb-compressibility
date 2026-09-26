/*
 * tbenum: enumerate every legal, symmetry-canonical position of one Syzygy
 * table in a fixed canonical order and write compact binary files:
 *
 *   valid.bits  packed bitmap over the raw index space (bit i = byte i>>3,
 *               bit (i&7), i.e. numpy bitorder='little'); 1 = the raw index is
 *               a legal canonical position.
 *   feat.u8     one byte per valid position (TB-free features, see below).
 *   labels.u8   one byte per valid position (only without --noprobe):
 *                 bits 0-2  true WDL from the side to move, 0..4
 *                           (0 loss, 1 blessed loss, 2 draw, 3 cursed win, 4 win)
 *                 bits 3-5  best capture value + 3, 0..5
 *                           (0 = no legal capture; else 1..5 = WDL -2..2 + 3)
 *   meta.json   layout of the raw index space and counts.
 *
 * Raw index space (most significant digit first):
 *   stm   : 0 = white to move, 1 = black to move (only 0 for tables with
 *           identical material on both sides; colour flip makes the other
 *           side redundant)
 *   wK    : index into the white-king domain (10 squares of the a1-d1-d4
 *           triangle for pawnless tables, 32 squares of files a-d otherwise)
 *   bK    : 0..63
 *   other : white pieces in table-name order, then black pieces; each 0..63,
 *           pawns 0..47 meaning squares a2..h7.
 *
 * "White" is always the side listed first in the table name (the file
 * KQvKR.rtbw also covers KR v KQ, which is the colour-flipped position).
 *
 * A raw index is valid iff
 *   (1) all squares differ,
 *   (2) identical pieces appear in increasing "square rank" order,
 *   (3) the side NOT to move is not in check (legal),
 *   (4) the position is the lexicographically smallest (by square rank)
 *       among its images under the board symmetries (8 for pawnless tables,
 *       left-right mirror for tables with pawns).
 * No castling rights, no en-passant square (as in Syzygy's index; the value
 * stored for a position is its value without e.p. rights).
 *
 * feat.u8: bit0 = side to move in check, bits1-2 = min(3,#legal captures),
 *          bits3-6 = min(15,#legal moves).
 *
 * Labels come from Fathom's probe_wdl(), i.e. the true WDL value including
 * Syzygy's capture resolution (so values Syzygy stores as "don't care" are
 * resolved). The best-capture value is computed by probing every child
 * after a legal capture (always a smaller table).
 */
#include "tbprobe.c"   /* Fathom, compiled into this unit to reach internals */
#include <omp.h>
#include <getopt.h>
#include <sys/stat.h>
#include <errno.h>

enum { T_P = 1, T_N, T_B, T_R, T_Q, T_K };

typedef struct {
  char name[32];
  int nslots;
  int color[8], type[8], radix[8];
  int group[8];            /* slots with identical (color,type) share group */
  int pawnful, symmetric, nstm, nsym;
  int dom[64], ndom;       /* white king domain */
  int sqrank[64];
  int T[8][64];            /* symmetry transforms */
  uint64_t raw_size, block_size, nblocks;
} Table;

static int piece_type(char c) {
  switch (c) { case 'P': return T_P; case 'N': return T_N; case 'B': return T_B;
    case 'R': return T_R; case 'Q': return T_Q; case 'K': return T_K; }
  return 0;
}

static void setup_table(Table *t, const char *name) {
  memset(t, 0, sizeof(*t));
  strncpy(t->name, name, 31);
  const char *v = strchr(name, 'v');
  if (!v || name[0] != 'K' || v[1] != 'K') { fprintf(stderr, "bad table %s\n", name); exit(1); }
  int n = 0;
  t->color[n] = 0; t->type[n] = T_K; n++;
  t->color[n] = 1; t->type[n] = T_K; n++;
  for (const char *c = name + 1; c < v; c++) { t->color[n] = 0; t->type[n] = piece_type(*c); n++; }
  for (const char *c = v + 2; *c; c++) { t->color[n] = 1; t->type[n] = piece_type(*c); n++; }
  t->nslots = n;
  for (int i = 0; i < n; i++) if (t->type[i] == T_P) t->pawnful = 1;
  size_t lw = v - name, lb = strlen(v + 1);
  t->symmetric = (lw == lb && strncmp(name, v + 1, lw) == 0);
  t->nstm = t->symmetric ? 1 : 2;
  /* groups of identical pieces */
  int g = 0;
  for (int i = 0; i < n; i++) {
    if (i > 0 && t->color[i] == t->color[i-1] && t->type[i] == t->type[i-1] && t->type[i] != T_K)
      t->group[i] = t->group[i-1];
    else t->group[i] = g++;
  }
  /* symmetries */
  for (int s = 0; s < 64; s++) {
    int f = s & 7, r = s >> 3;
    t->T[0][s] = s;
    t->T[1][s] = r * 8 + (7 - f);               /* mirror left-right */
    t->T[2][s] = (7 - r) * 8 + f;               /* mirror top-bottom */
    t->T[3][s] = (7 - r) * 8 + (7 - f);         /* rotate 180 */
    t->T[4][s] = f * 8 + r;                     /* a1-h8 diagonal */
    t->T[5][s] = (7 - f) * 8 + (7 - r);         /* a8-h1 diagonal */
    t->T[6][s] = f * 8 + (7 - r);               /* rotations */
    t->T[7][s] = (7 - f) * 8 + r;
  }
  t->nsym = t->pawnful ? 2 : 8;
  /* square rank: white king domain first */
  int used[64] = {0}, k = 0;
  if (!t->pawnful) {
    static const int tri[10] = {0, 1, 2, 3, 9, 10, 11, 18, 19, 27};
    for (int i = 0; i < 10; i++) { t->dom[i] = tri[i]; t->sqrank[tri[i]] = k++; used[tri[i]] = 1; }
    t->ndom = 10;
  } else {
    for (int s = 0; s < 64; s++) if ((s & 7) < 4) { t->dom[k] = s; t->sqrank[s] = k++; used[s] = 1; }
    t->ndom = 32;
  }
  for (int s = 0; s < 64; s++) if (!used[s]) t->sqrank[s] = k++;
  t->radix[0] = t->ndom; t->radix[1] = 64;
  for (int i = 2; i < n; i++) t->radix[i] = (t->type[i] == T_P) ? 48 : 64;
  t->block_size = 1;
  for (int i = 2; i < n; i++) t->block_size *= t->radix[i];
  t->nblocks = (uint64_t)t->nstm * t->ndom * 64;
  t->raw_size = t->nblocks * t->block_size;
}

/* decode block/offset into squares; returns stm */
static inline int decode(const Table *t, uint64_t blk, uint64_t off, int *sq) {
  for (int i = t->nslots - 1; i >= 2; i--) {
    int d = off % t->radix[i]; off /= t->radix[i];
    sq[i] = (t->type[i] == T_P) ? d + 8 : d;
  }
  sq[1] = blk % 64; blk /= 64;
  sq[0] = t->dom[blk % t->ndom]; blk /= t->ndom;
  return (int)blk;
}

static inline void make_pos(const Table *t, const int *sq, int stm, Pos *p) {
  memset(p, 0, sizeof(*p));
  for (int i = 0; i < t->nslots; i++) {
    uint64_t b = 1ULL << sq[i];
    if (t->color[i] == 0) p->white |= b; else p->black |= b;
    switch (t->type[i]) {
      case T_K: p->kings |= b; break; case T_Q: p->queens |= b; break;
      case T_R: p->rooks |= b; break; case T_B: p->bishops |= b; break;
      case T_N: p->knights |= b; break; case T_P: p->pawns |= b; break;
    }
  }
  p->turn = (stm == 0);   /* Fathom: turn == true means white to move */
}

/* sort key values within groups (insertion sort, tiny n) */
static inline void group_sort(const Table *t, int *key) {
  for (int i = 2; i < t->nslots; i++) {
    int j = i, v = key[i];
    while (j > 2 && t->group[j-1] == t->group[i] && key[j-1] > v) { key[j] = key[j-1]; j--; }
    key[j] = v;
  }
}

static int is_valid_raw(const Table *t, const int *sq, int stm, Pos *pos) {
  int n = t->nslots;
  uint64_t occ = 0;
  for (int i = 0; i < n; i++) {
    uint64_t b = 1ULL << sq[i];
    if (occ & b) return 0;
    occ |= b;
  }
  int key[8];
  for (int i = 0; i < n; i++) key[i] = t->sqrank[sq[i]];
  for (int i = 3; i < n; i++)
    if (t->group[i] == t->group[i-1] && key[i] <= key[i-1]) return 0;
  make_pos(t, sq, stm, pos);
  if (!is_legal(pos)) return 0;
  for (int g = 1; g < t->nsym; g++) {
    int k2[8];
    for (int i = 0; i < n; i++) k2[i] = t->sqrank[t->T[g][sq[i]]];
    group_sort(t, k2);
    for (int i = 0; i < n; i++) {
      if (k2[i] < key[i]) return 0;
      if (k2[i] > key[i]) break;
    }
  }
  return 1;
}

static uint8_t features(const Pos *pos) {
  TbMove moves[TB_MAX_MOVES];
  TbMove *end = gen_moves(pos, moves);
  int nleg = 0, ncap = 0;
  for (TbMove *m = moves; m < end; m++) {
    if (!legal_move(pos, *m)) continue;
    nleg++;
    if (is_capture(pos, *m)) ncap++;
  }
  int chk = is_check(pos);
  if (ncap > 3) ncap = 3;
  if (nleg > 15) nleg = 15;
  return (uint8_t)(chk | (ncap << 1) | (nleg << 3));
}

static uint8_t label(const Pos *pos0, const char *name) {
  Pos pos = *pos0;
  int success;
  int v = probe_wdl(&pos, &success);
  if (!success) { fprintf(stderr, "probe failed in %s\n", name); exit(2); }
  TbMove moves[TB_MAX_CAPTURES];
  TbMove *end = gen_captures(&pos, moves);
  int best = -3;
  for (TbMove *m = moves; m < end; m++) {
    Pos p1;
    if (!is_capture(&pos, *m)) continue;
    if (!do_move(&p1, &pos, *m)) continue;
    int s2;
    int vc = -probe_wdl(&p1, &s2);
    if (!s2) { fprintf(stderr, "child probe failed in %s\n", name); exit(2); }
    if (vc > best) best = vc;
  }
  if (best > v) { fprintf(stderr, "inconsistent capture value in %s\n", name); exit(3); }
  return (uint8_t)((v + 2) | ((best + 3) << 3));
}

static void write_file(const char *dir, const char *fn, const void *buf, size_t n) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, fn);
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(buf, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path); exit(4); }
  fclose(f);
}

int main(int argc, char **argv) {
  const char *tbdir = NULL, *outdir = NULL;
  int noprobe = 0, c;
  static struct option opts[] = {{"tb", 1, 0, 't'}, {"out", 1, 0, 'o'}, {"noprobe", 0, 0, 'n'}, {0, 0, 0, 0}};
  while ((c = getopt_long(argc, argv, "t:o:n", opts, NULL)) != -1) {
    if (c == 't') tbdir = optarg; else if (c == 'o') outdir = optarg; else if (c == 'n') noprobe = 1;
  }
  if (optind >= argc || !outdir || (!noprobe && !tbdir)) {
    fprintf(stderr, "usage: tbenum --tb DIR --out OUTDIR [--noprobe] TABLE\n");
    return 1;
  }
  Table t;
  setup_table(&t, argv[optind]);
  if (!noprobe) {
    if (!tb_init(tbdir) || TB_LARGEST < (unsigned)t.nslots) {
      fprintf(stderr, "tb_init failed / tables missing (largest=%u)\n", TB_LARGEST);
      return 1;
    }
  } else {
    tb_init("");   /* initialises attack tables only */
  }
  mkdir(outdir, 0755);

  uint64_t nb = t.nblocks, bs = t.block_size;
  uint8_t *bits = calloc(t.raw_size / 8 + 1, 1);
  uint64_t *cnt = calloc(nb + 1, sizeof(uint64_t));
  if (bs % 8) { fprintf(stderr, "block size not byte aligned\n"); return 1; }

  /* pass 1: validity */
  #pragma omp parallel for schedule(dynamic, 1)
  for (uint64_t b = 0; b < nb; b++) {
    uint64_t n = 0;
    uint8_t *bb = bits + b * bs / 8;
    for (uint64_t o = 0; o < bs; o++) {
      int sq[8]; Pos pos;
      int stm = decode(&t, b, o, sq);
      if (is_valid_raw(&t, sq, stm, &pos)) { bb[o >> 3] |= 1 << (o & 7); n++; }
    }
    cnt[b + 1] = n;
  }
  for (uint64_t b = 0; b < nb; b++) cnt[b + 1] += cnt[b];
  uint64_t N = cnt[nb];
  uint8_t *feat = malloc(N + 1), *lab = noprobe ? NULL : malloc(N + 1);

  /* pass 2: features and labels */
  #pragma omp parallel for schedule(dynamic, 1)
  for (uint64_t b = 0; b < nb; b++) {
    uint64_t k = cnt[b];
    const uint8_t *bb = bits + b * bs / 8;
    for (uint64_t o = 0; o < bs; o++) {
      if (!(bb[o >> 3] & (1 << (o & 7)))) continue;
      int sq[8]; Pos pos;
      int stm = decode(&t, b, o, sq);
      make_pos(&t, sq, stm, &pos);
      feat[k] = features(&pos);
      if (lab) lab[k] = label(&pos, t.name);
      k++;
    }
  }

  write_file(outdir, "valid.bits", bits, t.raw_size / 8);
  write_file(outdir, "feat.u8", feat, N);
  if (lab) write_file(outdir, "labels.u8", lab, N);

  char path[1024];
  snprintf(path, sizeof path, "%s/meta.json", outdir);
  FILE *f = fopen(path, "w");
  fprintf(f, "{\n  \"table\": \"%s\",\n  \"n_positions\": %llu,\n  \"raw_size\": %llu,\n"
             "  \"pawnful\": %d,\n  \"symmetric\": %d,\n  \"nstm\": %d,\n",
          t.name, (unsigned long long)N, (unsigned long long)t.raw_size, t.pawnful, t.symmetric, t.nstm);
  fprintf(f, "  \"color\": ["); for (int i = 0; i < t.nslots; i++) fprintf(f, "%s%d", i ? ", " : "", t.color[i]);
  fprintf(f, "],\n  \"type\": ["); for (int i = 0; i < t.nslots; i++) fprintf(f, "%s%d", i ? ", " : "", t.type[i]);
  fprintf(f, "],\n  \"radix\": ["); for (int i = 0; i < t.nslots; i++) fprintf(f, "%s%d", i ? ", " : "", t.radix[i]);
  fprintf(f, "],\n  \"wk_domain\": ["); for (int i = 0; i < t.ndom; i++) fprintf(f, "%s%d", i ? ", " : "", t.dom[i]);
  fprintf(f, "],\n  \"block_counts\": [");
  for (uint64_t b = 0; b < nb; b++) fprintf(f, "%s%llu", b ? ", " : "", (unsigned long long)(cnt[b + 1] - cnt[b]));
  fprintf(f, "]\n}\n");
  fclose(f);
  fprintf(stderr, "%s: %llu positions (raw %llu)\n", t.name, (unsigned long long)N, (unsigned long long)t.raw_size);
  return 0;
}
