"""Position -> fixed-length list of categorical feature ids.

Every feature is a categorical value mapped to one row of a single embedding
table; the first MLP layer is the sum of the selected rows (equivalent to a
linear layer on the concatenated one-hot vectors, but cheap to evaluate).

Features (documented in docs/METHOD.md):
  * piece-square one-hot: (colour, piece type, square) for every piece
    (identical pieces share rows, so the sum is permutation invariant)
  * side to move
  * Chebyshev distance between the two kings
  * for every non-king piece: Chebyshev distance to its own king and to the
    enemy king, per (colour, piece type)
  * for every pawn: (ranks to promotion, Chebyshev distance of the enemy king
    to the promotion square, whether the pawn's side is to move)
    -> the "rule of the square"
  * move-generator features from tbenum (no tablebase access): side to move
    in check, min(3, #legal captures), min(15, #legal moves)

A "vocabulary" describes which (colour, type) pairs exist. For per-table models
it contains only the table's own pieces; for joint (per piece-count) models it
contains all 12, and the per-table feature list is padded with PAD (row 0 of
the embedding, fixed to zero and not stored).
"""
import numpy as np

from .data import T_K, T_P

PAD = 0


class Vocab:
    def __init__(self, typecolors):
        # typecolors: sorted list of (colour, type)
        self.tcs = sorted(set(typecolors))
        self.tc_index = {tc: i for i, tc in enumerate(self.tcs)}
        ntc = len(self.tcs)
        nonking = [tc for tc in self.tcs if tc[1] != T_K]
        self.nk_index = {tc: i for i, tc in enumerate(nonking)}
        pawn_colours = sorted({c for c, t in self.tcs if t == T_P})
        self.pawn_index = {c: i for i, c in enumerate(pawn_colours)}
        off = 1  # row 0 = PAD
        self.ps = off; off += ntc * 64
        self.stm = off; off += 2
        self.kd = off; off += 8
        self.d_own = off; off += len(nonking) * 8
        self.d_enemy = off; off += len(nonking) * 8
        self.rule_sq = off; off += len(pawn_colours) * 7 * 8 * 2
        self.chk = off; off += 2
        self.ncap = off; off += 4
        self.nleg = off; off += 16
        self.rows = off

    @staticmethod
    def for_table(t):
        return Vocab(list(zip(t.color.tolist(), t.type.tolist())))

    @staticmethod
    def full():
        return Vocab([(c, ty) for c in (0, 1) for ty in range(1, 7)])

    def n_features(self, t):
        return t.nslots + 1 + 1 + 2 * (t.nslots - 2) + int((t.type == T_P).sum()) + 3


def cheb(a, b):
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    return np.maximum(np.abs((a & 7) - (b & 7)), np.abs((a >> 3) - (b >> 3)))


def feature_ids(t, vocab, sq, stm, feat, width=None):
    """t: data.Table; sq [n, nslots] int8; stm [n]; feat [n] uint8.
    Returns int32 array [n, width]."""
    n = len(stm)
    cols = []
    wk, bk = sq[:, 0], sq[:, 1]
    for i in range(t.nslots):
        tc = (int(t.color[i]), int(t.type[i]))
        cols.append(vocab.ps + vocab.tc_index[tc] * 64 + sq[:, i].astype(np.int32))
    cols.append(vocab.stm + stm.astype(np.int32))
    cols.append(vocab.kd + cheb(wk, bk).astype(np.int32))
    for i in range(2, t.nslots):
        tc = (int(t.color[i]), int(t.type[i]))
        own, enemy = (wk, bk) if tc[0] == 0 else (bk, wk)
        j = vocab.nk_index[tc] * 8
        cols.append(vocab.d_own + j + cheb(sq[:, i], own).astype(np.int32))
        cols.append(vocab.d_enemy + j + cheb(sq[:, i], enemy).astype(np.int32))
    for i in range(2, t.nslots):
        if t.type[i] != T_P:
            continue
        c = int(t.color[i])
        s = sq[:, i].astype(np.int16)
        f, r = s & 7, s >> 3
        if c == 0:
            dist, promo, enemy = 7 - r, 56 + f, bk
        else:
            dist, promo, enemy = r, f, wk
        tempo = (stm == c).astype(np.int32)
        ek = cheb(enemy, promo)
        cols.append(vocab.rule_sq + vocab.pawn_index[c] * 112 + (dist.astype(np.int32) * 8 + ek) * 2 + tempo)
    cols.append(vocab.chk + (feat & 1).astype(np.int32))
    cols.append(vocab.ncap + ((feat >> 1) & 3).astype(np.int32))
    cols.append(vocab.nleg + ((feat >> 3) & 15).astype(np.int32))
    ids = np.stack(cols, axis=1).astype(np.int32)
    if width is not None and width > ids.shape[1]:
        ids = np.concatenate([ids, np.full((n, width - ids.shape[1]), PAD, np.int32)], axis=1)
    return ids
