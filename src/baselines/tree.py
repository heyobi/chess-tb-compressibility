"""Decision-tree baseline: scikit-learn CART + the same exception coding.

The tree is serialised in pre-order (left child = next node) as four streams
(leaf bitmap, split feature, split threshold, leaf class), concatenated and
compressed with zstd -19. Inference for the size/verification numbers is done
from the deserialised arrays, not by scikit-learn.
"""
import struct

import numpy as np
import zstandard
from sklearn.tree import DecisionTreeClassifier

from model.data import T_P
from model.features import cheb


def numeric_features(t, sq, stm, feat):
    """Small non-negative integer features (all < 256)."""
    cols = []
    for i in range(t.nslots):
        s = sq[:, i].astype(np.int16)
        cols += [s, s & 7, s >> 3]
    cols.append(stm.astype(np.int16))
    wk, bk = sq[:, 0], sq[:, 1]
    cols.append(cheb(wk, bk))
    for i in range(2, t.nslots):
        own, enemy = (wk, bk) if t.color[i] == 0 else (bk, wk)
        cols += [cheb(sq[:, i], own), cheb(sq[:, i], enemy)]
        if t.type[i] == T_P:
            s = sq[:, i].astype(np.int16)
            f, r = s & 7, s >> 3
            if t.color[i] == 0:
                dist, promo = 7 - r, 56 + f
            else:
                dist, promo = r, f
            ek = cheb(enemy, promo)
            tempo = (stm == t.color[i]).astype(np.int16)
            cols += [dist, ek, ek - dist + (1 - tempo) + 8]  # "rule of the square" margin, shifted to be >= 0
    cols += [(feat & 1).astype(np.int16), ((feat >> 1) & 3).astype(np.int16), ((feat >> 3) & 15).astype(np.int16)]
    X = np.stack(cols, axis=1)
    assert X.min() >= 0 and X.max() < 256
    return X.astype(np.uint8)


def fit(X, y, max_leaf_nodes, seed=0):
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=seed)
    clf.fit(X, y)
    return clf


def serialize(clf):
    tr = clf.tree_
    leaf_bits, feats, thrs, classes = [], [], [], []

    def rec(n):
        if tr.children_left[n] == -1:
            leaf_bits.append(1)
            classes.append(int(np.argmax(tr.value[n][0])))
        else:
            leaf_bits.append(0)
            feats.append(int(tr.feature[n]))
            thrs.append(int(np.floor(tr.threshold[n])))
            rec(tr.children_left[n])
            rec(tr.children_right[n])

    import sys
    sys.setrecursionlimit(100000)
    rec(0)
    # classes are indices into clf.classes_
    cls_map = np.asarray(clf.classes_, dtype=np.uint8)
    classes = cls_map[np.asarray(classes, dtype=np.int64)] if classes else np.zeros(0, np.uint8)
    lb = np.packbits(np.asarray(leaf_bits, dtype=np.uint8), bitorder="little")
    raw = (struct.pack("<I", len(leaf_bits)) + lb.tobytes() + np.asarray(feats, np.uint8).tobytes()
           + np.asarray(np.clip(thrs, 0, 255), np.uint8).tobytes() + classes.astype(np.uint8).tobytes())
    return zstandard.ZstdCompressor(level=19).compress(raw)


class FlatTree:
    def __init__(self, blob):
        raw = zstandard.ZstdDecompressor().decompress(blob)
        (n,) = struct.unpack_from("<I", raw, 0)
        pos = 4
        nb = (n + 7) // 8
        leaf = np.unpackbits(np.frombuffer(raw, np.uint8, nb, pos), bitorder="little")[:n].astype(bool)
        pos += nb
        ni = int((~leaf).sum())
        nl = n - ni
        feats = np.frombuffer(raw, np.uint8, ni, pos); pos += ni
        thrs = np.frombuffer(raw, np.uint8, ni, pos); pos += ni
        classes = np.frombuffer(raw, np.uint8, nl, pos); pos += nl
        assert pos == len(raw)
        left = np.full(n, -1, np.int64)
        right = np.full(n, -1, np.int64)
        feat = np.zeros(n, np.int64)
        thr = np.zeros(n, np.int64)
        val = np.zeros(n, np.uint8)
        stack = []  # internal nodes waiting for their right child
        fi = li = 0
        for i in range(n):
            if i > 0:
                p, side = stack.pop()
                if side == 0:
                    left[p] = i
                    stack.append((p, 1))
                else:
                    right[p] = i
            if leaf[i]:
                val[i] = classes[li]; li += 1
            else:
                feat[i] = feats[fi]; thr[i] = thrs[fi]; fi += 1
                stack.append((i, 0))
        # after a node's left subtree completes, its right child is next: the
        # stack above pops (p,0) -> left, pushes (p,1); a later pop gives right.
        self.leaf, self.left, self.right, self.feat, self.thr, self.val = leaf, left, right, feat, thr, val

    def predict(self, X, chunk=1 << 21):
        out = np.empty(len(X), np.uint8)
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk]
            node = np.zeros(len(x), np.int64)
            active = np.arange(len(x))
            while len(active):
                nd = node[active]
                go_left = x[active, self.feat[nd]] <= self.thr[nd]
                nd = np.where(go_left, self.left[nd], self.right[nd])
                node[active] = nd
                active = active[~self.leaf[nd]]
            out[s:s + chunk] = self.val[node]
        return out
