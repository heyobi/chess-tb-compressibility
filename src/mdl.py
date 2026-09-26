"""MDL sweep for one table: model size vs. exception list size.

Container of an encoded table (what is counted as "our size"):
    varint len(model_blob) | model_blob | exception container
A constant predictor ("const") has model_blob = 1 byte (the class).
Decoding (variant a): label = model prediction, overwritten at exception
positions. Variant (b): decoded = max(that, best capture value), where the
best capture value comes from a 1-ply capture search into smaller tables.
"""
import copy
import hashlib
import math
import os
import time

import numpy as np
import torch

from baselines import raw as rawbase
from baselines import tree as treebase
from codec import exceptions as exc
from model.data import Table
from model.features import Vocab, feature_ids
from model.net import CONFIGS, Net, QNet, allowed_mask, quantize, train

FULL_LIMIT = 40_000_000  # tables up to this many positions keep all features in RAM


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class TableData:
    def __init__(self, name, vocab=None, width=None, train_max=20_000_000, seed=0, full=None, classes=5, no_movegen=False, eval_max=0, tree_feats=False):
        self.t = t = Table(name)
        if no_movegen:  # ablation: hide in-check / #captures / #legal moves
            t.feat[:] = 0
        self.name = name
        self.vocab = vocab or Vocab.for_table(t)
        self.width = width
        self.y = t.wdl.copy()
        self.bc = t.best_capture.copy()
        if classes == 3:
            # loss / draw / win under the 50-move rule: blessed loss and cursed
            # win become draws. The map is monotone, so max(., capture) commutes.
            m = np.array([0, 2, 2, 2, 4], np.uint8)
            self.y = m[self.y]
            bcl = self.bc.astype(np.int16) + 2
            self.bc = np.where(bcl >= 0, m[np.clip(bcl, 0, 4)].astype(np.int16) - 2, -3).astype(np.int8)
        self.n = t.n
        t.labels_raw = None  # y / bc hold everything needed; free memory
        self.full = (t.n <= FULL_LIMIT) if full is None else full
        rng = np.random.default_rng(seed)
        p = min(1.0, train_max / t.n)
        ids_all, Xs, sel_all = [], [], []
        pe = min(1.0, eval_max / t.n) if eval_max else 0.0
        if pe:
            # evaluation sample = 64 random blocks of consecutive positions, so
            # that the index locality of exceptions is preserved
            K = 64
            B = max(1, eval_max // K)
            starts = np.sort(rng.choice(max(1, t.n // B), size=min(K, max(1, t.n // B)), replace=False)) * B
            in_eval = np.zeros(t.n, bool)
            for st in starts:
                in_eval[st:st + B] = True
        ev_ids, ev_sel, ev_X = [], [], []
        used = np.zeros(self.vocab.rows, bool)
        idt = np.int16 if self.vocab.rows < 32768 else np.int32
        for s, e, sq, stm in t.iter_chunks():
            ids = feature_ids(t, self.vocab, sq, stm, t.feat[s:e], width)
            used[np.unique(ids)] = True
            if self.full:
                ids_all.append(ids.astype(idt))
            else:
                sel = np.flatnonzero(rng.random(e - s) < p)
                ids_all.append(ids[sel].astype(idt))
                sel_all.append(sel + s)
                if tree_feats:
                    Xs.append(treebase.numeric_features(t, sq[sel], stm[sel], t.feat[s:e][sel]))
                if pe:
                    es = np.flatnonzero(in_eval[s:e])
                    ev_ids.append(ids[es].astype(idt))
                    ev_sel.append(es + s)
                    if tree_feats:
                        ev_X.append(treebase.numeric_features(t, sq[es], stm[es], t.feat[s:e][es]))
        self.used_rows = used
        self.ids = np.concatenate(ids_all)
        self.train_idx = np.arange(t.n) if self.full else np.concatenate(sel_all)
        self.train_X = np.concatenate(Xs) if Xs else None
        if pe:
            self.eval_ids = np.concatenate(ev_ids)
            self.eval_idx = np.concatenate(ev_sel)
            self.eval_X = np.concatenate(ev_X) if ev_X else None
        if self.full and t.n > train_max:
            self.train_sel = np.sort(rng.choice(t.n, train_max, replace=False))
        else:
            self.train_sel = None

    def iter_ids(self, chunk=1 << 22):
        t = self.t
        if self.full:
            for s in range(0, self.n, chunk):
                yield s, min(self.n, s + chunk), self.ids[s:s + chunk]
        else:
            for s, e, sq, stm in t.iter_chunks(chunk):
                yield s, e, feature_ids(t, self.vocab, sq, stm, t.feat[s:e], self.width)

    def train_tensors(self, variant):
        ids = self.ids if self.train_sel is None else self.ids[self.train_sel]
        idx = self.train_idx if self.train_sel is None else self.train_idx[self.train_sel]
        allowed = allowed_mask(self.y[idx], self.bc[idx], variant)
        return torch.from_numpy(ids.astype(np.int32)), torch.from_numpy(allowed)

    def predict(self, q):
        p1 = np.empty(self.n, np.uint8)
        p2 = np.empty(self.n, np.uint8)
        for s, e, ids in self.iter_ids():
            a, b = q.predict(torch.from_numpy(ids.astype(np.int32)))
            p1[s:e] = a
            p2[s:e] = b
        return p1, p2


def decoded_values(pred, bc, variant):
    if variant == "a":
        return pred
    # best capture in label space; "no capture" (-3) -> 0, a no-op under max
    bcl = np.clip(bc, -2, 2).astype(np.int8) + np.int8(2)
    return np.maximum(pred, bcl.view(np.uint8))


def encode_container(model_blob, pred, ctx, y, bc, variant):
    wrong = np.flatnonzero(decoded_values(pred, bc, variant) != y)
    ex = exc.encode(wrong, y[wrong], ctx)
    return exc.varint(len(model_blob)) + model_blob + ex, len(wrong), len(ex)


def decode_container(buf, predict_fn, bc, variant):
    """predict_fn(model_blob) -> (pred, ctx). Returns decoded label array."""
    ln, pos = exc.read_varint(buf)
    model_blob = buf[pos:pos + ln]
    pred, ctx = predict_fn(model_blob)
    idx, lab = exc.decode(buf[pos + ln:], ctx)
    out = pred.copy()
    out[idx] = lab
    return decoded_values(out, bc, variant)


def steps_for(n, cfg, budget):
    """training steps: ~`epochs` passes over the data, clipped to a budget
    that shrinks for large configurations (CPU time)."""
    batch = 4096 if n > 200_000 else 1024
    epochs = 150 if n < 200_000 else 30
    s = int(epochs * n / batch)
    return int(np.clip(s, 2000, budget)), batch


def mlp_sweep(td, variant, cfgs, budget, bits_list=(8, 4), patience=2, seed=0, keep_best_blob=True):
    y, bc = td.y, td.bc
    res = []
    best = None
    worse = 0
    ids_t, allowed_t = td.train_tensors(variant)
    calib = ids_t[torch.randperm(len(ids_t), generator=torch.Generator().manual_seed(1))[:200_000]]
    for cfg in cfgs:
        torch.manual_seed(seed + cfg)
        net = Net(td.vocab.rows, cfg)
        h1 = CONFIGS[cfg][0]
        used_params = net.n_params() - (td.vocab.rows - 1 - int(td.used_rows[1:].sum())) * h1
        if best is not None and 0.25 * used_params >= best[0]["total_bytes"]:
            # even at 4 bits and perfect zstd this model alone would exceed the best total
            break
        steps, batch = steps_for(len(ids_t), cfg, budget)
        t0 = time.time()
        train(net, ids_t, allowed_t, steps, batch=batch, seed=seed + cfg)
        ttrain = time.time() - t0
        improved = False
        for bits in bits_list:
            t0 = time.time()
            qnet = net
            if bits == 4:
                # short quantisation-aware fine-tuning for 4-bit weights
                qnet = copy.deepcopy(net)
                qnet.qat_bits = 4
                train(qnet, ids_t, allowed_t, max(500, steps // 4), batch=batch, lr=5e-4, seed=seed + cfg + 100)
                qnet.qat_bits = None
            q = quantize(qnet, calib, bits, used_rows=td.used_rows)
            blob = q.serialize()
            p1, p2 = td.predict(q)
            ctx = p1 * 5 + p2
            cont, nexc, exc_bytes = encode_container(blob, p1, ctx, y, bc, variant)
            r = dict(model="mlp", cfg=cfg, arch=str(CONFIGS[cfg]), bits=bits, n_params=net.n_params(),
                     n_params_used=int(td.used_rows[1:].sum()) * CONFIGS[cfg][0] + net.n_params() - (td.vocab.rows - 1) * CONFIGS[cfg][0],
                     model_bytes=len(blob), n_exceptions=int(nexc), exception_bytes=exc_bytes,
                     total_bytes=len(cont), steps=steps, train_s=round(ttrain, 1), eval_s=round(time.time() - t0, 1))
            res.append(r)
            log(f"  [{td.name}/{variant}] mlp cfg{cfg} {bits}b params={r['n_params']} model={len(blob)} "
                f"exc={nexc} ({exc_bytes} B) total={len(cont)} train={ttrain:.0f}s")
            if best is None or len(cont) < best[0]["total_bytes"]:
                best = (r, cont, blob)
                improved = True
        worse = 0 if improved else worse + 1
        # larger configurations cannot help once the model alone costs more
        # than the best total, or once there are no exceptions left
        if worse >= patience or min(x["model_bytes"] for x in res[-len(bits_list):]) >= best[0]["total_bytes"] \
                or best[0]["n_exceptions"] == 0:
            break
    return res, best


def const_model(td, variant):
    y, bc = td.y, td.bc
    counts = np.bincount(y, minlength=5)
    best = None
    for c in range(5):
        pred = np.full(td.n, c, np.uint8)
        ctx = pred * 5 + pred
        cont, nexc, exb = encode_container(bytes([c]), pred, ctx, y, bc, variant)
        if best is None or len(cont) < best[0]["total_bytes"]:
            best = (dict(model="const", cfg=-1, bits=0, n_params=0, model_bytes=1, n_exceptions=int(nexc),
                         exception_bytes=exb, total_bytes=len(cont), cls=c), cont, bytes([c]))
    return best


def tree_sweep(td, variant, leaves_list, fit_max=4_000_000, patience=3, seed=0):
    t = td.t
    y, bc = td.y, td.bc
    # numeric features for the whole table (uint8)
    X = np.concatenate([treebase.numeric_features(t, sq, stm, t.feat[s:e]) for s, e, sq, stm in t.iter_chunks()])
    rng = np.random.default_rng(seed)
    if variant == "a":
        cand = np.arange(td.n)
    else:  # train only on positions whose value is not given by a capture
        cand = np.flatnonzero((bc.astype(np.int16) + 2) != y)
    fit_idx = cand if len(cand) <= fit_max else np.sort(rng.choice(cand, fit_max, replace=False))
    res, best, worse = [], None, 0
    for L in leaves_list:
        if L > len(fit_idx) // 2:
            break
        t0 = time.time()
        clf = treebase.fit(X[fit_idx], y[fit_idx], L, seed)
        blob = treebase.serialize(clf)
        ft = treebase.FlatTree(blob)
        pred = ft.predict(X)
        ctx = pred * 5 + pred
        cont, nexc, exb = encode_container(blob, pred, ctx, y, bc, variant)
        r = dict(model="tree", leaves=int(clf.get_n_leaves()), model_bytes=len(blob), n_exceptions=int(nexc),
                 exception_bytes=exb, total_bytes=len(cont), fit_s=round(time.time() - t0, 1))
        res.append(r)
        log(f"  [{td.name}/{variant}] tree L={L} model={len(blob)} exc={nexc} total={len(cont)} ({time.time()-t0:.0f}s)")
        if best is None or len(cont) < best[0]["total_bytes"]:
            best, worse = (r, cont, blob, X), 0
        else:
            worse += 1
        if worse >= patience or nexc == 0 or len(blob) >= best[0]["total_bytes"]:
            break
    return res, best


def verify_mlp(td, cont, variant):
    def pf(blob):
        q = QNet.deserialize(blob, td.vocab.rows)
        p1, p2 = td.predict(q)
        return p1, p1 * 5 + p2
    dec = decode_container(cont, pf, td.bc, variant)
    return int((dec != td.y).sum())


def verify_const(td, cont, variant):
    def pf(blob):
        pred = np.full(td.n, blob[0], np.uint8)
        return pred, pred * 5 + pred
    dec = decode_container(cont, pf, td.bc, variant)
    return int((dec != td.y).sum())


def verify_tree(td, cont, variant, X):
    def pf(blob):
        pred = treebase.FlatTree(blob).predict(X)
        return pred, pred * 5 + pred
    dec = decode_container(cont, pf, td.bc, variant)
    return int((dec != td.y).sum())


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]
