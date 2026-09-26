#!/usr/bin/env python3
"""Sample-based ESTIMATE of the achievable size for tables too large for a full
pass (used for 6-piece tables; calibrated on 5-piece tables with exact numbers).

  * the labels/features are memory-mapped; nothing of size N is loaded;
  * training sample: uniform random ranks (MLP 16M, tree 4M);
  * evaluation sample: 64 random blocks of consecutive positions;
  * size estimate = model bytes + (sum over blocks of the range-coded exception
    bytes of each block) * N / n_eval, + 8 bytes of header;
  * 95% interval: bootstrap over the 64 blocks (1000 resamples) of the
    exception part (model bytes are exact).
Everything written here is an estimate and is kept apart from the exhaustively
verified results (results/sampled/<TABLE>.json).
"""
import argparse
import copy
import json
import os
import sys
import time
import types

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import mdl  # noqa: E402
from baselines import tree as treebase  # noqa: E402
from codec import exceptions as exc  # noqa: E402
from model.data import TB_DIR, Table  # noqa: E402
from model.features import Vocab, feature_ids  # noqa: E402
from model.net import CONFIGS, Net, allowed_mask, quantize, train  # noqa: E402


def sample(t, train_max, n_blocks, block, seed=0):
    rng = np.random.default_rng(seed)
    tr = np.unique(rng.integers(0, t.n, train_max))
    nb = t.n // block
    starts = np.sort(rng.choice(nb, n_blocks, replace=False)) * block
    ev = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return tr, ev, starts


def gather(t, ranks, vocab):
    raw = t.ranks_to_raw(ranks)
    sq, stm = t.decode(raw)
    feat = np.asarray(t.feat[ranks])
    lab = np.asarray(t.labels_raw[ranks])
    ids = feature_ids(t, vocab, sq, stm, feat)
    X = treebase.numeric_features(t, sq, stm, feat)
    return ids, X, (lab & 7).astype(np.uint8), ((lab >> 3).astype(np.int8) - 3)


def block_bytes(pred, ctx, y, bc, v, n_blocks, block):
    """exception count and coded bytes per evaluation block"""
    wrong = mdl.decoded_values(pred, bc, v) != y
    cnt, byt = [], []
    for k in range(n_blocks):
        s = slice(k * block, (k + 1) * block)
        w = np.flatnonzero(wrong[s])
        cnt.append(len(w))
        byt.append(len(exc.encode(w, y[s][w], ctx[s])))
    return np.array(cnt), np.array(byt)


def estimate(model_bytes, cnt, byt, scale, rng):
    tot = model_bytes + byt.sum() * scale + 8
    bs = rng.integers(0, len(byt), (1000, len(byt)))
    boot = model_bytes + byt[bs].sum(axis=1) * scale + 8
    return dict(est_total_bytes=int(round(tot)), ci95=[int(np.percentile(boot, 2.5)), int(np.percentile(boot, 97.5))],
                est_n_exceptions=int(round(cnt.sum() * scale)), model_bytes=int(model_bytes),
                sample_exceptions=int(cnt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--cfgs", default="1,2,3,4,5,6,7")
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--leaves", default="256,1024,4096,16384,65536,262144")
    ap.add_argument("--variants", default="a,b")
    ap.add_argument("--train-max", type=int, default=16_000_000)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--block", type=int, default=62_500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sampled"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    t = Table(args.table, mmap=True)
    vocab = Vocab.for_table(t)
    tr, ev, starts = sample(t, args.train_max, args.blocks, args.block)
    mdl.log(f"{args.table}: {t.n} positions; train {len(tr)}, eval {len(ev)} in {args.blocks} blocks")
    ids_tr, X_tr, y_tr, bc_tr = gather(t, tr, vocab)
    ids_ev, X_ev, y_ev, bc_ev = gather(t, ev, vocab)
    used = np.zeros(vocab.rows, bool)
    used[np.unique(ids_tr)] = True
    used[np.unique(ids_ev)] = True
    scale = t.n / len(ev)
    rng = np.random.default_rng(1)
    res = dict(table=args.table, pieces=t.nslots, pawnful=t.pawnful, symmetric=t.symmetric, n_positions=int(t.n),
               status="ESTIMATE (sampled, not exhaustively verified)", train_sample=int(len(tr)),
               eval_sample=int(len(ev)), eval_blocks=args.blocks,
               syzygy_rtbw_bytes=os.path.getsize(os.path.join(TB_DIR, args.table + ".rtbw")),
               label_counts_eval_sample=np.bincount(y_ev, minlength=5).tolist(), variants={})
    tsel = np.sort(np.random.default_rng(2).choice(len(tr), min(len(tr), 4_000_000), replace=False))
    for v in args.variants.split(","):
        vr = {}
        # constant predictor
        best = None
        for c in range(5):
            p = np.full(len(ev), c, np.uint8)
            cnt, byt = block_bytes(p, p * 5 + p, y_ev, bc_ev, v, args.blocks, args.block)
            e = estimate(1, cnt, byt, scale, rng)
            if best is None or e["est_total_bytes"] < best["est_total_bytes"]:
                best = dict(e, cls=c)
        vr["const"] = best
        # tree
        sweep, best = [], None
        for L in [int(x) for x in args.leaves.split(",")]:
            clf = treebase.fit(X_tr[tsel], y_tr[tsel], L)
            blob = treebase.serialize(clf)
            p = treebase.FlatTree(blob).predict(X_ev)
            cnt, byt = block_bytes(p, p * 5 + p, y_ev, bc_ev, v, args.blocks, args.block)
            e = dict(estimate(len(blob), cnt, byt, scale, rng), leaves=int(clf.get_n_leaves()))
            sweep.append(e)
            mdl.log(f"  [{args.table}/{v}] tree L={L} est_total={e['est_total_bytes']} ci={e['ci95']}")
            if best is None or e["est_total_bytes"] < best["est_total_bytes"]:
                best = e
            elif len(blob) > best["est_total_bytes"]:
                break
        vr["tree_sweep"], vr["tree"] = sweep, best
        # MLP
        ids_t = torch.from_numpy(ids_tr.astype(np.int16))
        allowed_t = torch.from_numpy(allowed_mask(y_tr, bc_tr, v))
        ev_t = torch.from_numpy(ids_ev.astype(np.int32))
        calib = ev_t[:200_000]
        sweep, best, worse = [], None, 0
        for cfg in [int(c) for c in args.cfgs.split(",")]:
            torch.manual_seed(cfg)
            net = Net(vocab.rows, cfg)
            ts = time.time()
            train(net, ids_t, allowed_t, args.budget, batch=4096, seed=cfg)
            ttr = time.time() - ts
            improved = False
            for bits in (8, 4):
                qnet = net
                if bits == 4:
                    qnet = copy.deepcopy(net)
                    qnet.qat_bits = 4
                    train(qnet, ids_t, allowed_t, args.budget // 4, batch=4096, lr=5e-4, seed=cfg + 100)
                    qnet.qat_bits = None
                q = quantize(qnet, calib, bits, used_rows=used)
                blob = q.serialize()
                p1, p2 = q.predict(ev_t)
                cnt, byt = block_bytes(p1, p1 * 5 + p2, y_ev, bc_ev, v, args.blocks, args.block)
                e = dict(estimate(len(blob), cnt, byt, scale, rng), cfg=cfg, arch=str(CONFIGS[cfg]), bits=bits,
                         n_params=net.n_params(), train_s=round(ttr, 1))
                sweep.append(e)
                mdl.log(f"  [{args.table}/{v}] mlp cfg{cfg} {bits}b model={len(blob)} est_total={e['est_total_bytes']} "
                        f"ci={e['ci95']} train={ttr:.0f}s")
                if best is None or e["est_total_bytes"] < best["est_total_bytes"]:
                    best, improved = e, True
            worse = 0 if improved else worse + 1
            if worse >= 2 or min(x["model_bytes"] for x in sweep[-2:]) >= best["est_total_bytes"]:
                break
        vr["mlp_sweep"], vr["mlp"] = sweep, best
        res["variants"][v] = vr
        json.dump(res, open(os.path.join(args.out, f"{args.table}.json.partial"), "w"), indent=1)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(args.out, f"{args.table}.json"), "w"), indent=1)
    if os.path.exists(os.path.join(args.out, f"{args.table}.json.partial")):
        os.remove(os.path.join(args.out, f"{args.table}.json.partial"))
    mdl.log(f"{args.table}: done in {res['seconds']}s")


if __name__ == "__main__":
    main()
