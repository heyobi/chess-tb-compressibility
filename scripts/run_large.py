#!/usr/bin/env python3
"""Per-table experiment for large (5-piece) tables.

Differences to run_table.py (documented in docs/METHOD.md):
  * models are trained on a uniform random sample of the table;
  * the model-size sweep (MLP configs x {8,4} bits, tree sizes) is evaluated on
    a separate uniform random evaluation sample: exceptions are counted on the
    sample and the exception bytes are extrapolated as
    bytes(sample exceptions coded in sample-index space) * N / n_sample.
    These sweep points are marked "estimated";
  * the configuration with the best estimated total is then encoded exactly
    on the FULL table and decoded again from its bytes and compared with every
    position (verified). Only these exact numbers enter the main results.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import mdl  # noqa: E402
from baselines import raw as rawbase  # noqa: E402
from baselines import tree as treebase  # noqa: E402
from codec import exceptions as exc  # noqa: E402
from model.data import TB_DIR  # noqa: E402
from model.net import CONFIGS, Net, QNet, quantize, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def est_bytes(td, pred, ctx_s, v):
    """Estimated full-table exception bytes from the evaluation sample."""
    y, bc = td.y[td.eval_idx], td.bc[td.eval_idx]
    wrong = np.flatnonzero(mdl.decoded_values(pred, bc, v) != y)
    b = len(exc.encode(wrong, y[wrong], ctx_s))
    scale = td.n / len(td.eval_idx)
    return int(round(len(wrong) * scale)), int(round(b * scale)), len(wrong) / len(td.eval_idx)


def tree_full_predict(td, ft):
    out = np.empty(td.n, np.uint8)
    t = td.t
    for s, e, sq, stm in t.iter_chunks():
        out[s:e] = ft.predict(treebase.numeric_features(t, sq, stm, t.feat[s:e]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--cfgs", default="1,2,3,4,5,6,7")
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--leaves", default="256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288")
    ap.add_argument("--variants", default="a,b")
    ap.add_argument("--train-max", type=int, default=16_000_000)
    ap.add_argument("--eval-max", type=int, default=4_000_000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--classes", type=int, default=5, choices=(3, 5))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--skip-xz", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.table}.json")
    if os.path.exists(out_path):
        print("exists", out_path)
        return
    t0 = time.time()
    mdl.log(f"{args.table}: loading (large mode)")
    td = mdl.TableData(args.table, train_max=args.train_max, full=False, classes=args.classes,
                       eval_max=args.eval_max, tree_feats=True)
    t = td.t
    mdl.log(f"{args.table}: {t.n} positions, train sample {len(td.train_idx)}, eval sample {len(td.eval_idx)}")
    res = dict(table=args.table, classes=args.classes, mode="large", pieces=t.nslots, pawnful=t.pawnful,
               symmetric=t.symmetric, n_positions=int(t.n), raw_size=int(t.raw_size),
               train_sample=int(len(td.train_idx)), eval_sample=int(len(td.eval_idx)),
               syzygy_rtbw_bytes=os.path.getsize(os.path.join(TB_DIR, args.table + ".rtbw")),
               label_counts=np.bincount(td.y, minlength=5).tolist(),
               capture_resolvable=int(((td.bc.astype(np.int16) + 2) == td.y).sum()), variants={})
    encdir = os.path.join(t.dir, "enc" if args.classes == 5 else "enc3")
    os.makedirs(encdir, exist_ok=True)
    calib = torch.from_numpy(td.eval_ids[:200_000].astype(np.int32))
    for v in args.variants.split(","):
        vr = {}
        arr = td.y if v == "a" else rawbase.syzygy_style_fill(td.y, td.bc)
        if args.skip_xz:
            import zstandard
            vr["raw"] = {"zstd19": len(zstandard.ZstdCompressor(level=19, threads=-1).compress(arr.tobytes()))}
        else:
            vr["raw"] = rawbase.compressed_sizes(arr)
        del arr
        mdl.log(f"  [{args.table}/{v}] raw {vr['raw']}")

        r, cont, _ = mdl.const_model(td, v)
        r["verified_mismatches"] = mdl.verify_const(td, cont, v)
        r["sha"] = mdl.sha(cont)
        vr["const"] = r
        open(os.path.join(encdir, f"const_{v}.bin"), "wb").write(cont)
        mdl.log(f"  [{args.table}/{v}] const {r['total_bytes']}")

        # ---- tree: sweep on samples, exact on the best ----
        yt, bct = td.y[td.train_idx], td.bc[td.train_idx]
        cand = np.arange(len(yt)) if v == "a" else np.flatnonzero((bct.astype(np.int16) + 2) != yt)
        fit = cand[:4_000_000] if len(cand) > 4_000_000 else cand
        sweep, best, worse = [], None, 0
        for L in [int(x) for x in args.leaves.split(",")]:
            ts = time.time()
            clf = treebase.fit(td.train_X[fit], yt[fit], L)
            blob = treebase.serialize(clf)
            pe = treebase.FlatTree(blob).predict(td.eval_X)
            ne, eb, rate = est_bytes(td, pe, pe * 5 + pe, v)
            total = len(blob) + eb + 8
            rr = dict(model="tree", leaves=int(clf.get_n_leaves()), model_bytes=len(blob), est_n_exceptions=ne,
                      est_exception_bytes=eb, est_total_bytes=total, error_rate=rate, estimated=True)
            sweep.append(rr)
            mdl.log(f"  [{args.table}/{v}] tree L={L} model={len(blob)} est_exc={ne} est_total={total} ({time.time()-ts:.0f}s)")
            if best is None or total < best[0]["est_total_bytes"]:
                best, worse = (rr, blob), 0
            else:
                worse += 1
            if worse >= 3 or len(blob) >= best[0]["est_total_bytes"]:
                break
        rr, blob = best
        pred = tree_full_predict(td, treebase.FlatTree(blob))
        cont, nexc, exb = mdl.encode_container(blob, pred, pred * 5 + pred, td.y, td.bc, v)
        del pred

        def pf_tree(b):
            p = tree_full_predict(td, treebase.FlatTree(b))
            return p, p * 5 + p
        bad = int((mdl.decode_container(cont, pf_tree, td.bc, v) != td.y).sum())
        vr["tree_sweep"] = sweep
        vr["tree"] = dict(model="tree", leaves=rr["leaves"], model_bytes=len(blob), n_exceptions=int(nexc),
                          exception_bytes=exb, total_bytes=len(cont), verified_mismatches=bad, sha=mdl.sha(cont),
                          est_total_bytes=rr["est_total_bytes"])
        open(os.path.join(encdir, f"tree_{v}.bin"), "wb").write(cont)
        mdl.log(f"  [{args.table}/{v}] tree exact total={len(cont)} (est {rr['est_total_bytes']}) mismatches={bad}")

        # ---- MLP: sweep on samples, exact on the best ----
        ids_t, allowed_t = td.train_tensors(v)
        ev = torch.from_numpy(td.eval_ids.astype(np.int32))
        sweep, best, worse = [], None, 0
        for cfg in [int(c) for c in args.cfgs.split(",")]:
            torch.manual_seed(cfg)
            net = Net(td.vocab.rows, cfg)
            used_params = net.n_params() - (td.vocab.rows - 1 - int(td.used_rows[1:].sum())) * CONFIGS[cfg][0]
            if best is not None and 0.25 * used_params >= best[0]["est_total_bytes"]:
                break
            steps, batch = args.budget, 4096
            ts = time.time()
            train(net, ids_t, allowed_t, steps, batch=batch, seed=cfg)
            ttr = time.time() - ts
            improved = False
            for bits in (8, 4):
                qnet = net
                if bits == 4:
                    qnet = copy.deepcopy(net)
                    qnet.qat_bits = 4
                    train(qnet, ids_t, allowed_t, steps // 4, batch=batch, lr=5e-4, seed=cfg + 100)
                    qnet.qat_bits = None
                q = quantize(qnet, calib, bits, used_rows=td.used_rows)
                blob = q.serialize()
                p1, p2 = q.predict(ev)
                ne, eb, rate = est_bytes(td, p1, p1 * 5 + p2, v)
                total = len(blob) + eb + 8
                rr = dict(model="mlp", cfg=cfg, arch=str(CONFIGS[cfg]), bits=bits, n_params=net.n_params(),
                          model_bytes=len(blob), est_n_exceptions=ne, est_exception_bytes=eb, est_total_bytes=total,
                          error_rate=rate, steps=steps, train_s=round(ttr, 1), estimated=True)
                sweep.append(rr)
                mdl.log(f"  [{args.table}/{v}] mlp cfg{cfg} {bits}b model={len(blob)} est_exc={ne} "
                        f"est_total={total} train={ttr:.0f}s")
                if best is None or total < best[0]["est_total_bytes"]:
                    best, improved = (rr, blob), True
            worse = 0 if improved else worse + 1
            if worse >= 2 or min(x["model_bytes"] for x in sweep[-2:]) >= best[0]["est_total_bytes"]:
                break
        del ids_t, allowed_t
        rr, blob = best
        q = QNet.deserialize(blob, td.vocab.rows)
        p1, p2 = td.predict(q)
        cont, nexc, exb = mdl.encode_container(blob, p1, p1 * 5 + p2, td.y, td.bc, v)
        del p1, p2
        bad = mdl.verify_mlp(td, cont, v)
        vr["mlp_sweep"] = sweep
        vr["mlp"] = dict(model="mlp", cfg=rr["cfg"], arch=rr["arch"], bits=rr["bits"], n_params=rr["n_params"],
                         model_bytes=len(blob), n_exceptions=int(nexc), exception_bytes=exb, total_bytes=len(cont),
                         verified_mismatches=bad, sha=mdl.sha(cont), est_total_bytes=rr["est_total_bytes"])
        open(os.path.join(encdir, f"mlp_{v}.bin"), "wb").write(cont)
        mdl.log(f"  [{args.table}/{v}] best mlp exact total={len(cont)} (est {rr['est_total_bytes']}) "
                f"mismatches={bad}; tree {vr['tree']['total_bytes']}; syzygy {res['syzygy_rtbw_bytes']}")
        res["variants"][v] = vr
        with open(out_path + ".partial", "w") as f:
            json.dump(res, f, indent=1)
    res["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    os.remove(out_path + ".partial")
    mdl.log(f"{args.table}: done in {res['seconds']}s")


if __name__ == "__main__":
    main()
