#!/usr/bin/env python3
"""Run the full per-table experiment and write results/tables/<TABLE>.json.

  * baselines: Syzygy .rtbw size, zstd-19 / xz -9e of the label array,
    constant predictor + exceptions, decision tree + exceptions
  * MLP size sweep (8- and 4-bit), both variants (a) strict and (b) syzygy-style
  * the best encoding of every method is decoded again from its bytes and
    compared position by position with the Syzygy values (Fathom probe)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import mdl  # noqa: E402
from baselines import raw as rawbase  # noqa: E402
from model.data import TB_DIR  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--cfgs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--budget", type=int, default=30000, help="max training steps per model")
    ap.add_argument("--leaves", default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--variants", default="a,b")
    ap.add_argument("--train-max", type=int, default=20_000_000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--classes", type=int, default=5, choices=(3, 5))
    ap.add_argument("--no-movegen", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.table}.json")
    if os.path.exists(out_path) and not args.force:
        print("exists", out_path)
        return
    t0 = time.time()
    mdl.log(f"{args.table}: loading")
    td = mdl.TableData(args.table, train_max=args.train_max, classes=args.classes, no_movegen=args.no_movegen)
    t = td.t
    res = dict(table=args.table, classes=args.classes, no_movegen=args.no_movegen, pieces=t.nslots, pawnful=t.pawnful, symmetric=t.symmetric,
               n_positions=int(t.n), raw_size=int(t.raw_size),
               syzygy_rtbw_bytes=os.path.getsize(os.path.join(TB_DIR, args.table + ".rtbw")),
               label_counts=np.bincount(td.y, minlength=5).tolist(),
               capture_resolvable=int(((td.bc.astype(np.int16) + 2) == td.y).sum()),
               variants={})
    encdir = os.path.join(t.dir, ("enc" if args.classes == 5 else "enc3") + ("_nomg" if args.no_movegen else ""))
    os.makedirs(encdir, exist_ok=True)
    for v in args.variants.split(","):
        vr = {}
        arr = td.y if v == "a" else rawbase.syzygy_style_fill(td.y, td.bc)
        vr["raw"] = rawbase.compressed_sizes(arr)
        mdl.log(f"  [{args.table}/{v}] raw {vr['raw']}")

        r, cont, _ = mdl.const_model(td, v)
        r["verified_mismatches"] = mdl.verify_const(td, cont, v)
        r["sha"] = mdl.sha(cont)
        vr["const"] = r
        open(os.path.join(encdir, f"const_{v}.bin"), "wb").write(cont)

        tres, tbest = mdl.tree_sweep(td, v, [int(x) for x in args.leaves.split(",")])
        r, cont, _, X = tbest
        r = dict(r, verified_mismatches=mdl.verify_tree(td, cont, v, X), sha=mdl.sha(cont))
        vr["tree_sweep"] = tres
        vr["tree"] = r
        open(os.path.join(encdir, f"tree_{v}.bin"), "wb").write(cont)
        del X

        mres, mbest = mdl.mlp_sweep(td, v, [int(x) for x in args.cfgs.split(",")], args.budget)
        r, cont, _ = mbest
        r = dict(r, verified_mismatches=mdl.verify_mlp(td, cont, v), sha=mdl.sha(cont))
        vr["mlp_sweep"] = mres
        vr["mlp"] = r
        open(os.path.join(encdir, f"mlp_{v}.bin"), "wb").write(cont)
        mdl.log(f"  [{args.table}/{v}] best mlp {r['total_bytes']} B (verified mismatches: {r['verified_mismatches']}), "
                f"tree {vr['tree']['total_bytes']} B, syzygy {res['syzygy_rtbw_bytes']} B")
        res["variants"][v] = vr
    res["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    mdl.log(f"{args.table}: done in {res['seconds']}s")


if __name__ == "__main__":
    main()
