#!/usr/bin/env python3
"""One MLP shared by all tables of a given piece count (results/joint_<k>.json).

The model sees the full 12-(colour, piece type) vocabulary; per-table feature
lists are padded to a common width. Size = model bytes (once) + the sum of
the per-table exception containers. The best configuration is decoded again
from bytes and checked against every table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import mdl  # noqa: E402
from codec import exceptions as exc  # noqa: E402
from model.data import DATA_DIR, TB_DIR, list_tables, Table  # noqa: E402
from model.features import Vocab  # noqa: E402
from model.net import CONFIGS, Net, QNet, allowed_mask, quantize, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pieces", type=int)
    ap.add_argument("--tables", default=None, help="comma list; default: all enumerated tables with k pieces")
    ap.add_argument("--cfgs", default="2,3,4,5,6,7")
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--train-max", type=int, default=16_000_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", default="a,b")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.tables:
        names = args.tables.split(",")
    else:
        names = [n for n in list_tables(args.pieces) if len(n) - 1 == args.pieces
                 and os.path.exists(os.path.join(DATA_DIR, n, "labels.u8"))]
    vocab = Vocab.full()
    width = max(vocab.n_features(Table(n, load_labels=False)) for n in names)
    ntot = sum(Table(n, load_labels=False).n for n in names)
    mdl.log(f"joint {args.pieces}-piece: {len(names)} tables, {ntot} positions, width {width}")
    tds = [mdl.TableData(n, vocab=vocab, width=width, full=False,
                         train_max=max(1, int(args.train_max * Table(n, load_labels=False).n / ntot)))
           for n in names]
    used = np.zeros(vocab.rows, bool)
    for td in tds:
        used |= td.used_rows
    out = dict(pieces=args.pieces, tables=names, n_positions=int(ntot),
               syzygy_bytes=int(sum(os.path.getsize(os.path.join(TB_DIR, n + ".rtbw")) for n in names)),
               variants={})
    for v in args.variants.split(","):
        ids = torch.from_numpy(np.concatenate([td.ids for td in tds]).astype(np.int32))
        allowed = torch.from_numpy(np.concatenate([allowed_mask(td.y[td.train_idx], td.bc[td.train_idx], v) for td in tds]))
        calib = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(1))[:200_000]]
        sweep, best = [], None
        for cfg in [int(c) for c in args.cfgs.split(",")]:
            torch.manual_seed(cfg)
            net = Net(vocab.rows, cfg)
            steps = int(np.clip(10 * len(ids) / 4096, 3000, args.budget))
            t0 = time.time()
            train(net, ids, allowed, steps, batch=4096, seed=cfg)
            ttr = time.time() - t0
            improved = False
            for bits in (8, 4):
                q = quantize(net, calib, bits, used_rows=used)
                blob = q.serialize()
                conts, nexc = [], 0
                for td in tds:
                    p1, p2 = td.predict(q)
                    c, ne, _ = mdl.encode_container(b"", p1, p1 * 5 + p2, td.y, td.bc, v)
                    conts.append(c)
                    nexc += ne
                total = len(blob) + sum(len(c) for c in conts) + 4
                r = dict(cfg=cfg, arch=str(CONFIGS[cfg]), bits=bits, n_params=net.n_params(), model_bytes=len(blob),
                         n_exceptions=int(nexc), total_bytes=total, steps=steps, train_s=round(ttr, 1),
                         per_table_exception_bytes={n: len(c) for n, c in zip(names, conts)})
                sweep.append(r)
                mdl.log(f"  joint{args.pieces}/{v} cfg{cfg} {bits}b model={len(blob)} exc={nexc} total={total} "
                        f"(syzygy {out['syzygy_bytes']}) train={ttr:.0f}s")
                if best is None or total < best[0]["total_bytes"]:
                    best, improved = (r, blob, conts), True
            if not improved and len(sweep) >= 4 and sweep[-3]["total_bytes"] < sweep[-1]["total_bytes"]:
                break
            if min(x["model_bytes"] for x in sweep[-2:]) >= best[0]["total_bytes"]:
                break
        r, blob, conts = best
        # verification: decode every table from the stored bytes
        q = QNet.deserialize(blob, vocab.rows)
        bad = 0
        for td, c in zip(tds, conts):
            def pf(_):
                p1, p2 = td.predict(q)
                return p1, p1 * 5 + p2
            dec = mdl.decode_container(c, pf, td.bc, v)
            bad += int((dec != td.y).sum())
        r = dict(r, verified_mismatches=bad)
        mdl.log(f"  joint{args.pieces}/{v} best total={r['total_bytes']} verified mismatches={bad}")
        out["variants"][v] = dict(r, sweep=sweep)
        d = os.path.join(DATA_DIR, f"joint{args.pieces}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"model_{v}.bin"), "wb").write(blob)
        for n, c in zip(names, conts):
            open(os.path.join(d, f"{n}_{v}.bin"), "wb").write(c)
    with open(os.path.join(ROOT, "results", f"joint_{args.pieces}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
