#!/usr/bin/env python3
"""Where do the MLP's exceptions concentrate?

For every exhaustively evaluated table with >= 4 pieces (results/tables), the
stored best MLP container (variant a) is decoded; an exception is a position
where the quantised model's prediction differs from the Syzygy value.
Per position category we report the exception rate relative to the table's
overall rate ("enrichment"; 1 = no concentration):

  * true class (loss / blessed loss / draw / cursed win / win)
  * side to move in check; a legal capture exists; the value is achieved by a
    capture (capture-resolved)
  * zugzwang: the side to move would be better off if the other side were to
    move (looked up in the same table, colour-asymmetric tables only)
  * decisive positions: |DTZ| (plies to zeroing move) for exceptions vs. random
    decisive positions, from a sample probed with python-chess
Writes results/error_analysis.json, results/error_analysis.md and
figures/error_enrichment.png. Example FENs are included.
"""
import json
import os
import sys

import chess
import chess.syzygy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import mdl  # noqa: E402
from codec import exceptions as exc  # noqa: E402
from model.data import TB_DIR  # noqa: E402
from model.net import QNet  # noqa: E402

CLS = ["loss", "blessed loss", "draw", "cursed win", "win"]
POP8 = np.array([bin(i).count("1") for i in range(256)], np.int64)


def model_predictions(td, path):
    buf = open(path, "rb").read()
    ln, pos = exc.read_varint(buf)
    q = QNet.deserialize(buf[pos:pos + ln], td.vocab.rows)
    p1, _ = td.predict(q)
    return p1


def zugzwang(td):
    """bool array (n,) and validity mask: side to move is worse off than if
    the opponent were to move."""
    t = td.t
    n = t.n
    zz = np.zeros(n, bool)
    ok = np.zeros(n, bool)
    if t.nstm != 2:
        return zz, ok
    bits = t.bits()
    cum = np.concatenate([[0], np.cumsum(POP8[bits])])
    stride = t.raw_size // 2
    raw = t.raw_indices(0, n)
    stm = raw // stride
    other = raw + np.where(stm == 0, stride, -stride)
    byte = other >> 3
    bit = other & 7
    valid = (bits[byte] >> bit) & 1 == 1
    rank = cum[byte] + POP8[bits[byte] & ((1 << bit) - 1)]
    v = td.y.astype(np.int16) - 2
    vo = np.full(n, 0, np.int16)
    vo[valid] = td.y[rank[valid]].astype(np.int16) - 2
    zz[valid] = v[valid] < -vo[valid]
    ok[:] = valid
    return zz, ok


def dtz_sample(td, idx, tb, k=150, seed=0):
    rng = np.random.default_rng(seed)
    idx = idx if len(idx) <= k else rng.choice(idx, k, replace=False)
    out = []
    for i in idx:
        sq, stm = td.t.positions(int(i), int(i) + 1)
        out.append(abs(tb.probe_dtz(chess.Board(td.t.fen(sq[0], stm[0])))))
    return out


def main():
    rdir = os.path.join(ROOT, "results", "tables")
    rows, examples = [], {}
    tb = chess.syzygy.open_tablebase(TB_DIR)
    for f in sorted(os.listdir(rdir)):
        r = json.load(open(os.path.join(rdir, f)))
        if r["pieces"] < 4 or r.get("mode") == "large":
            continue
        name = r["table"]
        td = mdl.TableData(name, train_max=1)
        path = os.path.join(td.t.dir, "enc", "mlp_a.bin")
        p1 = model_predictions(td, path)
        e = p1 != td.y
        rate = e.mean()
        if rate == 0:
            continue
        feat = td.t.feat
        zz, zok = zugzwang(td)
        cats = {
            **{f"class={c}": td.y == i for i, c in enumerate(CLS)},
            "in check": (feat & 1) == 1,
            "capture available": ((feat >> 1) & 3) > 0,
            "value given by a capture": (td.bc.astype(np.int16) + 2) == td.y,
            "zugzwang": zz & zok,
            "stalemate/mate (no legal move)": (feat >> 3) == 0,
        }
        row = dict(table=name, pieces=r["pieces"], n=int(td.n), n_exceptions=int(e.sum()), rate=float(rate),
                   syzygy_bytes=r["syzygy_rtbw_bytes"], mlp_bytes=r["variants"]["a"]["mlp"]["total_bytes"])
        for k, m in cats.items():
            nm = int(m.sum())
            row[f"share[{k}]"] = nm / td.n
            row[f"enrich[{k}]"] = float(e[m].mean() / rate) if nm else None
        dec = (td.y != 2)
        ex_dec = np.flatnonzero(e & dec)
        all_dec = np.flatnonzero(dec)
        if len(ex_dec) and len(all_dec):
            row["median_absDTZ_exceptions"] = float(np.median(dtz_sample(td, ex_dec, tb)))
            row["median_absDTZ_all_decisive"] = float(np.median(dtz_sample(td, all_dec, tb, seed=1)))
        rows.append(row)
        rng = np.random.default_rng(0)
        pick = rng.choice(np.flatnonzero(e), min(4, int(e.sum())), replace=False)
        exs = []
        for i in sorted(pick):
            sq, stm = td.t.positions(int(i), int(i) + 1)
            exs.append(dict(fen=td.t.fen(sq[0], stm[0]), true=CLS[td.y[i]], predicted=CLS[p1[i]],
                            zugzwang=bool(zz[i] and zok[i])))
        examples[name] = exs
        print(f"{name}: rate {rate:.4f}", flush=True)
    rows.sort(key=lambda x: -x["n_exceptions"])
    json.dump(dict(rows=rows, examples=examples), open(os.path.join(ROOT, "results", "error_analysis.json"), "w"),
              indent=1)

    # markdown
    keys = ["in check", "capture available", "value given by a capture", "zugzwang",
            "class=blessed loss", "class=cursed win", "class=draw"]
    md = ["| table | positions | exceptions | rate | " + " | ".join(keys) + " | median abs(DTZ) exc / all |",
          "|---|---:|---:|---:|" + "---:|" * len(keys) + "---|"]
    for x in rows:
        cells = []
        for k in keys:
            v = x.get(f"enrich[{k}]")
            cells.append("–" if v is None else f"{v:.2f}")
        dtz = (f"{x['median_absDTZ_exceptions']:.0f} / {x['median_absDTZ_all_decisive']:.0f}"
               if "median_absDTZ_exceptions" in x else "–")
        md.append(f"| {x['table']} | {x['n']:,} | {x['n_exceptions']:,} | {x['rate']:.2%} | " + " | ".join(cells)
                  + f" | {dtz} |")
    md.append("")
    md.append("Enrichment = exception rate within the category / overall exception rate of the table "
              "(best MLP, variant a). '–' = category empty.")
    md.append("")
    md.append("## Example exceptions (FEN, true value, model prediction)")
    for name in [x["table"] for x in rows[:6]]:
        md.append(f"\n**{name}**\n")
        for ex in examples[name]:
            md.append(f"- `{ex['fen']}` — true: {ex['true']}, predicted: {ex['predicted']}"
                      + (" (zugzwang)" if ex["zugzwang"] else ""))
    open(os.path.join(ROOT, "results", "error_analysis.md"), "w").write("\n".join(md) + "\n")

    # figure: median enrichment per category across tables
    cats = ["in check", "capture available", "value given by a capture", "zugzwang", "class=draw",
            "class=blessed loss", "class=cursed win", "class=win", "class=loss"]
    med, lo, hi = [], [], []
    for k in cats:
        vals = [x[f"enrich[{k}]"] for x in rows if x.get(f"enrich[{k}]") is not None]
        vals = np.array(vals) if vals else np.array([np.nan])
        med.append(np.nanmedian(vals)); lo.append(np.nanpercentile(vals, 25)); hi.append(np.nanpercentile(vals, 75))
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(cats))
    ax.hlines(y, lo, hi, color="#2a78d6", lw=4, alpha=0.6)
    ax.plot(med, y, "o", color="#2a78d6", ms=8, mec="white", mew=2)
    ax.axvline(1, color="#52514e", lw=1, ls=":")
    ax.set_yticks(y, cats)
    ax.set_xscale("log")
    ax.set_xlabel("exception-rate enrichment (median over 4-piece tables, bar = IQR)", color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e4e3df")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "error_enrichment.png"), dpi=150)


if __name__ == "__main__":
    main()
