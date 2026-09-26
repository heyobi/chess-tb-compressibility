#!/usr/bin/env python3
"""Markdown tables for the README from results/*.json (writes results/report.md)."""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, d, "*.json")))]


def kb(x):
    return f"{x / 1024:,.1f}"


def ratio_cell(x, s):
    return f"{x / s:.3g}"


def by_pieces(rows, methods=("mlp", "tree", "const", "zstd19", "xz9e")):
    out = ["| pieces | variant | tables | Syzygy total (KB) | " + " | ".join(
        f"{m} pooled / median" for m in methods) + " | all verified |",
        "|---|---|---:|---:|" + "---|" * len(methods) + "---|"]
    for p in sorted({r["pieces"] for r in rows}):
        rp = [r for r in rows if r["pieces"] == p]
        for v in ("a", "b"):
            syz = sum(r["syzygy_rtbw_bytes"] for r in rp)
            cells, ok = [], True
            for m in methods:
                tot, rat = 0, []
                for r in rp:
                    vr = r["variants"][v]
                    x = vr["raw"][m] if m in ("zstd19", "xz9e") else vr[m]["total_bytes"]
                    if m not in ("zstd19", "xz9e"):
                        ok &= vr[m]["verified_mismatches"] == 0
                    tot += x
                    rat.append(x / r["syzygy_rtbw_bytes"])
                cells.append(f"{tot / syz:.3g} / {np.median(rat):.3g}")
            out.append(f"| {p} | {v} | {len(rp)} | {kb(syz)} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
    return out


def per_table(rows):
    out = ["| table | positions | Syzygy KB | MLP (a) | tree (a) | xz (a) | MLP (b) | tree (b) | xz (b) | "
           "MLP (a) choice | MLP bits/pos (a) | verified |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["pieces"], r["syzygy_rtbw_bytes"])):
        s = r["syzygy_rtbw_bytes"]
        a, b = r["variants"]["a"], r["variants"]["b"]
        ok = all(r["variants"][v][m]["verified_mismatches"] == 0 for v in ("a", "b") for m in ("mlp", "tree", "const"))
        m = a["mlp"]
        choice = f"{m.get('arch', '')} {m.get('bits', '')}b, {m.get('n_params', 0):,} p."
        out.append(f"| {r['table']} | {r['n_positions']:,} | {kb(s)} | {ratio_cell(a['mlp']['total_bytes'], s)} | "
                   f"{ratio_cell(a['tree']['total_bytes'], s)} | {ratio_cell(a['raw']['xz9e'], s)} | "
                   f"{ratio_cell(b['mlp']['total_bytes'], s)} | {ratio_cell(b['tree']['total_bytes'], s)} | "
                   f"{ratio_cell(b['raw']['xz9e'], s)} | {choice} | "
                   f"{8 * a['mlp']['total_bytes'] / r['n_positions']:.4f} | {'yes' if ok else 'NO'} |")
    return out


def main():
    rows = load("tables")
    md = ["## Scaling summary (ratio = our total bytes / Syzygy .rtbw bytes; < 1 means smaller than Syzygy)", "",
          "Pooled = sum over the tables of that piece count; median = median of per-table ratios.", ""]
    md += by_pieces(rows)
    md += ["", "## Per table (ratios to Syzygy)", ""]
    md += per_table(rows)
    for d, title in (("tables3", "3-class labels (loss / draw / win, 50-move rule applied)"),
                     ("ablation_nomovegen", "Ablation: no move-generator features")):
        rr = load(d)
        if rr:
            md += ["", f"## {title}", ""] + by_pieces(rr, ("mlp", "tree", "const", "xz9e")) + ["", ""] + per_table(rr)
    joints = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "joint_*.json")))]
    if joints:
        md += ["", "## Joint model per piece count (one MLP for all tables)", "",
               "| pieces | tables | Syzygy KB | variant | MLP arch | bits | model KB | exceptions KB | total / Syzygy | verified |",
               "|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
        for j in joints:
            for v, jr in j["variants"].items():
                exb = jr["total_bytes"] - jr["model_bytes"]
                md.append(f"| {j['pieces']} | {len(j['tables'])} | {kb(j['syzygy_bytes'])} | {v} | {jr['arch']} | "
                          f"{jr['bits']} | {kb(jr['model_bytes'])} | {kb(exb)} | "
                          f"{jr['total_bytes'] / j['syzygy_bytes']:.3g} | "
                          f"{'yes' if jr['verified_mismatches'] == 0 else 'NO'} |")
    open(os.path.join(RES, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
