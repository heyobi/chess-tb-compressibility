#!/usr/bin/env python3
"""Aggregate results/tables/*.json (+ results/joint_*.json, results/sampled_*.json)
into results/summary.csv and the figures in figures/."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# reference categorical palette (dataviz skill), fixed order by method
COL = {"mlp": "#2a78d6", "tree": "#eb6834", "const": "#1baf7a", "zstd19": "#eda100", "xz9e": "#e87ba4",
       "joint": "#4a3aa7"}
LABEL = {"mlp": "MLP + exceptions", "tree": "Decision tree + exceptions", "const": "No model (exceptions only)",
         "zstd19": "zstd -19 of label array", "xz9e": "xz -9e of label array", "joint": "Joint MLP per piece count"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VARIANT = {"a": "(a) strict: every position coded", "b": "(b) Syzygy-style: capture-resolved positions free"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "tables", "*.json"))):
        r = json.load(open(f))
        for v, vr in r["variants"].items():
            base = dict(table=r["table"], pieces=r["pieces"], pawnful=r["pawnful"], variant=v,
                        n_positions=r["n_positions"], syzygy_bytes=r["syzygy_rtbw_bytes"],
                        capture_resolvable=r["capture_resolvable"], status="exhaustive")
            for m in ("mlp", "tree", "const"):
                x = vr[m]
                rows.append(dict(base, method=m, total_bytes=x["total_bytes"], model_bytes=x["model_bytes"],
                                 exception_bytes=x["exception_bytes"], n_exceptions=x["n_exceptions"],
                                 config=x.get("arch", x.get("leaves", x.get("cls"))), bits=x.get("bits"),
                                 verified=x["verified_mismatches"] == 0))
            for m in ("zstd19", "xz9e"):
                rows.append(dict(base, method=m, total_bytes=vr["raw"][m], model_bytes=0, exception_bytes=0,
                                 n_exceptions=None, config=None, bits=None, verified=True))
    df = pd.DataFrame(rows)
    if len(df):
        df["ratio"] = df.total_bytes / df.syzygy_bytes
        df["bits_per_pos"] = 8 * df.total_bytes / df.n_positions
        df["syzygy_bits_per_pos"] = 8 * df.syzygy_bytes / df.n_positions
    return df


def load_joint():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "joint_*.json"))):
        j = json.load(open(f))
        for v, jr in j["variants"].items():
            rows.append(dict(pieces=j["pieces"], variant=v, tables=len(j["tables"]), total_bytes=jr["total_bytes"],
                             syzygy_bytes=j["syzygy_bytes"], n_positions=j["n_positions"],
                             verified=jr.get("verified_mismatches", 1) == 0))
    return pd.DataFrame(rows)


def scaling_plot(df, joint, ycol, fname, ylabel, logy=True, ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    methods = ["mlp", "tree", "const", "zstd19", "xz9e"]
    for ax, v in zip(axes, ("a", "b")):
        d = df[df.variant == v]
        pcs = sorted(d.pieces.unique())
        offs = np.linspace(-0.24, 0.24, len(methods))
        for m, o in zip(methods, offs):
            dm = d[d.method == m]
            if not len(dm):
                continue
            med = []
            for p in pcs:
                vals = dm[dm.pieces == p][ycol].values
                if not len(vals):
                    med.append(np.nan)
                    continue
                q1, q2, q3 = np.percentile(vals, [25, 50, 75])
                ax.vlines(p + o, vals.min(), vals.max(), color=COL[m], lw=1, alpha=0.45)
                ax.vlines(p + o, q1, q3, color=COL[m], lw=4, alpha=0.9)
                med.append(q2)
            ax.plot(np.array(pcs) + o, med, color=COL[m], lw=2, marker="o", ms=8, mec="white", mew=2,
                    label=LABEL[m])
        if len(joint) and ycol == "ratio":
            j = joint[joint.variant == v]
            ax.plot(j.pieces, j.total_bytes / j.syzygy_bytes, color=COL["joint"], lw=2, ls="--", marker="D",
                    ms=8, mec="white", mew=2, label=LABEL["joint"] + " (pooled)")
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1, ls=":")
            ax.text(pcs[0] - 0.35, ref * 1.08, "Syzygy .rtbw", color=INK2, fontsize=9)
        if ycol == "bits_per_pos":
            sz = d.drop_duplicates("table").groupby("pieces").syzygy_bits_per_pos.median()
            ax.plot(sz.index, sz.values, color=INK, lw=2, ls=":", marker="s", ms=7, label="Syzygy .rtbw")
        ax.set_xticks(pcs)
        ax.set_xlabel("pieces (incl. kings)", color=INK2)
        ax.set_title(VARIANT[v], fontsize=10, color=INK, loc="left")
        if logy:
            ax.set_yscale("log")
        style(ax)
    axes[0].set_ylabel(ylabel, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, 0.99, "Marker: median over tables; thick bar: interquartile range; thin bar: min-max. "
             "Exhaustively verified tables only.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def mdl_plot(tables):
    fig, axes = plt.subplots(1, len(tables), figsize=(4 * len(tables), 3.8))
    for ax, name in zip(np.atleast_1d(axes), tables):
        r = json.load(open(os.path.join(RES, "tables", f"{name}.json")))
        for bits, mk in ((8, "o"), (4, "s")):
            sw = [x for x in r["variants"]["a"]["mlp_sweep"] if x["bits"] == bits]
            p = [x["n_params"] for x in sw]
            tot = [x.get("total_bytes", x.get("est_total_bytes")) for x in sw]
            exb = [x.get("exception_bytes", x.get("est_exception_bytes")) for x in sw]
            ax.plot(p, tot, color=COL["mlp"], marker=mk, lw=2, ms=7, label=f"total, {bits}-bit")
            ax.plot(p, [x["model_bytes"] for x in sw], color=COL["mlp"], marker=mk, lw=1, ls=":", ms=5, alpha=0.7,
                    label=f"model only, {bits}-bit")
            ax.plot(p, exb, color=COL["tree"], marker=mk, lw=1, ls="--", ms=5,
                    alpha=0.8, label=f"exceptions only, {bits}-bit")
        ax.axhline(r["syzygy_rtbw_bytes"], color=INK2, lw=1, ls=":")
        ax.text(ax.get_xlim()[0] if False else min(p), r["syzygy_rtbw_bytes"] * 1.1, "Syzygy", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("MLP parameters", color=INK2)
        est = " — sample estimates" if r.get("mode") == "large" else ""
        ax.set_title(f"{name} ({r['n_positions']:,} pos.){est}", fontsize=9, color=INK, loc="left")
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("bytes (variant a)", color=INK2)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIG, "mdl_curves.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    joint = load_joint()
    df.to_csv(os.path.join(RES, "summary.csv"), index=False)
    if len(joint):
        joint.to_csv(os.path.join(RES, "summary_joint.csv"), index=False)
    # pooled per piece count
    pooled = (df.groupby(["pieces", "variant", "method"])
              .agg(tables=("table", "nunique"), total=("total_bytes", "sum"), syzygy=("syzygy_bytes", "sum"),
                   positions=("n_positions", "sum"), median_ratio=("ratio", "median"),
                   all_verified=("verified", "all")).reset_index())
    pooled["pooled_ratio"] = pooled.total / pooled.syzygy
    pooled.to_csv(os.path.join(RES, "summary_by_pieces.csv"), index=False)
    print(pooled.to_string())
    scaling_plot(df, joint, "ratio", "ratio_vs_pieces.png", "best total size / Syzygy WDL size", ref=1.0)
    scaling_plot(df, joint, "bits_per_pos", "bits_per_position.png", "bits per position")
    reps = [t for t in ("KPvK", "KRvKP", "KRBvKR", "KQPvKQ") if os.path.exists(os.path.join(RES, "tables", f"{t}.json"))]
    if reps:
        mdl_plot(reps[:4])


if __name__ == "__main__":
    main()
