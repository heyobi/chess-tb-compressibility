#!/usr/bin/env python3
"""Recompute the decision-tree baseline (tree_version 2: square-colour
feature added; variant (b) fitted on all positions with the true label) for
result files produced before that change, in place.

  python scripts/rerun_trees.py [--large] [RESULT_DIR ...]
(--large: only the sample-based large-mode results, otherwise only the others)
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import mdl  # noqa: E402

SMALL_LEAVES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
LARGE_LEAVES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]


def main():
    large = "--large" in sys.argv
    dirs = [a for a in sys.argv[1:] if a != "--large"] or [os.path.join(ROOT, "results", "tables")]
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(f))
            if r.get("tree_version") == 2 or (r.get("mode") == "large") != large:
                continue
            name = r["table"]
            nomg = r.get("no_movegen", False)
            classes = r.get("classes", 5)
            mdl.log(f"rerun trees {f}")
            if r.get("mode") == "large":
                td = mdl.TableData(name, train_max=16_000_000, full=False, classes=classes,
                                   eval_max=4_000_000, tree_feats=True, no_movegen=nomg)
            else:
                td = mdl.TableData(name, train_max=1, classes=classes, no_movegen=nomg)
            encdir = os.path.join(td.t.dir, ("enc" if classes == 5 else "enc3") + ("_nomg" if nomg else ""))
            for v, vr in r["variants"].items():
                if r.get("mode") == "large":
                    sweep, tr, cont = mdl.large_tree_sweep(td, v, LARGE_LEAVES, name)
                else:
                    sweep, best = mdl.tree_sweep(td, v, SMALL_LEAVES)
                    tr, cont, _, X = best
                    tr = dict(tr, verified_mismatches=mdl.verify_tree(td, cont, v, X), sha=mdl.sha(cont))
                    del X
                vr["tree_sweep"] = sweep
                vr["tree"] = tr
                open(os.path.join(encdir, f"tree_{v}.bin"), "wb").write(cont)
            r["tree_version"] = 2
            with open(f + ".tmp", "w") as fh:
                json.dump(r, fh, indent=1)
            os.replace(f + ".tmp", f)


if __name__ == "__main__":
    main()
