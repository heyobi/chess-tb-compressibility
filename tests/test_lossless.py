"""Lossless verification of every stored encoding.

For every table with results in results/tables/<T>.json:
  1. the position set and TB-free features are regenerated from scratch with
     `tbenum --noprobe` (no tablebase access) and must be byte-identical to the
     files the encoder used;
  2. every stored container (const / tree / mlp, variants a and b) is decoded
     from its bytes alone (+ the 1-ply capture values for variant b) and
     compared position by position with the Syzygy WDL values obtained by
     probing the .rtbw files through Fathom. A single mismatch fails the test.

Run: TB_DIR=~/tb TB_DATA=~/data pytest tests/test_lossless.py
Set LOSSLESS_MAXPOS to skip tables with more positions (default: no limit).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import mdl  # noqa: E402
from model.data import DATA_DIR  # noqa: E402

RESULT_DIRS = [os.path.join(ROOT, "results", d) for d in ("tables", "tables3", "ablation_nomovegen")]
TBENUM = os.path.join(ROOT, "src", "enumerate", "tbenum")
MAXPOS = int(os.environ.get("LOSSLESS_MAXPOS", "0")) or None


def tables():
    out = []
    for d in RESULT_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            r = json.load(open(os.path.join(d, f)))
            if MAXPOS and r["n_positions"] > MAXPOS:
                continue
            if os.path.exists(os.path.join(DATA_DIR, r["table"], "labels.u8")):
                out.append(os.path.join(d, f))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


@pytest.mark.parametrize("path", tables(), ids=lambda p: os.path.relpath(p, ROOT))
def test_table_lossless(path):
    res = json.load(open(path))
    name = res["table"]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([TBENUM, "--noprobe", "--out", tmp, name], stderr=subprocess.DEVNULL)
        for fn in ("valid.bits", "feat.u8"):
            assert md5(os.path.join(tmp, fn)) == md5(os.path.join(DATA_DIR, name, fn)), fn
    nomg = res.get("no_movegen", False)
    large = res.get("mode") == "large"
    td = mdl.TableData(name, train_max=1, classes=res.get("classes", 5), no_movegen=nomg,
                       full=False if large else None)
    encdir = os.path.join(td.t.dir, ("enc" if res.get("classes", 5) == 5 else "enc3") + ("_nomg" if nomg else ""))
    X = None
    for v, vr in res["variants"].items():
        for method in ("const", "tree", "mlp"):
            cont = open(os.path.join(encdir, f"{method}_{v}.bin"), "rb").read()
            assert mdl.sha(cont) == vr[method]["sha"]
            assert len(cont) == vr[method]["total_bytes"]
            if method == "const":
                bad = mdl.verify_const(td, cont, v)
            elif method == "tree" and large:  # chunked: the feature matrix would not fit in RAM
                bad = verify_tree_chunked(td, cont, v)
            elif method == "tree":
                if X is None:
                    from baselines import tree as treebase
                    t = td.t
                    X = np.concatenate([treebase.numeric_features(t, sq, stm, t.feat[s:e])
                                        for s, e, sq, stm in t.iter_chunks()])
                bad = mdl.verify_tree(td, cont, v, X)
            else:
                bad = mdl.verify_mlp(td, cont, v)
            assert bad == 0, f"{name} {method} variant {v}: {bad} mismatches"


def verify_tree_chunked(td, cont, v):
    from baselines import tree as treebase
    t = td.t

    def pf(blob):
        ft = treebase.FlatTree(blob)
        p = np.empty(td.n, np.uint8)
        for s, e, sq, stm in t.iter_chunks():
            p[s:e] = ft.predict(treebase.numeric_features(t, sq, stm, t.feat[s:e]))
        return p, p * 5 + p
    return int((mdl.decode_container(cont, pf, td.bc, v) != td.y).sum())
