"""Independent checks of the C enumerator (src/enumerate/tbenum) with python-chess.

1. For every 3-piece table: brute-force all boards with python-chess, keep the
   legal ones (no pawn on rank 1/8, side not to move not in check), reduce them
   to orbits under the board symmetries (D4 without pawns, left-right mirror
   with pawns; plus the colour flip for tables with identical material) and
   check that tbenum's position set contains exactly one member of every orbit.
2. Labels: tbenum's WDL (Fathom) is compared with python-chess's independent
   Syzygy prober on every position of the 3-piece tables and on a random
   sample of every other enumerated table.
3. A few textbook positions have their known values.
"""
import os
import sys

import chess
import chess.syzygy
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.data import DATA_DIR, TB_DIR, Table  # noqa: E402

THREE = ["KQvK", "KRvK", "KBvK", "KNvK", "KPvK"]


def available(names):
    return [n for n in names if os.path.exists(os.path.join(DATA_DIR, n, "meta.json"))]


def all_enumerated():
    return sorted(d for d in os.listdir(DATA_DIR) if os.path.exists(os.path.join(DATA_DIR, d, "labels.u8")))


def transforms(pawnful):
    fs = [lambda b: b, chess.flip_horizontal]
    if not pawnful:
        fs += [chess.flip_vertical, lambda b: chess.flip_vertical(chess.flip_horizontal(b)),
               chess.flip_diagonal, chess.flip_anti_diagonal,
               lambda b: chess.flip_horizontal(chess.flip_diagonal(b)),
               lambda b: chess.flip_vertical(chess.flip_diagonal(b))]
    return fs


def orbit_key(board, pawnful, symmetric):
    boards = [board.transform(f) for f in transforms(pawnful)]
    if symmetric:
        boards += [b.mirror() for b in boards]
    return min(b.fen() for b in boards)


def brute_force_orbits(name):
    white, black = name.split("v")
    pieces = [(chess.KING, True), (chess.KING, False)]
    pieces += [(chess.Piece.from_symbol(c).piece_type, True) for c in white[1:]]
    pieces += [(chess.Piece.from_symbol(c).piece_type, False) for c in black[1:]]
    assert len(pieces) == 3
    pawnful = any(p == chess.PAWN for p, _ in pieces)
    symmetric = white == black
    orbits = set()
    for a in range(64):
        for b in range(64):
            for c in range(64):
                sqs = (a, b, c)
                if len(set(sqs)) < 3:
                    continue
                board = chess.Board(None)
                ok = True
                for (pt, col), s in zip(pieces, sqs):
                    if pt == chess.PAWN and chess.square_rank(s) in (0, 7):
                        ok = False
                    board.set_piece_at(s, chess.Piece(pt, col))
                if not ok:
                    continue
                for turn in (chess.WHITE, chess.BLACK):
                    board.turn = turn
                    if board.was_into_check():
                        continue
                    orbits.add(orbit_key(board, pawnful, symmetric))
    return orbits, pawnful, symmetric


@pytest.mark.parametrize("name", available(THREE))
def test_position_set_matches_bruteforce(name):
    t = Table(name)
    orbits, pawnful, symmetric = brute_force_orbits(name)
    sq, stm = t.positions()
    keys = set()
    for i in range(t.n):
        b = chess.Board(t.fen(sq[i], stm[i]))
        keys.add(orbit_key(b, pawnful, symmetric))
    assert len(keys) == t.n, "two enumerated positions are symmetric images of each other"
    assert keys == orbits, f"{len(orbits - keys)} orbits missing, {len(keys - orbits)} extra"


def check_labels(name, sample=None, seed=0):
    t = Table(name)
    rng = np.random.default_rng(seed)
    idx = np.arange(t.n) if sample is None or sample >= t.n else np.sort(rng.choice(t.n, sample, replace=False))
    wdl = t.wdl
    with chess.syzygy.open_tablebase(TB_DIR) as tb:
        for i in idx:
            sq, stm = t.positions(int(i), int(i) + 1)
            b = chess.Board(t.fen(sq[0], stm[0]))
            assert tb.probe_wdl(b) + 2 == wdl[i], (name, i, b.fen())


@pytest.mark.parametrize("name", available(THREE))
def test_labels_exhaustive_3pc(name):
    check_labels(name)


@pytest.mark.parametrize("name", [n for n in all_enumerated() if len(n) > 4])
def test_labels_sample(name):
    check_labels(name, sample=300)


KNOWN = [
    ("7k/8/8/8/8/8/8/KQ6 w - - 0 1", 2),       # KQvK: win
    ("8/8/8/8/8/2k5/8/KB6 w - - 0 1", 0),      # KBvK: draw
    ("8/8/8/8/8/k7/8/KN6 b - - 0 1", 0),       # KNvK: draw
    ("8/3P4/8/8/8/8/k7/7K w - - 0 1", 2),      # unstoppable pawn
    ("4k3/4P3/4K3/8/8/8/8/8 b - - 0 1", 0),    # stalemate
    ("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1", 2),    # Kd6 Kf7 Kd7 wins
    ("k7/8/1K6/P7/8/8/8/8 w - - 0 1", 0),      # rook pawn vs cornered king: draw
    ("7k/8/8/8/8/8/8/KQ6 b - - 0 1", -2),      # same KQvK, loss for the defender
]


@pytest.mark.parametrize("fen,wdl", KNOWN)
def test_known_positions(fen, wdl):
    with chess.syzygy.open_tablebase(TB_DIR) as tb:
        assert tb.probe_wdl(chess.Board(fen)) == wdl
