"""Loading of the binary files written by src/enumerate/tbenum and decoding of
positions (squares per piece slot) from the canonical raw index, in numpy.

Layout of the raw index (see tbenum.c):
    stm, wK-domain index, bK square, other pieces (64 each, pawns 48 -> a2..h7)
"""
import json
import os

import numpy as np

T_P, T_N, T_B, T_R, T_Q, T_K = 1, 2, 3, 4, 5, 6
TYPE_CHAR = {T_P: "P", T_N: "N", T_B: "B", T_R: "R", T_Q: "Q", T_K: "K"}

DATA_DIR = os.environ.get("TB_DATA", os.path.expanduser("~/data"))
TB_DIR = os.environ.get("TB_DIR", os.path.expanduser("~/tb"))


class Table:
    def __init__(self, name, data_dir=None, load_labels=True, mmap=False):
        self.name = name
        self.dir = os.path.join(data_dir or DATA_DIR, name)
        with open(os.path.join(self.dir, "meta.json")) as f:
            m = json.load(f)
        self.meta = m
        self.n = m["n_positions"]
        self.raw_size = m["raw_size"]
        self.color = np.array(m["color"], dtype=np.int64)
        self.type = np.array(m["type"], dtype=np.int64)
        self.radix = np.array(m["radix"], dtype=np.int64)
        self.wk_domain = np.array(m["wk_domain"], dtype=np.int64)
        self.nstm = m["nstm"]
        self.pawnful = bool(m["pawnful"])
        self.symmetric = bool(m["symmetric"])
        self.nslots = len(self.color)
        self.block_counts = np.array(m["block_counts"], dtype=np.int64)
        self.block_size = int(np.prod(self.radix[2:]))
        self.block_offsets = np.concatenate([[0], np.cumsum(self.block_counts)])
        rd = (lambda f: np.memmap(f, dtype=np.uint8, mode="r")) if mmap else (lambda f: np.fromfile(f, dtype=np.uint8))
        self.feat = rd(os.path.join(self.dir, "feat.u8"))
        self.labels_raw = None
        if load_labels and os.path.exists(os.path.join(self.dir, "labels.u8")):
            self.labels_raw = rd(os.path.join(self.dir, "labels.u8"))
        self._bits = None

    @property
    def npieces(self):
        return self.nslots

    @property
    def wdl(self):
        """True WDL 0..4 (0 loss .. 4 win) from the side to move."""
        return self.labels_raw & 7

    @property
    def best_capture(self):
        """Best capture value -3..2 (-3 = no legal capture)."""
        return (self.labels_raw >> 3).astype(np.int8) - 3

    def bits(self):
        if self._bits is None:
            self._bits = np.memmap(os.path.join(self.dir, "valid.bits"), dtype=np.uint8, mode="r")
        return self._bits

    def raw_indices(self, start, stop):
        """Raw indices of valid positions with rank in [start, stop)."""
        # blocks covering the range
        b0 = int(np.searchsorted(self.block_offsets, start, side="right") - 1)
        b1 = int(np.searchsorted(self.block_offsets, stop, side="left"))
        bits = self.bits()
        bs8 = self.block_size // 8
        chunk = np.unpackbits(bits[b0 * bs8:b1 * bs8], bitorder="little")
        idx = np.flatnonzero(chunk).astype(np.int64) + b0 * self.block_size
        first = self.block_offsets[b0]
        return idx[start - first: stop - first]

    def decode(self, raw):
        """raw indices -> (squares [n, nslots] int8, stm [n] int8 (0 = white))."""
        raw = np.asarray(raw, dtype=np.int64)
        sq = np.empty((len(raw), self.nslots), dtype=np.int8)
        r = raw.copy()
        for i in range(self.nslots - 1, 1, -1):
            d = r % self.radix[i]
            r //= self.radix[i]
            sq[:, i] = d + 8 if self.type[i] == T_P else d
        sq[:, 1] = r % 64
        r //= 64
        sq[:, 0] = self.wk_domain[r % len(self.wk_domain)]
        r //= len(self.wk_domain)
        return sq, r.astype(np.int8)

    def ranks_to_raw(self, ranks):
        """raw indices for sorted ranks (any subset), block by block."""
        ranks = np.asarray(ranks, dtype=np.int64)
        out = np.empty(len(ranks), np.int64)
        blk = np.searchsorted(self.block_offsets, ranks, side="right") - 1
        bits = self.bits()
        bs8 = self.block_size // 8
        starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
        ends = np.r_[starts[1:], len(ranks)]
        for s, e in zip(starts, ends):
            b = int(blk[s])
            raw = np.flatnonzero(np.unpackbits(bits[b * bs8:(b + 1) * bs8], bitorder="little"))
            out[s:e] = raw[ranks[s:e] - self.block_offsets[b]] + b * self.block_size
        return out

    def positions(self, start=0, stop=None):
        stop = self.n if stop is None else stop
        return self.decode(self.raw_indices(start, stop))

    def iter_chunks(self, chunk=1 << 22):
        for s in range(0, self.n, chunk):
            e = min(self.n, s + chunk)
            sq, stm = self.positions(s, e)
            yield s, e, sq, stm

    def fen(self, sq_row, stm):
        board = ["."] * 64
        for i in range(self.nslots):
            c = TYPE_CHAR[int(self.type[i])]
            board[int(sq_row[i])] = c if self.color[i] == 0 else c.lower()
        rows = []
        for r in range(7, -1, -1):
            s, e = "", 0
            for f in range(8):
                p = board[r * 8 + f]
                if p == ".":
                    e += 1
                else:
                    if e:
                        s += str(e)
                        e = 0
                    s += p
            if e:
                s += str(e)
            rows.append(s)
        return "/".join(rows) + (" w" if stm == 0 else " b") + " - - 0 1"


def list_tables(maxpcs=5, tb_dir=None):
    """Official table names (from the generator checksum list order file)."""
    path = os.path.join(tb_dir or TB_DIR, "tables.lst")
    with open(path) as f:
        names = [l.strip() for l in f if l.strip()]
    return [n for n in names if len(n) - 1 <= maxpcs]
