"""MLP over summed categorical embeddings; training, post-training quantisation,
bit-exact integer inference and (de)serialisation.

Integer inference is bit-exact on any IEEE-754 machine:
  * weights are int8 (or int4) and activations uint7 (0..127), biases int32;
  * matrix products are computed in float32 on integer-valued tensors; every
    partial sum is an integer of magnitude < 2^24 (checked when quantising), so
    float32 represents all of them exactly and the result does not depend on
    the summation order used by the BLAS library;
  * re-quantisation between layers is floor(acc * M + 0.5) in float64 with a
    per-channel multiplier M computed from float32 scales stored in the model
    (one correctly rounded multiply + add -> deterministic);
  * the class is the first index of the maximal logit.
"""
import io
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard

torch.set_float32_matmul_precision("highest")

# (embedding width, hidden layer widths). The decoder knows this list; a model
# stores only the index of its configuration.
CONFIGS = [
    (4, []),
    (8, []),
    (16, [16]),
    (32, [32]),
    (64, [64]),
    (64, [128, 64]),
    (128, [256, 128]),
    (256, [512, 256]),
    (512, [1024, 512]),
]
NCLS = 5
EXACT = 1 << 24


class Net(nn.Module):
    def __init__(self, rows, cfg):
        super().__init__()
        h1, hidden = CONFIGS[cfg]
        self.cfg = cfg
        self.rows = rows
        self.emb = nn.Parameter(torch.randn(rows, h1) * (1.0 / math.sqrt(8)))
        self.b0 = nn.Parameter(torch.zeros(h1))
        dims = [h1] + hidden + [NCLS]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.emb.shape[1]  # PAD row excluded

    qat_bits = None  # when set, weights are fake-quantised (straight-through)

    def _fq(self, w, dim, per_tensor=False):
        qmax = 127 if self.qat_bits == 8 else 7
        a = w.detach().abs()
        s = (a.max() if per_tensor else a.amax(dim=dim, keepdim=True)).clamp_min(1e-8) / qmax
        return w + (torch.clamp(torch.round(w / s), -qmax, qmax) * s - w).detach()

    def forward(self, ids, return_acts=False):
        emb = self.emb if self.qat_bits is None else self._fq(self.emb, 0)
        h = F.relu(F.embedding_bag(ids, emb, mode="sum", padding_idx=0) + self.b0)
        acts = [h]
        for i, layer in enumerate(self.layers):
            w = layer.weight
            if self.qat_bits is not None:
                w = self._fq(w, 1, per_tensor=(i == len(self.layers) - 1))
            h = F.linear(h, w, layer.bias)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                acts.append(h)
        return (h, acts) if return_acts else h


def allowed_mask(y, best_cap, variant):
    """Set of stored values that decode to the true value y.
    (a) strict: only y.  (b) syzygy-style: decoded = max(stored, best capture),
    so every c with max(c, bc) == y is fine."""
    c = np.arange(NCLS)[None, :]
    if variant == "a":
        return c == y[:, None]
    bc = best_cap.astype(np.int64) + 2  # label space; no capture -> -1
    return np.maximum(c, bc[:, None]) == y[:, None]


def train(net, ids, allowed, steps, batch=4096, lr=3e-3, seed=0, log=None):
    """ids: int32 [N, F] (torch), allowed: bool [N, 5] (torch)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    N = ids.shape[0]
    net.train()
    for step in range(steps):
        bi = torch.randint(0, N, (min(batch, N),), generator=g)
        logits = net(ids[bi].int())
        logp = F.log_softmax(logits, dim=1)
        a = allowed[bi]
        loss = -(torch.logsumexp(logp.masked_fill(~a, -1e9), dim=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 2000 == 0 or step == steps - 1):
            log(f"    step {step}/{steps} loss {loss.item():.4f}")
    net.eval()
    return net


class QNet:
    """Quantised network (integer tensors stored as float32 for BLAS)."""

    def __init__(self, cfg, bits, rows, sE, Eq, b0q, sA, sW, Wq, bq, ver=2):
        # ver 1: float64 re-quantisation; ver 2: float32 (faster). Both are
        # exact IEEE operations, hence deterministic; the version is stored.
        self.ver = ver
        self.cfg, self.bits, self.rows = cfg, bits, rows
        self.sE, self.Eq, self.b0q, self.sA, self.sW, self.Wq, self.bq = sE, Eq, b0q, sA, sW, Wq, bq
        self._prep()

    def _prep(self):
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        self.tE = f32(self.Eq)
        self.tb0 = f32(self.b0q)
        self.tW = [f32(w) for w in self.Wq]
        self.tb = [f32(b) for b in self.bq]
        # per-channel requantisation multipliers, float64, from float32 scales
        sE = self.sE.astype(np.float64)
        sA = self.sA.astype(np.float64)
        self.M = [torch.from_numpy(sE / sA[0])]
        for l in range(len(self.Wq) - 1):
            self.M.append(torch.from_numpy(self.sW[l].astype(np.float64) * sA[l] / sA[l + 1]))
        if self.ver >= 2:
            self.M = [m.float() for m in self.M]

    def _requant(self, acc, M):
        if self.ver == 1:
            return torch.clamp(torch.floor(acc.double() * M + 0.5), 0, 127).float()
        # acc is an exact integer < 2^24 in float32; one rounded multiply, one
        # rounded add, floor, clamp -- all in place
        return acc.mul_(M).add_(0.5).floor_().clamp_(0, 127)

    @torch.no_grad()
    def logits(self, ids):
        ids = torch.as_tensor(ids)
        acc = F.embedding_bag(ids.long() if ids.dtype != torch.int32 else ids, self.tE, mode="sum", padding_idx=0) + self.tb0
        a = self._requant(acc, self.M[0])
        for l in range(len(self.tW)):
            acc = a @ self.tW[l].T + self.tb[l]
            if l < len(self.tW) - 1:
                a = self._requant(acc, self.M[l + 1])
        return acc

    @torch.no_grad()
    def predict(self, ids, chunk=1 << 16):
        """returns (first choice, second choice) uint8 arrays."""
        p1, p2 = [], []
        for s in range(0, len(ids), chunk):
            lg = self.logits(ids[s:s + chunk])
            a1 = torch.argmax(lg, dim=1)
            lg[torch.arange(len(lg)), a1] = -float("inf")
            a2 = torch.argmax(lg, dim=1)
            p1.append(a1.numpy().astype(np.uint8))
            p2.append(a2.numpy().astype(np.uint8))
        return np.concatenate(p1), np.concatenate(p2)

    # ---- serialisation -------------------------------------------------
    def serialize(self):
        buf = io.BytesIO()
        buf.write(struct.pack("<BBB", self.ver, self.cfg, self.bits))
        buf.write(self.sE.astype("<f4").tobytes())
        buf.write(self.sA.astype("<f4").tobytes())
        for s in self.sW:
            buf.write(s.astype("<f4").tobytes())
        ints = [self.Eq[1:]] + list(self.Wq)
        for w in ints:
            buf.write(pack_int(w, self.bits))
        buf.write(self.b0q.astype("<i4").tobytes())
        for b in self.bq:
            buf.write(b.astype("<i4").tobytes())
        return zstandard.ZstdCompressor(level=19).compress(buf.getvalue())

    @staticmethod
    def deserialize(blob, rows):
        raw = zstandard.ZstdDecompressor().decompress(blob)
        ver, cfg, bits = struct.unpack_from("<BBB", raw, 0)
        assert ver in (1, 2)
        pos = 3
        h1, hidden = CONFIGS[cfg]
        dims = [h1] + hidden + [NCLS]

        def take(n, dt):
            nonlocal pos
            a = np.frombuffer(raw, dtype=dt, count=n, offset=pos).copy()
            pos += n * np.dtype(dt).itemsize
            return a

        sE = take(h1, "<f4")
        sA = take(len(hidden) + 1, "<f4")
        sW = [take(o, "<f4") for o in dims[1:-1]]
        shapes = [(rows - 1, h1)] + [(o, i) for i, o in zip(dims[:-1], dims[1:])]
        ws = []
        for shp in shapes:
            n = shp[0] * shp[1]
            nbytes = n if bits == 8 else (n + 1) // 2
            ws.append(unpack_int(raw[pos:pos + nbytes], n, bits).reshape(shp))
            pos += nbytes
        Eq = np.concatenate([np.zeros((1, h1), np.int8), ws[0]])
        b0q = take(h1, "<i4")
        bq = [take(o, "<i4") for o in dims[1:]]
        assert pos == len(raw)
        return QNet(cfg, bits, rows, sE, Eq, b0q, sA, sW, ws[1:], bq, ver=ver)


def pack_int(w, bits):
    w = np.asarray(w, dtype=np.int8).ravel()
    if bits == 8:
        return w.tobytes()
    u = (w.astype(np.int16) + 8).astype(np.uint8)
    if len(u) % 2:
        u = np.append(u, 8)
    return (u[0::2] | (u[1::2] << 4)).tobytes()


def unpack_int(b, n, bits):
    a = np.frombuffer(b, dtype=np.uint8)
    if bits == 8:
        return a.view(np.int8)[:n].copy()
    out = np.empty(len(a) * 2, dtype=np.int8)
    out[0::2] = (a & 15).astype(np.int8) - 8
    out[1::2] = (a >> 4).astype(np.int8) - 8
    return out[:n]


@torch.no_grad()
def quantize(net, calib_ids, bits, used_rows=None, q=0.99999):
    qmax = 127 if bits == 8 else 7
    E = net.emb.detach().numpy().astype(np.float64).copy()
    E[0] = 0
    if used_rows is not None:
        E[~used_rows] = 0  # rows no position uses never influence a prediction
    b0 = net.b0.detach().numpy().astype(np.float64)
    _, acts = net(torch.as_tensor(calib_ids).int(), return_acts=True)
    sA = np.array([max(float(torch.quantile(a.flatten()[:1 << 24], q)) if a.numel() else 1.0, 1e-6) / 127
                   for a in acts], dtype=np.float32)
    sE = (np.maximum(np.abs(E).max(axis=0), 1e-8) / qmax).astype(np.float32)
    Eq = np.clip(np.round(E / sE), -qmax, qmax).astype(np.int8)
    b0q = np.clip(np.round(b0 / sE), -(1 << 22), 1 << 22).astype(np.int32)
    Wq, bq, sW = [], [], []
    for l, layer in enumerate(net.layers):
        W = layer.weight.detach().numpy().astype(np.float64)
        b = layer.bias.detach().numpy().astype(np.float64)
        s = (np.maximum(np.abs(W).max(axis=1), 1e-8) / qmax).astype(np.float32)
        if l == len(net.layers) - 1:
            # one scale for the output layer: argmax over integer logits must
            # equal argmax over the real-valued logits
            s = np.full_like(s, s.max())
        wq = np.clip(np.round(W / s[:, None]), -qmax, qmax).astype(np.int8)
        bb = np.round(b / (s.astype(np.float64) * sA[l]))
        limit = EXACT - 1 - 127 * np.abs(wq.astype(np.int64)).sum(axis=1)
        assert (limit > 0).all(), "layer too wide for exact float32 accumulation"
        bb = np.clip(bb, -limit, limit).astype(np.int32)
        Wq.append(wq)
        bq.append(bb)
        if l < len(net.layers) - 1:
            sW.append(s)
    # embedding sums: at most n_features * qmax + |b0q| << 2^24
    return QNet(net.cfg, bits, net.rows, sE, Eq, b0q, sA, sW, Wq, bq)
