"""Generic-compressor baselines on the raw label array (one byte per position
in canonical enumeration order)."""
import lzma

import numpy as np
import zstandard


def syzygy_style_fill(y, best_cap, chunk=1 << 24):
    """Variant (b) label array: for positions whose value is achieved by a
    capture, any stored value <= y decodes correctly (decoded = max(stored,
    best capture)). Fill them with the last stored value of a non-resolvable
    position, clipped to <= y, which keeps runs long. (Chunked for memory.)"""
    out = np.empty(len(y), np.uint8)
    carry = -1
    for s in range(0, len(y), chunk):
        yy = y[s:s + chunk].astype(np.uint8)
        res = (best_cap[s:s + chunk].astype(np.int16) + 2) == yy
        idx = np.where(~res, np.arange(len(yy), dtype=np.int32), -1)
        idx = np.maximum.accumulate(idx)
        prev = np.where(idx >= 0, yy[np.maximum(idx, 0)], carry if carry >= 0 else yy)
        out[s:s + chunk] = np.where(res, np.minimum(prev, yy), yy)
        nz = np.flatnonzero(~res)
        if len(nz):
            carry = int(yy[nz[-1]])
    return out


def compressed_sizes(arr):
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    z = zstandard.ZstdCompressor(level=19, threads=-1).compress(a)  # buffer protocol, no copy
    assert np.array_equal(np.frombuffer(zstandard.ZstdDecompressor().decompress(z), np.uint8), a)
    zl = len(z)
    del z
    x = lzma.compress(a, preset=9 | lzma.PRESET_EXTREME)
    assert np.array_equal(np.frombuffer(lzma.decompress(x), np.uint8), a)
    return {"zstd19": zl, "xz9e": len(x)}
