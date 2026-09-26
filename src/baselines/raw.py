"""Generic-compressor baselines on the raw label array (one byte per position
in canonical enumeration order)."""
import lzma

import numpy as np
import zstandard


def syzygy_style_fill(y, best_cap):
    """Variant (b) label array: for positions whose value is achieved by a
    capture, any stored value <= y decodes correctly (decoded = max(stored,
    best capture)). Fill them with the last stored value of a non-resolvable
    position, clipped to <= y, which keeps runs long."""
    y = y.astype(np.uint8)
    resolvable = (best_cap.astype(np.int16) + 2) == y
    idx = np.where(~resolvable, np.arange(len(y)), -1)
    idx = np.maximum.accumulate(idx)
    prev = np.where(idx >= 0, y[np.maximum(idx, 0)], y)
    out = np.where(resolvable, np.minimum(prev, y), y)
    return out.astype(np.uint8)


def compressed_sizes(arr):
    b = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    z = zstandard.ZstdCompressor(level=19, threads=-1).compress(b)
    assert zstandard.ZstdDecompressor().decompress(z) == b
    x = lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)
    assert lzma.decompress(x) == b
    return {"zstd19": len(z), "xz9e": len(x)}
