"""Python wrapper around the C range coder for exception lists, plus the
container format.

Container (all little-endian):
    varint  n_exceptions
    bytes   range-coded stream (see rangecoder.c)
The number of positions of the table and the context bytes are known to the
decoder (they come from the enumeration and from the model), so they are not
stored.
"""
import ctypes
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "librangecoder.so")


def _load():
    src = os.path.join(_HERE, "rangecoder.c")
    if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", _SO, src])
    lib = ctypes.CDLL(_SO)
    P = ctypes.c_void_p
    lib.exc_encode.restype = ctypes.c_size_t
    lib.exc_encode.argtypes = [P, P, P, ctypes.c_size_t, P, ctypes.c_size_t]
    lib.exc_decode.restype = ctypes.c_int
    lib.exc_decode.argtypes = [P, ctypes.c_size_t, ctypes.c_size_t, P, ctypes.c_uint64, P, P]
    return lib


_lib = _load()


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def read_varint(buf, pos=0):
    n = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, pos


def encode(idx, labels, ctx):
    """idx: sorted unique int64 positions; labels: uint8 true labels;
    ctx: uint8 context per position of the whole table."""
    idx = np.ascontiguousarray(idx, dtype=np.uint64)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    ctx = np.ascontiguousarray(ctx, dtype=np.uint8)
    assert len(idx) == len(labels)
    if len(idx) > 1:
        assert np.all(np.diff(idx.astype(np.int64)) > 0)
    cap = 64 + len(idx) * 8
    while True:
        out = np.empty(cap, dtype=np.uint8)
        n = _lib.exc_encode(_ptr(idx), _ptr(labels), _ptr(ctx), len(idx), _ptr(out), cap)
        if n <= cap:
            return varint(len(idx)) + out[:n].tobytes()
        cap = n + 64


def decode(buf, ctx):
    ctx = np.ascontiguousarray(ctx, dtype=np.uint8)
    n, pos = read_varint(buf)
    idx = np.empty(n, dtype=np.uint64)
    lab = np.empty(n, dtype=np.uint8)
    data = np.frombuffer(buf[pos:], dtype=np.uint8).copy()
    if n:
        r = _lib.exc_decode(_ptr(data), len(data), n, _ptr(ctx), len(ctx), _ptr(idx), _ptr(lab))
        if r:
            raise ValueError("corrupt exception stream")
    return idx.astype(np.int64), lab
