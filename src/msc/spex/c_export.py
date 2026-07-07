"""Binary SPEX exporter for the ds4 C-side loader.

Format, little-endian:
  header: 4s magic + 7 u32 = 32 bytes
  per-layer params: repeated (a, b, T) f32 triples
  markov rows: for each layer/current-expert row, m u16 then m*(eid u16, count u32)

The markov matrix for layer l encodes transitions from layer l to layer l+1.
The final layer is serialized as empty rows.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"SPEX"
VERSION = 1
PREDICTOR_MARKOV = 0
PREDICTOR_HIDDEN = 1
HEADER_STRUCT = struct.Struct("<4s7I")
LAYER_STRUCT = struct.Struct("<fff")
ROW_COUNT_STRUCT = struct.Struct("<H")
ENTRY_STRUCT = struct.Struct("<HI")


def write_markov_spex(
    path: str | Path,
    *,
    a: np.ndarray,
    b: np.ndarray,
    T: np.ndarray,
    C: list[np.ndarray],
    n_layer: int,
    n_expert: int,
    topN: int,
) -> None:
    """Write a markov SPEX predictor file for ds4.

    `C[l][cur, nxt]` must contain transition counts from current expert ids in
    layer `l` to candidate expert ids in layer `l + 1`.
    """

    n_layer = _u32("n_layer", n_layer)
    n_expert = _u32("n_expert", n_expert)
    topN = _u32("topN", topN)
    if n_expert > np.iinfo(np.uint16).max:
        raise ValueError("n_expert must fit uint16 expert ids")
    if topN > np.iinfo(np.uint16).max:
        raise ValueError("topN must fit uint16 row counts")

    aa = _float_array("a", a, n_layer)
    bb = _float_array("b", b, n_layer)
    tt = _float_array("T", T, n_layer)
    mats = _markov_mats(C, n_layer, n_expert)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(
            HEADER_STRUCT.pack(
                MAGIC,
                VERSION,
                PREDICTOR_MARKOV,
                n_layer,
                n_expert,
                0,  # n_embd is used only by the hidden predictor.
                topN,
                0,
            )
        )
        for layer in range(n_layer):
            fh.write(LAYER_STRUCT.pack(float(aa[layer]), float(bb[layer]), float(tt[layer])))
        for mat in mats:
            for row in mat:
                entries = _top_entries(row, topN)
                fh.write(ROW_COUNT_STRUCT.pack(len(entries)))
                for eid, count in entries:
                    fh.write(ENTRY_STRUCT.pack(eid, count))


def _float_array(name: str, value: np.ndarray, n_layer: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < n_layer:
        raise ValueError(f"{name} must contain at least {n_layer} values")
    return arr[:n_layer]


def _markov_mats(C: list[np.ndarray], n_layer: int, n_expert: int) -> list[np.ndarray]:
    if len(C) < n_layer:
        raise ValueError(f"C must contain at least {n_layer} layer matrices")
    mats = []
    for layer in range(n_layer):
        mat = np.asarray(C[layer])
        if mat.shape != (n_expert, n_expert):
            raise ValueError(
                f"C[{layer}] has shape {mat.shape}, expected {(n_expert, n_expert)}"
            )
        if np.any(mat < 0):
            raise ValueError(f"C[{layer}] contains negative counts")
        mats.append(mat.astype(np.uint64, copy=False))
    return mats


def _top_entries(row: np.ndarray, topN: int) -> list[tuple[int, int]]:
    nz = np.flatnonzero(row)
    if nz.size == 0 or topN == 0:
        return []

    counts = row[nz]
    if nz.size > topN:
        pick = np.argpartition(-counts, topN - 1)[:topN]
        nz = nz[pick]
        counts = counts[pick]

    order = np.lexsort((nz, -counts))
    entries = []
    for eid, count in zip(nz[order], counts[order]):
        eid_i = int(eid)
        count_i = _u32("count", int(count))
        if count_i > 0:
            entries.append((eid_i, count_i))
    return entries


def _u32(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit uint32")
    return value
