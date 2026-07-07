from __future__ import annotations

import struct

import numpy as np
import pytest

from msc.spex.c_export import HEADER_STRUCT, write_markov_spex
from msc.spex.spex_loop import export_c


def _read_rows(blob: bytes, offset: int, n_rows: int):
    rows = []
    for _ in range(n_rows):
        (m,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        row = []
        for _ in range(m):
            eid, count = struct.unpack_from("<HI", blob, offset)
            offset += 6
            row.append((eid, count))
        rows.append(row)
    return rows, offset


def test_markov_spex_binary_contract(tmp_path):
    out = tmp_path / "params.spex"
    C = [np.zeros((4, 4), dtype=np.uint32) for _ in range(3)]
    C[0][0] = [0, 5, 7, 1]
    C[0][2] = [3, 3, 0, 9]
    C[1][1] = [4, 0, 4, 2]

    write_markov_spex(
        out,
        a=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        b=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        T=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        C=C,
        n_layer=3,
        n_expert=4,
        topN=2,
    )

    blob = out.read_bytes()
    header = HEADER_STRUCT.unpack_from(blob, 0)
    assert header == (b"SPEX", 1, 0, 3, 4, 0, 2, 0)

    offset = HEADER_STRUCT.size
    triples = []
    for _ in range(3):
        triples.append(struct.unpack_from("<fff", blob, offset))
        offset += 12
    np.testing.assert_allclose(
        np.array(triples, dtype=np.float32),
        np.array([[0.1, -1.0, 1.0], [0.2, -2.0, 2.0], [0.3, -3.0, 3.0]], dtype=np.float32),
    )

    rows, offset = _read_rows(blob, offset, n_rows=3 * 4)
    assert rows[0] == [(2, 7), (1, 5)]  # top counts, descending; tie-break by eid.
    assert rows[2] == [(3, 9), (0, 3)]
    assert rows[5] == [(0, 4), (2, 4)]
    assert rows[-1] == []
    assert offset == len(blob)


def test_export_c_from_npz_writes_markov_file(tmp_path):
    npz = tmp_path / "trace.npz"
    out = tmp_path / "trace.spex"
    experts = np.array(
        [
            [[0, 1], [1, 2], [2, 3]],
            [[1, 2], [2, 3], [3, 0]],
            [[2, 3], [3, 0], [0, 1]],
            [[3, 0], [0, 1], [1, 2]],
        ],
        dtype=np.int64,
    )
    np.savez(npz, experts=experts, doclens=np.array([2, 2]), n_experts=np.array(4))

    meta = export_c(str(npz), str(out), seed=0, topN=2, pred="markov")

    assert meta["predictor"] == "markov"
    assert meta["L"] == 3
    assert out.exists() and out.stat().st_size > HEADER_STRUCT.size
    assert HEADER_STRUCT.unpack_from(out.read_bytes(), 0) == (b"SPEX", 1, 0, 3, 4, 0, 2, 0)


def test_export_c_rejects_hidden_until_c_loader_contract_exists(tmp_path):
    with pytest.raises(ValueError, match="only --predictor markov"):
        export_c("unused.npz", str(tmp_path / "x.spex"), seed=0, topN=2, pred="hidden")
