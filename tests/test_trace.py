"""test_trace.py — round-trip e robustezza dello schema/I-O della traccia di attivazione.

CPU-only, deterministico, isolato: usa solo `tmp_path` e la stdlib (nessun torch/transformers,
nessuna dipendenza da moduli implementati da altri agent).
"""

from __future__ import annotations

import json

from msc.instrument.trace import ActivationRecord, TraceReader, TraceWriter


def _make_record(i: int, layer: int) -> ActivationRecord:
    """Costruisce un record deterministico in funzione dell'indice e del layer."""
    return ActivationRecord(
        session_id="sess-0",
        step=i,
        layer=layer,
        token_pos=i,
        topk_ids=(layer, i % 8, (i + 1) % 8),
        gate_w=(0.7, 0.2, 0.1),
        ctx_len=4096,
        model_id="OLMoE-1B-7B",
    )


def test_to_json_schema_e_tipi():
    """to_json rispetta lo schema §6: chiavi attese, tuple -> liste, JSON-serializzabile."""
    rec = _make_record(3, layer=2)
    d = rec.to_json()

    assert set(d.keys()) == {
        "session_id", "step", "layer", "token_pos",
        "topk_ids", "gate_w", "ctx_len", "model_id",
    }
    # Le tuple devono diventare liste (JSON non ha tuple).
    assert d["topk_ids"] == [2, 3, 4]
    assert isinstance(d["topk_ids"], list)
    assert isinstance(d["gate_w"], list)
    # Deve essere realmente serializzabile in JSON.
    assert json.loads(json.dumps(d)) == d


def test_round_trip_write_read(tmp_path):
    """Scrivere N record e rileggerli restituisce gli stessi oggetti, nello stesso ordine."""
    path = str(tmp_path / "trace.jsonl")
    n = 50
    originals = [_make_record(i, layer=i % 3) for i in range(n)]

    # buffer_size piccolo per esercitare più flush durante la scrittura.
    writer = TraceWriter(path, buffer_size=8)
    for rec in originals:
        writer.write(rec)
    writer.close()

    read_back = list(TraceReader(path).records())
    assert len(read_back) == n
    assert read_back == originals  # dataclass frozen -> __eq__ per valore


def test_filtro_per_layer(tmp_path):
    """records(layer=L) restituisce solo i record di quel layer, preservando l'ordine."""
    path = str(tmp_path / "trace.jsonl")
    originals = [_make_record(i, layer=i % 3) for i in range(30)]

    with TraceWriter(path, buffer_size=4) as writer:
        for rec in originals:
            writer.write(rec)

    layer1 = list(TraceReader(path).records(layer=1))
    attesi = [r for r in originals if r.layer == 1]
    assert layer1 == attesi
    assert all(r.layer == 1 for r in layer1)

    # Layer inesistente -> nessun record.
    assert list(TraceReader(path).records(layer=999)) == []


def test_file_vuoto(tmp_path):
    """Un file vuoto produce zero record senza errori (robustezza)."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(TraceReader(str(path)).records()) == []
    assert list(TraceReader(str(path)).records(layer=0)) == []


def test_writer_vuoto_crea_file_vuoto(tmp_path):
    """Un writer aperto e chiuso senza write() produce un file vuoto leggibile come zero record."""
    path = str(tmp_path / "noop.jsonl")
    TraceWriter(path).close()
    assert list(TraceReader(path).records()) == []


def test_righe_vuote_ignorate(tmp_path):
    """Righe bianche nel jsonl vengono ignorate, non causano errori di parsing."""
    path = tmp_path / "with_blanks.jsonl"
    rec = _make_record(0, layer=0)
    line = json.dumps(rec.to_json())
    path.write_text(f"\n{line}\n\n{line}\n\n", encoding="utf-8")

    read_back = list(TraceReader(str(path)).records())
    assert read_back == [rec, rec]


def test_close_idempotente_e_write_dopo_close(tmp_path):
    """close() è idempotente; write() dopo close() solleva ValueError."""
    path = str(tmp_path / "t.jsonl")
    writer = TraceWriter(path)
    writer.write(_make_record(0, layer=0))
    writer.close()
    writer.close()  # idempotente: nessun errore

    try:
        writer.write(_make_record(1, layer=0))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("write() dopo close() deve sollevare ValueError")
