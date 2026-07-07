"""trace.py — schema della traccia di attivazione + I/O (jsonl).

Una traccia = la sequenza di expert attivati per (layer, token) in una sessione, con i metadati di
lunghezza di contesto. È l'input di workingset/ (stima + concentrazione) e di experiment/ (metriche).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

# Buffer di default: numero di record accumulati prima di un flush su disco.
# La strumentazione gira dentro generate(); bufferizzare in batch tiene basso l'overhead di I/O (R4).
_DEFAULT_BUFFER_SIZE = 256


@dataclass(frozen=True)
class ActivationRecord:
    """Una riga di traccia: top-k expert di UN layer per UN token.

    Vedi docs/00_architecture.md §6 per lo schema jsonl.
    """

    session_id: str
    step: int            # indice di generazione/forward
    layer: int           # indice del layer MoE
    token_pos: int       # posizione del token nel contesto
    topk_ids: tuple[int, ...]
    gate_w: tuple[float, ...]
    ctx_len: int         # lunghezza di contesto della sessione (asse C)
    model_id: str

    def to_json(self) -> dict:
        """Serializza il record nel dict dello schema jsonl (§6).

        Le tuple `topk_ids`/`gate_w` diventano liste (JSON non ha tuple); l'ordine delle chiavi
        segue lo schema documentato per leggibilità del file.
        """
        return {
            "session_id": self.session_id,
            "step": self.step,
            "layer": self.layer,
            "token_pos": self.token_pos,
            "topk_ids": list(self.topk_ids),
            "gate_w": list(self.gate_w),
            "ctx_len": self.ctx_len,
            "model_id": self.model_id,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "ActivationRecord":
        """Ricostruisce un ActivationRecord dal dict letto dal jsonl (inverso di to_json)."""
        return cls(
            session_id=obj["session_id"],
            step=obj["step"],
            layer=obj["layer"],
            token_pos=obj["token_pos"],
            topk_ids=tuple(obj["topk_ids"]),
            gate_w=tuple(obj["gate_w"]),
            ctx_len=obj["ctx_len"],
            model_id=obj["model_id"],
        )


class TraceWriter:
    """Scrive ActivationRecord in append su un file jsonl (uno per sessione).

    Deve essere a basso overhead: la strumentazione gira durante generate(); bufferizzare e
    scrivere in batch (vedi rischio R4, overhead del predittore/stima).

    Usabile anche come context manager (`with TraceWriter(path) as w: ...`) per garantire il flush.
    """

    def __init__(self, path: str, buffer_size: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._path = path
        self._buffer_size = max(1, int(buffer_size))
        self._buffer: list[str] = []
        # Apertura in append: una traccia per sessione, scritture incrementali durante generate().
        self._fh = open(path, "a", encoding="utf-8")
        self._closed = False

    def write(self, rec: ActivationRecord) -> None:
        """Accoda un record al buffer; effettua il flush quando il buffer è pieno."""
        if self._closed:
            raise ValueError("TraceWriter già chiuso")
        self._buffer.append(json.dumps(rec.to_json(), ensure_ascii=False, separators=(",", ":")))
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        """Svuota il buffer su disco in un'unica scrittura (una riga jsonl per record)."""
        if not self._buffer:
            return
        # Un solo write() per batch: ogni record termina con '\n' (formato jsonl).
        self._fh.write("".join(line + "\n" for line in self._buffer))
        self._fh.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush finale del buffer e chiusura del file (idempotente)."""
        if self._closed:
            return
        self._flush()
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TraceReader:
    """Legge una traccia jsonl, opzionalmente filtrando per layer / range di token."""

    def __init__(self, path: str) -> None:
        self._path = path

    def records(self, layer: int | None = None) -> Iterator[ActivationRecord]:
        """Itera sui record della traccia, opzionalmente filtrando per `layer`.

        Lettura lazy riga-per-riga (le tracce possono essere grandi). Le righe vuote vengono
        ignorate, così un file vuoto o con righe in bianco produce semplicemente zero record.
        """
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if layer is not None and obj.get("layer") != layer:
                    continue
                yield ActivationRecord.from_json(obj)
