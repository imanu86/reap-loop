"""base.py — interfaccia del validatore deterministico.

REQUISITI NON NEGOZIABILI (docs/00_architecture.md §8, rischio R5):
  - verdetto BINARIO 100% verificabile (no giudizio soft, no LLM-as-judge)
  - DETERMINISTICO: stesso (prompt, modello, policy) -> stesso verdetto byte-per-byte
    (greedy T=0, seed fissi, prompt fissi). Imposto da tests/test_validator_determinism.py.
  - valutazione a CONTESTO CRESCENTE: ogni item viene valutato a ciascuna lunghezza target e
    l'accuratezza è riportata come CURVA vs lunghezza, MAI come media (Vincolo B).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class ValItem:
    """Un caso di valutazione."""

    item_id: str
    prompt: str
    payload: dict   # dipende dal task: unit test, risposta attesa, posizione del needle, ...


@dataclass(frozen=True)
class ValVerdict:
    """Esito binario di UN item a UNA lunghezza di contesto."""

    item_id: str
    ctx_len: int
    correct: bool          # il segnale binario
    detail: str = ""       # es. quali unit test sono falliti (per il debug, non per lo score)


@dataclass(frozen=True)
class ContextLengthResult:
    """Accuratezza aggregata a UNA lunghezza (un punto della curva)."""

    ctx_len: int
    n_items: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_items if self.n_items else 0.0


class Validator(abc.ABC):
    """Genera col modello (sotto una data policy/miss_mode) e verifica in modo deterministico."""

    name: str

    @abc.abstractmethod
    def items(self) -> list[ValItem]:
        """Il dataset di valutazione (deterministico, ordinato)."""
        raise NotImplementedError

    @abc.abstractmethod
    def verify(self, item: ValItem, model_output: str) -> bool:
        """Verifica binaria 100% verificabile dell'output. TODO nelle sottoclassi."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_at_lengths(self, *, generate_fn, ctx_lengths: list[int], filler) -> list[ContextLengthResult]:
        """Per ciascuna lunghezza: riempi il contesto, genera, verifica, aggrega.

        `generate_fn(prompt) -> str` incapsula modello+policy+miss_mode (greedy, deterministico).
        Ritorna UNA curva (un ContextLengthResult per lunghezza). TODO.
        """
        raise NotImplementedError
