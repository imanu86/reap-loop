"""closed_form.py — task SECONDARIO/triangolazione: risposte chiuse + probe long-context.

Due validatori:
  - ClosedFormValidator: domanda con risposta chiusa controllabile (uguaglianza esatta / regex).
  - NeedleInHaystackValidator: la risposta RICHIEDE di attendere a contesto distante.

Il secondo è il test specifico per il rischio R1 / Vincolo B: un task di codegen corto può non
toccare mai l'attention a lungo raggio e quindi NON esporre i fallimenti che collassano a contesto
lungo. Il needle li forza: piazza il fatto-chiave a profondità variabile nel contesto riempito e
chiede di recuperarlo. Se l'accuratezza del needle crolla con ctx mentre quella a ctx corto regge,
è esattamente la modalità di fallimento da dichiarare FALLITA (docs/00_architecture.md §12).
"""

from __future__ import annotations

import random
import re

from msc.validator.base import ContextLengthResult, ValItem, Validator


def _normalize(text: str) -> str:
    """Normalizzazione per il match: strip, lower, whitespace collassato.

    Deterministica e senza dipendenze. Serve sia al closed-form (match tollerante) sia al needle
    (recupero del valore atteso ovunque appaia nell'output).
    """
    return re.sub(r"\s+", " ", text).strip().lower()


class ClosedFormValidator(Validator):
    name = "closed-form"

    # Dataset chiuso e deterministico: domande con risposta esatta controllabile. Niente file
    # esterni -> riproducibile e CPU-only. `answer` è la stringa attesa; `aliases` risposte
    # equivalenti accettate (match normalizzato).
    _DATASET: tuple[dict, ...] = (
        {"item_id": "cf-00", "prompt": "Quanto fa 2 + 2? Rispondi col solo numero.",
         "answer": "4", "aliases": ()},
        {"item_id": "cf-01", "prompt": "Capitale dell'Italia? Una parola.",
         "answer": "Roma", "aliases": ()},
        {"item_id": "cf-02", "prompt": "7 moltiplicato 6?",
         "answer": "42", "aliases": ("quarantadue",)},
        {"item_id": "cf-03", "prompt": "Vero o falso: il cielo e' verde?",
         "answer": "falso", "aliases": ("false",)},
    )

    def items(self) -> list[ValItem]:
        """Dataset di valutazione chiuso, in ordine deterministico."""
        return [
            ValItem(
                item_id=row["item_id"],
                prompt=row["prompt"],
                payload={"answer": row["answer"], "aliases": tuple(row.get("aliases", ()))},
            )
            for row in self._DATASET
        ]

    def verify(self, item: ValItem, model_output: str) -> bool:
        """Uguaglianza esatta / match normalizzato con item.payload['answer'].

        Accetta sia il match esatto sia il caso in cui la risposta attesa compaia (normalizzata)
        come token nell'output del modello: i modelli tendono a rispondere a frase intera.
        """
        expected = item.payload["answer"]
        aliases = item.payload.get("aliases", ())
        out_norm = _normalize(model_output)
        for candidate in (expected, *aliases):
            cand_norm = _normalize(candidate)
            if not cand_norm:
                continue
            if out_norm == cand_norm:
                return True
            # match come parola intera dentro l'output (evita falsi positivi tipo "42" in "423")
            if re.search(rf"(?<!\w){re.escape(cand_norm)}(?!\w)", out_norm):
                return True
        return False

    def evaluate_at_lengths(self, *, generate_fn, ctx_lengths, filler) -> list[ContextLengthResult]:
        """Per ciascuna lunghezza: riempi il contesto, genera, verifica, aggrega in UN punto."""
        items = self.items()
        results: list[ContextLengthResult] = []
        for ctx in ctx_lengths:
            n_correct = 0
            for item in items:
                filled_prompt, _eff_len = filler.fill(item.prompt, ctx)
                output = generate_fn(filled_prompt)
                if self.verify(item, output):
                    n_correct += 1
            results.append(ContextLengthResult(ctx_len=ctx, n_items=len(items), n_correct=n_correct))
        return results


class NeedleInHaystackValidator(Validator):
    name = "needle-in-haystack"

    # Spazio chiuso e deterministico per generare i needle: una coppia (codice, valore) per item.
    # Il valore è ciò che il modello deve recuperare; la domanda lo richiede esplicitamente.
    _MAGIC_CODES: tuple[str, ...] = (
        "ALPHA-7731", "BRAVO-1024", "CHARLIE-4096", "DELTA-2718", "ECHO-3141",
    )

    def __init__(self, needle_depths: tuple[float, ...] = (0.1, 0.5, 0.9)) -> None:
        # needle_depths: profondità relative a cui piazzare il fatto-chiave nel contesto riempito.
        self.needle_depths = needle_depths

    def _value_for(self, idx: int, depth: float) -> str:
        """Valore-chiave deterministico per (indice item, profondità). Stabile e ricostruibile."""
        rng = random.Random((idx, round(depth, 6)).__hash__())
        base = self._MAGIC_CODES[idx % len(self._MAGIC_CODES)]
        return f"{base}-{rng.randint(1000, 9999)}"

    def items(self) -> list[ValItem]:
        """Genera (needle, domanda, profondità) deterministici.

        Un item per (indice base, profondità): il needle è una frase che dichiara il valore-chiave,
        la domanda chiede di recuperarlo, il payload porta valore atteso e profondità.
        """
        items: list[ValItem] = []
        for idx in range(len(self._MAGIC_CODES)):
            for depth in self.needle_depths:
                value = self._value_for(idx, depth)
                needle = f"Il codice magico segreto e' {value}."
                question = (
                    "Nel testo seguente e' nascosto un codice magico segreto. "
                    "Qual e' il codice magico segreto? Rispondi col solo codice."
                )
                item_id = f"nih-{idx:02d}-d{int(round(depth * 100)):03d}"
                items.append(
                    ValItem(
                        item_id=item_id,
                        prompt=question,
                        payload={"needle": needle, "answer": value, "needle_depth": depth},
                    )
                )
        return items

    def verify(self, item: ValItem, model_output: str) -> bool:
        """Il modello ha recuperato il needle? match esatto col valore atteso (normalizzato)."""
        expected = _normalize(item.payload["answer"])
        out_norm = _normalize(model_output)
        if not expected:
            return False
        return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", out_norm) is not None

    def evaluate_at_lengths(self, *, generate_fn, ctx_lengths, filler) -> list[ContextLengthResult]:
        """Curva accuratezza-recupero vs ctx (per profondità). Cattura il fallimento R1.

        Per ogni lunghezza, riempi il contesto col needle alla profondità dell'item, genera,
        verifica il recupero, aggrega in UN punto per ciascuna ctx.
        """
        items = self.items()
        results: list[ContextLengthResult] = []
        for ctx in ctx_lengths:
            n_correct = 0
            for item in items:
                filled_prompt, _eff_len = filler.fill(
                    item.prompt,
                    ctx,
                    needle=item.payload["needle"],
                    needle_depth=item.payload["needle_depth"],
                )
                output = generate_fn(filled_prompt)
                if self.verify(item, output):
                    n_correct += 1
            results.append(ContextLengthResult(ctx_len=ctx, n_items=len(items), n_correct=n_correct))
        return results
