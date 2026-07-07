"""context_filler.py — porta ogni item alle lunghezze target {1k,4k,16k,64k,max}.

Due meccanismi (docs/00_architecture.md §8.2), entrambi necessari:
  - PADDING_DISTRACTORS: il TASK resta fisso; si riempie il contesto con codice/testo distrattore
    fino alla lunghezza target. Isola l'effetto della *lunghezza* a difficoltà del task costante.
  - RELEVANT_HAYSTACK: materiale rilevante in cui è immerso il fatto-chiave (per il needle).

Vincolo di determinismo: il riempimento deve essere RIPRODUCIBILE (stesso seed -> stesso contesto),
altrimenti il segnale a contesto lungo diventa rumore (rischio R5).
"""

from __future__ import annotations

import enum
import random


class FillStrategy(str, enum.Enum):
    PADDING_DISTRACTORS = "padding-distractors"
    RELEVANT_HAYSTACK = "relevant-haystack"


# Vocabolario distrattore deterministico. Serve a generare riempitivo riproducibile senza dipendere
# da risorse esterne. Il contenuto è volutamente "rumore plausibile" (parole di codice/testo) per
# stressare l'attention senza introdurre per caso un secondo fatto-chiave.
_DISTRACTOR_WORDS: tuple[str, ...] = (
    "def", "return", "value", "compute", "buffer", "index", "token", "layer",
    "expert", "router", "gate", "weight", "cache", "offset", "result", "data",
    "node", "edge", "graph", "matrix", "vector", "scalar", "tensor", "stream",
    "the", "and", "of", "to", "in", "for", "while", "with", "from", "import",
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
)


class ContextFiller:
    """Costruisce un prompt della lunghezza target a partire dal task base.

    Misura le lunghezze in TOKEN del tokenizer del modello (non in caratteri), e ritorna anche la
    lunghezza effettiva ottenuta (può non essere esattamente il target). Rispetta la context window
    massima del modello (docs/02_models.md §5): se il target supera il max, segnala e tronca.
    """

    def __init__(self, tokenizer, strategy: FillStrategy, seed: int = 0) -> None:
        self._tok = tokenizer
        self._strategy = strategy
        self._seed = seed

    # --- helper interni -------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        """Numero di token = len(encode(text)). Il tokenizer iniettato espone `encode`/`decode`."""
        return list(self._tok.encode(text))

    def _decode(self, ids) -> str:
        return self._tok.decode(list(ids))

    def _n_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _distractor_text(self, n_tokens_needed: int, rng: random.Random) -> str:
        """Genera testo distrattore con (circa) `n_tokens_needed` token, in modo deterministico.

        Genera parola per parola dal vocabolario fisso usando l'RNG seedato, accumulando finché il
        conteggio in token (secondo il tokenizer iniettato) raggiunge il target. Deterministico:
        stesso seed -> stessa sequenza di parole.
        """
        if n_tokens_needed <= 0:
            return ""
        words: list[str] = []
        # Generiamo a blocchi e ri-misuriamo: evita una chiamata al tokenizer per ogni parola, ma
        # resta esatto perché controlliamo il conteggio reale prima di restituire.
        while True:
            # blocco proporzionato al residuo (almeno 1 parola)
            remaining = n_tokens_needed - (self._n_tokens(" ".join(words)) if words else 0)
            if remaining <= 0:
                break
            block = max(1, remaining)
            for _ in range(block):
                words.append(rng.choice(_DISTRACTOR_WORDS))
            # taglia eventuale eccesso parola-per-parola fino a non superare il target
            while words and self._n_tokens(" ".join(words)) > n_tokens_needed:
                words.pop()
                # se siamo scesi sotto il target ci fermeremo comunque al controllo esterno
                if self._n_tokens(" ".join(words)) <= n_tokens_needed:
                    break
            if self._n_tokens(" ".join(words)) >= n_tokens_needed:
                break
        return " ".join(words)

    def _max_ctx(self) -> int | None:
        """Context window massima dichiarata dal tokenizer/modello, se nota; altrimenti None."""
        for attr in ("model_max_length", "max_context", "max_length"):
            val = getattr(self._tok, attr, None)
            if isinstance(val, int) and 0 < val < 10**12:  # scarta i sentinella tipo 1e30 di HF
                return val
        return None

    # --- API pubblica ---------------------------------------------------

    def fill(self, base_prompt: str, target_tokens: int, *, needle: str | None = None,
             needle_depth: float | None = None) -> tuple[str, int]:
        """Ritorna (prompt_riempito, lunghezza_effettiva_in_token). Deterministico dato il seed.

        - Il riempitivo distrattore è generato da un RNG seedato (`self._seed`): stesso seed +
          stesso target -> stesso prompt e stessa lunghezza.
        - Se `needle` è dato, viene inserito a profondità relativa `needle_depth` (0.0 = inizio,
          1.0 = fine) tra il base_prompt e il riempitivo: serve al NeedleInHaystackValidator.
        - Rispetta la context window massima (se il tokenizer la espone): se il target la supera,
          tronca al massimo consentito.
        """
        # RNG locale: deterministico e isolato (non tocca lo stato globale di random).
        rng = random.Random(self._seed)

        # Rispetto della context window: tronca il target se eccede il massimo del modello.
        max_ctx = self._max_ctx()
        effective_target = target_tokens
        if max_ctx is not None and effective_target > max_ctx:
            effective_target = max_ctx

        base_tokens = self._n_tokens(base_prompt)
        needle_tokens = self._n_tokens(needle) if needle else 0

        # Budget per il riempitivo distrattore = target - (base + needle), mai negativo.
        fill_budget = effective_target - base_tokens - needle_tokens
        if fill_budget < 0:
            fill_budget = 0

        distractor = self._distractor_text(fill_budget, rng)

        # Composizione del prompt.
        if needle is not None:
            depth = 0.5 if needle_depth is None else float(needle_depth)
            depth = min(1.0, max(0.0, depth))
            # Spezza il distrattore in due metà secondo la profondità (in parole, deterministico).
            d_words = distractor.split(" ") if distractor else []
            cut = int(round(depth * len(d_words)))
            head = " ".join(d_words[:cut])
            tail = " ".join(d_words[cut:])
            # base_prompt (domanda) in testa per restare visibile; needle immerso nel distrattore.
            parts = [p for p in (base_prompt, head, needle, tail) if p]
            prompt = "\n".join(parts)
        else:
            parts = [p for p in (base_prompt, distractor) if p]
            prompt = "\n".join(parts)

        return prompt, self._n_tokens(prompt)
