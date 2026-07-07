"""Retrieval over the CONVERSATION HISTORY (not a domain KB).

Rung-2 of the eventual cascade is recall-reinject (BM25+e5 over an external KB); here,
for the Step-1 always-RAG baseline (B1), we retrieve the most relevant PAST TURNS.
Reuse-first: we import the pure/in-memory `BM25Index` from an external KB project (retrieval/) via
sys.path (spec: reuse retrieval/bm25.py read-only). If that module is not importable
(different machine, moved repo), a self-contained Okapi-BM25 fallback keeps the
harness portable. Which backend was used is recorded in the run manifest.

e5 (embed.py) is deferred to rung-1 (Step 2): it needs sentence-transformers+torch
and a ~650MB model download, so it is NOT a dependency of the Step-1 baseline.
"""
from __future__ import annotations
import os
import re
import sys
import math
import unicodedata

import config

_EXTERNAL_OK = False
_BM25Index = None


def _try_import_external():
    global _EXTERNAL_OK, _BM25Index
    d = config.EXTERNAL_RETRIEVAL_DIR
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
    try:
        from bm25 import BM25Index  # type: ignore
        _BM25Index = BM25Index
        _EXTERNAL_OK = True
    except Exception:
        _EXTERNAL_OK = False
    return _EXTERNAL_OK


_try_import_external()


# --- self-contained fallback ------------------------------------------------
def _tokenize(text: str):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]{2,}|\d+", text)


class _FallbackBM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(c["bm25_text"]) for c in chunks]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.tf = [{} for _ in self.docs]
        for i, d in enumerate(self.docs):
            for t in d:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1

    def _idf(self, t):
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query, k=10, filters=None):
        q = _tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            dl = len(d)
            s = 0.0
            for t in q:
                if t not in self.tf[i]:
                    continue
                f = self.tf[i][t]
                s += self._idf(t) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class MemoryRetriever:
    """Indexes conversation messages; retrieves the top-k most relevant ones."""
    def __init__(self, messages):
        self.messages = messages
        chunks = [{"bm25_text": m.get("content", ""), "fields": {}, "source_type": m.get("role", "")}
                  for m in messages]
        if _EXTERNAL_OK:
            self.index = _BM25Index(chunks)
            self.backend = "external.BM25Index"
        else:
            self.index = _FallbackBM25(chunks)
            self.backend = "fallback.Okapi"

    def search(self, query, k=5):
        hits = self.index.search(query, k=k)
        return [{"idx": i, "score": float(s), **self.messages[i]} for i, s in hits]


def backend_name():
    return "external.BM25Index" if _EXTERNAL_OK else "fallback.Okapi"
