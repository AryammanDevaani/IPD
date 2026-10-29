"""Minimal in-memory BM25 (Okapi) over passages, for lexical candidate retrieval.

Adequate for a local MVP corpus (thousands to low hundreds of thousands of passages).
Swap for OpenSearch/Elasticsearch when the corpus outgrows memory.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .fingerprints import tokenize

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with".split())


def terms(text: str) -> list[str]:
    return [t.norm for t in tokenize(text) if t.norm not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add(self, doc_id: str, text: str) -> None:
        tf = Counter(terms(text))
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_len.append(sum(tf.values()))
        for term, c in tf.items():
            self.postings[term].append((idx, c))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = len(self.doc_ids)
        if n == 0:
            return []
        avg = sum(self.doc_len) / n
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for d, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[d] / avg)
                scores[d] += idf * tf * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [(self.doc_ids[d], s) for d, s in top]
