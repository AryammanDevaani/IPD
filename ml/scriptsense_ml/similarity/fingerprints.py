"""Word n-gram fingerprints with winnowing (Schleimer, Wilkerson & Aiken, 2003) and
exact passage alignment that keeps character offsets into the original texts.

Tokens are case/punctuation-normalised words, each remembering its original span, so a
match can always be highlighted in both the submission and the source.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Token:
    norm: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(unicodedata.normalize("NFKC", m.group()).casefold(), m.start(), m.end())
            for m in _WORD.finditer(text)]


def ngram_hashes(tokens: list[Token], n: int) -> list[int]:
    return [zlib.crc32(" ".join(t.norm for t in tokens[i:i + n]).encode()) for i in range(len(tokens) - n + 1)]


def winnow(hashes: list[int], window: int) -> list[tuple[int, int]]:
    """Return (hash, position) fingerprints: the rightmost minimum of each window.
    Guarantees any shared run of >= window + n - 1 tokens shares at least one fingerprint."""
    if not hashes:
        return []
    if len(hashes) <= window:
        m = min(hashes)
        return [(m, max(i for i, h in enumerate(hashes) if h == m))]
    out, last = [], -1
    for s in range(len(hashes) - window + 1):
        win = hashes[s:s + window]
        m = min(win)
        pos = s + max(i for i, h in enumerate(win) if h == m)
        if pos != last:
            out.append((m, pos))
            last = pos
    return out


@dataclass
class Match:
    sub_start: int   # character offsets in the submission
    sub_end: int
    src_start: int   # character offsets in the source
    src_end: int
    n_words: int


def align_exact(sub_tokens: list[Token], src_tokens: list[Token], n: int = 5, min_words: int = 8) -> list[Match]:
    """Maximal runs of identical normalised words (seeded by shared n-grams) of >= min_words."""
    if len(sub_tokens) < n or len(src_tokens) < n:
        return []
    src_index: dict[int, list[int]] = {}
    for j, h in enumerate(ngram_hashes(src_tokens, n)):
        src_index.setdefault(h, []).append(j)
    matches, covered_until = [], -1
    sub_hashes = ngram_hashes(sub_tokens, n)
    i = 0
    while i < len(sub_hashes):
        best = None
        for j in src_index.get(sub_hashes[i], []):
            if any(sub_tokens[i + k].norm != src_tokens[j + k].norm for k in range(n)):
                continue  # crc collision
            # extend left (only within not-yet-covered submission text) and right
            a, b = i, j
            while a > covered_until + 1 and b > 0 and sub_tokens[a - 1].norm == src_tokens[b - 1].norm:
                a, b = a - 1, b - 1
            length = 0
            while a + length < len(sub_tokens) and b + length < len(src_tokens) and \
                    sub_tokens[a + length].norm == src_tokens[b + length].norm:
                length += 1
            if best is None or length > best[2]:
                best = (a, b, length)
        if best and best[2] >= min_words:
            a, b, length = best
            matches.append(Match(sub_tokens[a].start, sub_tokens[a + length - 1].end,
                                 src_tokens[b].start, src_tokens[b + length - 1].end, length))
            covered_until = a + length - 1
            i = a + length
        else:
            i += 1
    return matches
