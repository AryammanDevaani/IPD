"""Exact and near-duplicate detection.

Exact duplicates: identical ``text_hash`` (aggressive normalization).
Near duplicates: MinHash over word 5-gram shingles + LSH banding, then candidate
pairs are confirmed by estimated Jaccard >= ``threshold``.

Everything here is deterministic: word hashes use CRC32 (not Python's salted
``hash``) and permutation parameters come from a seeded generator.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .normalization import dedup_key

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def shingle_hashes(text: str, n: int = 5) -> np.ndarray:
    words = dedup_key(text).split()
    if not words:
        return np.zeros(1, dtype=np.uint64)
    wh = np.fromiter((zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words))
    if len(words) < n:
        n = len(words)
    # Polynomial combination of n consecutive word hashes (wraps mod 2^64).
    acc = np.zeros(len(words) - n + 1, dtype=np.uint64)
    base = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for i in range(n):
            acc = acc * base + wh[i: len(words) - n + 1 + i]
    return np.unique(acc)


@dataclass
class MinHasher:
    num_perm: int = 128
    seed: int = 1234

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.a = rng.integers(1, 2**63, size=self.num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=self.num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        sh = shingle_hashes(text)
        with np.errstate(over="ignore"):
            # multiply-shift universal hashing, one column per permutation
            h = (sh[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)
        return h.min(axis=0).astype(np.uint32)


def exact_duplicate_groups(hashes: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    return {h: idx for h, idx in groups.items() if len(idx) > 1}


def near_duplicate_pairs(
    signatures: np.ndarray, bands: int = 16, threshold: float = 0.8
) -> list[tuple[int, int, float]]:
    """Return (i, j, estimated_jaccard) for i<j whose signatures collide in an LSH band
    and whose estimated Jaccard similarity is at least ``threshold``."""
    n, num_perm = signatures.shape
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    seen: set[tuple[int, int]] = set()
    out = []
    for b in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = np.ascontiguousarray(signatures[:, b * rows:(b + 1) * rows])
        for i in range(n):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 5000:  # giant buckets = degenerate (e.g. empty) texts
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pair = (members[x], members[y])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    j = float(np.mean(signatures[pair[0]] == signatures[pair[1]]))
                    if j >= threshold:
                        out.append((pair[0], pair[1], j))
    return out
