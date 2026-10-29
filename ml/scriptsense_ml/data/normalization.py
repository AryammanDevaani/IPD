"""Text normalization.

Two levels, deliberately different:

* ``normalize_for_model`` is conservative and is applied identically at training
  and inference time. It keeps casing, punctuation and paragraph breaks, and only
  removes invisible characters that carry no linguistic content (zero-width
  spaces, BOMs, soft hyphens). Removing them is intentional: RAID's
  ``zero_width_space`` attack would otherwise be a trivially exploitable signal.
* ``dedup_key`` is aggressive (NFKC, casefold, punctuation-insensitive) and is
  used only to detect duplicates. It is never shown to a model or user.

Raw text is always stored separately so offsets can be mapped back.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_HSPACE = re.compile(r"[^\S\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_SPACE_AROUND_NL = re.compile(r" *\n *")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_for_model(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_INVISIBLE)
    text = _HSPACE.sub(" ", text)
    text = _SPACE_AROUND_NL.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def dedup_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).casefold()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    """Stable exact-duplicate hash over the aggressive dedup key."""
    return hashlib.sha1(dedup_key(text).encode("utf-8")).hexdigest()


def stable_id(*parts: object) -> str:
    """Deterministic short id for hashing provenance values (e.g. prompts)."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


def word_count(text: str) -> int:
    return len(text.split())


def decode_mixed_lines(raw: bytes) -> tuple[str, dict[str, int]]:
    """Decode a byte stream whose lines use a mix of UTF-8 and Windows-1252.

    Each physical line is tried as UTF-8, then cp1252, then latin-1 (which never
    fails). Counts are returned so the manifest can record what happened.
    """
    stats = {"utf8": 0, "cp1252": 0, "latin1_fallback": 0}
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    out = []
    for line in raw.split(b"\n"):
        try:
            out.append(line.decode("utf-8"))
            stats["utf8"] += 1
            continue
        except UnicodeDecodeError:
            pass
        try:
            out.append(line.decode("cp1252"))
            stats["cp1252"] += 1
        except UnicodeDecodeError:
            out.append(line.decode("latin-1"))
            stats["latin1_fallback"] += 1
    return "\n".join(out), stats


def unique_trigram_ratio(text: str, min_trigrams: int = 20) -> float:
    """Unique word trigrams / all trigrams (1.0 for texts too short to judge)."""
    w = text.lower().split()
    grams = list(zip(w, w[1:], w[2:]))
    return len(set(grams)) / len(grams) if len(grams) >= min_trigrams else 1.0
