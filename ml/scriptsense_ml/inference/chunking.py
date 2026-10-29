"""Eligible-prose detection and overlapping token windows with original character offsets.

Windows are computed on the ORIGINAL extracted text so every chunk maps back to exact
character offsets (and therefore pages). The model sees normalize_for_model(chunk), the
same normalization used in training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCE_HEADINGS = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|reference list|sources)\s*:?\s*$", re.I)
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Block:
    start: int
    end: int
    eligible: bool
    reason: str = ""


@dataclass
class Window:
    start: int        # character offsets into the original text
    end: int
    n_tokens: int
    new_chars: int    # characters not already covered by the previous window (aggregation weight)


def _alpha_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def segment_blocks(text: str, min_words: int = 8) -> list[Block]:
    """Split into paragraphs and mark which are eligible prose.

    Excluded: everything after a references/bibliography heading, short lines (headings,
    captions), and low-alphabetic blocks (tables, formulas, code).
    """
    blocks, pos, in_refs = [], 0, False
    for m in list(_PARA_SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        lead = len(seg) - len(seg.lstrip())
        s, e = pos + lead, pos + len(seg.rstrip())
        if e > s:
            body = text[s:e]
            first_line = body.split("\n", 1)[0]
            if REFERENCE_HEADINGS.match(first_line):
                in_refs = True
            if in_refs:
                blocks.append(Block(s, e, False, "references"))
            elif len(body.split()) < min_words:
                blocks.append(Block(s, e, False, "too_short"))
            elif _alpha_ratio(body) < 0.6:
                blocks.append(Block(s, e, False, "non_prose"))
            else:
                blocks.append(Block(s, e, True))
        pos = m.end() if m else len(text)
    return blocks


def eligible_spans(blocks: list[Block]) -> list[tuple[int, int]]:
    """Merge consecutive eligible blocks into spans; windows never cross excluded material."""
    spans: list[list[int]] = []
    prev_eligible = False
    for b in blocks:
        if b.eligible and prev_eligible:
            spans[-1][1] = b.end
        elif b.eligible:
            spans.append([b.start, b.end])
        prev_eligible = b.eligible
    return [(s, e) for s, e in spans]


def token_windows(text: str, span: tuple[int, int], tokenizer, max_tokens: int = 384,
                  overlap: int = 64) -> list[Window]:
    s0, e0 = span
    enc = tokenizer(text[s0:e0], add_special_tokens=False, return_offsets_mapping=True)
    offs = [o for o in enc["offset_mapping"] if o[1] > o[0]]
    if not offs:
        return []
    body = max_tokens - tokenizer.num_special_tokens_to_add()
    stride = max(1, body - overlap)
    windows, covered_to, i = [], s0, 0
    while True:
        chunk = offs[i:i + body]
        ws, we = s0 + chunk[0][0], s0 + chunk[-1][1]
        windows.append(Window(ws, we, len(chunk), max(0, we - max(ws, covered_to))))
        covered_to = max(covered_to, we)
        if i + body >= len(offs):
            break
        i += stride
    return windows


def offset_to_page(offset: int, page_starts: list[int]) -> int:
    """1-based page for a character offset given each page's starting offset."""
    page = 1
    for i, start in enumerate(page_starts):
        if offset >= start:
            page = i + 1
    return page
