"""Lineage-grouped split assignment and leakage auditing.

Rows are connected when they share any group key (prompt, source document,
generation lineage, exact-duplicate hash, near-duplicate cluster). Each connected
component is assigned to exactly one split, so variants of one original never
straddle train and evaluation data.

Publisher-fixed splits (e.g. the SemEval official dev set) pin their component.
If a component contains rows pinned to different evaluation splits, it cannot be
assigned without leakage and is reported as a conflict for removal.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .deduplication import UnionFind


def connected_components(group_keys: list[list[str]]) -> list[int]:
    """Return a component id (smallest row index in the component) per row."""
    uf = UnionFind(len(group_keys))
    first_seen: dict[str, int] = {}
    for i, keys in enumerate(group_keys):
        for k in keys:
            if not k:
                continue
            if k in first_seen:
                uf.union(first_seen[k], i)
            else:
                first_seen[k] = i
    return [uf.find(i) for i in range(len(group_keys))]


def _unit_interval(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_splits(
    components: list[int],
    component_names: dict[int, str],
    fixed_splits: list[str],
    ratios: dict[str, float],
    seed: int,
) -> tuple[list[str], dict[int, list[str]]]:
    """Assign one split per component.

    ``fixed_splits[i]`` is the publisher split for row i or "" if free.
    ``ratios`` maps split name -> probability for free components.
    ``component_names`` gives a stable, order-independent name per component
    (e.g. its smallest row_id) so assignment is reproducible across runs.
    Returns per-row splits ("" for conflicting components) and the conflicts.
    """
    total = sum(ratios.values())
    cumulative, acc = [], 0.0
    for name, r in ratios.items():
        acc += r / total
        cumulative.append((acc, name))

    pins: dict[int, set[str]] = defaultdict(set)
    for comp, fixed in zip(components, fixed_splits):
        if fixed:
            pins[comp].add(fixed)

    decided: dict[int, str] = {}
    conflicts: dict[int, list[str]] = {}
    for comp in set(components):
        pinned = pins.get(comp, set())
        eval_pins = pinned - {"train"}
        if len(eval_pins) > 1:
            conflicts[comp] = sorted(pinned)
            decided[comp] = ""
        elif eval_pins:
            decided[comp] = next(iter(eval_pins))  # evaluation pin wins; train rows in it get moved/dropped
        elif pinned:
            decided[comp] = "train"
        else:
            u = _unit_interval(component_names[comp], seed)
            decided[comp] = next(name for edge, name in cumulative if u < edge or edge == cumulative[-1][0])
    return [decided[c] for c in components], conflicts


def leakage_report(splits: list[str], group_keys: list[list[str]]) -> dict:
    """Every group key must live in exactly one split. Returns offending keys."""
    key_splits: dict[str, set[str]] = defaultdict(set)
    for s, keys in zip(splits, group_keys):
        if not s:
            continue
        for k in keys:
            if k:
                key_splits[k].add(s)
    leaks = {k: sorted(v) for k, v in key_splits.items() if len(v) > 1}
    kinds = Counter(k.split(":", 2)[0] + ":" + k.split(":", 2)[1] if k.count(":") >= 2 else k for k in leaks)
    return {"num_leaking_keys": len(leaks), "by_key_type": dict(kinds), "examples": dict(list(leaks.items())[:20])}


class LeakageError(RuntimeError):
    pass


def assert_no_leakage(splits: list[str], group_keys: list[list[str]]) -> None:
    rep = leakage_report(splits, group_keys)
    if rep["num_leaking_keys"]:
        raise LeakageError(f"split leakage detected: {rep}")
