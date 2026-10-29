import numpy as np
import pytest
from transformers import AutoTokenizer

from scriptsense_ml.inference.aggregation import Policy, aggregate
from scriptsense_ml.inference.chunking import eligible_spans, offset_to_page, segment_blocks, token_windows

PROSE = ("Researchers in the coastal town measured how quickly the tide eroded the old sea wall, "
         "and they compared those readings with records kept by the harbour office since the war. ")


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


def test_segment_blocks_excludes_headings_tables_and_references():
    text = ("Introduction\n\n" + PROSE * 2 + "\n\n| 1.2 | 3.4 | 5.6 | 7.8 | 9.0 | 1.1 | 2.2 | 3.3 |\n\n"
            + PROSE + "\n\nReferences\n\nSmith, J. (2020). A study of tides and walls in coastal towns. Journal.")
    blocks = segment_blocks(text)
    reasons = [b.reason for b in blocks]
    assert reasons == ["too_short", "", "non_prose", "", "references", "references"]
    spans = eligible_spans(blocks)
    assert len(spans) == 2  # windows never cross the excluded table
    for s, e in spans:
        assert text[s:e].startswith("Researchers")


def test_windows_cover_span_with_overlap_and_exact_offsets(tok):
    text = "Title\n\n" + PROSE * 40
    (span,) = eligible_spans(segment_blocks(text))
    wins = token_windows(text, span, tok, max_tokens=128, overlap=32)
    assert len(wins) > 3
    assert wins[0].start == span[0] and wins[-1].end == span[1]
    for a, b in zip(wins, wins[1:]):
        assert b.start < a.end  # overlap
        assert a.n_tokens <= 126
    # overlap-aware weights add up to exactly the covered characters
    assert sum(w.new_chars for w in wins) == span[1] - span[0]
    assert text[wins[1].start:wins[1].end] in text


def test_offset_to_page():
    assert [offset_to_page(o, [0, 100, 250]) for o in (0, 99, 100, 300)] == [1, 1, 2, 3]


def test_aggregate_insufficient_evidence():
    out = aggregate(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 40, Policy(0.8, "fpr_0.01"))
    assert out["status"] == "insufficient_evidence" and out["class_probabilities"] is None


def test_aggregate_weighted_and_thresholded():
    p4 = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.1, 0.7, 0.1]])
    pai = np.array([0.1, 0.95])
    pol = Policy(threshold=0.8, operating_point="fpr_0.01")
    out = aggregate(p4, pai, np.array([3.0, 1.0]), 500, pol)
    assert out["status"] == "complete"
    assert out["flagged_text_share"] == pytest.approx(0.25)
    assert out["ai_signal_score"] == pytest.approx(0.75 * 0.1 + 0.25 * 0.95)
    assert out["document_class"] == "human_only"
    out = aggregate(p4, pai, np.array([1.0, 9.0]), 500, pol)
    assert out["document_class"] == "ai_paraphrased_or_refined"


def test_aggregate_disagreeing_heads_are_uncertain():
    out = aggregate(np.array([[0.2, 0.6, 0.1, 0.1]]), np.array([0.3]), np.array([1.0]), 500, Policy(0.8, "x"))
    assert out["document_class"] == "uncertain" and out["reasons"]
