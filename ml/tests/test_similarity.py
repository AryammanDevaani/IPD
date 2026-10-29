from scriptsense_ml.similarity.fingerprints import align_exact, ngram_hashes, tokenize, winnow

SRC = ("The industrial revolution transformed rural economies as factories drew workers into rapidly "
       "growing cities, where housing shortages and poor sanitation created new public health crises.")


def test_tokenize_keeps_offsets():
    toks = tokenize("Hello, World!")
    assert [t.norm for t in toks] == ["hello", "world"]
    assert "Hello, World!"[toks[1].start:toks[1].end] == "World"


def test_winnow_guarantee_shared_run_shares_a_fingerprint():
    a = tokenize("intro words here " + SRC + " outro")
    b = tokenize("different start " + SRC)
    fa = {h for h, _ in winnow(ngram_hashes(a, 5), 4)}
    fb = {h for h, _ in winnow(ngram_hashes(b, 5), 4)}
    assert fa & fb


def test_align_exact_finds_copied_passage_with_offsets_case_insensitive():
    sub = "My essay begins here. " + SRC.upper() + " Then my own conclusion follows."
    m = align_exact(tokenize(sub), tokenize(SRC), n=5, min_words=8)
    assert len(m) == 1
    assert m[0].n_words == len(tokenize(SRC))
    assert sub[m[0].sub_start:m[0].sub_end].lower() == SRC.rstrip(".").lower()
    assert SRC[m[0].src_start:m[0].src_end] == SRC.rstrip(".")


def test_align_exact_ignores_short_common_phrases():
    sub = "on the other hand the results were mixed"
    src = "on the other hand the weather was fine"
    assert align_exact(tokenize(sub), tokenize(src), n=5, min_words=8) == []


def test_bm25_ranks_relevant_passage_first():
    from scriptsense_ml.similarity.lexical_search import BM25Index

    idx = BM25Index()
    idx.add("src:1", SRC)
    idx.add("src:2", "Glaciers retreat as global temperatures rise, exposing bedrock and meltwater lakes.")
    idx.add("src:3", "Recipes for sourdough bread require a starter culture and long fermentation.")
    hits = idx.search("factories drew rural workers into cities with poor sanitation", k=2)
    assert hits[0][0] == "src:1" and len(hits) == 1
