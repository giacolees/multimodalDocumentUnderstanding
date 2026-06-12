import unittest.mock as mock


def test_chunks_respects_max_chars():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=50)
    text = "Hello world.\n\nThis is a second paragraph that is definitely longer than fifty chars."
    result = r.chunks(text)
    for c in result:
        assert len(c) <= 50, f"Chunk too long ({len(c)}): {c!r}"


def test_chunks_preserves_content():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=100)
    text = "Line one.\nLine two.\nLine three."
    chunks = r.chunks(text)
    combined = " ".join(chunks)
    assert "Line one" in combined
    assert "Line two" in combined
    assert "Line three" in combined


def test_chunks_empty_text_returns_something():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=50)
    result = r.chunks("   ")
    assert isinstance(result, list)


def test_transcribe_calls_model_generate(tmp_path):
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(cache_dir=str(tmp_path))
    item = {"document_path": "data/raw/doc.png", "page_index": 0}
    mock_model = mock.MagicMock()
    mock_model.generate.return_value = "The year is 1975. Table 1 shows values."

    text = r.transcribe(item, mock_model)

    assert text == "The year is 1975. Table 1 shows values."
    mock_model.generate.assert_called_once()
    call_args = mock_model.generate.call_args
    assert call_args[0][0] == "data/raw/doc.png"


def test_transcribe_cache_hit_skips_generate(tmp_path):
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(cache_dir=str(tmp_path))
    item = {"document_path": "data/raw/doc.png", "page_index": 0}
    mock_model = mock.MagicMock()
    mock_model.generate.return_value = "Cached text."

    r.transcribe(item, mock_model)   # first call writes cache
    r.transcribe(item, mock_model)   # second call should read cache

    assert mock_model.generate.call_count == 1


def test_retrieve_rrf_ranks_relevant_chunk_first():
    """RRF fuses BM25 + dense; the chunk winning both signals appears first."""
    import sys
    import numpy as np

    # Mock rank_bm25 at the module level so the lazy import inside retrieve() uses it.
    mock_bm25_module = mock.MagicMock()
    mock_bm25_instance = mock.MagicMock()
    mock_bm25_module.BM25Okapi.return_value = mock_bm25_instance
    # chunk 0 = relevant, wins sparse
    mock_bm25_instance.get_scores.return_value = np.array([10.0, 0.1])

    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(top_k=1, chunk_max_chars=200)

    # Pre-set embedder to avoid sentence_transformers import
    mock_embedder = mock.MagicMock()
    # chunk 0 embedding is most similar to query (dot product: [0.9, 0.1])
    mock_embedder.encode.side_effect = [
        np.array([[0.9, 0.1], [0.1, 0.9]]),  # chunk embeddings (2 chunks)
        np.array([[1.0, 0.0]]),               # query embedding — chunk 0 wins dense too
    ]
    r._embedder = mock_embedder

    item = {"document_path": "doc.png", "page_index": 0}
    relevant = "net profit for 1975 was high"
    irrelevant = "company founded in 1960"
    full_text = f"{relevant}\n\n{irrelevant}"

    with mock.patch.object(r, "transcribe", return_value=full_text), \
         mock.patch.dict(sys.modules, {"rank_bm25": mock_bm25_module}):
        chunks = r.retrieve(item, "net profit 1975", mock.MagicMock())

    assert len(chunks) == 1
    assert chunks[0] == relevant


def test_retrieve_top_k_respected():
    """retrieve() returns at most top_k chunks."""
    import sys
    import numpy as np

    mock_bm25_module = mock.MagicMock()
    mock_bm25_instance = mock.MagicMock()
    mock_bm25_module.BM25Okapi.return_value = mock_bm25_instance
    mock_bm25_instance.get_scores.return_value = np.array([3.0, 2.0, 1.0])

    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(top_k=2, chunk_max_chars=30)
    mock_embedder = mock.MagicMock()
    mock_embedder.encode.side_effect = [
        np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
    ]
    r._embedder = mock_embedder

    text = "Chunk A text here.\n\nChunk B text here.\n\nChunk C text here."
    item = {"document_path": "doc.png", "page_index": 0}
    with mock.patch.object(r, "transcribe", return_value=text), \
         mock.patch.dict(sys.modules, {"rank_bm25": mock_bm25_module}):
        chunks = r.retrieve(item, "query", mock.MagicMock())

    assert len(chunks) == 2


def test_rag_strategy_prompt_contains_context_and_question_placeholder():
    from src.mitigation.strategies.rag import RagStrategy
    strategy = RagStrategy({})
    item = {"document_path": "doc.png", "page_index": 0, "corrupted_question": "What year?"}
    mock_model = mock.MagicMock()

    with mock.patch.object(strategy.retriever, "retrieve",
                           return_value=["Relevant passage about 1975."]):
        prompt = strategy.build_prompt(item, mock_model)

    assert "Relevant passage about 1975." in prompt
    assert "{question}" in prompt          # placeholder left for VllmModel.predict_unanswerable
    assert "UNANSWERABLE" in prompt


def test_rag_strategy_empty_retrieval_still_returns_prompt():
    from src.mitigation.strategies.rag import RagStrategy
    strategy = RagStrategy({})
    item = {"document_path": "doc.png", "page_index": 0, "corrupted_question": "Anything?"}
    mock_model = mock.MagicMock()

    with mock.patch.object(strategy.retriever, "retrieve", return_value=[]):
        prompt = strategy.build_prompt(item, mock_model)

    assert "{question}" in prompt
