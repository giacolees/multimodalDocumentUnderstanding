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
