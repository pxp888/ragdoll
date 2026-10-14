import pytest
from unittest.mock import AsyncMock
from ragdoll.pipeline.chunker import (
    split_coarse_chunks,
    split_atomic_units,
    split_fine_chunks_with_kev,
)
from ragdoll.clients.kev import KevClient


def test_split_coarse_chunks_empty():
    assert split_coarse_chunks("") == []
    assert split_coarse_chunks("   \n\n  ") == []


def test_split_coarse_chunks_single_small():
    text = "Short text under 2000 characters."
    chunks = split_coarse_chunks(text, min_chars=100, max_chars=500)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["byte_offset_start"] == 0
    assert chunks[0]["byte_offset_end"] == len(text.encode("utf-8"))


def test_split_coarse_chunks_headers_and_paragraphs():
    p1 = "Paragraph 1 with sufficient content to create an initial section. " * 5
    p2 = "## Section 2 Header\n" + ("Paragraph 2 content that follows the header. " * 5)
    full_text = f"{p1}\n\n{p2}"

    chunks = split_coarse_chunks(full_text, min_chars=200, max_chars=400)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i
        assert c["byte_offset_start"] < c["byte_offset_end"]


def test_split_atomic_units():
    text = "First sentence. Second sentence! Third sentence? And line 2\nLine 3."
    units = split_atomic_units(text)
    assert len(units) > 0
    assert "".join(units) == text


@pytest.mark.asyncio
async def test_split_fine_chunks_with_mock_kev():
    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.decide_boundary.return_value = True

    sample_text = (
        "Sentence one discusses system architecture. " * 10
        + "Sentence two begins a completely different discussion about financials. " * 10
    )
    byte_len = len(sample_text.encode("utf-8"))

    coarse_chunk = {
        "id": 42,
        "chunk_index": 0,
        "text": sample_text,
        "byte_offset_start": 100,
        "byte_offset_end": 100 + byte_len,
    }

    fine_chunks = await split_fine_chunks_with_kev(
        coarse_chunk,
        mock_kev,
        min_chars=200,
        max_chars=400,
    )

    assert len(fine_chunks) >= 2
    for fc in fine_chunks:
        assert fc["parent_chunk_id"] == 42
        assert fc["byte_offset_start"] >= 100
        assert fc["byte_offset_end"] <= 100 + byte_len
        assert len(fc["text"]) > 0
