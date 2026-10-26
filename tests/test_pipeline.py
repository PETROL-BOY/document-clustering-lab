import pytest
import numpy as np
import pandas as pd
import io

from app.pipeline import (
    chunk_documents,
    ingest_file,
    sanitize_csv_cell,
    compute_c_tfidf,
    run_clustering_analysis,
    search_and_rag,
    read_text_from_pdf,
)
from app.schemas import DocumentChunk


# ---------------------------------------------------------------------------
# Ingestion & Security Tests
# ---------------------------------------------------------------------------

def test_csv_formula_injection_sanitization():
    assert sanitize_csv_cell("=1+2") == "'=1+2"
    assert sanitize_csv_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_csv_cell("-100") == "'-100"
    assert sanitize_csv_cell("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_csv_cell("Normal Text") == "Normal Text"
    assert sanitize_csv_cell("") == ""
    assert sanitize_csv_cell(None) == ""


def test_chunk_single_short_string():
    docs = [{"title": "Doc1", "source": "test.txt", "text": "This is a brief text for testing."}]
    chunks = chunk_documents(docs, chunk_size=1100, overlap=140, min_len=10)
    assert len(chunks) == 1
    assert chunks[0].title == "Doc1"
    assert chunks[0].chunk_index == 0


def test_chunk_overlap_and_boundary():
    # Long text with repeated sentences
    sentence = "Data mining discovers patterns in large text datasets. "
    long_text = sentence * 30  # ~1650 chars
    docs = [{"title": "Doc2", "source": "test.txt", "text": long_text}]
    chunks = chunk_documents(docs, chunk_size=1100, overlap=140, min_len=30)
    assert len(chunks) >= 2
    # Ensure IDs are distinct
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunk_id_determinism():
    docs = [{"title": "Doc", "source": "file.csv", "text": "Identical content across runs."}]
    c1 = chunk_documents(docs)
    c2 = chunk_documents(docs)
    assert c1[0].chunk_id == c2[0].chunk_id


def test_duplicate_chunk_deduplication():
    docs = [
        {"title": "Doc", "source": "file.csv", "text": "This is identical sample text content with more than thirty characters."},
        {"title": "Doc", "source": "file.csv", "text": "This is identical sample text content with more than thirty characters."}
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1


def test_csv_ingestion_with_alternate_columns():
    csv_content = b"headline,abstract,topic\nNews Title,This is the abstract body of news.,Technology\n"
    docs = ingest_file("sample.csv", csv_content)
    assert len(docs) == 1
    assert docs[0]["title"] == "News Title"
    assert "abstract body" in docs[0]["text"]
    assert docs[0]["category"] == "Technology"


def test_csv_encoding_fallbacks():
    # Test latin-1 encoded character like copyright sign
    csv_latin1 = "title,text\nDoc,Text with latin-1: \xa9 2026\n".encode("latin-1")
    docs = ingest_file("latin1.csv", csv_latin1)
    assert len(docs) == 1
    assert "2026" in docs[0]["text"]


def test_scanned_pdf_heuristic_rejection(monkeypatch):
    # Create fake PDF reader that extracts < 50 chars per page
    class MockPage:
        def extract_text(self):
            return "too short"

    class MockReader:
        def __init__(self, stream):
            self.pages = [MockPage(), MockPage()]

    monkeypatch.setattr("app.pipeline.PdfReader", MockReader)

    with pytest.raises(ValueError, match="scanned or image-based PDF"):
        read_text_from_pdf(b"%PDF-1.4 dummy content")


# ---------------------------------------------------------------------------
# Mathematical & Clustering Tests
# ---------------------------------------------------------------------------

def test_k_greater_than_sample_error():
    chunks = [
        DocumentChunk(chunk_id="c1", title="T1", source="s", chunk_index=0, text="Doc one content text."),
        DocumentChunk(chunk_id="c2", title="T2", source="s", chunk_index=0, text="Doc two content text.")
    ]
    with pytest.raises(ValueError, match="must be strictly smaller"):
        run_clustering_analysis(chunks, k=2, method="tfidf")


def test_c_tfidf_score_computation():
    chunks = [
        DocumentChunk(chunk_id="c1", title="T1", source="s", chunk_index=0, text="solar energy photovoltaic array power"),
        DocumentChunk(chunk_id="c2", title="T2", source="s", chunk_index=0, text="solar panels renewable power generation"),
        DocumentChunk(chunk_id="c3", title="T3", source="s", chunk_index=0, text="phishing email credential attack vector"),
        DocumentChunk(chunk_id="c4", title="T4", source="s", chunk_index=0, text="phishing scam cybersecurity malicious link")
    ]
    labels = np.array([0, 0, 1, 1])
    keywords = compute_c_tfidf(chunks, labels, top_n=3)

    assert 0 in keywords and 1 in keywords
    # Cluster 0 should contain solar / energy
    assert any("solar" in kw or "power" in kw for kw in keywords[0])
    # Cluster 1 should contain phishing
    assert any("phishing" in kw for kw in keywords[1])


def test_kmeans_representative_chunks_selection():
    # Provide two clearly separated semantic topics: solar vs ransomware
    chunks = [
        DocumentChunk(
            chunk_id=f"solar_{i}",
            title=f"Solar Doc {i}",
            source="s",
            chunk_index=0,
            text=f"Photovoltaic solar array renewable energy generation topic {i} sunlight megawatts"
        )
        for i in range(4)
    ] + [
        DocumentChunk(
            chunk_id=f"sec_{i}",
            title=f"Security Doc {i}",
            source="s",
            chunk_index=0,
            text=f"Ransomware phishing cybersecurity malware intrusion topic {i} vulnerability payload"
        )
        for i in range(4)
    ]
    analysis = run_clustering_analysis(chunks, k=2, method="tfidf", generate_llm_themes=False)

    for cluster in analysis["clusters"]:
        assert len(cluster.representatives) > 0
        # Check representatives are sorted by distance ascending
        dists = [r.distance_to_centroid for r in cluster.representatives]
        assert dists == sorted(dists)


def test_similarity_threshold_rejection():
    # If no similarity >= 0.65, search_and_rag returns explicit message
    chunks = [DocumentChunk(chunk_id="c1", title="Solar", source="s", chunk_index=0, text="Solar energy panels")]
    embeddings = np.array([[1.0, 0.0]])  # Dimension 2

    # Mock embed_query_gemini to return orthogonal vector
    def mock_embed_query(q):
        return np.array([0.0, 1.0])

    import app.pipeline as pl
    pl.embed_query_gemini = mock_embed_query

    result = search_and_rag("medical question", chunks, embeddings)
    assert result["threshold_passed"] is False
    assert "not contain sufficient evidence" in result["answer"].lower()
