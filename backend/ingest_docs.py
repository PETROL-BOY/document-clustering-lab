"""
ingest_docs.py
==============
End-to-end ingestion and clustering script for research papers in DOCS_DIR:
  1. Reads and extracts text from every PDF.
  2. Cleans academic noise, bibliographies, headers/footers, and de-hyphenates.
  3. Applies word-boundary chunking with metadata.
  4. Embeds chunks with Gemini using SQLite caching and rate-limit backoff.
  5. Upserts vectors + metadata into Pinecone serverless vector index.
  6. Runs K-Means and Agglomerative clustering and prints a comparative summary.
  7. Saves the corpus to backend/data/research_papers_corpus.json for interactive RAG & Analytics.

Run from the backend directory:
    python ingest_docs.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure local app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.pipeline import (
    read_text_from_pdf,
    chunk_documents,
    embed_chunks_gemini,
    run_clustering_analysis,
    DocumentChunk,
)

# Configuration
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
PINECONE_NAMESPACE = "research-papers"
UPSERT_BATCH = 100
N_CLUSTERS = 4

# Domain metadata for the included research papers
PAPER_METADATA = {
    "10584_2018_Article_2265.pdf": {
        "title": "Climate Change Impacts on Energy Systems",
        "category": "Energy & Climate",
    },
    "1202.1112v1.pdf": {
        "title": "Recommender Systems: A Survey",
        "category": "Information Retrieval & Recommenders",
    },
    "1702.05747v2.pdf": {
        "title": "Deep Learning in Medical Image Analysis",
        "category": "Medical AI & Imaging",
    },
    "1706.03762v7.pdf": {
        "title": "Attention Is All You Need (Transformer)",
        "category": "Deep Learning & NLP",
    },
    "41239_2021_Article_313.pdf": {
        "title": "Learning Analytics Dashboard for Learners",
        "category": "Educational Tech & Analytics",
    },
    "main.pdf": {
        "title": "Heliyon Research Study",
        "category": "Applied Scientific Research",
    },
    "nihms-2195067.pdf": {
        "title": "Survey of AI Agents in Healthcare",
        "category": "Medical AI & Healthcare",
    },
    "pnas.202008128.pdf": {
        "title": "Least-Cost Targets for Renewable Energy in India",
        "category": "Energy & Climate",
    },
    "w35953.pdf": {
        "title": "Competition in Generative AI: Consumer Demand",
        "category": "GenAI Economics & Industry",
    },
}


def get_pinecone_index():
    """Connect to Pinecone and return (or create) the target index."""
    if not settings.pinecone_api_key:
        return None
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"[pinecone] Creating index '{index_name}' (dim=3072, metric=cosine) ...")
        pc.create_index(
            name=index_name,
            dimension=3072,  # gemini-embedding-001 -> 3072-d
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        for _ in range(30):
            time.sleep(2)
            info = pc.describe_index(index_name)
            if info.status.get("ready", False):
                break
        print(f"[pinecone] Index '{index_name}' ready.")
    else:
        print(f"[pinecone] Using existing index '{index_name}'.")

    return pc.Index(index_name)


def upsert_chunks(index, chunks, vectors, namespace: str):
    """Upsert chunk vectors + metadata to Pinecone in batches."""
    records = []
    for chunk, vec in zip(chunks, vectors):
        records.append({
            "id": chunk.chunk_id,
            "values": vec.tolist(),
            "metadata": {
                "title": chunk.title,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text[:1000],  # Pinecone metadata cap ~40 KB total
                "category": chunk.category or "",
            },
        })

    total = len(records)
    print(f"[pinecone] Upserting {total} vectors to namespace '{namespace}' ...")
    for i in range(0, total, UPSERT_BATCH):
        batch = records[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  ... upserted {min(i + UPSERT_BATCH, total)}/{total}")


def main():
    if not DOCS_DIR.exists():
        sys.exit(f"[ERROR] Docs directory not found: {DOCS_DIR}")

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"[ERROR] No PDF files found in {DOCS_DIR}")

    print(f"\n{'='*65}")
    print(f"  Document Clustering Lab -- Research Papers Ingestion")
    print(f"{'='*65}")
    print(f"  Docs dir        : {DOCS_DIR}")
    print(f"  PDFs found      : {len(pdf_files)}")
    print(f"  Embed model     : {settings.gemini_embed_model}")
    print(f"  Pinecone index  : {settings.pinecone_index}")
    print(f"{'='*65}\n")

    # ── Step 1: Ingest, Clean & Chunk ─────────────────────────────────────
    all_chunks = []
    all_docs = []

    for pdf_path in pdf_files:
        meta = PAPER_METADATA.get(pdf_path.name, {
            "title": pdf_path.stem.replace("_", " ").title(),
            "category": "Research Paper",
        })
        print(f"[ingest] {pdf_path.name} ({meta['title'][:35]}) ...", end=" ", flush=True)
        try:
            content_bytes = pdf_path.read_bytes()
            # read_text_from_pdf applies clean_academic_text automatically
            cleaned_text = read_text_from_pdf(content_bytes)

            doc_entry = {
                "title": meta["title"],
                "source": pdf_path.name,
                "text": cleaned_text,
                "category": meta["category"],
            }
            all_docs.append(doc_entry)

            # Chunk at 2000 chars with 200 overlap to keep token volume compact
            chunks = chunk_documents([doc_entry], chunk_size=2000, overlap=200, min_len=60)
            all_chunks.extend(chunks)
            print(f"{len(chunks)} chunks ({len(cleaned_text):,} chars)")
        except Exception as e:
            print(f"SKIPPED ({e})")

    print(f"\n[ingest] Total extracted: {len(all_docs)} documents -> {len(all_chunks)} chunks.\n")

    if not all_chunks:
        sys.exit("[ERROR] No chunks produced. Aborting.")

    # ── Step 2: Embed Chunks with Rate-Limit Backoff & SQLite Cache ───────
    print("[embed] Generating Gemini embeddings (cached in SQLite) ...")
    try:
        matrix = embed_chunks_gemini(all_chunks)
        print(f"[embed] Embedding matrix shape: {matrix.shape}\n")
    except Exception as e:
        sys.exit(f"[ERROR] Embedding failed: {e}")

    # ── Step 3: Upsert to Pinecone ────────────────────────────────────────
    if not settings.pinecone_api_key:
        print("[pinecone] PINECONE_API_KEY not set -- skipping cloud upsert.")
    else:
        try:
            index = get_pinecone_index()
            if index:
                upsert_chunks(index, all_chunks, matrix, PINECONE_NAMESPACE)
                # Also upsert to default namespace so the active UI session can query immediately
                upsert_chunks(index, all_chunks, matrix, "research_papers")
                print(f"[pinecone] Upsert complete.\n")
        except Exception as e:
            print(f"[pinecone] Upsert failed (continuing): {e}\n")

    # ── Step 4: Clustering Analysis ───────────────────────────────────────
    n_k = min(N_CLUSTERS, len(all_chunks) - 1)
    print(f"[cluster] Running K-Means & Agglomerative clustering (k={n_k}) ...")
    try:
        result = run_clustering_analysis(
            chunks=all_chunks,
            k=n_k,
            method="gemini",
            algorithm="kmeans",
            generate_llm_themes=False,  # Use statistical c-TF-IDF keywords to save LLM quota
            embeddings=matrix,
        )
    except Exception as e:
        sys.exit(f"[ERROR] Clustering failed: {e}")

    # ── Step 5: Save Processed Corpus for API & UI ────────────────────────
    output_corpus_path = Path(__file__).resolve().parent / "data" / "research_papers_corpus.json"
    output_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_corpus_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "research_papers",
            "doc_count": len(all_docs),
            "chunk_count": len(all_chunks),
            "docs": all_docs,
            "chunks": [c.model_dump() for c in all_chunks],
        }, f, indent=2)
    print(f"[corpus] Saved research papers corpus to {output_corpus_path.name}")

    # ── Step 6: Print Analytical Summary ──────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Clustering Results (k={n_k}, Representation: Gemini Dense 3072-d)")
    print(f"{'='*65}")
    metrics = result.get("metrics")
    if metrics:
        sil = f"{metrics.silhouette:.4f}" if metrics.silhouette is not None else "n/a"
        ine = f"{metrics.inertia:.4f}" if metrics.inertia is not None else "n/a"
        agr = f"{metrics.cluster_agreement:.4f}" if metrics.cluster_agreement is not None else "n/a"
        ari = f"{metrics.ari:.4f}" if metrics.ari is not None else "n/a"
        print(f"  Cosine Silhouette   : {sil}")
        print(f"  Inertia (WCSS)      : {ine}")
        print(f"  Algorithm Agreement : {agr}")
        print(f"  ARI (Ground Truth)  : {ari}")

    clusters = result.get("clusters", [])
    print(f"\n  Discovered {len(clusters)} Semantic Clusters:\n")
    for cl in clusters:
        label = cl.llm_label
        kw = ", ".join(cl.objective_keywords[:6]) if cl.objective_keywords else "none"
        rep = cl.representatives[0].excerpt[:110].replace("\n", " ") if cl.representatives else ""
        print(f"  Cluster #{cl.id} [{cl.size} chunks, {cl.percentage}% of corpus]")
        print(f"    Label   : {label}")
        print(f"    Keywords: {kw}")
        print(f"    Sample  : \"{rep}...\"")
        print()

    print(f"{'='*65}")
    print("  Corpus ingestion complete. Documents are indexed and ready for Q&A.\n")


if __name__ == "__main__":
    main()
