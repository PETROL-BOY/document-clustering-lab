"""
ingest_docs.py
==============
One-shot script that:
  1. Reads every PDF from DOCS_DIR
  2. Chunks each PDF with the project's pipeline
  3. Embeds chunks with Gemini (with rate-limit retry)
  4. Upserts vectors + metadata to Pinecone
  5. Runs KMeans + Agglomerative clustering and prints a summary

Run from the backend directory:
    python ingest_docs.py
"""

import os
import sys
import time
from pathlib import Path

# ── Make sure local app package is importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.pipeline import (
    ingest_file,
    chunk_documents,
    embed_chunks_gemini,
    run_clustering_analysis,
)

# ── Config ─────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
PINECONE_NAMESPACE = "research-papers"
EMBED_BATCH_SLEEP = 6   # seconds between embedding batches (free-tier safety)
UPSERT_BATCH = 100      # Pinecone vectors per upsert call
N_CLUSTERS = 4          # how many clusters to request


def get_pinecone_index():
    """Connect to Pinecone and return (or create) the target index."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"[pinecone] Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=768,          # gemini-embedding-001 / text-embedding-004 -> 768-d
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait until ready
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
                "text": chunk.text[:1000],   # Pinecone metadata cap ~40 KB total
                "category": chunk.category or "",
            },
        })

    total = len(records)
    print(f"[pinecone] Upserting {total} vectors ...")
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

    print(f"\n{'='*60}")
    print(f"  Document Clustering Lab -- Ingestion Script")
    print(f"{'='*60}")
    print(f"  Docs dir   : {DOCS_DIR}")
    print(f"  PDFs found : {len(pdf_files)}")
    print(f"  Embed model: {settings.gemini_embed_model}")
    print(f"  Pinecone index: {settings.pinecone_index}")
    print(f"{'='*60}\n")

    # ── Step 1: Ingest & Chunk ────────────────────────────────────────────
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"[ingest] {pdf_path.name} ...", end=" ", flush=True)
        try:
            content_bytes = pdf_path.read_bytes()
            docs = ingest_file(pdf_path.name, content_bytes)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            print(f"{len(chunks)} chunks")
        except Exception as e:
            print(f"SKIPPED ({e})")

    print(f"\n[ingest] Total chunks: {len(all_chunks)}\n")

    if not all_chunks:
        sys.exit("[ERROR] No chunks produced. Aborting.")

    # ── Step 2: Embed ─────────────────────────────────────────────────────
    print("[embed] Generating Gemini embeddings (with rate-limit retry) ...")
    try:
        matrix = embed_chunks_gemini(all_chunks)
        print(f"[embed] Embedding matrix shape: {matrix.shape}\n")
    except Exception as e:
        sys.exit(f"[ERROR] Embedding failed: {e}")

    # ── Step 3: Upsert to Pinecone ────────────────────────────────────────
    if not settings.pinecone_api_key:
        print("[pinecone] PINECONE_API_KEY not set -- skipping upsert.")
    else:
        try:
            index = get_pinecone_index()
            upsert_chunks(index, all_chunks, matrix, PINECONE_NAMESPACE)
            print(f"\n[pinecone] Upsert complete.\n")
        except Exception as e:
            print(f"[pinecone] Upsert failed (continuing to cluster): {e}\n")

    # ── Step 4: Cluster ───────────────────────────────────────────────────
    n_k = min(N_CLUSTERS, len(all_chunks) - 1)
    print(f"[cluster] Running KMeans + Agglomerative with k={n_k} ...")
    try:
        result = run_clustering_analysis(
            chunks=all_chunks,
            embeddings=matrix,
            n_clusters=n_k,
            use_gemini=False,   # c-TF-IDF themes; avoids burning gen-AI quota
        )
    except Exception as e:
        sys.exit(f"[ERROR] Clustering failed: {e}")

    # ── Step 5: Print Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Clustering Results  (method: {result.method})")
    print(f"{'='*60}")
    if result.metrics:
        m = result.metrics
        sil = f"{m.silhouette_score:.4f}" if m.silhouette_score is not None else "n/a"
        db  = f"{m.davies_bouldin_index:.4f}" if m.davies_bouldin_index is not None else "n/a"
        print(f"  Silhouette     : {sil}")
        print(f"  Davies-Bouldin : {db}")
        if m.adjusted_rand_index is not None:
            print(f"  ARI            : {m.adjusted_rand_index:.4f}")

    print(f"\n  {len(result.clusters)} Clusters:\n")
    for cl in result.clusters:
        theme = cl.theme.label if cl.theme else "--"
        kw    = ", ".join(cl.theme.keywords[:6]) if cl.theme else ""
        rep   = cl.representative_chunks[0].text[:120].replace("\n", " ") if cl.representative_chunks else ""
        print(f"  Cluster {cl.cluster_id}  [{cl.size} chunks]")
        print(f"    Theme   : {theme}")
        print(f"    Keywords: {kw}")
        print(f"    Sample  : {rep}...")
        print()

    print(f"{'='*60}")
    print("  Done. The corpus is ready for RAG queries.\n")


if __name__ == "__main__":
    main()
