import csv
import io
import json
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.pipeline import (
    ingest_file,
    chunk_documents,
    run_clustering_analysis,
    run_k_sweep_pipeline,
    search_and_rag,
    upsert_to_pinecone,
    sanitize_csv_cell,
)
from app.schemas import (
    AnalyzeRequest,
    SweepRequest,
    ChatRequest,
    AnalyticsResponse,
    ChatResponse,
    DocumentChunk,
    RAGLogEntry,
    SystemOverviewResponse,
)

app = FastAPI(
    title="Document Intelligence API",
    description="Automated Document Clustering, Thematic Analysis and Grounded RAG Engine",
    version="2.0.0"
)

# Narrow CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory server session state
session_state: Dict[str, Any] = {
    "run_id": "initial",
    "timestamp": datetime.now().isoformat(),
    "dataset_name": "sample_documents.csv",
    "chunks": [],
    "embeddings": None,
    "last_analysis": None,
    "sweep_data": None
}


def load_default_corpus():
    """Load research papers if available, else load bundled sample_documents.csv."""
    papers_path = Path(__file__).resolve().parent.parent / "data" / "research_papers_corpus.json"
    if papers_path.exists():
        try:
            with open(papers_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [DocumentChunk(**c) for c in data["chunks"]]
            session_state["docs"] = data["docs"]
            session_state["chunks"] = chunks
            session_state["dataset_name"] = data.get("dataset_name", "research_papers")
            return
        except Exception as e:
            print(f"Notice: Could not load research papers corpus: {e}")

    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_documents.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            content = f.read()
        docs = ingest_file("sample_documents.csv", content)
        chunks = chunk_documents(docs)
        session_state["docs"] = docs
        session_state["chunks"] = chunks
        session_state["dataset_name"] = "sample_documents.csv"


# Initialize with default corpus
try:
    load_default_corpus()
except Exception as e:
    print(f"Warning: Failed to load default corpus: {e}")


@app.post("/api/corpus/select")
def select_active_corpus(corpus_name: str = Query(..., description="'sample' or 'research_papers'")):
    """Switch active corpus between sample documents and research papers."""
    if corpus_name in ("research_papers", "papers"):
        papers_path = Path(__file__).resolve().parent.parent / "data" / "research_papers_corpus.json"
        if not papers_path.exists():
            raise HTTPException(status_code=404, detail="Research papers corpus not found. Please run ingest_docs.py first.")
        with open(papers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [DocumentChunk(**c) for c in data["chunks"]]
        session_state["docs"] = data["docs"]
        session_state["chunks"] = chunks
        session_state["dataset_name"] = "research_papers"
        session_state["embeddings"] = None
        session_state["last_analysis"] = None
        session_state["sweep_data"] = None
        return {"status": "success", "dataset_name": "research_papers", "chunk_count": len(chunks)}
    else:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_documents.csv"
        with open(sample_path, "rb") as f:
            content = f.read()
        docs = ingest_file("sample_documents.csv", content)
        chunks = chunk_documents(docs)
        session_state["docs"] = docs
        session_state["chunks"] = chunks
        session_state["dataset_name"] = "sample_documents.csv"
        session_state["embeddings"] = None
        session_state["last_analysis"] = None
        session_state["sweep_data"] = None
        return {"status": "success", "dataset_name": "sample_documents.csv", "chunk_count": len(chunks)}


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """Check health and provider connectivity."""
    return {
        "status": "healthy",
        "gemini_configured": bool(settings.gemini_api_key),
        "pinecone_configured": bool(settings.pinecone_api_key),
        "embed_model": settings.gemini_embed_model,
        "gen_model": settings.gemini_gen_model,
        "corpus_loaded": len(session_state["chunks"]) > 0,
        "chunk_count": len(session_state["chunks"]),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document file (.csv, .txt, .md, .pdf)."""
    valid_exts = (".csv", ".txt", ".md", ".pdf")
    if not file.filename.lower().endswith(valid_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Supported extensions are: {', '.join(valid_exts)}"
        )

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 15 MB.")

    try:
        docs = ingest_file(file.filename, content)
        chunks = chunk_documents(docs)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    session_state["docs"] = docs
    session_state["chunks"] = chunks
    session_state["dataset_name"] = file.filename
    session_state["embeddings"] = None
    session_state["last_analysis"] = None
    session_state["sweep_data"] = None

    has_gt = any(c.category is not None for c in chunks)

    return {
        "status": "success",
        "dataset_name": file.filename,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "has_ground_truth": has_gt
    }


@app.post("/api/analyze", response_model=AnalyticsResponse)
def analyze_corpus(req: AnalyzeRequest):
    """Run clustering, c-TF-IDF keyword extraction, LLM themes, and PCA."""
    chunks = session_state.get("chunks", [])
    if not chunks:
        load_default_corpus()
        chunks = session_state.get("chunks", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="Corpus is empty. Please upload a document first.")

    if req.k >= len(chunks):
        raise HTTPException(
            status_code=400,
            detail=f"K ({req.k}) must be smaller than the number of chunks ({len(chunks)})."
        )

    try:
        analysis = run_clustering_analysis(
            chunks=chunks,
            k=req.k,
            method=req.method,
            algorithm=req.algorithm,
            generate_llm_themes=True
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(exc)}")

    run_id = f"run_{int(time.time())}"
    timestamp = datetime.now().isoformat()

    session_state["run_id"] = run_id
    session_state["timestamp"] = timestamp
    session_state["embeddings"] = analysis["X"]
    session_state["last_analysis"] = analysis
    session_state["last_method"] = req.method
    session_state["last_algorithm"] = req.algorithm
    session_state["last_k"] = req.k

    # Upsert to Pinecone if using Gemini embeddings and Pinecone key configured
    if req.method == "gemini" and settings.pinecone_api_key:
        try:
            upsert_to_pinecone(chunks, analysis["X"], analysis["labels"], namespace=session_state["dataset_name"])
        except Exception as e:
            print(f"Pinecone background upsert notice: {e}")

    return AnalyticsResponse(
        run_id=run_id,
        timestamp=timestamp,
        dataset_name=session_state["dataset_name"],
        doc_count=len(set(c.source for c in chunks)),
        chunk_count=len(chunks),
        method=req.method,
        algorithm=req.algorithm,
        k=req.k,
        metrics=analysis["metrics"],
        clusters=analysis["clusters"],
        pca_points=analysis["pca_points"],
        sweep_data=session_state.get("sweep_data")
    )


@app.post("/api/sweep")
def sweep_k_parameters(req: SweepRequest):
    """Run parameter sweep over candidate K values on cached vectors."""
    chunks = session_state.get("chunks", [])
    if not chunks:
        load_default_corpus()
        chunks = session_state.get("chunks", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="Corpus is empty.")

    try:
        sweep_results = run_k_sweep_pipeline(
            chunks=chunks,
            k_min=req.k_min,
            k_max=req.k_max,
            method=req.method
        )
        session_state["sweep_data"] = sweep_results
        return sweep_results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"K sweep failed: {str(exc)}")


@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics():
    """Retrieve the current active run analytics, or compute default analysis if none exists."""
    chunks = session_state.get("chunks", [])
    if not chunks:
        load_default_corpus()
        chunks = session_state.get("chunks", [])

    if session_state.get("last_analysis") is None:
        # Run default K=4 analysis
        default_req = AnalyzeRequest(k=4, method="gemini" if settings.gemini_api_key else "tfidf", algorithm="kmeans")
        return analyze_corpus(default_req)

    analysis = session_state["last_analysis"]
    return AnalyticsResponse(
        run_id=session_state.get("run_id", "default"),
        timestamp=session_state.get("timestamp", datetime.now().isoformat()),
        dataset_name=session_state["dataset_name"],
        doc_count=len(set(c.source for c in chunks)),
        chunk_count=len(chunks),
        method=session_state.get("last_method", "gemini"),
        algorithm=session_state.get("last_algorithm", "kmeans"),
        k=session_state.get("last_k", 4),
        metrics=analysis["metrics"],
        clusters=analysis["clusters"],
        pca_points=analysis["pca_points"],
        sweep_data=session_state.get("sweep_data")
    )


# In-memory query logging & observability
log_lock = threading.Lock()
rag_logs: deque = deque(maxlen=100)

query_metrics = {
    "total_questions": 0,
    "successful_queries": 0,
    "insufficient_evidence_queries": 0,
    "error_queries": 0,
    "total_rag_latency_ms": 0.0,
    "total_rejected_latency_ms": 0.0,
    "rag_count": 0,
    "rejected_count": 0,
}


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_corpus(req: ChatRequest):
    """Answer question with source attribution, threshold validation, and latency logging."""
    chunks = session_state.get("chunks", [])
    embeddings = session_state.get("embeddings")

    if not chunks or embeddings is None:
        # Run default analysis first so vectors exist with safe k
        safe_k = min(4, max(2, len(chunks) - 1)) if len(chunks) > 2 else 2
        default_req = AnalyzeRequest(k=safe_k, method="gemini" if settings.gemini_api_key else "tfidf", algorithm="kmeans")
        analyze_corpus(default_req)
        chunks = session_state.get("chunks", [])
        embeddings = session_state.get("embeddings")

    t_start = time.perf_counter()
    try:
        result = search_and_rag(
            question=req.question,
            chunks=chunks,
            embeddings=embeddings,
            top_k=req.top_k
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Thread-safe log recording
        with log_lock:
            query_metrics["total_questions"] += 1
            if result["threshold_passed"]:
                query_metrics["successful_queries"] += 1
                query_metrics["rag_count"] += 1
                query_metrics["total_rag_latency_ms"] += elapsed_ms
                status = "success"
            else:
                query_metrics["insufficient_evidence_queries"] += 1
                query_metrics["rejected_count"] += 1
                query_metrics["total_rejected_latency_ms"] += elapsed_ms
                status = "insufficient_evidence"

            rag_logs.appendleft({
                "id": f"log_{query_metrics['total_questions']}",
                "timestamp": datetime.now().isoformat(),
                "question": req.question,
                "status": status,
                "citations_count": len(result.get("citations", [])),
                "similarity_scores": [c.score for c in result.get("citations", [])],
                "latency_ms": elapsed_ms,
                "error": None
            })

        return ChatResponse(
            question=req.question,
            answer=result["answer"],
            threshold_passed=result["threshold_passed"],
            citations=result["citations"],
            latency_ms=elapsed_ms
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        with log_lock:
            query_metrics["total_questions"] += 1
            query_metrics["error_queries"] += 1
            rag_logs.appendleft({
                "id": f"log_{query_metrics['total_questions']}",
                "timestamp": datetime.now().isoformat(),
                "question": req.question,
                "status": "error",
                "citations_count": 0,
                "similarity_scores": [],
                "latency_ms": elapsed_ms,
                "error": str(exc)
            })
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(exc)}")


@app.get("/api/analytics/overview", response_model=SystemOverviewResponse)
def get_system_overview():
    """Return aggregated system performance and query observability metrics."""
    with log_lock:
        total_q = query_metrics["total_questions"]
        succ = query_metrics["successful_queries"]
        insuff = query_metrics["insufficient_evidence_queries"]
        err = query_metrics["error_queries"]
        rag_c = query_metrics["rag_count"]
        rej_c = query_metrics["rejected_count"]
        pass_rate = round((succ / total_q) * 100, 1) if total_q > 0 else 100.0
        avg_rag = round(query_metrics["total_rag_latency_ms"] / max(1, rag_c), 1)
        avg_rej = round(query_metrics["total_rejected_latency_ms"] / max(1, rej_c), 1)

    chunks = session_state.get("chunks", [])
    last_analysis = session_state.get("last_analysis") or {}
    clusters = last_analysis.get("clusters", [])
    k_val = len(clusters) if clusters else 4

    return SystemOverviewResponse(
        total_docs=len(set(c.source for c in chunks)),
        total_chunks=len(chunks),
        total_clusters=k_val,
        total_questions=total_q,
        successful_queries=succ,
        insufficient_evidence_queries=insuff,
        error_queries=err,
        evidence_pass_rate_pct=pass_rate,
        avg_rag_latency_ms=avg_rag,
        avg_rejected_latency_ms=avg_rej
    )


@app.get("/api/analytics/logs", response_model=List[RAGLogEntry])
def get_rag_logs(
    search: Optional[str] = Query(None, description="Search keyword in question"),
    status: Optional[str] = Query(None, description="Filter by status: success, insufficient_evidence, error"),
    limit: int = Query(50, ge=1, le=100)
):
    """Retrieve filterable and searchable question-answering request logs."""
    with log_lock:
        items = list(rag_logs)

    if status and status.lower() != "all":
        items = [x for x in items if x["status"].lower() == status.lower()]

    if search and search.strip():
        s_lower = search.strip().lower()
        items = [x for x in items if s_lower in x["question"].lower()]

    return [RAGLogEntry(**x) for x in items[:limit]]


@app.get("/api/export/csv")
def export_csv():
    """Export labeled document chunks with formula injection sanitization."""
    analysis = session_state.get("last_analysis")
    chunks = session_state.get("chunks", [])

    if not analysis or not chunks:
        raise HTTPException(status_code=400, detail="No active analysis to export.")

    labels = analysis["labels"]
    clusters_dict = {c.id: c for c in analysis["clusters"]}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "chunk_id",
        "title",
        "source",
        "chunk_index",
        "cluster_id",
        "cluster_label",
        "objective_keywords",
        "text_preview"
    ])

    for ch, lbl in zip(chunks, labels):
        c_info = clusters_dict.get(int(lbl))
        lbl_str = c_info.llm_label if c_info else f"Cluster {lbl}"
        obj_kw_str = ", ".join(c_info.objective_keywords) if c_info else ""
        writer.writerow([
            sanitize_csv_cell(ch.chunk_id),
            sanitize_csv_cell(ch.title),
            sanitize_csv_cell(ch.source),
            sanitize_csv_cell(ch.chunk_index),
            sanitize_csv_cell(int(lbl)),
            sanitize_csv_cell(lbl_str),
            sanitize_csv_cell(obj_kw_str),
            sanitize_csv_cell(ch.text[:200])
        ])

    csv_data = output.getvalue().encode("utf-8-sig")
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=clustered_chunks_{session_state['run_id']}.csv"}
    )


@app.get("/api/export/json")
def export_json():
    """Export complete analytic report as structured JSON."""
    analysis = session_state.get("last_analysis")
    if not analysis:
        raise HTTPException(status_code=400, detail="No active analysis to export.")

    analytics_obj = get_analytics()
    return JSONResponse(
        content=analytics_obj.model_dump(),
        headers={"Content-Disposition": f"attachment; filename=clustering_report_{session_state['run_id']}.json"}
    )


@app.get("/api/documents/{chunk_id}")
def get_document_details(chunk_id: str):
    """Retrieve full document and chunk context by chunk ID, title, or index."""
    chunks = session_state.get("chunks", [])
    if not chunks:
        load_default_corpus()
        chunks = session_state.get("chunks", [])

    target_chunk = None
    # 1. Exact match by chunk_id
    for c in chunks:
        if c.chunk_id == chunk_id:
            target_chunk = c
            break

    # 2. Case-insensitive title match if not found by chunk_id
    if not target_chunk:
        for c in chunks:
            if c.title.strip().lower() == chunk_id.strip().lower():
                target_chunk = c
                break

    # 3. Numeric index match
    if not target_chunk and chunk_id.isdigit():
        idx = int(chunk_id)
        if 0 <= idx < len(chunks):
            target_chunk = chunks[idx]

    if not target_chunk:
        raise HTTPException(status_code=404, detail=f"Document with identifier '{chunk_id}' not found.")

    # Find all chunks from this document
    matching_chunks = [
        c for c in chunks
        if c.title == target_chunk.title and c.source == target_chunk.source
    ]
    matching_chunks.sort(key=lambda x: x.chunk_index)

    # Check if raw docs has the full original text
    raw_docs = session_state.get("docs", [])
    raw_text = None
    for d in raw_docs:
        if d.get("title") == target_chunk.title and d.get("source") == target_chunk.source:
            raw_text = d.get("text")
            break

    full_text = raw_text if raw_text else "\n\n".join(c.text for c in matching_chunks)

    return {
        "chunk_id": target_chunk.chunk_id,
        "title": target_chunk.title,
        "source": target_chunk.source,
        "chunk_index": target_chunk.chunk_index,
        "total_chunks": len(matching_chunks),
        "category": target_chunk.category,
        "chunk_text": target_chunk.text,
        "full_text": full_text,
        "all_chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "length": len(c.text)
            }
            for c in matching_chunks
        ]
    }

