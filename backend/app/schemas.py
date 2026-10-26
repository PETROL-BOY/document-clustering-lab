from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Schema for structured Gemini generation output
class ClusterThemeSchema(BaseModel):
    label: str = Field(description="A concise, descriptive academic title for this semantic cluster (3 to 6 words)")
    summary: str = Field(description="A 1-2 sentence coherent summary synthesizing the shared thematic topic of these excerpts")
    keywords: List[str] = Field(description="List of 3 to 5 key domain-specific terms characterizing this cluster")

# Chunk representation
class DocumentChunk(BaseModel):
    chunk_id: str
    title: str
    source: str
    chunk_index: int
    text: str
    category: Optional[str] = None

# Representative chunk excerpt
class RepresentativeChunk(BaseModel):
    chunk_id: str
    title: str
    source: str
    chunk_index: int
    excerpt: str
    distance_to_centroid: float
    full_text: Optional[str] = None

# Cluster result in analytics
class ClusterResult(BaseModel):
    id: int
    size: int
    percentage: float
    objective_keywords: List[str] = Field(default_factory=list, description="Top c-TF-IDF terms")
    llm_label: str
    llm_summary: str
    llm_keywords: List[str] = Field(default_factory=list)
    representatives: List[RepresentativeChunk] = Field(default_factory=list)

# 2D PCA point
class PCAPoint(BaseModel):
    x: float
    y: float
    cluster_id: int
    title: str
    source: str
    preview: str

# Evaluation metrics
class EvaluationMetrics(BaseModel):
    silhouette: Optional[float] = None
    inertia: Optional[float] = None
    ari: Optional[float] = None
    nmi: Optional[float] = None
    cluster_agreement: Optional[float] = Field(default=None, description="Agreement between K-Means and Agglomerative (ARI)")

# Analysis Request
class AnalyzeRequest(BaseModel):
    k: int = Field(default=4, ge=2, le=20)
    method: str = Field(default="gemini", description="'gemini' or 'tfidf'")
    algorithm: str = Field(default="kmeans", description="'kmeans' or 'agglomerative'")

# Sweep Request
class SweepRequest(BaseModel):
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=6, ge=2)
    method: str = Field(default="gemini", description="'gemini' or 'tfidf'")

# Sweep Point
class SweepPoint(BaseModel):
    k: int
    inertia: Optional[float] = None
    silhouette: Optional[float] = None
    sizes: List[int]

# Full Analytics Response
class AnalyticsResponse(BaseModel):
    run_id: str
    timestamp: str
    dataset_name: str
    doc_count: int
    chunk_count: int
    method: str
    algorithm: str
    k: int
    metrics: EvaluationMetrics
    clusters: List[ClusterResult]
    pca_points: List[PCAPoint]
    sweep_data: Optional[List[SweepPoint]] = None

# RAG Chat
class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)

class SourceCitation(BaseModel):
    chunk_id: str
    title: str
    source: str
    chunk_index: int
    score: float
    excerpt: str
    full_text: Optional[str] = None
    category: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    threshold_passed: bool
    citations: List[SourceCitation]
    latency_ms: Optional[float] = None

# Observability schemas
class RAGLogEntry(BaseModel):
    id: str
    timestamp: str
    question: str
    status: str = Field(description="'success', 'insufficient_evidence', or 'error'")
    citations_count: int
    similarity_scores: List[float] = Field(default_factory=list)
    latency_ms: float
    error: Optional[str] = None

class SystemOverviewResponse(BaseModel):
    total_docs: int
    total_chunks: int
    total_clusters: int
    total_questions: int
    successful_queries: int
    insufficient_evidence_queries: int
    error_queries: int
    evidence_pass_rate_pct: float
    avg_rag_latency_ms: float
    avg_rejected_latency_ms: float

