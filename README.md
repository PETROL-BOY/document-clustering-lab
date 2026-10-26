# Document Cluster Lab

> **Academic Project:** Automated Document Clustering and Thematic Analysis  
> **Course:** Data Warehousing and Mining (DWM) Mini-Project  
> **Stack:** Python 3.10+ · FastAPI · Google GenAI (`google-genai`) · Pinecone · React + Vite  

---

## 1. Overview & Problem Statement

Unlabeled text corpora—such as academic papers, engineering logs, customer inquiries, and intelligence reports—are labor-intensive to organize manually. Standard keyword search frequently fails when documents use varied vocabulary or synonyms to discuss identical concepts.

**Document Cluster Lab** is an end-to-end data mining and information retrieval application built for the DWM curriculum. It implements:
1. **Document Ingestion & Preprocessing:** Supports CSV, TXT, Markdown, and text-based PDF files with scanned-page detection.
2. **Text Chunking:** Word-boundary-aligned 1,100-character windows with 140-character overlap and SHA-256 deduplication.
3. **Dual Representation Space:**
   - **Baseline:** Classical sparse TF-IDF with English stop-word filtering and sublinear term frequency.
   - **Target:** Google GenAI dense 768-dimensional embeddings (`text-embedding-004`) with explicit `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types, backed by a local SQLite cache.
4. **Normalized Vector Space Clustering:** Strict **L2 normalization** ($\|x\| = 1$) so that Euclidean minimization in K-Means corresponds mathematically to maximizing cosine similarity.
5. **Multi-Algorithm Support:** Both **K-Means** (Lloyd's algorithm) and **Agglomerative Hierarchical Clustering** (Ward's minimum variance linkage).
6. **Dual Thematic Extraction:**
   - **Objective:** Class-based TF-IDF (**c-TF-IDF**) extracting data-driven statistical keywords per cluster.
   - **Generative:** Structured JSON synthesis via Gemini (`gemini-1.5-flash`) producing academic titles and coherent summaries.
7. **Traceability:** Extracts actual **centroid-nearest representative excerpts** per cluster, logging Euclidean distance to centroid.
8. **Interactive Visualizations:** 2D Principal Component Analysis (**PCA**) projection and dynamic **Elbow (Inertia/WCSS) & Silhouette K-sweep** curves.
9. **Grounded RAG Assistant:** Pinecone cloud vector retrieval gated by a cosine similarity threshold ($\ge 0.65$), providing grounded question answering with direct source citations.
10. **Excel-Safe Exports:** Labeled chunks CSV with formula injection sanitization (`=`, `+`, `-`, `@`) and full JSON analytical reports.

---

## 2. Cloud Privacy & Data Governance Disclosure

> [!IMPORTANT]
> This application uses cloud APIs for dense embedding, generative thematic synthesis, and managed vector storage:
> - **Google GenAI API:** Document text chunks and user queries are sent over HTTPS to Google's GenAI endpoints (`text-embedding-004` and `gemini-1.5-flash`).
> - **Pinecone Vector DB:** Document chunk vector representations and excerpt metadata (up to 2,000 characters) are stored remotely in a Pinecone serverless cloud index.
>
> **Do not upload confidential, proprietary, or sensitive personal data without appropriate authorization.** API keys are kept strictly server-side in `.env` and are never committed to version control or exposed to the client-side React application.

---

## 3. Measured DWM Experimental Results

### A. Measured TF-IDF Baseline (24-Document Corpus)

*Settings: `TfidfVectorizer(stop_words='english', min_df=1, sublinear_tf=True, norm='l2')`, K-Means (`random_state=42`, `n_init=10`).*

| K | Inertia (WCSS) | Cosine Silhouette | Cluster Size Distribution | Observations |
|---:|---:|---:|---|---|
| 2 | 19.8402 | 0.0626 | [17, 7] | High imbalance; clusters merge dissimilar engineering topics. |
| 3 | 18.3850 | 0.0806 | [10, 3, 11] | Partial separation of sensors and energy. |
| 4 | 17.0510 | 0.0783 | [6, 8, 6, 4] | Moderate balance across primary corpus categories. |
| 5 | 15.8445 | 0.0878 | [7, 3, 5, 5, 4] | Cohesive grouping of renewable energy documents. |
| 6 | 14.4619 | 0.1150 | [4, 3, 7, 4, 4, 2] | Diagnostic split; small clusters exhibit vocabulary-level sharing. |

> **Comparative Guidance:**
> Inertia is **not comparable** between TF-IDF (sparse space) and dense embedding space (768 dimensions). When comparing representations, evaluate only Cosine Silhouette score, cluster size balance, and qualitative coherence.

---

## 4. Local Setup & Execution Guide (Windows PowerShell)

### Prerequisites
- Python 3.10+ (tested on Python 3.13.3)
- Node.js 18+ (tested on Node v23.11.0)
- Google Gemini API Key
- Pinecone API Key

### Step 1: Backend Virtual Environment
```powershell
# Open Windows PowerShell
Set-ExecutionPolicy -Scope Process RemoteSigned -Force
cd d:\College\DWM\Mini_Project\Code\document-clustering-lab\backend

# Activate virtualenv
.\venv\Scripts\Activate.ps1

# Configure secrets
Copy-Item .env.example .env
# Open backend/.env in your editor and insert GEMINI_API_KEY and PINECONE_API_KEY
```

### Step 2: Run Unit Tests
```powershell
cd d:\College\DWM\Mini_Project\Code\document-clustering-lab
.\backend\venv\Scripts\pytest.exe tests/ -v
```

### Step 3: Run the Backend API Server
```powershell
cd d:\College\DWM\Mini_Project\Code\document-clustering-lab\backend
.\venv\Scripts\Activate.ps1
uvicorn app.api:app --reload --port 8000
```
*Health check URL: `http://127.0.0.1:8000/api/health`*  
*Interactive Swagger docs: `http://127.0.0.1:8000/docs`*

### Step 4: Run the React + Vite Frontend (New Terminal)
```powershell
cd d:\College\DWM\Mini_Project\Code\document-clustering-lab\frontend
npm run dev
```
*Access the user interface at: `http://localhost:5173`*

---

## 5. Standalone CLI K-Sweep Experiment

You can run automated DWM parameter sweeps directly from the terminal without launching the UI:

```powershell
cd d:\College\DWM\Mini_Project\Code\document-clustering-lab\backend
.\venv\Scripts\Activate.ps1

# Run TF-IDF baseline sweep
python -m experiments.run_k_sweep --dataset data/sample_documents.csv --method tfidf --k-min 2 --k-max 6

# Run Gemini dense embedding sweep (uses local cache)
python -m experiments.run_k_sweep --dataset data/sample_documents.csv --method gemini --k-min 2 --k-max 6
```
*Results are automatically output as formatted tables and saved to `backend/experiments/k_sweep_results.csv`.*

---

## 6. Known Limitations

1. **Small Sample Corpus:** The included 24-document dataset serves as an academic smoke test and benchmark. Silhouette scores on small corpora are typically compressed ($< 0.20$).
2. **Lossy 2D PCA:** Dimensionality reduction to 2 dimensions for scatter plotting discards higher-order variance. Vectors are clustered in the full normalized embedding space, never in the 2D PCA space.
3. **No OCR Support:** Scanned/image-based PDFs without selectable text layers are detected by heuristic ($< 50$ characters/page) and cleanly rejected.
4. **Subjective Generative Themes:** While LLM theme summaries provide human-readable synthesis, they should always be evaluated alongside the purely statistical **c-TF-IDF objective keywords** and centroid-nearest representative excerpts.

---

## 7. Troubleshooting Guide

| Issue | Root Cause | Solution |
|---|---|---|
| `GEMINI_API_KEY is not configured` | Missing key in `.env` | Add key to `backend/.env` and restart backend server. |
| `Scanned or image-based PDF` error | PDF has no selectable text | Upload a digital, text-based PDF or CSV/TXT file. |
| `K must be strictly smaller than chunk count` | Selected K too large | Lower K in the UI or upload more documents. |
| `Failed to connect to backend engine` | Backend server not running | Ensure `uvicorn app.api:app --reload --port 8000` is active. |
| `Insufficient semantic overlap` in Chat | Cosine similarity $< 0.65$ | The question is out-of-corpus; the threshold safely prevents LLM hallucination. |
