import React, { useState, useEffect, useCallback } from 'react';
import {
  Upload,
  Play,
  TrendingUp,
  Download,
  FileText,
  Layers,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  RefreshCw,
  Search,
  Activity,
  CheckCircle2,
  XCircle,
  HelpCircle,
  BarChart2,
  Database,
  ExternalLink,
} from 'lucide-react';
import DocumentModal from '../components/DocumentModal';
import {
  getAnalytics,
  getOverview,
  getLogs,
  analyzeCorpus,
  runSweep,
  uploadFile,
} from '../api/client';
import PCAPlot from '../components/PCAPlot';
import SweepChart from '../components/SweepChart';

export default function AnalyticsPage() {
  const [data, setData] = useState(null);
  const [overview, setOverview] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [sweeping, setSweeping] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedReps, setExpandedReps] = useState({});
  const [selectedDoc, setSelectedDoc] = useState(null);

  // Mining Controls
  const [k, setK] = useState(4);
  const [method, setMethod] = useState('gemini');
  const [algorithm, setAlgorithm] = useState('kmeans');

  // Log filter controls
  const [logSearch, setLogSearch] = useState('');
  const [logStatus, setLogStatus] = useState('all');

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [analyticsRes, overviewRes, logsRes] = await Promise.all([
        getAnalytics(),
        getOverview(),
        getLogs({ limit: 50 }),
      ]);
      setData(analyticsRes.data);
      setOverview(overviewRes.data);
      setLogs(logsRes.data);
      setK(analyticsRes.data.k || 4);
      setMethod(analyticsRes.data.method || 'gemini');
      setAlgorithm(analyticsRes.data.algorithm || 'kmeans');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to connect to backend engine.');
    } finally {
      setLoading(false);
    }
  };

  const fetchFilteredLogs = useCallback(async () => {
    setLogsLoading(true);
    try {
      const params = { limit: 50 };
      if (logSearch.trim()) params.search = logSearch.trim();
      if (logStatus !== 'all') params.status = logStatus;
      const res = await getLogs(params);
      setLogs(res.data);
    } catch (err) {
      console.warn('Failed to fetch filtered logs:', err);
    } finally {
      setLogsLoading(false);
    }
  }, [logSearch, logStatus]);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchFilteredLogs();
    }, 250);
    return () => clearTimeout(timer);
  }, [logSearch, logStatus, fetchFilteredLogs]);

  const handleRunAnalysis = async () => {
    setAnalyzing(true);
    setError(null);
    try {
      const res = await analyzeCorpus({ k: parseInt(k, 10), method, algorithm });
      setData(res.data);
      const ov = await getOverview();
      setOverview(ov.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis execution failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleRunSweep = async () => {
    setSweeping(true);
    setError(null);
    try {
      const res = await runSweep({ k_min: 2, k_max: 6, method });
      setData((prev) => ({ ...prev, sweep_data: res.data }));
    } catch (err) {
      setError(err.response?.data?.detail || 'K-sweep execution failed.');
    } finally {
      setSweeping(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await uploadFile(file);
      await handleRunAnalysis();
    } catch (err) {
      setError(err.response?.data?.detail || 'File ingestion failed.');
    } finally {
      setUploading(false);
    }
  };

  const toggleReps = (cId) => {
    setExpandedReps((prev) => ({ ...prev, [cId]: !prev[cId] }));
  };

  if (loading) {
    return (
      <div className="main-container" style={{ textAlign: 'center', padding: '6rem 0' }}>
        <RefreshCw className="animate-spin" size={36} style={{ margin: '0 auto 1rem', color: 'var(--text-secondary)' }} />
        <h3 style={{ fontWeight: 600 }}>Initializing Document Intelligence Engine...</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Loading corpus, calculating dense embeddings, and retrieving observability metrics.
        </p>
      </div>
    );
  }

  return (
    <div className="main-container">
      {selectedDoc && (
        <DocumentModal doc={selectedDoc} onClose={() => setSelectedDoc(null)} />
      )}
      {error && (
        <div
          style={{
            background: '#fff1f2',
            border: '1px solid #fecdd3',
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            color: '#be123c',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontSize: '0.9rem',
          }}
        >
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* SECTION A: SYSTEM OVERVIEW KPI CARDS */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Total Documents</div>
          <div className="kpi-value">{overview?.total_docs ?? data?.doc_count ?? 0}</div>
          <div className="kpi-sub">Across active knowledge base</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Total Chunks</div>
          <div className="kpi-value">{overview?.total_chunks ?? data?.chunk_count ?? 0}</div>
          <div className="kpi-sub">Normalized vector representations</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Discovered Clusters</div>
          <div className="kpi-value">{overview?.total_clusters ?? data?.k ?? 0}</div>
          <div className="kpi-sub">{algorithm === 'kmeans' ? "K-Means Lloyd's" : 'Agglomerative Ward'}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Questions Handled</div>
          <div className="kpi-value">{overview?.total_questions ?? 0}</div>
          <div className="kpi-sub">
            {overview?.successful_queries ?? 0} passed &bull; {overview?.insufficient_evidence_queries ?? 0} rejected
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Evidence Pass Rate</div>
          <div className="kpi-value">{overview?.evidence_pass_rate_pct ?? 100}%</div>
          <div className="kpi-sub">
            Avg RAG: {overview?.avg_rag_latency_ms ? `${overview.avg_rag_latency_ms}ms` : 'N/A'}
          </div>
        </div>
      </div>

      {/* SECTION B: MINING CONTROL & INGESTION */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Layers size={18} />
            <span>Mining Control &amp; Dataset Ingestion</span>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Active Corpus: <b>{data?.dataset_name || 'sample_documents.csv'}</b> ({data?.chunk_count || 0} chunks)
            </span>
            <label className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.78rem' }}>
              <Upload size={14} />
              <span>{uploading ? 'Ingesting...' : 'Upload File'}</span>
              <input
                type="file"
                accept=".csv,.txt,.md,.pdf"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </label>
          </div>
        </div>

        <div className="controls-grid">
          <div className="form-group">
            <label className="form-label">Representation Method</label>
            <select className="form-select" value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="gemini">Google Gemini Dense (gemini-embedding-001, 3072-d)</option>
              <option value="tfidf">TF-IDF Baseline (Sparse L2 Norm)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Clustering Algorithm</label>
            <select className="form-select" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              <option value="kmeans">K-Means (Euclidean / Lloyd's)</option>
              <option value="agglomerative">Agglomerative Hierarchical (Ward)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Clusters (K)</label>
            <input
              type="number"
              className="form-input"
              min="2"
              max={Math.max(2, (data?.chunk_count || 10) - 1)}
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className="btn btn-primary"
              style={{ flex: 1 }}
              onClick={handleRunAnalysis}
              disabled={analyzing}
            >
              <Play size={15} />
              <span>{analyzing ? 'Clustering...' : 'Execute Clustering'}</span>
            </button>
            <button
              className="btn btn-secondary"
              onClick={handleRunSweep}
              disabled={sweeping}
              title="Compute Elbow and Silhouette over candidate K values"
            >
              <TrendingUp size={15} />
              <span>{sweeping ? 'Sweeping...' : 'K-Sweep'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* SECTION C: EVALUATION METRICS BANNER */}
      <div className="metrics-banner" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="kpi-card">
          <div className="kpi-label">Cosine Silhouette</div>
          <div className="kpi-value" style={{ fontSize: '1.4rem' }}>
            {data?.metrics?.silhouette !== null && data?.metrics?.silhouette !== undefined
              ? data.metrics.silhouette.toFixed(4)
              : 'N/A'}
          </div>
          <div className="kpi-sub">Cluster separation score (-1 to +1)</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Inertia (WCSS)</div>
          <div className="kpi-value" style={{ fontSize: '1.4rem' }}>
            {data?.metrics?.inertia !== null && data?.metrics?.inertia !== undefined
              ? data.metrics.inertia.toFixed(2)
              : 'N/A'}
          </div>
          <div className="kpi-sub">{algorithm === 'kmeans' ? 'Within-cluster sum of squares' : 'N/A for Agglomerative'}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Algorithm Agreement</div>
          <div className="kpi-value" style={{ fontSize: '1.4rem' }}>
            {data?.metrics?.cluster_agreement !== null && data?.metrics?.cluster_agreement !== undefined
              ? data.metrics.cluster_agreement.toFixed(4)
              : 'N/A'}
          </div>
          <div className="kpi-sub">K-Means vs Agglomerative ARI</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Adjusted Rand Index</div>
          <div className="kpi-value" style={{ fontSize: '1.4rem' }}>
            {data?.metrics?.ari !== null && data?.metrics?.ari !== undefined
              ? data.metrics.ari.toFixed(4)
              : 'N/A (Unlabeled)'}
          </div>
          <div className="kpi-sub">Ground-truth label agreement</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Normalized MI</div>
          <div className="kpi-value" style={{ fontSize: '1.4rem' }}>
            {data?.metrics?.nmi !== null && data?.metrics?.nmi !== undefined
              ? data.metrics.nmi.toFixed(4)
              : 'N/A (Unlabeled)'}
          </div>
          <div className="kpi-sub">Mutual information with ground truth</div>
        </div>
      </div>

      {/* SECTION D: PCA 2D MAP & K-SWEEP VISUALIZATIONS */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: data?.sweep_data ? '3fr 2fr' : '1fr',
          gap: '1.5rem',
          marginBottom: '1.5rem',
        }}
      >
        <div className="panel" style={{ marginBottom: 0 }}>
          <div className="panel-header">
            <div className="panel-title">
              <BarChart2 size={18} />
              <span>2D PCA Projection Map</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Projected from {method === 'gemini' ? '3072-d dense space' : 'TF-IDF sparse space'}
            </span>
          </div>
          <PCAPlot points={data?.pca_points} k={data?.k} />
        </div>

        {data?.sweep_data && (
          <div className="panel" style={{ marginBottom: 0 }}>
            <div className="panel-header">
              <div className="panel-title">
                <TrendingUp size={18} />
                <span>K-Sweep Optimization Curves</span>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                Elbow &amp; Silhouette
              </span>
            </div>
            <SweepChart sweepData={data.sweep_data} />
          </div>
        )}
      </div>

      {/* SECTION E: SYNTHESIZED SEMANTIC CLUSTERS */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
        }}
      >
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Synthesized Semantic Clusters</h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <a
            href="/api/export/csv"
            download
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', textDecoration: 'none' }}
          >
            <Download size={14} />
            <span>Download CSV (Excel Safe)</span>
          </a>
          <a
            href="/api/export/json"
            download
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', textDecoration: 'none' }}
          >
            <FileText size={14} />
            <span>Export JSON Report</span>
          </a>
        </div>
      </div>

      <div className="clusters-grid">
        {data?.clusters?.map((cluster) => (
          <div key={cluster.id} className="cluster-card">
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderBottom: '1px solid var(--border-light)',
                paddingBottom: '0.5rem',
              }}
            >
              <span
                style={{
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  fontSize: '0.82rem',
                  background: 'var(--bg-muted)',
                  padding: '0.15rem 0.5rem',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                Cluster #{cluster.id}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {cluster.size} documents ({cluster.percentage}%)
              </span>
            </div>

            <div>
              <div style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '0.35rem' }}>
                {cluster.llm_label}
              </div>
              <div style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {cluster.llm_summary}
              </div>
            </div>

            {/* Objective Keywords */}
            <div>
              <div
                style={{
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  marginBottom: '0.35rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                Objective Keywords (c-TF-IDF):
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                {cluster.objective_keywords.length > 0 ? (
                  cluster.objective_keywords.map((kw, i) => (
                    <span key={i} className="tag-objective">
                      {kw}
                    </span>
                  ))
                ) : (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    No dominant terms
                  </span>
                )}
              </div>
            </div>

            {/* Centroid-Nearest Representatives */}
            <div style={{ marginTop: 'auto', paddingTop: '0.5rem', borderTop: '1px solid var(--border-light)' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{
                  width: '100%',
                  justifyContent: 'space-between',
                  padding: '0.4rem 0.75rem',
                  fontSize: '0.78rem',
                }}
                onClick={() => toggleReps(cluster.id)}
              >
                <span>Centroid-Nearest Representatives ({cluster.representatives?.length || 0})</span>
                {expandedReps[cluster.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>

              {expandedReps[cluster.id] && (
                <div style={{ marginTop: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {cluster.representatives?.map((rep, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--bg-subtle)',
                        border: '1px solid var(--border-light)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.6rem 0.85rem',
                        fontSize: '0.78rem',
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                      }}
                      className="source-card"
                      onClick={() => setSelectedDoc(rep)}
                      title={`Click to open full document: ${rep.title}`}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontWeight: 600,
                          marginBottom: '0.2rem',
                        }}
                      >
                        <span style={{ color: 'var(--text-primary)' }}>{rep.title}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                            d={rep.distance_to_centroid}
                          </span>
                          <ExternalLink size={11} color="var(--text-muted)" />
                        </div>
                      </div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', lineHeight: 1.4 }}>
                        {rep.excerpt}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* SECTION F: RAG REQUEST LOGS & OBSERVABILITY */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Activity size={18} />
            <span>RAG Query Observability Logs</span>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            {/* Search Input */}
            <div style={{ position: 'relative', width: '240px' }}>
              <Search
                size={14}
                style={{
                  position: 'absolute',
                  left: '0.65rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-muted)',
                }}
              />
              <input
                type="text"
                placeholder="Filter questions..."
                value={logSearch}
                onChange={(e) => setLogSearch(e.target.value)}
                className="form-input"
                style={{ paddingLeft: '2rem', width: '100%', fontSize: '0.8rem', padding: '0.35rem 0.65rem 0.35rem 2rem' }}
              />
            </div>

            {/* Status Filter */}
            <select
              value={logStatus}
              onChange={(e) => setLogStatus(e.target.value)}
              className="form-select"
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.65rem' }}
            >
              <option value="all">All Statuses</option>
              <option value="success">Success</option>
              <option value="insufficient_evidence">Insufficient Evidence</option>
              <option value="error">Error</option>
            </select>

            <button
              type="button"
              className="btn btn-secondary"
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
              onClick={fetchFilteredLogs}
              title="Refresh logs table"
            >
              <RefreshCw size={13} className={logsLoading ? 'animate-spin' : ''} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Logs Table */}
        <div className="logs-table-wrapper">
          <table className="logs-table">
            <thead>
              <tr>
                <th style={{ width: '90px' }}>Time</th>
                <th>Question</th>
                <th style={{ width: '150px' }}>Status</th>
                <th style={{ width: '120px' }}>Max Similarity</th>
                <th style={{ width: '110px' }}>Citations</th>
                <th style={{ width: '95px' }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {logs.length === 0 ? (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                    No matching query logs recorded yet. Ask a question in the Grounded Q&amp;A tab to record live telemetry.
                  </td>
                </tr>
              ) : (
                logs.map((entry) => {
                  const maxSim = entry.similarity_scores?.length
                    ? Math.max(...entry.similarity_scores).toFixed(3)
                    : 'N/A';
                  const timeFormatted = new Date(entry.timestamp).toLocaleTimeString();

                  let statusBadgeClass = 'status-success';
                  let statusLabel = 'Success';
                  let StatusIcon = CheckCircle2;

                  if (entry.status === 'insufficient_evidence') {
                    statusBadgeClass = 'status-insufficient';
                    statusLabel = 'Rejected (Cutoff)';
                    StatusIcon = AlertCircle;
                  } else if (entry.status === 'error') {
                    statusBadgeClass = 'status-error';
                    statusLabel = 'Error';
                    StatusIcon = XCircle;
                  }

                  return (
                    <tr key={entry.id}>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {timeFormatted}
                      </td>
                      <td style={{ fontWeight: 500 }}>{entry.question}</td>
                      <td>
                        <span className={`status-badge ${statusBadgeClass}`}>
                          <StatusIcon size={12} />
                          <span>{statusLabel}</span>
                        </span>
                      </td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                        {maxSim}
                      </td>
                      <td>
                        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                          {entry.citations_count} chunk{entry.citations_count !== 1 ? 's' : ''}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                        {entry.latency_ms?.toFixed(0)} ms
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
