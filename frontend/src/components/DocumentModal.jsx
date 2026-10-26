import React, { useState, useEffect } from 'react';
import {
  X,
  FileText,
  Copy,
  Check,
  Tag,
  Folder,
  Layers,
  Sparkles,
  ExternalLink,
  BookOpen,
  Hash,
} from 'lucide-react';
import { getDocument } from '../api/client';

export default function DocumentModal({ doc, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState('full'); // 'full' or 'chunk'
  const [selectedChunkIdx, setSelectedChunkIdx] = useState(doc?.chunk_index || 0);

  useEffect(() => {
    if (!doc) return;

    // Close on Escape key press
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);

    // Fetch full document details from backend
    const fetchFullDetails = async () => {
      setLoading(true);
      try {
        const identifier = doc.chunk_id || doc.title;
        if (identifier) {
          const res = await getDocument(identifier);
          setDetails(res.data);
          if (res.data.chunk_index !== undefined) {
            setSelectedChunkIdx(res.data.chunk_index);
          }
        }
      } catch (err) {
        console.warn('Could not fetch extra document details from server:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchFullDetails();

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [doc, onClose]);

  if (!doc) return null;

  const title = details?.title || doc.title || 'Untitled Document';
  const source = details?.source || doc.source || 'Uploaded Corpus';
  const category = details?.category || doc.category;
  const score = doc.score;
  const chunkIndex = details?.chunk_index ?? doc.chunk_index ?? 0;
  const totalChunks = details?.total_chunks || (doc.chunk_index !== undefined ? 1 : null);

  // Content selection
  const chunkText =
    (details?.all_chunks && details.all_chunks[selectedChunkIdx]?.text) ||
    details?.chunk_text ||
    doc.full_text ||
    doc.text ||
    doc.excerpt ||
    '';

  const fullDocText = details?.full_text || doc.full_text || doc.text || doc.excerpt || '';

  const activeContent = activeTab === 'chunk' && details?.all_chunks?.length > 1 ? chunkText : fullDocText;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.warn('Clipboard write failed:', e);
    }
  };

  const wordCount = activeContent.trim() ? activeContent.trim().split(/\s+/).length : 0;
  const charCount = activeContent.length;
  const readingTimeSec = Math.max(1, Math.round((wordCount / 200) * 60));

  return (
    <div
      className="doc-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="doc-modal-title"
    >
      <div className="doc-modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Top Header Bar */}
        <div className="doc-modal-header">
          <div className="doc-modal-title-group">
            <div className="doc-modal-icon-badge">
              <FileText size={18} />
            </div>
            <div>
              <h2 id="doc-modal-title" className="doc-modal-title">
                {title}
              </h2>
              <div className="doc-modal-meta-row">
                <span className="doc-modal-meta-item">
                  <Folder size={12} />
                  <span>{source}</span>
                </span>
                {category && (
                  <span className="doc-modal-meta-item doc-modal-badge-category">
                    <Tag size={11} />
                    <span>{category}</span>
                  </span>
                )}
                {typeof score === 'number' && (
                  <span className="doc-modal-meta-item doc-modal-badge-sim">
                    <Sparkles size={11} />
                    <span>similarity: {score.toFixed(3)}</span>
                  </span>
                )}
                {totalChunks && totalChunks > 1 && (
                  <span className="doc-modal-meta-item">
                    <Layers size={11} />
                    <span>{totalChunks} chunks</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="doc-modal-actions">
            <button
              type="button"
              className="btn btn-secondary doc-modal-btn"
              onClick={handleCopy}
              title="Copy text to clipboard"
            >
              {copied ? (
                <>
                  <Check size={14} color="#16a34a" />
                  <span style={{ color: '#16a34a' }}>Copied</span>
                </>
              ) : (
                <>
                  <Copy size={14} />
                  <span>Copy</span>
                </>
              )}
            </button>
            <button
              type="button"
              className="doc-modal-close-btn"
              onClick={onClose}
              title="Close modal (Esc)"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* View mode switcher if document has multiple chunks */}
        {details?.all_chunks && details.all_chunks.length > 1 && (
          <div className="doc-modal-tabs">
            <button
              type="button"
              className={`doc-modal-tab ${activeTab === 'full' ? 'active' : ''}`}
              onClick={() => setActiveTab('full')}
            >
              <BookOpen size={13} />
              <span>Full Document ({details.all_chunks.length} Chunks Combined)</span>
            </button>
            <button
              type="button"
              className={`doc-modal-tab ${activeTab === 'chunk' ? 'active' : ''}`}
              onClick={() => setActiveTab('chunk')}
            >
              <Layers size={13} />
              <span>Referenced Chunk (#{chunkIndex + 1})</span>
            </button>
          </div>
        )}

        {/* Chunk selector pill bar if in chunk mode */}
        {activeTab === 'chunk' && details?.all_chunks && details.all_chunks.length > 1 && (
          <div className="doc-modal-chunk-selector">
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              SELECT CHUNK:
            </span>
            {details.all_chunks.map((chk, idx) => (
              <button
                key={chk.chunk_id || idx}
                type="button"
                className={`doc-chunk-pill ${selectedChunkIdx === idx ? 'active' : ''} ${
                  idx === chunkIndex ? 'referenced' : ''
                }`}
                onClick={() => setSelectedChunkIdx(idx)}
                title={`Chunk ${idx + 1} (${chk.length || chk.text?.length} chars)${
                  idx === chunkIndex ? ' - Directly referenced in answer' : ''
                }`}
              >
                Chunk {idx + 1} {idx === chunkIndex && '★'}
              </button>
            ))}
          </div>
        )}

        {/* Main Document Text Content */}
        <div className="doc-modal-body">
          <div className="doc-content-card">
            <div className="doc-content-text">
              {activeContent}
            </div>
          </div>
        </div>

        {/* Bottom Status / Stats Footer */}
        <div className="doc-modal-footer">
          <div className="doc-modal-stats">
            <span>{wordCount} words</span>
            <span>&bull;</span>
            <span>{charCount} characters</span>
            <span>&bull;</span>
            <span>~{readingTimeSec}s reading time</span>
            {doc.chunk_id && (
              <>
                <span>&bull;</span>
                <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                  ID: {doc.chunk_id}
                </span>
              </>
            )}
          </div>
          <button
            type="button"
            className="btn btn-primary"
            style={{ padding: '0.35rem 0.85rem', fontSize: '0.78rem' }}
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
