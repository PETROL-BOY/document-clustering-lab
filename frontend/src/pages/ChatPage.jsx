import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Send,
  Sparkles,
  AlertCircle,
  Trash2,
  ChevronDown,
  ChevronUp,
  FileText,
  HelpCircle,
  BookOpen,
  ArrowRight,
  ShieldCheck,
  ExternalLink,
  Upload,
} from 'lucide-react';
import { useChat } from '../context/ChatContext';
import DocumentModal from '../components/DocumentModal';

export default function ChatPage() {
  const { messages, sendMessage, clearConversation, loading, uploadCorpus, activeDataset } = useChat();
  const [input, setInput] = useState('');
  const [openSources, setOpenSources] = useState({});
  const [activeDoc, setActiveDoc] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const sampleQuestions = [
    'How do smart traffic prediction systems operate?',
    'What methods are used for solar power forecasting?',
    'How does ransomware incident response isolate endpoints?',
    'What sensors monitor soil moisture for precision irrigation?',
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;
    sendMessage(input.trim());
    setInput('');
  };

  const handlePromptClick = (prompt) => {
    if (loading) return;
    sendMessage(prompt);
  };

  const toggleSource = (msgId) => {
    setOpenSources((prev) => ({
      ...prev,
      [msgId]: prev[msgId] === undefined ? false : !prev[msgId],
    }));
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadCorpus(file);
    } catch (err) {
      console.warn('Upload error:', err);
    } finally {
      e.target.value = '';
    }
  };

  const renderContentWithCitations = (text, citations) => {
    if (!text) return null;
    // Replace [Source 1], [Source 2], etc. with markdown links [Source 1](#source-1)
    const formattedText = text.replace(/\[Source\s*(\d+)\]/gi, (match, num) => {
      return `[Source ${num}](#source-${num})`;
    });

    return (
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            if (href && href.startsWith('#source-')) {
              const num = parseInt(href.replace('#source-', ''), 10);
              const citation = citations && citations[num - 1];
              return (
                <button
                  type="button"
                  className="citation-pill"
                  onClick={(e) => {
                    e.preventDefault();
                    if (citation) {
                      setActiveDoc(citation);
                    }
                  }}
                  title={citation ? `Open referenced document: ${citation.title}` : `Source ${num}`}
                >
                  <FileText size={10} style={{ opacity: 0.8 }} />
                  <span>{children}</span>
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {formattedText}
      </ReactMarkdown>
    );
  };

  const isInitialState = messages.length <= 1;

  return (
    <div className="chat-container">
      {/* Document Preview Modal */}
      {activeDoc && (
        <DocumentModal doc={activeDoc} onClose={() => setActiveDoc(null)} />
      )}

      {/* Top Bar with Status and Actions */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.75rem 1.5rem',
          borderBottom: '1px solid var(--border-light)',
          background: 'var(--bg-surface)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={16} color="var(--accent-charcoal)" />
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Grounded Semantic RAG
          </span>
          <span
            style={{
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              borderLeft: '1px solid var(--border-light)',
              paddingLeft: '0.5rem',
            }}
          >
            Threshold: cosine &ge; 0.65
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label
            className="btn btn-secondary"
            style={{
              padding: '0.3rem 0.7rem',
              fontSize: '0.75rem',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
            title="Upload research paper PDF (.pdf) or text/csv corpus"
          >
            <Upload size={13} />
            <span>Upload Paper (PDF)</span>
            <input
              type="file"
              accept=".pdf,.csv,.txt,.md"
              onChange={handleFileChange}
              disabled={loading}
              style={{ display: 'none' }}
            />
          </label>

          {messages.length > 1 && (
            <button
              type="button"
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={clearConversation}
              title="Clear message history"
            >
              <Trash2 size={13} />
              <span>Clear Chat</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Conversation or Welcome Screen */}
      <div className="chat-scroll-area">
        {isInitialState && (
          <div className="welcome-screen">
            <div className="welcome-badge">
              <Sparkles size={13} />
              <span>DWM Mini-Project &bull; Document Intelligence</span>
            </div>
            <h1 className="welcome-title">Ask anything across your repository</h1>
            <p className="welcome-subtitle">
              Synthesizing semantically grounded answers from your uploaded corpus with strict cosine threshold verification and direct source citations.
            </p>

            <div className="prompts-grid">
              {sampleQuestions.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="prompt-card"
                  onClick={() => handlePromptClick(q)}
                  disabled={loading}
                >
                  <span>{q}</span>
                  <ArrowRight size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => {
          const isUser = msg.role === 'user';
          const sourcesVisible = openSources[msg.id] !== false;

          return (
            <div
              key={msg.id}
              className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}
            >
              <div className={isUser ? 'user-bubble' : 'assistant-bubble'}>
                {isUser ? (
                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                ) : (
                  <div>
                    <div className="markdown-body">
                      {renderContentWithCitations(msg.text, msg.citations)}
                    </div>

                    {/* Insufficient Evidence Warning Callout */}
                    {msg.threshold_passed === false && (
                      <div
                        style={{
                          marginTop: '0.75rem',
                          padding: '0.65rem 0.85rem',
                          background: '#fffbeb',
                          border: '1px solid #fef3c7',
                          borderRadius: 'var(--radius-sm)',
                          color: '#92400e',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                          fontSize: '0.8rem',
                        }}
                      >
                        <AlertCircle size={15} style={{ flexShrink: 0 }} />
                        <span>
                          Similarity threshold cutoff: No retrieved document chunks met the cosine similarity threshold (&ge; 0.65). The response was rejected due to insufficient evidence.
                        </span>
                      </div>
                    )}

                    {/* Collapsible Source Evidence Drawer */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="sources-drawer">
                        <button
                          type="button"
                          className="sources-header-btn"
                          onClick={() => toggleSource(msg.id)}
                        >
                          <BookOpen size={13} />
                          <span>Retrieved Evidence ({msg.citations.length} Chunks)</span>
                          {sourcesVisible ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </button>

                        {sourcesVisible && (
                          <div className="source-cards-list">
                            {msg.citations.map((c, cIdx) => (
                              <div
                                key={cIdx}
                                className="source-card"
                                onClick={() => setActiveDoc(c)}
                                role="button"
                                tabIndex={0}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' || e.key === ' ') {
                                    e.preventDefault();
                                    setActiveDoc(c);
                                  }
                                }}
                                title={`Click to open "${c.title || `Document ${cIdx + 1}`}"`}
                              >
                                <div className="source-card-top">
                                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                                    <FileText size={13} color="var(--text-secondary)" />
                                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                      {c.title || c.source || `Chunk ${cIdx + 1}`}
                                    </span>
                                  </span>
                                  <div className="source-card-actions">
                                    <span className="source-badge">
                                      sim: {typeof c.score === 'number' ? c.score.toFixed(3) : c.score}
                                    </span>
                                    <span className="source-view-hint">
                                      <span>Open doc</span>
                                      <ExternalLink size={11} />
                                    </span>
                                  </div>
                                </div>
                                <div className="source-excerpt">
                                  {c.excerpt || c.text || 'No preview available'}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Latency info if available */}
                    {msg.latency_ms && (
                      <div
                        style={{
                          marginTop: '0.5rem',
                          fontSize: '0.68rem',
                          color: 'var(--text-muted)',
                          fontFamily: 'monospace',
                        }}
                      >
                        synthesized in {msg.latency_ms.toFixed(0)} ms
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Loading Indicator */}
        {loading && (
          <div className="message-row assistant-row">
            <div
              className="assistant-bubble"
              style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', color: 'var(--text-secondary)' }}
            >
              <Sparkles size={16} className="animate-spin" />
              <span style={{ fontSize: '0.88rem' }}>
                Querying vector space &amp; evaluating semantic evidence...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Composer Area */}
      <div className="composer-area">
        <form onSubmit={handleSubmit}>
          <div className="composer-box">
            <textarea
              ref={textareaRef}
              className="composer-textarea"
              placeholder="Ask a question about the document collection..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={2}
            />
            <div className="composer-footer">
              <span className="composer-hint">
                Press <kbd style={{ padding: '0.1rem 0.3rem', background: 'var(--bg-muted)', borderRadius: '3px', border: '1px solid var(--border-medium)' }}>Enter</kbd> to send, <kbd style={{ padding: '0.1rem 0.3rem', background: 'var(--bg-muted)', borderRadius: '3px', border: '1px solid var(--border-medium)' }}>Shift+Enter</kbd> for new line
              </span>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !input.trim()}
                style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem' }}
              >
                <Send size={14} />
                <span>Ask Question</span>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

