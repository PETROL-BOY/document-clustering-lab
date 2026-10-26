import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Database, MessageSquare, BarChart3 } from 'lucide-react';
import { ChatProvider } from './context/ChatContext';
import ChatPage from './pages/ChatPage';
import AnalyticsPage from './pages/AnalyticsPage';

function Header() {
  const location = useLocation();

  return (
    <header className="app-header">
      <Link to="/" className="brand">
        <div className="brand-icon">
          <Database size={18} />
        </div>
        <div>
          <span className="brand-title">Document Intelligence</span>
        </div>
        <span className="brand-badge">DWM Mini-Project</span>
      </Link>

      <nav className="nav-tabs">
        <Link
          to="/"
          className={`nav-tab ${location.pathname === '/' ? 'active' : ''}`}
        >
          <MessageSquare size={15} />
          <span>Grounded Q&amp;A</span>
        </Link>
        <Link
          to="/analytics"
          className={`nav-tab ${location.pathname === '/analytics' ? 'active' : ''}`}
        >
          <BarChart3 size={15} />
          <span>Analytics &amp; Observability</span>
        </Link>
      </nav>
    </header>
  );
}

export default function App() {
  return (
    <ChatProvider>
      <BrowserRouter>
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>
          <Header />
          <main style={{ flex: 1 }}>
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </ChatProvider>
  );
}
