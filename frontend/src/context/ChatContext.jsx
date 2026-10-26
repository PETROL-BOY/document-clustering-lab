import React, { createContext, useContext, useState, useEffect } from 'react';
import { chatWithCorpus, uploadFile } from '../api/client';

const ChatContext = createContext(null);

const STORAGE_KEY = 'doc_intel_chat_history';

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.warn('Failed to parse saved chat history:', e);
    }
    return [
      {
        id: 'initial_welcome',
        role: 'assistant',
        text: 'Welcome to **Document Intelligence**. You can ask natural-language questions about your uploaded documents, and I will generate grounded answers with direct source attribution.',
        citations: [],
        timestamp: new Date().toISOString(),
      },
    ];
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeDataset, setActiveDataset] = useState('sample_documents.csv');

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.warn('Failed to save chat history to sessionStorage:', e);
    }
  }, [messages]);

  const sendMessage = async (question) => {
    if (!question || !question.trim() || loading) return;

    const userMsg = {
      id: `user_${Date.now()}`,
      role: 'user',
      text: question.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setError(null);

    try {
      const res = await chatWithCorpus({ question: question.trim(), top_k: 4 });
      const assistantMsg = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        text: res.data.answer,
        threshold_passed: res.data.threshold_passed,
        citations: res.data.citations || [],
        latency_ms: res.data.latency_ms,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'An error occurred during query generation.';
      setError(errorMsg);
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          role: 'assistant',
          text: `⚠️ **Request Error:** ${errorMsg}`,
          isError: true,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const uploadCorpus = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await uploadFile(file);
      setActiveDataset(res.data.dataset_name);
      const isPdf = file.name.toLowerCase().endsWith('.pdf');
      const noticeText = isPdf
        ? `📄 **Research Paper Ingested:** \`${file.name}\`\n\n- Extracted: **${res.data.chunk_count} semantic chunks** across document pages\n- Vector indexing initialized.\n\nYou can now ask specific questions, request summaries, or query methodologies directly from this research paper!`
        : `📁 **Dataset Ingested:** \`${file.name}\`\n\n- Loaded: **${res.data.doc_count} document(s)** with **${res.data.chunk_count} chunks**\n- You can now query this dataset!`;

      setMessages((prev) => [
        ...prev,
        {
          id: `upload_${Date.now()}`,
          role: 'assistant',
          text: noticeText,
          citations: [],
          timestamp: new Date().toISOString(),
        },
      ]);
      return res.data;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'File ingestion failed.';
      setError(errorMsg);
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          role: 'assistant',
          text: `⚠️ **Upload Failed:** ${errorMsg}`,
          isError: true,
          timestamp: new Date().toISOString(),
        },
      ]);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const clearConversation = () => {
    const initial = [
      {
        id: 'initial_welcome',
        role: 'assistant',
        text: 'Welcome to **Document Intelligence**. You can ask natural-language questions about your uploaded documents, and I will generate grounded answers with direct source attribution.',
        citations: [],
        timestamp: new Date().toISOString(),
      },
    ];
    setMessages(initial);
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(initial));
    } catch (e) {
      // ignore
    }
  };

  return (
    <ChatContext.Provider
      value={{
        messages,
        sendMessage,
        uploadCorpus,
        activeDataset,
        clearConversation,
        loading,
        error,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
