import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
});

export const getHealth = () => api.get('/health');
export const getAnalytics = () => api.get('/analytics');
export const getOverview = () => api.get('/analytics/overview');
export const getLogs = (params) => api.get('/analytics/logs', { params });
export const analyzeCorpus = (data) => api.post('/analyze', data);
export const runSweep = (data) => api.post('/sweep', data);
export const chatWithCorpus = (data) => api.post('/chat', data);
export const getDocument = (chunkIdOrTitle) => api.get(`/documents/${encodeURIComponent(chunkIdOrTitle)}`);
export const uploadFile = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export default api;

