// frontend/src/api/client.js
import axios from 'axios';

const base = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const api = axios.create({
  baseURL: base,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

export const investigateAccount = async (accountId, file = null) => {
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/investigate?account_id=${encodeURIComponent(accountId)}&hop_radius=2&time_window_days=30`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  }
  return api.post(`/investigate?account_id=${encodeURIComponent(accountId)}&hop_radius=2&time_window_days=30`);
};

export const investigateTransaction = async (transactionId) => {
  return api.post(`/investigate/transaction?transaction_id=${encodeURIComponent(transactionId)}`);
};

export default api;