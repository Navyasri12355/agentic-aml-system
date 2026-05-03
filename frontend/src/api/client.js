import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 60000, // 60 seconds (LangGraph is heavy)
  headers: {
    'Content-Type': 'application/json',
  },
});

export const investigateAccount = async (accountId, file = null) => {
  // If there's a file, we must use FormData
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    // Add query params
    return api.post(`/investigate?account_id=${encodeURIComponent(accountId)}&hop_radius=2&time_window_days=30`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  // If no file, we just hit the endpoint with query params
  return api.post(`/investigate?account_id=${encodeURIComponent(accountId)}&hop_radius=2&time_window_days=30`);
};

export default api;