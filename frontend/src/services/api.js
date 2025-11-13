import axios from 'axios';

const API_BASE_URL = '/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const processQuery = async (queryData) => {
  const response = await api.post('/query', queryData);
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};
