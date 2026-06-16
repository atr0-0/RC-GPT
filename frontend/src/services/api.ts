import axios from "axios";

// Type definitions matching backend models
export interface QueryRequest {
  query: string;
  year_range?: [number, number];
  tort_types?: string[];
  max_sources?: number;
}

export interface Source {
  case_name: string;
  citation: string;
  excerpt: string;
  confidence: string;
  section_type: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  search_type: string;
}

export interface HealthResponse {
  status: string;
  retrieval_chain_loaded: boolean;
  hybrid_search: boolean;
  documents_loaded: number;
}

// In production (Vercel), VITE_API_BASE_URL must point to the deployed backend
// (e.g. https://rcgpt-backend.fly.dev). In local dev the Vite proxy rewrites
// /api → http://localhost:8000, so the fallback "/api" just works.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export const processQuery = async (
  queryData: QueryRequest
): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>("/query", queryData);
  return response.data;
};

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>("/health");
  return response.data;
};

export default api;
