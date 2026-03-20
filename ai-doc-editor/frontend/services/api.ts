import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
});

export type Suggestion = {
  id: string;
  type: "grammar" | "clarity" | "style" | "structure";
  original_text: string;
  suggested_text: string;
  explanation: string;
  start?: number;
  end?: number;
};

export type AnalysisResponse = {
  suggestions: Suggestion[];
  scores: {
    readability: number;
    grammar: number;
    clarity: number;
    argument_strength: number;
  };
  summary: string;
};

export async function analyzeText(payload: { document_id: string; text: string }) {
  const response = await api.post<AnalysisResponse>("/ai/analyze", payload);
  return response.data;
}

export async function chatWithAI(payload: {
  document_id: string;
  message: string;
  text: string;
  command?: string;
}) {
  const response = await api.post<{ reply: string; suggestions: Suggestion[] }>("/ai/chat", payload);
  return response.data;
}
