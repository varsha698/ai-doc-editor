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

/** Fast path: rule / spell / optional LanguageTool ΓÇö separate from LLM suggestions */
export type GrammarIssue = {
  type: "grammar" | "punctuation" | "spelling" | "clarity";
  start: number;
  end: number;
  suggestion: string;
  message?: string;
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
  grammar_issues?: GrammarIssue[];
  style_profile?: {
    avg_sentence_length: number;
    vocabulary_usage: { top_terms: string[]; type_token_ratio: number };
    tone: string;
    structure: string;
  };
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
