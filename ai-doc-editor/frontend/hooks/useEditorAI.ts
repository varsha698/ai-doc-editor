import { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisResponse, analyzeText } from "@/services/api";

type UseEditorAIOptions = {
  documentId: string;
  debounceMs?: number;
};

export function useEditorAI({ documentId, debounceMs = 2000 }: UseEditorAIOptions) {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const triggerAnalysis = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setIsAnalyzing(true);
      try {
        const result = await analyzeText({ document_id: documentId, text });
        setAnalysis(result);
      } finally {
        setIsAnalyzing(false);
      }
    },
    [documentId]
  );

  const queueAnalysis = useCallback(
    (text: string) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        void triggerAnalysis(text);
      }, debounceMs);
    },
    [debounceMs, triggerAnalysis]
  );

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return { analysis, isAnalyzing, queueAnalysis, triggerAnalysis };
}
