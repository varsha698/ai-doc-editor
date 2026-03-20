import { useMemo, useState } from "react";
import { Suggestion } from "@/services/api";

export function useSuggestions() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);

  const grouped = useMemo(() => {
    return suggestions.reduce<Record<string, Suggestion[]>>((acc, item) => {
      if (!acc[item.type]) acc[item.type] = [];
      acc[item.type].push(item);
      return acc;
    }, {});
  }, [suggestions]);

  function acceptSuggestion(id: string) {
    setSuggestions((prev) => prev.filter((s) => s.id !== id));
  }

  function rejectSuggestion(id: string) {
    setSuggestions((prev) => prev.filter((s) => s.id !== id));
  }

  return { suggestions, setSuggestions, grouped, acceptSuggestion, rejectSuggestion };
}
