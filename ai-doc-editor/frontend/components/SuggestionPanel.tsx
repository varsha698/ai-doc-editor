import { Suggestion } from "@/services/api";

type Props = {
  suggestions: Suggestion[];
  onAccept: (s: Suggestion) => void;
  onReject: (id: string) => void;
};

export default function SuggestionPanel({ suggestions, onAccept, onReject }: Props) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-700">Suggestions</h3>
      {suggestions.length === 0 && <p className="text-sm text-slate-500">No suggestions yet.</p>}
      {suggestions.map((s) => (
        <div key={s.id} className="rounded border border-slate-200 bg-white p-3 shadow-sm">
          <p className="mb-1 text-xs uppercase tracking-wide text-slate-500">{s.type}</p>
          <p className="text-sm text-slate-600 line-through">{s.original_text}</p>
          <p className="text-sm font-medium text-slate-900">{s.suggested_text}</p>
          <p className="mt-1 text-xs text-slate-500">{s.explanation}</p>
          <div className="mt-2 flex gap-2">
            <button className="rounded bg-emerald-600 px-2 py-1 text-xs text-white" onClick={() => onAccept(s)}>
              Accept
            </button>
            <button className="rounded bg-slate-200 px-2 py-1 text-xs text-slate-700" onClick={() => onReject(s.id)}>
              Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
