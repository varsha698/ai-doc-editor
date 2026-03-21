import { useMemo, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import Editor from "@/components/Editor";
import AIChatPanel from "@/components/AIChatPanel";
import SuggestionPanel from "@/components/SuggestionPanel";
import { chatWithAI, Suggestion } from "@/services/api";
import { useEditorAI } from "@/hooks/useEditorAI";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useAIStream } from "@/hooks/useAIStream";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export default function EditorPage() {
  const documentId = useMemo(() => uuidv4(), []);
  const [documentText, setDocumentText] = useState("Start writing your document...");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const { analysis, isAnalyzing, queueAnalysis } = useEditorAI({ documentId });
  const { suggestions, setSuggestions, acceptSuggestion, rejectSuggestion } = useSuggestions();
  const editorRef = useRef<import("@/components/Editor").EditorHandle | null>(null);
  const assistantIndexRef = useRef<number>(-1);
  const { streamChat, isStreaming } = useAIStream();

  function handleEditorChange(text: string) {
    setDocumentText(text);
    queueAnalysis(text);
  }

  async function handleSend(message: string) {
    if (isStreaming) return;
    setMessages((prev) => {
      assistantIndexRef.current = prev.length + 1;
      return [
        ...prev,
        { role: "user", content: message },
        { role: "assistant", content: "" },
      ];
    });

    await streamChat(
      { document_id: documentId, message, text: documentText, command: null },
      {
        onSuggestions: (incoming) => setSuggestions(incoming),
        onToken: (token) => {
          const idx = assistantIndexRef.current;
          if (idx < 0) return;
          setMessages((prev) => {
            return prev.map((m, i) => {
              if (i !== idx) return m;
              return { ...m, content: m.content + token };
            });
          });
        },
      }
    );
  }

  async function handleSlashCommand(command: string, text: string) {
    const promptMap: Record<string, string> = {
      rewrite: "Rewrite this section for clarity.",
      summarize: "Summarize this section.",
      expand: "Expand this section with more detail.",
      "make-professional": "Make this writing more professional.",
      "create-outline": "Create an outline from this content."
    };
    const instruction = promptMap[command] || `Run command: ${command}`;
    const reply = await chatWithAI({ document_id: documentId, message: instruction, text, command });
    setMessages((prev) => [...prev, { role: "assistant", content: reply.reply }]);
    setSuggestions(reply.suggestions);
  }

  function handleAccept(s: Suggestion) {
    const applied = editorRef.current?.applySuggestion(s) ?? false;
    if (!applied) return;
    acceptSuggestion(s.id);
  }

  return (
    <main className="grid h-full grid-cols-12 gap-4 p-4">
      <section className="col-span-8 rounded-lg border border-slate-200 bg-white p-4">
        <h1 className="mb-3 text-xl font-semibold">AI Document Editor</h1>
        <Editor
          ref={editorRef}
          content={documentText}
          onChange={handleEditorChange}
          onSlashCommand={handleSlashCommand}
          suggestions={suggestions}
        />
      </section>

      <aside className="col-span-4 space-y-4">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <h2 className="text-sm font-semibold">Real-time Analysis</h2>
          <p className="text-xs text-slate-500">{isAnalyzing ? "Analyzing..." : "Idle"}</p>
          {analysis && (
            <div className="mt-2 space-y-1 text-xs text-slate-700">
              <p>Readability: {analysis.scores.readability}</p>
              <p>Grammar: {analysis.scores.grammar}</p>
              <p>Clarity: {analysis.scores.clarity}</p>
              <p>Argument Strength: {analysis.scores.argument_strength}</p>
              <p className="pt-1 text-slate-600">{analysis.summary}</p>
            </div>
          )}
        </div>
        <SuggestionPanel suggestions={suggestions} onAccept={handleAccept} onReject={rejectSuggestion} />
        <AIChatPanel onSend={handleSend} messages={messages} />
      </aside>
    </main>
  );
}
