import { FormEvent, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type Props = {
  onSend: (message: string) => Promise<void>;
  messages: Message[];
};

export default function AIChatPanel({ onSend, messages }: Props) {
  const [value, setValue] = useState("");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    setValue("");
    await onSend(trimmed);
  }

  return (
    <div className="flex h-[350px] flex-col rounded border border-slate-200 bg-white">
      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {messages.map((m, idx) => (
          <div key={idx} className={`rounded p-2 text-sm ${m.role === "assistant" ? "bg-slate-100" : "bg-blue-100"}`}>
            {m.content}
          </div>
        ))}
      </div>
      <form className="flex gap-2 border-t p-2" onSubmit={handleSubmit}>
        <input
          className="flex-1 rounded border px-2 py-1 text-sm"
          placeholder="Ask AI to rewrite, summarize, or improve..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <button className="rounded bg-blue-600 px-3 py-1 text-sm text-white" type="submit">
          Send
        </button>
      </form>
    </div>
  );
}
