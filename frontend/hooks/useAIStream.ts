import { useCallback, useRef, useState } from "react";
import type { Suggestion } from "@/services/api";

type ChatPayload = {
  document_id: string;
  message: string;
  text: string;
  command?: string | null;
};

type StreamHandlers = {
  onToken?: (token: string) => void;
  onSuggestions?: (suggestions: Suggestion[]) => void;
};

function toWsUrl(httpUrl: string) {
  // http://host -> ws://host, https://host -> wss://host
  return httpUrl.replace(/^http/, "ws");
}

export function useAIStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const streamChat = useCallback(
    async (payload: ChatPayload, handlers: StreamHandlers = {}): Promise<void> => {
      // Close any in-flight socket.
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }

      setIsStreaming(true);

      const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
      const url = `${toWsUrl(base)}/ai/chat/ws`;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      return new Promise((resolve, reject) => {
        ws.onopen = () => {
          ws.send(JSON.stringify(payload));
        };

        ws.onmessage = (event) => {
          let msg: any;
          try {
            msg = JSON.parse(event.data);
          } catch {
            return;
          }

          if (msg.type === "suggestions") {
            handlers.onSuggestions?.(msg.suggestions as Suggestion[]);
            return;
          }
          if (msg.type === "token") {
            handlers.onToken?.(String(msg.token ?? ""));
            return;
          }
          if (msg.type === "done") {
            try {
              ws.close();
            } catch {
              // ignore
            }
            resolve();
            return;
          }
          if (msg.type === "error") {
            try {
              ws.close();
            } catch {
              // ignore
            }
            reject(new Error(String(msg.error ?? "Unknown stream error")));
          }
        };

        ws.onerror = () => {
          try {
            ws.close();
          } catch {
            // ignore
          }
          reject(new Error("WebSocket error"));
        };

        ws.onclose = () => {
          setIsStreaming(false);
        };
      });
    },
    []
  );

  return { isStreaming, streamChat };
}

