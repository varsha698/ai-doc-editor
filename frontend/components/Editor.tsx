import { useEffect, useImperativeHandle, useMemo, useRef, useState, forwardRef } from "react";
import { EditorContent, useEditor } from "@tiptap/react";
import { Extension } from "@tiptap/core";
import StarterKit from "@tiptap/starter-kit";
import Link from "@tiptap/extension-link";
import Table from "@tiptap/extension-table";
import TableRow from "@tiptap/extension-table-row";
import TableCell from "@tiptap/extension-table-cell";
import TableHeader from "@tiptap/extension-table-header";
import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import { Decoration, DecorationSet } from "prosemirror-view";
import { Plugin, PluginKey } from "prosemirror-state";
import { common, createLowlight } from "lowlight";
import Toolbar from "@/components/Toolbar";
import type { Suggestion } from "@/services/api";

const lowlight = createLowlight(common);

type Props = {
  content: string;
  onChange: (text: string) => void;
  onSlashCommand?: (command: string, text: string) => void;
  suggestions?: Suggestion[];
};

const slashHint = `Type slash commands:
/rewrite
/summarize
/expand
/make-professional
/create-outline`;

export type EditorHandle = {
  applySuggestion: (s: Suggestion) => boolean;
};

function _buildTextIndex(doc: any) {
  // Walk text nodes and build a mapping from editor.getText() offsets to ProseMirror positions.
  // tiptap/editor.getText() inserts '\n' between textblocks, so we mirror that by
  // incrementing our offset when we enter a new textblock.
  const parts: Array<{
    textStart: number;
    textEnd: number;
    posStart: number;
    posEnd: number;
  }> = [];
  let offset = 0;
  let firstBlock = true;
  doc.descendants((node: any, pos: number) => {
    if (node.isTextblock) {
      if (!firstBlock) offset += 1; // blockSeparator: '\n'
      firstBlock = false;
    }
    if (!node.isText) return true;
    const t = String(node.text ?? "");
    const start = offset;
    const end = offset + t.length;
    parts.push({ textStart: start, textEnd: end, posStart: pos, posEnd: pos + t.length });
    offset = end;
    return true;
  });
  return parts;
}

function _mapTextOffsetToPos(index: any[], textOffset: number) {
  if (index.length === 0) return null;
  for (const seg of index) {
    // Allow boundary mapping to the end position of the segment.
    if (textOffset >= seg.textStart && textOffset <= seg.textEnd) {
      const within = textOffset - seg.textStart;
      return seg.posStart + within;
    }
  }
  return null;
}

function _mapTextRangeToPosRange(index: any[], start: number, end: number) {
  const from = _mapTextOffsetToPos(index, start);
  const to = _mapTextOffsetToPos(index, end);
  if (from == null || to == null) return null;
  if (to < from) return null;
  return { from, to };
}

const suggestionPluginKey = new PluginKey("suggestion-range-decorations");

const highlightStyle = "background: rgba(245, 158, 11, 0.25); border-bottom: 1px solid rgba(245, 158, 11, 0.65);";

export default forwardRef<EditorHandle, Props>(function Editor(
  { content, onChange, onSlashCommand, suggestions }: Props,
  ref
) {
  const lastCommandRef = useRef<string>("");
  const suggestionsRef = useRef<Suggestion[]>(suggestions || []);
  const [editorReady, setEditorReady] = useState(false);
  const [editorInstance, setEditorInstance] = useState<any>(null);

  const extensions = useMemo(
    () => [
      StarterKit,
      Link.configure({ openOnClick: true }),
      Table.configure({ resizable: true }),
      TableRow,
      TableCell,
      TableHeader,
      CodeBlockLowlight.configure({ lowlight }),
      Extension.create({
        name: "suggestion-decorations",
        addProseMirrorPlugins() {
          return [
            new Plugin({
              key: suggestionPluginKey,
              state: {
                init: () => DecorationSet.empty,
                apply: (tr, oldState) => {
                  const meta = tr.getMeta(suggestionPluginKey);
                  if (!meta || !meta.suggestions) return oldState;

                  const doc = tr.doc;
                  const index = _buildTextIndex(doc);
                  const decos: any[] = [];

                  try {
                    for (const s of meta.suggestions as Suggestion[]) {
                      if (s.start == null || s.end == null) continue;
                      const mapped = _mapTextRangeToPosRange(index, s.start, s.end);
                      if (!mapped) continue;

                      // Safety: only highlight if the extracted text matches the suggestion's original_text.
                      const extracted = doc.textBetween(mapped.from, mapped.to, "\n", "\n");
                      if (extracted !== s.original_text) continue;

                      decos.push(
                        Decoration.inline(mapped.from, mapped.to, {
                          style: highlightStyle,
                        })
                      );
                    }
                  } catch {
                    // If highlighting fails, keep old decorations.
                    return oldState;
                  }
                  return DecorationSet.create(doc, decos);
                },
              },
              props: {
                decorations(state) {
                  return suggestionPluginKey.getState(state) as DecorationSet;
                },
              },
            }),
          ];
        },
      }),
    ],
    []
  );

  const editor = useEditor({
    extensions,
    content,
    onUpdate({ editor }) {
      const text = editor.getText();
      onChange(text);
      const slashMatch = text.match(/\/(rewrite|summarize|expand|make-professional|create-outline)\s*$/);
      if (slashMatch && onSlashCommand) {
        const signature = `${slashMatch[1]}:${text.length}`;
        if (lastCommandRef.current !== signature) {
          lastCommandRef.current = signature;
          onSlashCommand(slashMatch[1], text);
        }
      }
    },
    editorProps: {
      attributes: {
        class: "ProseMirror"
      }
    }
  });

  useEffect(() => {
    suggestionsRef.current = suggestions || [];
    if (!editor) return;
    // Update decorations without altering the document.
    editor.view.dispatch(editor.state.tr.setMeta(suggestionPluginKey, { suggestions: suggestionsRef.current }));
  }, [editor, suggestions]);

  useEffect(() => {
    if (editor && !editorReady) {
      setEditorReady(true);
      setEditorInstance(editor);
    }
  }, [editor, editorReady]);

  useImperativeHandle(ref, () => ({
    applySuggestion: (s: Suggestion) => {
      if (!editor) return false;
      if (s.start == null || s.end == null) return false;

      try {
        const doc = editor.state.doc;
        const index = _buildTextIndex(doc);
        const mapped = _mapTextRangeToPosRange(index, s.start, s.end);
        if (!mapped) return false;

        const extracted = doc.textBetween(mapped.from, mapped.to, "\n", "\n");
        if (extracted !== s.original_text) return false;

        const tr = editor.state.tr;
        // Replace the exact matched span with plain text.
        // This keeps the surrounding document structure intact (we only touch the mapped range).
        tr.replaceWith(
          mapped.from,
          mapped.to,
          editor.state.schema.text(s.suggested_text)
        );
        editor.view.dispatch(tr);
        return true;
      } catch (e) {
        // If mapping fails or ProseMirror rejects the range, do not apply.
        // eslint-disable-next-line no-console
        console.warn("applySuggestion failed:", e);
        return false;
      }
    },
  }));

  return (
    <div>
      <Toolbar editor={editor} />
      <p className="mb-2 text-xs whitespace-pre-wrap text-slate-500">{slashHint}</p>
      <EditorContent editor={editor} />
    </div>
  );
});
