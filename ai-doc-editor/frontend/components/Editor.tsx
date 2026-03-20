import { useMemo, useRef } from "react";
import { EditorContent, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Link from "@tiptap/extension-link";
import Table from "@tiptap/extension-table";
import TableRow from "@tiptap/extension-table-row";
import TableCell from "@tiptap/extension-table-cell";
import TableHeader from "@tiptap/extension-table-header";
import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import { common, createLowlight } from "lowlight";
import Toolbar from "@/components/Toolbar";

const lowlight = createLowlight(common);

type Props = {
  content: string;
  onChange: (text: string) => void;
  onSlashCommand?: (command: string, text: string) => void;
};

const slashHint = `Type slash commands:
/rewrite
/summarize
/expand
/make-professional
/create-outline`;

export default function Editor({ content, onChange, onSlashCommand }: Props) {
  const lastCommandRef = useRef<string>("");
  const extensions = useMemo(
    () => [
      StarterKit,
      Link.configure({ openOnClick: true }),
      Table.configure({ resizable: true }),
      TableRow,
      TableCell,
      TableHeader,
      CodeBlockLowlight.configure({ lowlight })
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

  return (
    <div>
      <Toolbar editor={editor} />
      <p className="mb-2 text-xs whitespace-pre-wrap text-slate-500">{slashHint}</p>
      <EditorContent editor={editor} />
    </div>
  );
}
