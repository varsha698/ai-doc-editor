import { Editor } from "@tiptap/react";

type Props = {
  editor: Editor | null;
};

export default function Toolbar({ editor }: Props) {
  if (!editor) return null;

  return (
    <div className="mb-3 flex flex-wrap gap-2">
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleBold().run()}>
        Bold
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleItalic().run()}>
        Italic
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleBulletList().run()}>
        Bullet List
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleOrderedList().run()}>
        Numbered List
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleCodeBlock().run()}>
        Code
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().setParagraph().run()}>
        Paragraph
      </button>
      <button className="rounded border px-2 py-1" onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}>
        H2
      </button>
    </div>
  );
}
