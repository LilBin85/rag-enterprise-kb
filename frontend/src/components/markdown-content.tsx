import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function normalizeMarkdownTables(markdown: string) {
  return markdown
    .replace(
      /\s*(\|\s*:?-{3,}\s*\|\s*:?-{3,}\s*\|)\s*/g,
      "\n$1\n",
    )
    .replace(/\|\s+\|(?=\s*\*\*[^*\n]+\*\*\s*\|)/g, "|\n|");
}

export function MarkdownContent({ children }: { children: string }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>
        {normalizeMarkdownTables(children)}
      </ReactMarkdown>
    </div>
  );
}
