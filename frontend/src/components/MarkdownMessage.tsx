import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
}

/**
 * Renders an AG-UI TEXT_MESSAGE_CHUNK payload as formatted markdown.
 *
 * AG-UI has no native markdown rendering — the protocol streams plain text and
 * leaves presentation to the frontend. This component wraps react-markdown with
 * the remark-gfm plugin so GitHub-flavored markdown (tables, task lists,
 * strikethrough) renders correctly alongside standard markdown (code blocks,
 * bold, lists, links).
 */
export function MarkdownMessage({ content }: Props) {
  return (
    <div className="agent-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
