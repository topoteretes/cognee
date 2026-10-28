import ReactMarkdown from "react-markdown";

const MONO_FONT = 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace';

export default function MarkdownContent({ text }: { text: string }) {
  return (
    <div style={{ fontSize: 14, color: "#EDECEA", lineHeight: "22px", wordBreak: "break-word", minWidth: 0 }}>
      <ReactMarkdown
        components={{
          p: ({ children }) => (
            <p style={{ margin: "0 0 10px", fontSize: 14, lineHeight: "22px", color: "#EDECEA" }}>{children}</p>
          ),
          h1: ({ children }) => (
            <h1 style={{ fontSize: 18, fontWeight: 700, margin: "0 0 8px", color: "#EDECEA", lineHeight: "26px" }}>{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px", color: "#EDECEA", lineHeight: "24px" }}>{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 style={{ fontSize: 14, fontWeight: 700, margin: "0 0 6px", color: "#EDECEA" }}>{children}</h3>
          ),
          strong: ({ children }) => (
            <strong style={{ fontWeight: 700, color: "#EDECEA" }}>{children}</strong>
          ),
          em: ({ children }) => (
            <em style={{ fontStyle: "italic" }}>{children}</em>
          ),
          code: ({ className, children }) => {
            const isBlock = Boolean(className?.startsWith("language-"));
            if (isBlock) {
              return (
                <code style={{
                  display: "block", background: "rgba(0,0,0,0.4)", color: "#CDD6F4",
                  padding: "12px 16px", fontSize: 12,
                  fontFamily: MONO_FONT,
                  overflowX: "auto", lineHeight: "20px", whiteSpace: "pre",
                }}>
                  {children}
                </code>
              );
            }
            return (
              <code style={{
                background: "rgba(255,255,255,0.1)", borderRadius: 3, padding: "1px 5px",
                fontSize: 12, fontFamily: MONO_FONT, color: "#CBA6F7",
              }}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre style={{ margin: "0 0 10px", borderRadius: 8, overflow: "hidden", background: "rgba(0,0,0,0.4)" }}>{children}</pre>
          ),
          ul: ({ children }) => (
            <ul style={{ margin: "0 0 10px", paddingLeft: 20, lineHeight: "22px" }}>{children}</ul>
          ),
          ol: ({ children }) => (
            <ol style={{ margin: "0 0 10px", paddingLeft: 20, lineHeight: "22px" }}>{children}</ol>
          ),
          li: ({ children }) => (
            <li style={{ fontSize: 14, color: "#EDECEA", marginBottom: 4 }}>{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote style={{ borderLeft: "3px solid rgba(188,155,255,0.35)", paddingLeft: 12, margin: "0 0 10px", color: "rgba(237,236,234,0.6)" }}>{children}</blockquote>
          ),
          hr: () => (
            <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.1)", margin: "12px 0" }} />
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "#BC9BFF", textDecoration: "underline" }}>{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
