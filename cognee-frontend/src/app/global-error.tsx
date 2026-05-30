"use client";

export default function GlobalError({ reset }: { error: Error; reset: () => void }) {
  return (
    <html>
      <body>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", fontFamily: '"Inter", system-ui, sans-serif', gap: 16, padding: 32 }}>
          <h1 style={{ fontSize: 24, fontWeight: 600, color: "#18181B", margin: 0 }}>Something went wrong</h1>
          <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>An unexpected error occurred. Please try again.</p>
          <button
            onClick={reset}
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, cursor: "pointer", marginTop: 8 }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
