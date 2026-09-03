"use client";

interface BusinessErrorFallbackProps {
  resetErrorBoundary: () => void;
}

// Matches src/app/error.tsx's look, scoped to just the view instead of the
// whole page — the canvas RAF loop and force simulation are prone to
// edge-case runtime errors (NaN coordinates, division by zero on an empty
// bbox), and without this boundary one of those crashed the entire
// /business or /knowledge-graph route instead of a recoverable panel.
export default function BusinessErrorFallback({ resetErrorBoundary }: BusinessErrorFallbackProps): React.JSX.Element {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        gap: 16,
        padding: 32,
        backgroundColor: "#000000",
      }}
    >
      <h1 style={{ fontSize: 14, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Failed to load business view</h1>
      <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0, textAlign: "center", maxWidth: 400 }}>
        Something went wrong rendering the graph. Please try again.
      </p>
      <button
        onClick={resetErrorBoundary}
        style={{
          background: "#6510F4",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          padding: "8px 20px",
          fontSize: 14,
          fontWeight: 500,
          cursor: "pointer",
          marginTop: 8,
        }}
      >
        Try again
      </button>
    </div>
  );
}
