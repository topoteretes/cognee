"use client";

// Full-screen "Setting up workspace" screen — shown after Stripe checkout
// returns while TenantProvider polls for the new tenant in the background.
export default function WorkspaceSetupScreen({ name }: { name: string }): React.JSX.Element {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        background: "#0a0a0a",
        gap: "1.5rem",
      }}
    >
      <style>{`
        @keyframes ws-spin { to { transform: rotate(360deg); } }
        @keyframes ws-pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
      `}</style>

      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: "50%",
          border: "3px solid rgba(188,155,255,0.2)",
          borderTopColor: "#bc9bff",
          animation: "ws-spin 0.9s linear infinite",
        }}
      />

      <div style={{ textAlign: "center", maxWidth: 320 }}>
        <p
          style={{
            margin: "0 0 0.5rem",
            fontSize: "1.125rem",
            fontWeight: 600,
            color: "#EDECEA",
            fontFamily: '"TWKLausanne", sans-serif',
          }}
        >
          Creating your new workspace{name ? ` "${name}"` : ""}
        </p>
        <p
          style={{
            margin: 0,
            fontSize: "0.875rem",
            color: "rgba(237,236,234,0.45)",
            animation: "ws-pulse 2s ease-in-out infinite",
          }}
        >
          This takes just a moment…
        </p>
      </div>
    </div>
  );
}
