"use client";

/**
 * Slim sticky banner shown at the top of the app shell while the tenant pod is
 * still provisioning (`!tenantReady`). Derived from context per-render, so it
 * survives a page refresh and clears automatically when the pod comes online.
 *
 * Suppressed only where WorkspaceWaking is on screen saying the same thing.
 * On the routes that have no waking screen — settings, members, integrations,
 * billing and the rest — this banner is the only signal a wake is running, so
 * it stays and takes the wake wording instead.
 */
export default function ProvisioningBanner({ waking = false }: { waking?: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 16px",
        background: "rgba(188,155,255,0.12)",
        borderBottom: "1px solid rgba(188,155,255,0.25)",
        flexShrink: 0,
      }}
    >
      <style>{`@keyframes provbanner-spin { to { transform: rotate(360deg); } }`}</style>
      <div
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          border: "2px solid rgba(237,236,234,0.2)",
          borderTopColor: "#BC9BFF",
          animation: "provbanner-spin 0.8s linear infinite",
          flexShrink: 0,
        }}
      />
      <span role="status" aria-live="polite" style={{ fontSize: 13, color: "#EDECEA" }}>
        {waking
          ? "Waking your workspace — it went to sleep after a quiet spell. Nothing was lost, and it should be back in under a minute."
          : "Setting up your workspace — this can take a minute. Some features unlock once it's ready."}
      </span>
    </div>
  );
}
