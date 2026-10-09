"use client";

/**
 * Slim sticky banner shown at the top of the app shell while the tenant pod is
 * still provisioning (`!tenantReady`). Derived from context per-render, so it
 * survives a page refresh and clears automatically when the pod comes online.
 */
export default function ProvisioningBanner() {
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
      <span style={{ fontSize: 13, color: "#EDECEA" }}>
        Setting up your workspace — this can take a minute. Some features unlock once it&apos;s ready.
      </span>
    </div>
  );
}
