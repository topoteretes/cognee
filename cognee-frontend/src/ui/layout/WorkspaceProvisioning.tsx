"use client";

import SkeletonBar from "@/ui/elements/SkeletonBar";

/**
 * Shared placeholder for pod-dependent pages while the tenant workspace is still
 * provisioning. Rendered by CustomAppShell in place of the route's children so
 * those pages never show broken/empty data against an unready pod.
 *
 * A workspace being WOKEN gets WorkspaceWaking instead: it already exists, so
 * "being set up" would be wrong there.
 */
export default function WorkspaceProvisioning() {
  return (
    <div
      // An indeterminate ~1 minute wait otherwise leaves a screen reader on the
      // previous page's content with no indication anything is happening.
      // role="status" (rather than only aria-live) is what gets this announced
      // at all: the region mounts already populated, and a bare live region is
      // only reliably announced when its content changes after mount.
      role="status"
      aria-live="polite"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        gap: 16,
        padding: 48,
        textAlign: "center",
      }}
    >
      <style>{`@keyframes wsprov-spin { to { transform: rotate(360deg); } }`}</style>
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          border: "3px solid rgba(237,236,234,0.12)",
          borderTopColor: "#BC9BFF",
          animation: "wsprov-spin 0.8s linear infinite",
        }}
      />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <SkeletonBar width={220} height={8} />
        <SkeletonBar width={160} height={8} />
      </div>
      <p style={{ margin: 0, fontSize: 14, color: "rgba(237,236,234,0.6)", maxWidth: 360, lineHeight: "22px" }}>
        Your workspace is being set up — usually takes under a minute.
        This page unlocks as soon as it&apos;s ready.
      </p>
    </div>
  );
}
