"use client";

import DashboardSkeleton from "@/app/(app)/dashboard/DashboardSkeleton";

// Full-screen wrapper shown right after workspace creation, while
// TenantProvider polls for the new tenant record before switchTenant's hard
// navigation lands on /dashboard. Renders the exact same DashboardSkeleton
// the destination page shows while its pod isn't ready yet, inside a
// background that matches CustomAppShell's — so the wait reads as one
// continuous screen across the navigation instead of two different loading
// treatments (a spinner here, a skeleton there) back to back (CLO-244).
export default function WorkspaceSetupScreen(): React.JSX.Element {
  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#000000",
        backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
      }}
    >
      <DashboardSkeleton />
    </div>
  );
}
