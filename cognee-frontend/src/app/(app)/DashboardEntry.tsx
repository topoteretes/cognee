"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import OverviewPage from "./dashboard/OverviewPage";

export default function DashboardEntry() {
  const router = useRouter();
  // Synchronous, no network round-trip: local mode is exactly one browser
  // talking to exactly one backend, so these flags reliably answer "has this
  // user ever finished (or skipped) onboarding". Brand new / freshly-wiped
  // installs go straight to /onboarding instead of mounting the dashboard
  // first, which would otherwise flash on screen while OverviewPage's own
  // async check (datasets/runs + /api/user-app-state) resolves.
  const [shouldShowDashboard] = useState(() => {
    if (typeof window === "undefined") return true;
    try {
      return (
        sessionStorage.getItem("cognee-onboarding-skipped") !== null ||
        localStorage.getItem("cognee-onboarding-complete") !== null
      );
    } catch {
      return true;
    }
  });

  useEffect(() => {
    if (!shouldShowDashboard) {
      router.replace("/onboarding");
    }
  }, [shouldShowDashboard, router]);

  // Renders nothing while redirecting — OverviewPage still re-validates this
  // decision against real backend data once it mounts (existing datasets/
  // runs + /api/user-app-state check), so a stale flag can't strand a
  // returning user with real data on /onboarding.
  if (!shouldShowDashboard) return null;
  return <OverviewPage />;
}
