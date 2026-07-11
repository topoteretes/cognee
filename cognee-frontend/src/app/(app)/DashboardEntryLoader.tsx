"use client";

import dynamicImport from "next/dynamic";

// ssr: false is load-bearing: DashboardEntry's redirect decision reads
// localStorage, which doesn't exist during server rendering. Without this,
// the server-rendered HTML falls back to "show the dashboard" (no window),
// so the browser paints a full dashboard for a moment before client JS
// hydrates, runs the real check, and swaps to /onboarding. next/dynamic's
// ssr:false option is only usable from a Client Component, hence this
// wrapper around the Server Component page.tsx.
const DashboardEntry = dynamicImport(() => import("./DashboardEntry"), { ssr: false });

export default function DashboardEntryLoader() {
  return <DashboardEntry />;
}
