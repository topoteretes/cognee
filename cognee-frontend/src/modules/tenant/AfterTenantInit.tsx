"use client";

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { NEW_ONBOARDING_PATH } from "@/utils/onboardingPaths";
import { isSessionlessPath } from "@/utils/sessionRoutes";
import { useTenant } from "./TenantContext";

// Routes that render before tenant init settles: the dashboard brings its own
// skeleton, and the first-run screens are what init waits behind.
const RENDERS_DURING_INIT_PATHS = ["/dashboard", "/welcome", "/onboarding", NEW_ONBOARDING_PATH];

/**
 * Holds its children back until tenant init has settled, on the routes where
 * TenantProvider used to hold the whole app back (removed in CLO-244).
 *
 * For code that reads what init writes rather than waiting for it: the page
 * view tracker reads the selected tenant and plan type from sessionStorage
 * once, so mounted before init it reports the page without them. Wrap only
 * such readers; the app itself no longer waits.
 */
export default function AfterTenantInit({ children }: { children: ReactNode }): ReactNode {
  const pathname = usePathname();
  const { isInitializing } = useTenant();

  const waits = isInitializing && !isSessionlessPath(pathname) && !RENDERS_DURING_INIT_PATHS.includes(pathname);

  return waits ? null : children;
}
