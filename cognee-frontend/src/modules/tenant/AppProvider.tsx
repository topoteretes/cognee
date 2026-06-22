"use client";

import { LocalProvider } from "./LocalProvider";

/**
 * Open-source version — always uses LocalProvider.
 * The SaaS version conditionally routes to TenantProvider (cloud) or LocalProvider (local).
 */
export default function AppProvider({ children }: { children: React.ReactNode }) {
  return <LocalProvider>{children}</LocalProvider>;
}
