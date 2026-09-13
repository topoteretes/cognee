// Bridges the pod HTTP client's error interceptor (plain module code, no React)
// to the InsufficientCreditsProvider (a React tree, mounted once). Mirrors the
// existing activeLogger/setLogger pattern in services/http/pod.ts rather than a
// generic multi-subscriber bus, since there is only ever one active listener.
export type InsufficientCreditsEvent = {
  operation: string | null;
  message: string;
  at: number;
  // Extracted from the failing request's URL, not read from "whichever tenant
  // is active now" — a 402 can arrive after the user has switched workspaces,
  // and the persistent notice must be attributed to the tenant that actually
  // failed, not whoever is active when the response lands.
  tenantId: string | null;
};

export type InsufficientCreditsListener = (event: InsufficientCreditsEvent) => void;

let activeListener: InsufficientCreditsListener | null = null;

export function setInsufficientCreditsListener(listener: InsufficientCreditsListener | null): void {
  activeListener = listener;
}

export function notifyInsufficientCredits(event: InsufficientCreditsEvent): void {
  try {
    activeListener?.(event);
  } catch {
    // Listener must never crash the request that triggered it.
  }
}
