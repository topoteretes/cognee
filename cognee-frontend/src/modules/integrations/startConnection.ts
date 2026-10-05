"use server";

export interface StartConnectionResult {
  authorizeUrl?: string;
  error?: string;
}

const UNAVAILABLE =
  "Connecting a shared data source needs Cognee Cloud — it authorizes the workspace through the hosted control plane, which a self-hosted install does not run.";

/**
 * Open-source stub — team-scoped OAuth is brokered by the cloud control
 * plane, which holds the provider credentials and the per-workspace redirect.
 *
 * Returns the reason rather than an empty result: the caller opens a popup on
 * `authorizeUrl`, so a silently empty response would look like a connect
 * attempt that quietly did nothing.
 */
export default async function startConnection(
  _provider: string,
  _tenantId: string,
): Promise<StartConnectionResult> {
  return { error: UNAVAILABLE };
}
