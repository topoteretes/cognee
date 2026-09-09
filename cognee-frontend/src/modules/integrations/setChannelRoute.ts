"use server";

export interface SetChannelRouteResult {
  success: boolean;
  error?: string;
}

const UNAVAILABLE =
  "Routing a channel to another workspace needs Cognee Cloud — a self-hosted install has a single workspace to route to.";

/** Open-source stub — see getChannelRoutes.ts. */
export default async function setChannelRoute(
  _provider: string,
  _tenantId: string,
  _resourceId: string,
  _resourceName: string,
  _targetTenantId: string,
): Promise<SetChannelRouteResult> {
  return { success: false, error: UNAVAILABLE };
}
