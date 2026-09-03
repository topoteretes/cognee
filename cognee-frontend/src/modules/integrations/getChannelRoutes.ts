"use server";

export interface ChannelRoute {
  resourceId: string;
  resourceName: string | null;
  tenantId: string;
}

/**
 * Open-source stub — routing a channel to one of several workspaces is a
 * multi-tenant concept; a self-hosted install has exactly one, so the honest
 * answer is that no channel is routed anywhere. Unlike the calls above this
 * needs no error channel: "no routes" is true here, not a failed read.
 */
export default async function getChannelRoutes(
  _provider: string,
  _tenantId: string,
): Promise<ChannelRoute[]> {
  return [];
}
