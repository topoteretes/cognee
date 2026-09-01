"use server";

export interface DeleteChannelRouteResult {
  success: boolean;
  error?: string;
}

const UNAVAILABLE =
  "Channel routing needs Cognee Cloud, so there is no route to remove here.";

/** Open-source stub — see getChannelRoutes.ts. */
export default async function deleteChannelRoute(
  _provider: string,
  _tenantId: string,
  _resourceId: string,
): Promise<DeleteChannelRouteResult> {
  return { success: false, error: UNAVAILABLE };
}
