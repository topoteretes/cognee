"use server";

export interface DisconnectResult {
  success: boolean;
  error?: string;
}

const UNAVAILABLE =
  "Shared data-source connections live in Cognee Cloud, so there is nothing to disconnect here.";

/** Open-source stub — see startConnection.ts. */
export default async function disconnectConnection(
  _provider: string,
  _tenantId: string,
): Promise<DisconnectResult> {
  return { success: false, error: UNAVAILABLE };
}
