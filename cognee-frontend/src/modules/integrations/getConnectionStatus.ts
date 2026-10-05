"use server";

export interface ConnectionStatus {
  connected: boolean;
  failed: boolean;
  teamName?: string;
  accountId?: string;
  connectedByUserId?: string;
  connectedAt?: string;
  syncStatus?: string;
  lastSyncedAt?: string;
  viaRouting?: boolean;
  routedTeamName?: string;
  routedChannelCount?: number;
}

/**
 * Open-source stub — connector connection status requires the cloud
 * management API. Always reports disconnected so the dashboard memory graph
 * simply shows every data source as not yet connected in OSS mode.
 */
export default async function getConnectionStatus(
  _provider: string,
  _tenantId: string,
): Promise<ConnectionStatus> {
  return { connected: false, failed: false };
}
