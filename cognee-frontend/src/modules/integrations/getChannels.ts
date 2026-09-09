"use server";

export interface IntegrationChannel {
  id: string;
  name: string;
  isPrivate: boolean;
  allowed: boolean;
}

export interface ChannelsResult {
  channels: IntegrationChannel[];
  /** Set when the list could not be read; `channels` is then empty. */
  error: string | null;
}

const UNAVAILABLE =
  "Channel lists come from a connected workspace in Cognee Cloud, which a self-hosted install does not have.";

/**
 * Open-source stub — the channel list is read from the provider through the
 * cloud control plane's stored connection.
 *
 * Reports the reason in `error` rather than returning an empty list: callers
 * treat an empty list as "connected, no channels yet", which is a different
 * and wrong story to tell here.
 */
export default async function getChannels(
  _provider: string,
  _tenantId: string,
): Promise<ChannelsResult> {
  return { channels: [], error: UNAVAILABLE };
}
