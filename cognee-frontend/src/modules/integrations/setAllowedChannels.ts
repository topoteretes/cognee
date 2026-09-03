"use server";

export interface SetAllowedChannelsResult {
  success: boolean;
  error?: string;
}

const UNAVAILABLE =
  "Choosing which channels to ingest needs a workspace connected through Cognee Cloud.";

/** Open-source stub — see getChannels.ts. */
export default async function setAllowedChannels(
  _provider: string,
  _tenantId: string,
  _channelIds: string[],
): Promise<SetAllowedChannelsResult> {
  return { success: false, error: UNAVAILABLE };
}
