"use server";

export interface LinkSlackResult {
  success: boolean;
  error?: string;
}

const UNAVAILABLE =
  "Linking a Slack account to a Cognee login is handled by Cognee Cloud, which issues and redeems the /cognee-link code.";

/** Open-source stub — see startConnection.ts. */
export default async function linkSlackAccount(
  _tenantId: string,
  _code: string,
): Promise<LinkSlackResult> {
  return { success: false, error: UNAVAILABLE };
}
