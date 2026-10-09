export interface CreatedApiKey {
  id: string;
  key: string;
}

// Mirrors the SaaS module's result shape — the synced ApiKeysPage and
// CreateApiKeyButton read `.ok`/`.error`/`.key` and must not care which build
// they're running in.
export type CreateApiKeyResult =
  | { ok: true; key: CreatedApiKey | null }
  | { ok: false; error: string; status?: number };

export default async function createApiKey(
  _options: { name?: string; noRedirectOnAuth?: boolean } = {},
): Promise<CreateApiKeyResult> {
  console.warn("API key creation requires Cognee Cloud.");
  return { ok: false, error: "API key creation requires Cognee Cloud." };
}
