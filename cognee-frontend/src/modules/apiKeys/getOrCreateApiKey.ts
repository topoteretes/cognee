/**
 * Open-source stub — API key management not available in local mode.
 *
 * Mirrors the real action's discriminated-union return (see
 * modules/apiKeys/getOrCreateApiKey.ts) rather than its old Promise<string>
 * shape, so this stays a faithful stand-in even though every consumer of the
 * real action is itself excluded from the public sync today.
 */
type GetOrCreateApiKeyResult =
  | { ok: true; key: string }
  | { ok: false; error: string; status?: number };

export default async function getOrCreateApiKey(): Promise<GetOrCreateApiKeyResult> {
  return { ok: true, key: "" };
}
