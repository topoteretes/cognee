export interface CreatedApiKey {
  id: string;
  key: string;
}

export default async function createApiKey(
  _options: { name?: string; noRedirectOnAuth?: boolean } = {},
): Promise<CreatedApiKey> {
  console.warn("API key creation requires Cognee Cloud.");
  return { id: "", key: "" };
}
