/**
 * Open-source stub — API keys are managed via the local backend directly.
 */

export interface ApiKey {
  id: string;
  api_key: string;
  label: string;
  name?: string;
}

export default async function getApiKeys(_instance?: unknown): Promise<ApiKey[]> {
  return [];
}
