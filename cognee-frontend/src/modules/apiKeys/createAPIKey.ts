import { CogneeInstance } from "../instances/types";

export interface CreatedApiKey {
  id: string;
  key: string;
}

export default async function createApiKey(
  instance: CogneeInstance,
  options: { name?: string } = {},
): Promise<CreatedApiKey> {
  const response = await instance.fetch("/auth/api-keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: options.name ?? null }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create API key (${response.status})`);
  }
  return response.json();
}
