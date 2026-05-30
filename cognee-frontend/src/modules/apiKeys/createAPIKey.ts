import localFetch from "@/modules/instances/localFetch";

export interface CreatedApiKey {
  id: string;
  key: string;
  label?: string;
  name?: string;
}

export default async function createApiKey(
  options: { name?: string; noRedirectOnAuth?: boolean } = {},
): Promise<CreatedApiKey> {
  const response = await localFetch("/v1/auth/api-keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: options.name ?? null }),
  });
  return response.json();
}
