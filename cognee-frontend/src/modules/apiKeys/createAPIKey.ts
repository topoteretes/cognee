import localFetch from "@/modules/instances/localFetch";

export default async function createApiKey(
  options?: { name?: string; noRedirectOnAuth?: boolean } | string,
): Promise<string> {
  const name = typeof options === "string" ? options : options?.name || null;

  const response = await localFetch("/v1/auth/api-keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    const err = await response.text().catch(() => "Failed to create API key");
    throw new Error(err);
  }

  const data = await response.json();
  return data.api_key ?? data.token ?? "";
}
