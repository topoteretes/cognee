import createApiKey from "./createAPIKey";
import getApiKeys from "./getApiKeys";

export default async function getOrCreateApiKey(): Promise<string> {
  const keys = await getApiKeys();
  const existing = keys.find((key) => key.key && !key.key.includes("*"));
  if (existing) return existing.key;

  const created = await createApiKey({ name: "Local dev" });
  return created.key;
}
