import localFetch from "@/modules/instances/localFetch";

export interface ApiKey {
  id: string;
  key: string;
  label: string;
  name: string;
}

export default async function getApiKeys(): Promise<ApiKey[]> {
  try {
    const response = await localFetch("/v1/auth/api-keys");
    if (!response.ok) return [];
    const data = await response.json();
    if (!Array.isArray(data)) return [];
    return data.map((item: Record<string, unknown>) => ({
      id: String(item.id ?? ""),
      key: String(item.key ?? ""),
      label: String(item.label ?? ""),
      name: String(item.name ?? ""),
    }));
  } catch {
    return [];
  }
}
