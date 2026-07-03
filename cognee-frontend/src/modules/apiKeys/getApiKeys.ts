import { CogneeInstance } from "../instances/types";

export interface ApiKey {
  id: string;
  key: string;
  label: string;
  name: string;
}

export default async function getApiKeys(instance: CogneeInstance): Promise<ApiKey[]> {
  const response = await instance.fetch("/auth/api-keys");
  if (!response.ok) return [];
  const data = await response.json();
  return Array.isArray(data) ? data : [];
}
