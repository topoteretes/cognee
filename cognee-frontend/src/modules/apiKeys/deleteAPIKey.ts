import { CogneeInstance } from "../instances/types";

export default async function deleteApiKey(instance: CogneeInstance, keyId: string): Promise<void> {
  const response = await instance.fetch(`/auth/api-keys/${keyId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Failed to delete API key (${response.status})`);
  }
}
