import localFetch from "@/modules/instances/localFetch";

export default async function deleteApiKey(keyId: string): Promise<void> {
  const response = await localFetch(`/v1/auth/api-keys/${keyId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const err = await response.text().catch(() => "Failed to delete API key");
    throw new Error(err);
  }
}
