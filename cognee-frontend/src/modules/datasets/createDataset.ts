import { CogneeInstance } from "../instances/types";

export default async function createDataset(
  dataset: { name: string },
  instance: CogneeInstance,
  tenantId?: string | null,
  signal?: AbortSignal,
) {
  const response = await instance.fetch(`/v1/datasets/`, {
    method: "POST",
    body: JSON.stringify(dataset),
    headers: { "Content-Type": "application/json" },
    signal,
  });
  const created = await response.json();

  // Grant tenant-level read+write so all tenant members can see this dataset
  if (tenantId && created.id) {
    const body = JSON.stringify([created.id]);
    try {
      await Promise.all([
        instance.fetch(`/v1/permissions/datasets/${tenantId}?permission_name=read`, {
          method: "POST",
          body,
          headers: { "Content-Type": "application/json" },
          signal,
        }),
        instance.fetch(`/v1/permissions/datasets/${tenantId}?permission_name=write`, {
          method: "POST",
          body,
          headers: { "Content-Type": "application/json" },
          signal,
        }),
      ]);
    } catch {
      // Non-fatal — dataset was created, just not shared yet
    }
  }

  return created;
}
