import { notifySurveyTrigger } from "@/services/survey/surveyTriggerBridge";
import { CogneeInstance } from "../instances/types";

export default async function createDataset(
  dataset: { name: string },
  instance: CogneeInstance,
  signal?: AbortSignal,
) {
  const response = await instance.fetch(`/v1/datasets/`, {
    method: "POST",
    body: JSON.stringify(dataset),
    headers: { "Content-Type": "application/json" },
    signal,
  });
  const created = await response.json();

  notifySurveyTrigger("datasource_added");

  return created;
}

/**
 * Give every member of the workspace read and write on a dataset, so the whole
 * team sees it. Apart from createDataset on purpose: the dataset exists, and is
 * usable by its creator, as soon as it is created, while each of these grants is
 * a pod call that takes seconds. Best effort: a failure leaves the dataset
 * created and just not shared yet.
 */
export async function shareDatasetWithWorkspace(
  datasetId: string,
  tenantId: string,
  instance: CogneeInstance,
  signal?: AbortSignal,
): Promise<void> {
  const body = JSON.stringify([datasetId]);
  const grant = (permission: "read" | "write") =>
    instance.fetch(`/v1/permissions/datasets/${tenantId}?permission_name=${permission}`, {
      method: "POST",
      body,
      headers: { "Content-Type": "application/json" },
      signal,
    });
  try {
    await Promise.all([grant("read"), grant("write")]);
  } catch (err) {
    console.error("Failed to share the new dataset with the workspace:", err);
  }
}
