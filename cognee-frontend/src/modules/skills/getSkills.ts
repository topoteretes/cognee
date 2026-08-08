import { CogneeInstance } from "../instances/types";
import { mapSkill, type Skill, type SkillRaw } from "./types";

/**
 * List the skills available in a dataset, with publisher metadata.
 * Backed by GET /v1/skills?dataset_id=... on the tenant cognee pod.
 */
export default function getSkills(
  instance: CogneeInstance,
  datasetId: string,
  includeInactive = false,
): Promise<Skill[]> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  if (includeInactive) params.set("include_inactive", "true");

  return instance
    .fetch(`/v1/skills/?${params.toString()}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json())
    .then((data: SkillRaw[]) => (Array.isArray(data) ? data.map(mapSkill) : []));
}
