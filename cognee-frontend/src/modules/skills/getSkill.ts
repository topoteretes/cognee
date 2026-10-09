import { CogneeInstance } from "../instances/types";
import { mapSkill, type Skill, type SkillRaw } from "./types";

/**
 * Fetch a single skill including its full `procedure` body.
 * Backed by GET /v1/skills/{skill_id}?dataset_id=... on the tenant cognee pod.
 * `dataset_id` is required by the backend even on the by-id lookup.
 */
export default function getSkill(
  instance: CogneeInstance,
  datasetId: string,
  skillId: string,
): Promise<Skill> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  return instance
    .fetch(`/v1/skills/${encodeURIComponent(skillId)}?${params.toString()}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json())
    .then((data: SkillRaw) => mapSkill(data));
}
