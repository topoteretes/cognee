import { CogneeInstance } from "../instances/types";
import { mapSkill, type Skill, type SkillRaw } from "./types";

/**
 * The server caps `GET /v1/skills` at 200 skills unless asked otherwise, and
 * rejects anything above 1000 (`limit: int = Query(default=200, ge=1, le=1000)`
 * in `get_skills_router.py`). A request without a limit therefore returns a
 * page, not the dataset — ask for the largest page the server will serve, and
 * compare the result against `total` to know whether it was the whole thing.
 */
export const SKILLS_PAGE_LIMIT = 1000;

export interface SkillsPage {
  skills: Skill[];
  /** Skills in the dataset, ignoring pagination. `skills.length` is a page size. */
  total: number;
}

/**
 * List the skills available in a dataset, with publisher metadata.
 * Backed by GET /v1/skills?dataset_id=... on the tenant cognee pod.
 */
export default function getSkills(
  instance: CogneeInstance,
  datasetId: string,
  includeInactive = false,
): Promise<SkillsPage> {
  const params = new URLSearchParams({
    dataset_id: datasetId,
    limit: String(SKILLS_PAGE_LIMIT),
  });
  if (includeInactive) params.set("include_inactive", "true");

  const page = instance
    .fetch(`/v1/skills/?${params.toString()}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json())
    .then((data: SkillRaw[]) => (Array.isArray(data) ? data.map(mapSkill) : []));

  return Promise.all([page, getSkillsCount(instance, datasetId, includeInactive)])
    .then(([skills, total]) => ({ skills, total: Math.max(total, skills.length) }));
}

/**
 * Number of skills in a dataset.
 *
 * A separate call because the list endpoint's length saturates at the limit: a
 * page of exactly 1000 rows and a dataset of exactly 1000 skills look the same
 * from the list alone, which is how the page came to claim "200 skills" for a
 * dataset holding 378.
 */
export function getSkillsCount(
  instance: CogneeInstance,
  datasetId: string,
  includeInactive = false,
): Promise<number> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  if (includeInactive) params.set("include_inactive", "true");

  return instance
    .fetch(`/v1/skills/count?${params.toString()}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json())
    .then((body: { count?: number }) =>
      Number.isSafeInteger(body?.count) && body.count! >= 0 ? body.count! : 0,
    )
    // An older pod has no /count route. Fall back to "no separate total known"
    // rather than failing the whole listing.
    .catch(() => 0);
}
