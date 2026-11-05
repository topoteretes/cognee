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
  return getAllSkills(instance, datasetId, includeInactive);
}

const SKILLS_PAGE_SIZE = 1000;

async function getAllSkills(
  instance: CogneeInstance,
  datasetId: string,
  includeInactive: boolean,
): Promise<Skill[]> {
  const skills: Skill[] = [];
  const seen = new Set<string>();
  let offset = 0;

  for (;;) {
    const params = new URLSearchParams({
      dataset_id: datasetId,
      limit: String(SKILLS_PAGE_SIZE),
      offset: String(offset),
    });
    if (includeInactive) params.set("include_inactive", "true");

    const response = await instance.fetch(`/v1/skills/?${params.toString()}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    const data: unknown = await response.json();
    if (!Array.isArray(data)) return [];

    const page = data as SkillRaw[];
    if (page.length > SKILLS_PAGE_SIZE) {
      throw new Error("Server returned more skills than requested");
    }

    if (page.length > 0 && page.every((raw) => seen.has(raw.id))) {
      throw new Error("Server did not honor skills pagination");
    }

    for (const raw of page) {
      if (!seen.has(raw.id)) {
        seen.add(raw.id);
        skills.push(mapSkill(raw));
      }
    }

    offset += page.length;
    if (page.length < SKILLS_PAGE_SIZE) return skills;
  }
}
