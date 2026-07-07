import { CogneeInstance } from "../instances/types";
import type { Skill } from "./types";

export interface RememberSkillResponse {
  status: string;
  dataset_name: string | null;
  dataset_id: string | null;
  pipeline_run_id: string | null;
  error?: string;
}

/**
 * Ingest a single skill into ONE dataset as a dataset-scoped Skill node, via
 * POST /v1/remember with content_type="skills".
 *
 * The cognee parser only picks up files literally named `SKILL.md`, and derives
 * the skill's name from the file's PARENT directory. So we always send the
 * content under `<slug>/SKILL.md` — that guarantees it's parsed and named `slug`.
 *
 * The remember endpoint accepts one dataset per call, so attaching a skill to
 * multiple datasets means calling this once per dataset (see SkillUploadModal).
 */
export default async function rememberSkill(
  datasetId: string,
  slug: string,
  content: Blob,
  instance: CogneeInstance,
): Promise<RememberSkillResponse> {
  const formData = new FormData();
  // Third arg sets the multipart filename regardless of the source File's name.
  formData.append("data", content, `${slug}/SKILL.md`);
  formData.append("datasetId", datasetId);
  formData.append("content_type", "skills");
  // Skills ingestion is quick; run synchronously so the UI can refresh on success.
  formData.append("run_in_background", "false");

  const response = await instance.fetch("/v1/remember", {
    method: "POST",
    body: formData,
  });
  const body = await response.json();
  if (!response.ok || body?.error) {
    throw new Error(body?.error || `Skill ingestion failed (HTTP ${response.status})`);
  }
  return body as RememberSkillResponse;
}

/** Slugify a skill name into a safe directory name used as the skill's identity. */
export function slugifySkillName(name: string): string {
  return (
    name
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "skill"
  );
}

/**
 * Reconstruct a SKILL.md (YAML frontmatter + procedure body) from an existing
 * skill, so it can be re-ingested into another dataset. Used by the share /
 * "add to brains" action — the cognee parser round-trips these field aliases.
 * Values are JSON-encoded, which is valid YAML for scalars and flow sequences.
 */
export function skillToMarkdown(skill: Skill): string {
  const lines: string[] = [];
  if (skill.description) lines.push(`description: ${JSON.stringify(skill.description)}`);
  if (skill.declaredTools.length) lines.push(`allowed-tools: [${skill.declaredTools.map((t) => JSON.stringify(t)).join(", ")}]`);
  if (skill.maintainer) lines.push(`maintainer: ${JSON.stringify(skill.maintainer)}`);
  if (skill.maintainerUrl) lines.push(`maintainer_url: ${JSON.stringify(skill.maintainerUrl)}`);
  if (skill.version) lines.push(`version: ${JSON.stringify(skill.version)}`);
  if (skill.tags.length) lines.push(`tags: [${skill.tags.map((t) => JSON.stringify(t)).join(", ")}]`);
  if (skill.license) lines.push(`license: ${JSON.stringify(skill.license)}`);
  const frontmatter = lines.length ? `---\n${lines.join("\n")}\n---\n\n` : "";
  return `${frontmatter}# ${skill.name}\n\n${skill.procedure ?? ""}`;
}
