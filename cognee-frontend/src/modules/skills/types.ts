export interface Skill {
  id: string;
  name: string;
  description: string;
  /** Publishing company / team that maintains the skill. */
  maintainer: string;
  /** Maintainer homepage or repository URL. */
  maintainerUrl?: string;
  version?: string;
  tags: string[];
  license?: string;
  /** Tools the skill is allowed to invoke. */
  declaredTools: string[];
  /** Dataset UUIDs this skill is scoped to. */
  datasetScope: string[];
  isActive: boolean;
  sourceRepoUrl?: string;
  sourceDir?: string;
  /** Full skill instruction body. Only populated by the detail endpoint. */
  procedure?: string;
}

/** Raw shape returned by GET /v1/skills (snake_case from the cognee pod). */
export interface SkillRaw {
  id: string;
  name: string;
  description?: string;
  maintainer?: string;
  maintainer_url?: string;
  version?: string;
  tags?: string[];
  license?: string;
  declared_tools?: string[];
  dataset_scope?: string[];
  is_active?: boolean;
  source_repo_url?: string;
  source_dir?: string;
  /** Present only on GET /v1/skills/{id} (the detail endpoint). */
  procedure?: string;
}

export function mapSkill(raw: SkillRaw): Skill {
  return {
    id: raw.id,
    name: raw.name,
    description: raw.description ?? "",
    maintainer: raw.maintainer ?? "",
    maintainerUrl: raw.maintainer_url || undefined,
    version: raw.version || undefined,
    tags: Array.isArray(raw.tags) ? raw.tags : [],
    license: raw.license || undefined,
    declaredTools: Array.isArray(raw.declared_tools) ? raw.declared_tools : [],
    datasetScope: Array.isArray(raw.dataset_scope) ? raw.dataset_scope : [],
    isActive: raw.is_active ?? true,
    sourceRepoUrl: raw.source_repo_url || undefined,
    sourceDir: raw.source_dir || undefined,
    procedure: raw.procedure || undefined,
  };
}
