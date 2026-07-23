import type { CogneeInstance } from "@/modules/instances/types";
import { loadGraphModelsConfig, saveCustomPrompt, assignPromptToDataset, clearDatasetOutdated } from "@/modules/configuration/userConfiguration";

export type CreateBrainTemplateKey = "documents" | "support-tickets" | "legal";

export interface CreateBrainTemplate {
  key: CreateBrainTemplateKey;
  label: string;
  // Name the prompt is saved under — shared/reusable across every brain that
  // picks this template, same as any other named custom prompt.
  promptName: string;
  promptText: string;
}

export const CREATE_BRAIN_TEMPLATES: CreateBrainTemplate[] = [
  {
    key: "documents",
    label: "Documents",
    promptName: "Documents template",
    promptText:
      "Extract the key entities, concepts, and facts from this document. Identify people, organizations, dates, and the relationships between them. Keep entity names consistent across the text (use the same identifier for an entity every time it's mentioned).",
  },
  {
    key: "support-tickets",
    label: "Support tickets",
    promptName: "Support tickets template",
    promptText:
      "Extract customers, agents, products, and the issues being reported. For each issue, capture its status, root cause if mentioned, and how it was resolved. Link issues to the product or feature they relate to, and to the customer who reported them.",
  },
  {
    key: "legal",
    label: "Legal",
    promptName: "Legal template",
    promptText:
      "Extract the parties, contracts, obligations, and key dates (effective date, termination date, deadlines) from this document. Capture clauses and the obligations they create, and link each obligation to the party responsible for it.",
  },
];

// Applies a template's prompt to a newly created dataset: saves the template's
// default prompt text under its shared name (without overwriting it if a user
// already customized a prompt with that same name), then assigns it to the
// dataset. Used right after brain creation — a lightweight alternative to a
// full predefined graph model schema (CLO-292).
export async function applyCreateBrainTemplate(
  instance: CogneeInstance,
  datasetId: string,
  templateKey: CreateBrainTemplateKey,
): Promise<void> {
  const template = CREATE_BRAIN_TEMPLATES.find((t) => t.key === templateKey);
  if (!template) return;

  const cfg = await loadGraphModelsConfig(instance);
  if (!cfg.customPrompts?.[template.promptName]) {
    await saveCustomPrompt(instance, template.promptName, template.promptText);
  }
  await assignPromptToDataset(instance, datasetId, template.promptName);
  // assignPromptToDataset unconditionally flags the dataset "outdated" — right
  // for the interactive "change prompt on a brain that already has a build"
  // flow, wrong here: a brand-new, never-built dataset can't have an outdated
  // graph. Clear it right back so the detail page doesn't show a misleading
  // "rebuild graph" banner before the first upload.
  await clearDatasetOutdated(instance, datasetId);
}
