// Public copy of the SaaS module of the same path. The directory is excluded
// from the sync (it contains cloud-only billing/config actions), but this file
// itself is portable and its consumers are shared UI — keep it compatible with
// the SaaS original; the sync build gate fails if the signatures drift.
import type { CogneeInstance } from "@/modules/instances/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { loadGraphModelsConfig, findModelForDataset, findPromptForDataset, findOntologyForDataset } from "./userConfiguration";

export interface CognifyOptions {
  graphModel?: object;
  customPrompt?: string;
  ontologyKey?: string[];
}

// Loads the graph model / custom prompt / ontology assigned to a dataset from
// the shared backend config and converts them into cognifyDataset's expected
// shape. Used so every upload path applies the dataset's saved customization
// (CLO-292): before this, only the detail page — which keeps that state
// loaded locally for its MemoryCustomizationBar — passed it through; the
// brains list page uploaded with default options regardless of what was
// configured for the dataset.
export async function buildCognifyOptionsForDataset(
  instance: CogneeInstance,
  datasetId: string,
): Promise<CognifyOptions> {
  const cfg = await loadGraphModelsConfig(instance);
  const opts: CognifyOptions = {};

  const model = findModelForDataset(cfg.models, datasetId);
  if (model) {
    opts.graphModel = toGraphModelSchema(toCleanSchema(model.schema));
  }

  const promptName = findPromptForDataset(cfg.promptAssignments ?? {}, datasetId);
  if (promptName && cfg.customPrompts?.[promptName]) {
    opts.customPrompt = cfg.customPrompts[promptName];
  }

  const ontologyKey = findOntologyForDataset(cfg.ontologyAssignments ?? {}, datasetId);
  if (ontologyKey) {
    opts.ontologyKey = [ontologyKey];
  }

  return opts;
}
