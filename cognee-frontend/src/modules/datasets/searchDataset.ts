import { CogneeInstance } from "../instances/types";
import { getPipelineSettingsFromStorage } from "../configuration/pipelineSettings";

export default function searchDataset(instance: CogneeInstance, request: SearchRequest) {
  const settings = getPipelineSettingsFromStorage();
  const requestWithDefaults = {
    ...request,
    topK: request.topK ?? settings.topK,
    // Server defaults this to false; the frontend defaults it to whatever
    // the user has set in Extraction Settings (which itself defaults to on).
    include_references: request.include_references ?? settings.includeReferences,
  };
  return instance.fetch("/v1/search", {
    method: "POST",
    body: JSON.stringify(requestWithDefaults),
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => response.json());
}

type SearchRequest = {
    searchType?: string;
    datasets?: string[];
    datasetIds?: string[];
    query?: string;
    systempPrompt?: string;
    nodeName?: string[];
    topK?: number;
    onlyContext?: boolean;
    useCombinedContext?: boolean;
    include_references?: boolean;
}

export type SearchResponse = Array<{search_result: string[], dataset_id: string, dataset_name: string}>;