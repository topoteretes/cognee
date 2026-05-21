import { CogneeInstance } from "../instances/types";

export default function searchDataset(instance: CogneeInstance, request: SearchRequest) {
  return instance.fetch("/v1/search", {
    method: "POST",
    body: JSON.stringify(request),
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
}

export type SearchResponse = Array<{search_result: string[], dataset_id: string, dataset_name: string}>;
