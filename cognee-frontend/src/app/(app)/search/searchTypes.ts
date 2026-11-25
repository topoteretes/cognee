import type { ConversationSearchType } from "@/modules/searchHistory/conversations";

/**
 * The two ways the Search tab can retrieve. The names are the user's; what
 * each one sends is decided where it is sent, because the two do not differ by
 * a field — they go to different retrievers with different contracts.
 *
 * Text is a question: POST /v1/recall, answered by a model over what the
 * hybrid retriever finds. Code is a lookup: POST /v1/search with a structured
 * code_query, answered by the code graph with no model involved (codeSearch.ts).
 * A repository is only reachable the second way — its nodes are not embedded,
 * so nothing the hybrid retriever searches contains them.
 */
export type SearchTypeId = ConversationSearchType;

export interface SearchTypeOption {
  id: SearchTypeId;
  label: string;
  /** Hover copy on the control, since two one-word labels cannot say this. */
  hint: string;
  /** Says what to type, which is the difference the user actually has to know. */
  placeholder: string;
  /** Starting points for the empty state, each one something this type can answer. */
  suggestions: readonly string[];
}

// Keyed rather than listed so a third type cannot be added without its
// option: the record is exhaustive over SearchTypeId at compile time, which a
// find() over a list with a fallback never was.
const OPTIONS: Record<SearchTypeId, SearchTypeOption> = {
  text: {
    id: "text",
    label: "Text",
    hint: "Ask a question about documents and the knowledge graph.",
    placeholder: "Ask a question about your data...",
    suggestions: [
      "What are the main entities?",
      "Summarize the uploaded documents",
      "What relationships exist in the data?",
    ],
  },
  code: {
    id: "code",
    label: "Code",
    hint: "Look up a function, class or file by name in a connected repository.",
    placeholder: "Search a function, class or file name...",
    suggestions: ["Show the architecture", "List the API routes", "List the modules"],
  },
};

/** In the order the control shows them. */
export const SEARCH_TYPES: readonly SearchTypeOption[] = [OPTIONS.text, OPTIONS.code];

export const DEFAULT_SEARCH_TYPE: SearchTypeId = "text";

export const GITHUB_HINT = "For GitHub repositories, use Code — it looks up functions, classes and files by name.";

export function searchTypeOption(id: SearchTypeId): SearchTypeOption {
  return OPTIONS[id];
}

/**
 * Guards what comes back out of storage: the key is per pod and long-lived, so
 * a value written by an older build (or a hand-edited one) must not decide
 * which retriever a search runs on.
 */
export function isSearchTypeId(value: string | null): value is SearchTypeId {
  return value !== null && Object.prototype.hasOwnProperty.call(OPTIONS, value);
}

// Every connected repository lands in a dataset named `github_<account>`: the
// control plane's connectors name theirs `{provider}_{slug}`
// (cloud-backend framework/base.py dataset_name_for), and cognee's own GitHub
// sync uses the same prefix (modules/integrations/github/sync.py).
const REPOSITORY_DATASET_PREFIX = "github_";

/** Whether this dataset holds a connected repository, so Code has something to find. */
export function isRepositoryDataset(name: string): boolean {
  return name.startsWith(REPOSITORY_DATASET_PREFIX);
}
