// Wire shapes for the two payloads the Business view joins: the governance
// graph (GET /v1/schema/provenance/json) and the content graph (GET
// /v1/visualize/json, GET /v1/visualize/brains). Both go through cognee-core's
// `preprocess()` and come back in this exact shape — see
// build_visualization_payload / build_brains_summary_payload in
// cognee/modules/visualization/cognee_network_visualization.py.
//
// Known fields are typed explicitly because computeBrainState and the canvas
// hit-test/draw code read them by name; everything else a node/link may carry
// (type-specific extras) is untyped on purpose — narrow with a guard at the
// point of use rather than widening this shape.
export interface BusinessGraphNode {
  id: string;
  name?: string;
  // Retained when a structured row's content-derived wire name is shortened.
  originalName?: string;
  type?: string;
  // 'entity' | 'document' | 'chunk' | 'summary' | 'context' | 'type' | 'other'
  stage?: string;
  importance?: number;
  label_priority?: boolean;
  is_unnamed?: boolean;
  source_node_set?: string;
  belongs_to_set?: string[];
  // Present on EVERY node since cognee 1.6.1 (preprocessor.py), not just
  // code ones — falls back to the node's own class name ("Entity") when no
  // is_a/instance_of edge resolved a real type. That makes it a reliable
  // type-bucket fallback (computeBrainState's buildTypeLayer) only for a
  // code node, whose entity_type always carries its actual class
  // (CodeSymbol, CodeModule, …) with no such generic value to guard against.
  entity_type?: string;
  // Set on some code-graph file-reference nodes ("document" for a plain
  // file, as opposed to a function/class/method symbol) — read by
  // codeDisplayName to tell a filename apart from a dotted symbol path.
  symbol_kind?: string;
  // Names the repository a code node belongs to (matches its part_of
  // CodeRepository); read by normalizeCodeNodes to give each synced repository
  // its own territory.
  repo?: string;
  [key: string]: unknown;
}

export interface BusinessGraphLink {
  source: string;
  target: string;
  relation?: string;
  edge_class?: string;
  [key: string]: unknown;
}

export interface SessionEvent {
  kind?: string;
  time?: string;
  qa_id?: string;
  question?: string;
  node_ids?: string[];
  [key: string]: unknown;
}

export interface VisualizationPayload {
  nodes: BusinessGraphNode[];
  links: BusinessGraphLink[];
  color_maps: Record<string, Record<string, string>>;
  search_events?: SessionEvent[];
  [key: string]: unknown;
}

// GET /v1/visualize/brains-summary value shape — one per dataset the caller
// can read, built from relational metadata rather than a graph read, so it
// carries no nodes or links at all (COG-6258). node_count covers the whole
// graph as of the dataset's latest cognify run, not only entity nodes, and
// is 0 for a dataset that has never been cognified.
export interface BrainSummary {
  name: string;
  source_names: string[];
  node_count: number;
  node_set_colors: Record<string, string>;
}

export type BrainsSummaryPayload = Record<string, BrainSummary>;

export interface LiveEventsPayload {
  events: SessionEvent[];
  cursor: string | null;
}
