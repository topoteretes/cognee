// Wire shapes for the two payloads the Business view joins: the governance
// graph (GET /v1/schema/provenance/json) and the content graph (GET
// /v1/visualize/json, GET /v1/visualize/brains). Both go through cognee-core's
// `preprocess()` and come back in this exact shape — see
// build_visualization_payload / build_brain_summary_payload in
// cognee/modules/visualization/cognee_network_visualization.py.
//
// Known fields are typed explicitly because computeBrainState and the canvas
// hit-test/draw code read them by name; everything else a node/link may carry
// (type-specific extras) is untyped on purpose — narrow with a guard at the
// point of use rather than widening this shape.
export interface BusinessGraphNode {
  id: string;
  name?: string;
  type?: string;
  // 'entity' | 'document' | 'chunk' | 'summary' | 'context' | 'type' | 'other'
  stage?: string;
  importance?: number;
  label_priority?: boolean;
  is_unnamed?: boolean;
  source_node_set?: string;
  belongs_to_set?: string[];
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

// GET /v1/visualize/brains value shape — one per dataset the caller can read.
export interface BrainSummary {
  name: string;
  nodes: BusinessGraphNode[];
  links: BusinessGraphLink[];
  node_set_colors: Record<string, string>;
}

export type BrainsPayload = Record<string, BrainSummary>;

export interface LiveEventsPayload {
  events: SessionEvent[];
  cursor: string | null;
}
