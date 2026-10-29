import type { BusinessGraphNode } from "./types";

// The code-graph node types cognee's GitHub connector produces (probed on a
// ~15.8k-node dataset, cognee 1.6.1). cognee's own /visualize preprocessing
// only assigns a real `stage` to TextDocument/DocumentChunk/Entity/
// EntityType/TextSummary/schema nodes, so every one of these arrives with
// stage "other" and renders as an untyped grey record until promoted below
// (CLO-627).
export const CODE_NODE_TYPES = new Set([
  "CodeSymbol",
  "ExternalDependency",
  "CodeFileReference",
  "CodeTestReference",
  "CodeModule",
  "ApiEndpoint",
  "CodeInsight",
  "CodeRepository",
  "CodeExtractionAccount",
  "StorageResource",
]);

// belongs_to_set/source_node_set are null on every code node observed so
// far, the same shape UNCATEGORIZED_SOURCE (computeBrainState.ts) exists
// for — but code content isn't "no source", it's its own source, so it gets
// a tile/hull of its own rather than piling into uncategorized uploads.
// CODE_SOURCE is only the last resort: a node that names its repository
// (every code node observed carries a `repo` field matching its `part_of`
// CodeRepository) is attributed to that repository instead, so two synced
// repositories are two territories rather than one merged "code" blob.
export const CODE_SOURCE = "code";

// Exported for computeBrainState's type-bucket resolver — entity_type is
// present on every node since cognee 1.6.1, not just code ones (see
// types.ts), so anything reading it as a real type label has to gate on
// this the same way normalizeCodeNodes and codeDisplayName already do.
export function isCodeNode(node: BusinessGraphNode): boolean {
  return typeof node.type === "string" && CODE_NODE_TYPES.has(node.type);
}

// The CodeRepository node itself carries no `repo` field, but its own name is
// the repository.
function repositoryOf(node: BusinessGraphNode): string {
  if (typeof node.repo === "string" && node.repo) return node.repo;
  if (node.type === "CodeRepository" && typeof node.name === "string" && node.name) return node.name;
  return CODE_SOURCE;
}

function hasRealSource(node: BusinessGraphNode): boolean {
  return (Array.isArray(node.belongs_to_set) && node.belongs_to_set.length > 0) || Boolean(node.source_node_set);
}

// The fields to overwrite on a code node's sources. A single name sent as a
// plain string (the SDK's preprocessor accepts it) is a real source and is
// kept as an array, which is the only shape setsOf reads.
function sourceFields(node: BusinessGraphNode): Pick<BusinessGraphNode, "belongs_to_set"> {
  const sets: unknown = node.belongs_to_set;
  if (typeof sets === "string" && sets) return { belongs_to_set: [sets] };
  return hasRealSource(node) ? {} : { belongs_to_set: [repositoryOf(node)] };
}

// Promotes code-graph nodes into the entity layer before computeBrainState
// reads stage/belongs_to_set, in one O(n) pass — every downstream consumer
// (byId, the entity filter, buildTypeLayer, the plumbing split, setsOf)
// then sees an already-normalized node and needs no code-specific branch of
// its own. Only nodes cognee tags with one of the code types above are
// touched; a node already staged "entity" or carrying a real node_set is
// returned as-is. entity_type is left untouched — cognee already stamps it
// equal to `type` on every code node, and buildTypeLayer reads it directly
// as the type-bucket fallback when no is_a/instance_of edge exists.
export function normalizeCodeNodes(nodes: BusinessGraphNode[]): BusinessGraphNode[] {
  return nodes.map((node) => {
    if (!isCodeNode(node) || node.stage === "entity") return node;
    return {
      ...node,
      stage: "entity",
      ...sourceFields(node),
    };
  });
}

// Extensions cognee's code graph names a file by directly (symbol_kind isn't
// always populated — probed payload had it only on some file references),
// so a node whose last path segment ends in one of these is treated as a
// filename even without that field.
const FILE_EXTENSIONS = new Set([
  "md", "ts", "tsx", "js", "jsx", "json", "yml", "yaml", "py", "txt", "css",
  "scss", "html", "sh", "go", "rs", "java", "rb", "php", "toml", "cfg", "ini",
  "xml", "sql", "lock",
]);

function isFileLikeSegment(node: BusinessGraphNode, segment: string): boolean {
  if (node.symbol_kind === "document") return true;
  const dot = segment.lastIndexOf(".");
  if (dot <= 0) return false;
  return FILE_EXTENSIONS.has(segment.slice(dot + 1).toLowerCase());
}

// cognee names a code node by its fully qualified path
// ("cognee-saas-frontend/src/modules/tenant.isFreshlyCreatedTenant"), which
// is unreadable as a canvas label or a tooltip/panel title. This derives the
// short form at render time rather than writing it back onto node.name —
// search, shortest-path and identity all match on the raw name elsewhere in
// this module, and a second stored field would just be one more copy of the
// same string to keep in sync for no benefit, since this is a pure function
// of fields already on the node. Non-code nodes pass through untouched;
// everything else in the graph already has a short, human name.
export function codeDisplayName(node: BusinessGraphNode): string {
  const name = typeof node.name === "string" ? node.name : "";
  if (!name || !isCodeNode(node)) return name;
  const segment = name.slice(name.lastIndexOf("/") + 1);
  if (!segment) return name;
  if (isFileLikeSegment(node, segment)) return segment;
  // A dotted symbol ("tenant.isFreshlyCreatedTenant") shortens to the part
  // after the last dot; a plain module name ("tenant") has none and passes
  // through as-is. Guards against a trailing dot leaving nothing behind it.
  const dot = segment.lastIndexOf(".");
  return dot > 0 && dot < segment.length - 1 ? segment.slice(dot + 1) : segment;
}
