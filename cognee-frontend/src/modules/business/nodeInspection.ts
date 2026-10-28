import type { CodeFact, CodeNeighbor } from "@/modules/datasets/searchCode";
import type { BusinessEntity, SemanticLink } from "./sceneTypes";
import type { BusinessGraphNode } from "./types";

const CODE_TYPE_LABELS: Record<string, string> = {
  CodeModule: "Module", CodeSymbol: "Code symbol", ExternalDependency: "External dependency",
  CodeFileReference: "File reference", CodeTestReference: "Test reference", ApiEndpoint: "API endpoint",
  CodeInsight: "Architecture finding", CodeRepository: "Repository", StorageResource: "Storage resource",
  CodeExtractionAccount: "Extraction coverage", CodeService: "Service", CodeFlow: "Code flow",
  CodeIntent: "Declared architecture", CodeAssociation: "Model relationship", CodeLintFinding: "Code finding",
};

export function inspectionType(node: BusinessGraphNode): string {
  return CODE_TYPE_LABELS[node.type ?? ""] ?? node.type ?? "Record";
}

export function codeFactEntity(fact: CodeFact & { type: string }): BusinessEntity {
  return {
    ...fact,
    name: fact.name ?? fact.id,
    stage: "entity",
    file_path: fact.file,
    belongs_to_set: fact.repo ? [fact.repo] : [],
  };
}

export interface InspectionConnection {
  relation: string;
  direction: CodeNeighbor["direction"];
  node: BusinessGraphNode;
  /** Overrides the relation-derived group, for links that are not graph relationships. */
  group?: string;
}

/** Only links and endpoints actually present in this view can be inspected locally. */
export function loadedConnections(
  nodeId: string,
  links: SemanticLink[],
  byId: Record<string, BusinessGraphNode>,
  group?: string,
): InspectionConnection[] {
  const connections: InspectionConnection[] = [];
  for (const link of links) {
    if (link._sid === nodeId && byId[link._tid]) {
      connections.push({ relation: link.relation || "unlabelled relationship", direction: "out", node: byId[link._tid], group });
    }
    if (link._tid === nodeId && byId[link._sid]) {
      connections.push({ relation: link.relation || "unlabelled relationship", direction: "in", node: byId[link._sid], group });
    }
  }
  return connections;
}

export function connectionGroup({ relation, direction, group }: InspectionConnection): string {
  if (group) return group;
  const outgoing = direction === "out";
  if (["imports", "calls", "depends_on"].includes(relation)) return outgoing ? "Uses" : "Used by";
  // Imported declarations can point from the declared item to its module.
  // Keep their stored direction visible without inferring containment from it.
  if (relation === "declares") return "Declarations";
  if (["contains", "has_method"].includes(relation)) return outgoing ? "Contains" : "Contained by";
  if (relation === "part_of") return outgoing ? "Belongs to" : "Contains";
  if (relation === "evidences") return outgoing ? "Supported by" : "Evidence for";
  return outgoing ? "Outgoing relationships" : "Incoming relationships";
}
