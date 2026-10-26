import type { BusinessEntity } from "./sceneTypes";

// Code entity types in the fixed order their sectors take around a territory.
// The sector angle is derived from a type's index in THIS list, never from
// which types happen to be present: a type arriving later (a first
// ApiEndpoint in a live-growing repo) must not rotate every anchor already
// placed. The code classes are a closed set, unlike LLM-invented entity types,
// which are too fragmented to place by (68 types on 237 entities in a probed
// text dataset), so text entities stay at their territory's center until a
// coarse category exists to group them by.
export const CODE_GROUP_ORDER = ["CodeModule", "CodeSymbol", "ExternalDependency", "CodeInsight", "ApiEndpoint"];

export interface LayoutGroups {
  groupOf: Record<string, string | null>;
  groupOrder: string[];
}

// Which sector of its source territory each entity is pulled toward. This is
// layout only: nearby entities of one kind.
export function resolveLayoutGroups(
  entities: BusinessEntity[],
  entityType: Record<string, string>,
): LayoutGroups {
  const groupOf: Record<string, string | null> = {};
  entities.forEach((e) => {
    const type = entityType[e.id];
    groupOf[e.id] = type && CODE_GROUP_ORDER.includes(type) ? type : null;
  });
  return { groupOf, groupOrder: CODE_GROUP_ORDER };
}
