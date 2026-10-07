"use client";

import { useMemo, useState } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import governanceLayer from "./layers/governanceLayer";
import contentLayer from "./layers/contentLayer";
import type { BusinessLayerContext, BusinessLayerResult } from "./layers/types";
import type { VisualizationPayload } from "./types";
import { computeBrainState } from "./computeBrainState";
import type { BrainState } from "./sceneTypes";

export interface BusinessScene {
  brainState: BrainState | null;
  isLoading: boolean;
  // The focused dataset's graph fetch specifically — true only while that
  // per-dataset query has no data yet (first load of a dataset, or switching
  // to one that isn't cached). Unlike `isLoading`, a background layer query
  // (governance) can't hold this true over an already-rendered graph.
  isContentLoading: boolean;
  // The focused dataset's graph fetch failure. Without this the view can't
  // tell "dataset genuinely has no graph content" apart from "/visualize/json
  // returned 500" — both render as a null content payload.
  contentError: unknown;
  // The governance fetch's failure, for the same reason contentError exists:
  // an empty `datasets` list means "this workspace has no brains" and a
  // failed /visualize governance call also produces one, and the view has to
  // tell those two apart before telling anyone to go create a brain.
  governanceError: unknown;
  // Raw per-layer payloads, keyed by layer id — governance and content have
  // dedicated rail UIs that read their own layer directly (e.g. the
  // operators rail needs governance's actor nodes, not the merged scene).
  layerData: Record<string, VisualizationPayload | null>;
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
}

function mergeLayers(payloads: VisualizationPayload[]): {
  nodes: VisualizationPayload["nodes"];
  links: VisualizationPayload["links"];
  colorsMap: Record<string, string>;
} {
  const colorsMap: Record<string, string> = {};
  payloads.forEach((p) => Object.assign(colorsMap, p.color_maps?.node_set));
  return {
    nodes: payloads.flatMap((p) => p.nodes),
    links: payloads.flatMap((p) => p.links),
    colorsMap,
  };
}

// Joins whichever layers are enabled into one canvas scene. Node/link arrays
// concatenate cleanly across layers because computeBrainState only reacts to
// fields a layer actually sets (stage/importance/belongs_to_set for content,
// nothing lookalike in governance's actor nodes) — so a third layer merges
// the same way without this function changing.
//
// Registering a layer takes three small, mechanical edits: the layer file
// itself (layers/types.ts contract), one line in layers/registry.ts (so
// anything that lists "every layer" sees it), and one explicit `useData`
// call below — React's rules-of-hooks forbid calling hooks from a loop over
// a dynamic-length array, so the registry alone can't drive this call.
export function useBusinessScene(cogniInstance: CogneeInstance): BusinessScene {
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const ctx: BusinessLayerContext = { cogniInstance, activeDatasetId };

  const governance: BusinessLayerResult = governanceLayer.useData(ctx);
  const content: BusinessLayerResult = contentLayer.useData(ctx);

  const layerData = useMemo(
    () => ({ [governanceLayer.id]: governance.data, [contentLayer.id]: content.data }),
    [governance.data, content.data],
  );

  const brainState = useMemo(() => {
    const payloads = Object.values(layerData).filter((d): d is VisualizationPayload => d !== null);
    if (!payloads.length) return null;
    const { nodes, links, colorsMap } = mergeLayers(payloads);
    return computeBrainState(nodes, links, colorsMap);
  }, [layerData]);

  return {
    brainState,
    isLoading: governance.isLoading || content.isLoading,
    isContentLoading: content.isLoading,
    contentError: content.error,
    governanceError: governance.error,
    layerData,
    activeDatasetId,
    setActiveDatasetId,
  };
}
