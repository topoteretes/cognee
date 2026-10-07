"use client";

import { useBrainGraph } from "../useBrainGraph";
import type { BusinessLayer, BusinessLayerContext, BusinessLayerResult } from "./types";

export const CONTENT_LAYER_ID = "content";

// "What is in the brain" — the focused dataset's entities/documents/chunks,
// with the stage/importance/belongs_to_set fields computeBrainState reads.
// Fetches ONE dataset's graph (GET /v1/visualize/json) instead of reading it
// out of the every-dataset /brains payload — the switcher list still comes
// from /brains, but nobody pays for graphs they aren't looking at (CLO-597).
// No dataset focused yet means no content to show; the governance layer
// still renders on its own in that state.
//
// The payload reference must stay stable across same-data refetches — a new
// object per render tore down and reseeded the whole force simulation on
// every poll (COG-6233 "perpetually trembling entities"). React Query's
// structural sharing guarantees that here: an unchanged response keeps the
// previous object identity.
const contentLayer: BusinessLayer = {
  id: CONTENT_LAYER_ID,
  label: "Content",
  defaultEnabled: true,
  useData(ctx: BusinessLayerContext): BusinessLayerResult {
    const { data, isLoading, error } = useBrainGraph(ctx.cogniInstance, ctx.activeDatasetId);
    return { data: data ?? null, isLoading, error: error ?? null };
  },
};

export default contentLayer;
