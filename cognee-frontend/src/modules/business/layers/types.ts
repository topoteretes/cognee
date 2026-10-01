import type { CogneeInstance } from "@/modules/instances/types";
import type { VisualizationPayload } from "../types";

// What a layer's data hook is given to fetch with. `activeDatasetId` is the
// currently-selected brain (null while the primary/governance-only view is
// showing) — a layer that doesn't need it (e.g. governance) just ignores it.
export interface BusinessLayerContext {
  cogniInstance: CogneeInstance;
  activeDatasetId: string | null;
}

export interface BusinessLayerResult {
  data: VisualizationPayload | null;
  isLoading: boolean;
  error: unknown;
}

// The extension point CLO-402 asked for: the canvas scene is a merge of
// however many of these are enabled, not a hardcoded two-source join. Adding
// a third data source (e.g. a billing/risk layer later) means writing one
// more file matching this contract and pushing it into the registry — see
// registry.ts — nothing in useBusinessScene or the canvas changes.
export interface BusinessLayer {
  id: string;
  label: string;
  defaultEnabled: boolean;
  useData(ctx: BusinessLayerContext): BusinessLayerResult;
}
