import type { BusinessLayer } from "./types";
import governanceLayer from "./governanceLayer";
import contentLayer from "./contentLayer";

// Every data source the Business scene draws from. Adding a layer later
// (e.g. a billing/risk source) means writing one file matching BusinessLayer
// and pushing it in here — useBusinessScene and the canvas don't know or
// care how many there are.
export const BUSINESS_LAYERS: BusinessLayer[] = [governanceLayer, contentLayer];
