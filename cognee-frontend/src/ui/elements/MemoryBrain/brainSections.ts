/**
 * The brain is divided the way a brain is, into lobes, and each lobe is one of
 * the things a workspace connects or builds:
 *
 *   parietal lobe     teams          top             who shares it
 *   occipital lobe    data sources   back            the senses: what comes in
 *   cerebellum        integrations   bottom back     coordination
 *   frontal lobe      agents         front           the part that acts
 *   temporal lobe     sessions       front, low      what agents have worked on
 *   brain stem        connections    below           what links the brain to the rest
 *
 * Each lobe fills with lavender cells, spreading out from where its spokes
 * land, as its count grows toward the point where it reads as full.
 */
export type BrainSection =
  | "teams"
  | "dataSources"
  | "integrations"
  | "sessions"
  | "agents"
  | "connections";

/** How many of each the workspace has: what fills each lobe. */
export type BrainAmounts = Record<BrainSection, number>;

export const BRAIN_SECTIONS: readonly BrainSection[] = [
  "teams",
  "dataSources",
  "agents",
  "integrations",
  "sessions",
  "connections",
];

/** The count at which a lobe reads as full. */
export const FULL: Readonly<BrainAmounts> = {
  teams: 10,
  dataSources: 10,
  integrations: 10,
  sessions: 10,
  agents: 10,
  connections: 40,
};

export const SECTION_LABEL: Readonly<Record<BrainSection, string>> = {
  teams: "Teams",
  dataSources: "Data sources",
  agents: "Agents",
  integrations: "Integrations",
  sessions: "Sessions",
  connections: "Connections",
};

/**
 * Each lobe's shade of lavender once it fills: light to deep, alternating so
 * no two neighbouring lobes sit a step apart.
 */
export const SECTION_FILL: Readonly<Record<BrainSection, string>> = {
  teams: "fill-brain-lavender-1",
  dataSources: "fill-brain-lavender-4",
  integrations: "fill-brain-lavender-6",
  sessions: "fill-brain-lavender-2",
  agents: "fill-brain-lavender-3",
  connections: "fill-brain-lavender-7",
};

/** Which side of the brain each lobe's tiles are on: where its spokes land. */
export const SECTION_SIDE: Readonly<
  Record<Exclude<BrainSection, "connections">, "left" | "right">
> = {
  teams: "left",
  dataSources: "left",
  integrations: "left",
  sessions: "right",
  agents: "right",
};
