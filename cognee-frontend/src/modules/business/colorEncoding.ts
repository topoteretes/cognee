import { CODE_NODE_TYPES } from "./codeGraph";

// What the canvas, the minimap and the rails paint a node, a type or a source
// with. Every layer asks this instead of reading BrainState.setColor itself:
// the rule "fill = the source's color" used to be written separately in five
// canvas files, so adding a second encoding (category) meant an if/else in
// each of them. Inputs are set NAMES (setsOf already resolves a node's sets),
// which keeps this module free of any node shape and of computeBrainState.
export type ColorMode = "type" | "source";

export interface ColorEncoding {
  // A code entity is painted by its code type; everything else by its source
  // (source mode) or by its entity type (type mode).
  entityFill: (sets: string[], type?: string) => string;
  // The second ring around an entity that spans several sources or carries
  // agent session memory; null when it has neither.
  entityRing: (sets: string[]) => string | null;
  documentFill: (sets: string[]) => string;
  // An L0 type node is one color, not a proportional pie: the dominant NON-session
  // source decides it ("mostly crm with a sliver of google_drive"), or the
  // focused source when a source lens is active. A code type is the exception:
  // its palette color is the more specific signal, so a source lens never
  // recolors it.
  typeFill: (memberSets: Record<string, number>, focusSets: Set<string> | null, typeName?: string) => string;
  // A source's own color: its territory hull and every rail swatch. In type
  // mode a source has no color of its own, so territories and swatches stay
  // neutral instead of putting a second color system on screen.
  sourceColor: (name: string) => string;
}

// One source (a repository) holds thousands of code nodes, so source color
// cannot tell a module from a symbol from a dependency; the code type can.
// Validated on the canvas color (#0E1526) in CODE_GROUP_ORDER order, the order
// the types sit around a territory, so neighbouring sectors differ for
// color-vision deficiency and for full-color readers (worst neighbour dE 9.2
// and 16.5). Not every pair clears those floors, which no six colors can, so
// two non-neighbouring types can be close. None is near the reserved amber,
// cyan or green. Types without an entry here (file and test references,
// extraction accounts, storage) keep their source color.
export const CODE_TYPE_COLORS: Record<string, string> = {
  CodeModule: "#cb5ab7",
  CodeSymbol: "#7a7af5",
  ExternalDependency: "#768b09",
  CodeInsight: "#c43766",
  ApiEndpoint: "#0a72da",
  CodeRepository: "#bd4d00",
};

// The colors the biggest entity types of a dataset take, in rank order. They
// are the validated ring of eight on the canvas color (#0E1526): consecutive
// ranks are ring neighbours, which are the pairs that differ for color-vision
// deficiency (worst neighbour dE 16.8) and for full-color readers (19.4).
// None is near the reserved amber, cyan or green.
export const TYPE_PALETTE = [
  "#0a72da", "#e55551", "#7a7af5", "#c43766", "#8654cc", "#768b09", "#cb5ab7", "#bd4d00",
];
// Types past the palette, and entities with no resolved type, stay in one
// muted tone: eight colors is as many as a reader can tell apart.
export const TYPE_TAIL_COLOR = "#5B6880";
// The bucket entities with no resolved type fall into (see buildTypeLayer).
export const UNTYPED_BUCKET = "other";

// Color maps are keyed by names an LLM chose (entity types) or a user typed
// (sources), and a plain object answers "constructor" or "toString" with an
// inherited function. Only own keys count.
function own(map: Record<string, string>, key: string | undefined): string | undefined {
  return key !== undefined && Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined;
}

export const ENTITY_FALLBACK_COLOR = "#8A7BD8";
export const SOURCE_FALLBACK_COLOR = "#7E8CA6";
// Amber is reserved for what is happening now (the live answer, the asking
// agent); an entity that spans sources or carries agent session memory is
// ringed in neutral white instead.
const BRIDGE_RING_COLOR = "#E9EEF6";

export interface TypeCount {
  name: string;
  count: number;
}

// Biggest types first take the palette in order. Colors follow rank inside one
// dataset, so the same type can differ between datasets. Every code type is
// left out, including the ones with no palette entry of their own: they would
// take the top slots on a code dataset and land on colors the code palette
// already uses. The untyped bucket never takes a color either.
export function assignTypeColors(types: TypeCount[]): Record<string, string> {
  return Object.fromEntries(
    types
      .filter((t) => t.name !== UNTYPED_BUCKET && !CODE_NODE_TYPES.has(t.name))
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
      .slice(0, TYPE_PALETTE.length)
      .map((t, i) => [t.name, TYPE_PALETTE[i]]),
  );
}

export function createSourceEncoding(
  setColor: Record<string, string>,
  isSessionSet: (name: string) => boolean,
): ColorEncoding {
  const colorOr = (name: string | undefined, fallback: string): string =>
    own(setColor, name) || fallback;
  const sourcesOf = (sets: string[]): string[] => sets.filter((s) => !isSessionSet(s));

  return {
    entityFill(sets, type) {
      return own(CODE_TYPE_COLORS, type) || colorOr(sourcesOf(sets)[0] || sets[0], ENTITY_FALLBACK_COLOR);
    },
    entityRing(sets) {
      const sources = sourcesOf(sets);
      const sessionMember = sets.length > sources.length;
      if (sources.length <= 1) return sessionMember ? BRIDGE_RING_COLOR : null;
      return colorOr(sources[1], BRIDGE_RING_COLOR);
    },
    documentFill(sets) {
      return colorOr(sets[0], SOURCE_FALLBACK_COLOR);
    },
    typeFill(memberSets, focusSets, typeName) {
      const codeColor = own(CODE_TYPE_COLORS, typeName);
      if (codeColor) return codeColor;
      const ranked = Object.keys(memberSets).sort((a, b) => memberSets[b] - memberSets[a]);
      const dominant = focusSets ? [...focusSets][0] : ranked.find((s) => !isSessionSet(s)) || ranked[0];
      return colorOr(dominant, ENTITY_FALLBACK_COLOR);
    },
    sourceColor(name) {
      return colorOr(name, SOURCE_FALLBACK_COLOR);
    },
  };
}

// The type-mode counterpart: an entity is painted by its entity type (the
// biggest types get a color each, the tail a shared muted tone) and sources
// are neutral. Code keeps its own palette, and a ring still marks an entity
// that spans sources or carries session memory.
export function createTypeEncoding(
  typeColors: Record<string, string>,
  isSessionSet: (name: string) => boolean,
): ColorEncoding {
  const typeColor = (type: string | undefined): string =>
    own(CODE_TYPE_COLORS, type) || own(typeColors, type) || TYPE_TAIL_COLOR;
  const sourcesOf = (sets: string[]): string[] => sets.filter((s) => !isSessionSet(s));

  return {
    entityFill(_sets, type) {
      return typeColor(type);
    },
    entityRing(sets) {
      const sources = sourcesOf(sets);
      return sources.length > 1 || sets.length > sources.length ? BRIDGE_RING_COLOR : null;
    },
    documentFill() {
      return SOURCE_FALLBACK_COLOR;
    },
    typeFill(_memberSets, _focusSets, typeName) {
      return typeColor(typeName);
    },
    sourceColor() {
      return SOURCE_FALLBACK_COLOR;
    },
  };
}
