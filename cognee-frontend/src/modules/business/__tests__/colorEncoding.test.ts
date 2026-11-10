import { CODE_NODE_TYPES } from "../codeGraph";
import {
  assignTypeColors,
  CODE_TYPE_COLORS,
  createSourceEncoding,
  createTypeEncoding,
  ENTITY_FALLBACK_COLOR,
  SOURCE_FALLBACK_COLOR,
  TYPE_PALETTE,
  TYPE_TAIL_COLOR,
} from "../colorEncoding";

const SETS = { crm: "#111111", drive: "#222222", "session_learnings:claude_1": "#333333" };
const isSession = (name: string) => name.startsWith("session_learnings");
const encoding = createSourceEncoding(SETS, isSession);

describe("CODE_TYPE_COLORS", () => {
  it("only colors types that draw as code", () => {
    Object.keys(CODE_TYPE_COLORS).forEach((type) => expect(CODE_NODE_TYPES.has(type)).toBe(true));
  });
});

describe("createSourceEncoding", () => {
  describe("entityFill", () => {
    it("uses the first source's color", () => {
      expect(encoding.entityFill(["crm", "drive"])).toBe("#111111");
    });

    it("skips a session set when a real source is present", () => {
      expect(encoding.entityFill(["session_learnings:claude_1", "drive"])).toBe("#222222");
    });

    it("falls back to the session set's color when it is the only set", () => {
      expect(encoding.entityFill(["session_learnings:claude_1"])).toBe("#333333");
    });

    it("paints a code entity by its code type, whatever its source", () => {
      expect(encoding.entityFill(["crm"], "CodeModule")).toBe("#cb5ab7");
      expect(encoding.entityFill(["drive"], "CodeModule")).toBe("#cb5ab7");
    });

    it("keeps the source color for a type with no code color", () => {
      expect(encoding.entityFill(["crm"], "country")).toBe("#111111");
      expect(encoding.entityFill(["crm"], "CodeFileReference")).toBe("#111111");
    });

    it("uses the entity fallback for a set with no color", () => {
      expect(encoding.entityFill(["unknown"])).toBe(ENTITY_FALLBACK_COLOR);
    });
  });

  describe("entityRing", () => {
    it("is null for an entity in a single source", () => {
      expect(encoding.entityRing(["crm"])).toBeNull();
    });

    it("takes the second source's color for an entity spanning two sources", () => {
      expect(encoding.entityRing(["crm", "drive"])).toBe("#222222");
    });

    it("is the neutral ring for session memory in at most one source, never amber", () => {
      expect(encoding.entityRing(["crm", "session_learnings:claude_1"])).toBe("#E9EEF6");
      expect(encoding.entityRing(["session_learnings:claude_1"])).toBe("#E9EEF6");
    });

    it("is the neutral bridge ring when the second source has no color", () => {
      expect(encoding.entityRing(["crm", "unknown"])).toBe("#E9EEF6");
    });

    it("is neutral when a session member also spans sources whose second has no color", () => {
      expect(encoding.entityRing(["crm", "unknown", "session_learnings:claude_1"])).toBe("#E9EEF6");
    });
  });

  describe("typeFill", () => {
    it("uses the dominant non-session source", () => {
      const counts = { "session_learnings:claude_1": 9, crm: 2, drive: 5 };
      expect(encoding.typeFill(counts, null)).toBe("#222222");
    });

    it("uses the focused source when a source lens is active", () => {
      expect(encoding.typeFill({ crm: 9, drive: 1 }, new Set(["drive"]))).toBe("#222222");
    });

    it("paints the type node of a code type with its code color", () => {
      expect(encoding.typeFill({ crm: 3 }, null, "CodeSymbol")).toBe("#7a7af5");
    });

    it("keeps a code type's color when a source lens is active", () => {
      expect(encoding.typeFill({ crm: 3, drive: 1 }, new Set(["drive"]), "CodeSymbol")).toBe("#7a7af5");
    });

    it("uses the entity fallback for a type with no members", () => {
      expect(encoding.typeFill({}, null)).toBe(ENTITY_FALLBACK_COLOR);
    });
  });

  describe("documentFill and sourceColor", () => {
    it("paints a document with its own first set, without the session filter", () => {
      expect(encoding.documentFill(["session_learnings:claude_1", "crm"])).toBe("#333333");
    });

    it("uses the neutral source fallback for an unknown source", () => {
      expect(encoding.sourceColor("unknown")).toBe(SOURCE_FALLBACK_COLOR);
      expect(encoding.documentFill(["unknown"])).toBe(SOURCE_FALLBACK_COLOR);
    });

    it("returns a known source's color", () => {
      expect(encoding.sourceColor("crm")).toBe("#111111");
    });
  });
});

describe("assignTypeColors", () => {
  const counts = (entries: Record<string, number>) => Object.entries(entries).map(([name, count]) => ({ name, count }));

  it("gives the biggest types the palette in rank order", () => {
    const colors = assignTypeColors(counts({ person: 5, country: 9, concept: 7 }));
    expect(colors).toEqual({ country: TYPE_PALETTE[0], concept: TYPE_PALETTE[1], person: TYPE_PALETTE[2] });
  });

  it("breaks a tie by name so the assignment is deterministic", () => {
    const colors = assignTypeColors(counts({ b: 3, a: 3 }));
    expect(colors).toEqual({ a: TYPE_PALETTE[0], b: TYPE_PALETTE[1] });
  });

  it("leaves types past the palette without a color", () => {
    const many = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`t${String(i).padStart(2, "0")}`, 100 - i]));
    const colors = assignTypeColors(counts(many));
    expect(Object.keys(colors)).toHaveLength(TYPE_PALETTE.length);
    expect(colors.t11).toBeUndefined();
  });

  it("never colors the untyped bucket or a code type", () => {
    const colors = assignTypeColors(counts({ other: 50, CodeSymbol: 40, person: 1 }));
    expect(colors).toEqual({ person: TYPE_PALETTE[0] });
  });

  it("leaves out a code type that has no palette entry too, so it cannot take a business type's slot", () => {
    const colors = assignTypeColors(counts({ CodeFileReference: 90, StorageResource: 80, person: 1 }));
    expect(colors).toEqual({ person: TYPE_PALETTE[0] });
  });

  it("keeps a type named like an Object.prototype member as an own key", () => {
    const colors = assignTypeColors(counts({ constructor: 5, __proto__: 4 }));
    expect(Object.keys(colors)).toEqual(expect.arrayContaining(["constructor"]));
    expect(colors.constructor).toBe(TYPE_PALETTE[0]);
  });
});

describe("createTypeEncoding", () => {
  const typeEncoding = createTypeEncoding({ person: "#111111" }, isSession);

  it("paints an entity by its type, whatever its source", () => {
    expect(typeEncoding.entityFill(["crm"], "person")).toBe("#111111");
    expect(typeEncoding.entityFill(["drive"], "person")).toBe("#111111");
  });

  it("paints a type outside the palette, or an untyped entity, in the tail color", () => {
    expect(typeEncoding.entityFill(["crm"], "country")).toBe(TYPE_TAIL_COLOR);
    expect(typeEncoding.entityFill(["crm"], undefined)).toBe(TYPE_TAIL_COLOR);
  });

  it("paints a code type with no palette entry in the tail color, not a business color", () => {
    expect(typeEncoding.entityFill(["repo"], "CodeFileReference")).toBe(TYPE_TAIL_COLOR);
    expect(typeEncoding.typeFill({ repo: 4 }, null, "StorageResource")).toBe(TYPE_TAIL_COLOR);
  });

  it("does not answer a prototype member name with an inherited function", () => {
    expect(typeEncoding.entityFill(["crm"], "constructor")).toBe(TYPE_TAIL_COLOR);
    expect(typeEncoding.entityFill(["crm"], "toString")).toBe(TYPE_TAIL_COLOR);
    expect(typeof encoding.entityFill(["crm"], "constructor")).toBe("string");
    expect(typeof encoding.typeFill({ crm: 1 }, null, "valueOf")).toBe("string");
  });

  it("keeps the code palette for a code type", () => {
    expect(typeEncoding.entityFill(["repo"], "CodeModule")).toBe(CODE_TYPE_COLORS.CodeModule);
    expect(typeEncoding.typeFill({ repo: 4 }, null, "CodeModule")).toBe(CODE_TYPE_COLORS.CodeModule);
  });

  it("colors an L0 type node by its type, not by its dominant source", () => {
    expect(typeEncoding.typeFill({ crm: 9 }, new Set(["crm"]), "person")).toBe("#111111");
  });

  it("leaves sources and documents neutral", () => {
    expect(typeEncoding.sourceColor("crm")).toBe(SOURCE_FALLBACK_COLOR);
    expect(typeEncoding.documentFill(["crm"])).toBe(SOURCE_FALLBACK_COLOR);
  });

  it("rings an entity that spans sources or carries session memory in white, never amber", () => {
    expect(typeEncoding.entityRing(["crm", "drive"])).toBe("#E9EEF6");
    expect(typeEncoding.entityRing(["crm", "session_learnings:claude_1"])).toBe("#E9EEF6");
    expect(typeEncoding.entityRing(["crm"])).toBeNull();
  });
});
