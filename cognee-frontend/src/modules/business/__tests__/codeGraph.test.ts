import type { BusinessGraphNode } from "../types";
import { CODE_SOURCE, codeDisplayName, normalizeCodeNodes } from "../codeGraph";

function node(id: string, overrides: Partial<BusinessGraphNode> = {}): BusinessGraphNode {
  return { id, stage: "other", ...overrides };
}

describe("normalizeCodeNodes", () => {
  it("promotes a code node to stage entity", () => {
    const [result] = normalizeCodeNodes([node("a", { type: "CodeSymbol" })]);
    expect(result.stage).toBe("entity");
  });

  it("does not promote a non-code node with stage other", () => {
    const [result] = normalizeCodeNodes([node("a", { type: "TextSummary" })]);
    expect(result.stage).toBe("other");
  });

  it("does not promote a node with no type at all", () => {
    const [result] = normalizeCodeNodes([node("a")]);
    expect(result.stage).toBe("other");
  });

  it("tags a code node with no real source with CODE_SOURCE", () => {
    const [result] = normalizeCodeNodes([node("a", { type: "CodeModule" })]);
    expect(result.belongs_to_set).toEqual([CODE_SOURCE]);
  });

  it("attributes a code node to the repository named in its repo field", () => {
    const [result] = normalizeCodeNodes([node("a", { type: "CodeSymbol", repo: "cognee-website" })]);
    expect(result.belongs_to_set).toEqual(["cognee-website"]);
  });

  it("attributes a CodeRepository node, which has no repo field, to its own name", () => {
    const [result] = normalizeCodeNodes([node("r", { type: "CodeRepository", name: "cognee-website" })]);
    expect(result.belongs_to_set).toEqual(["cognee-website"]);
  });

  it("falls back to CODE_SOURCE when the repo field is not a non-empty string", () => {
    const results = normalizeCodeNodes([
      node("a", { type: "CodeSymbol", repo: "" }),
      // A malformed payload can carry a non-string repo despite the declared type.
      Object.assign(node("b", { type: "CodeSymbol" }), { repo: 7 }),
    ]);
    expect(results.map((n) => n.belongs_to_set)).toEqual([[CODE_SOURCE], [CODE_SOURCE]]);
  });

  it("keeps a code node's source sent as a plain string, as an array, instead of the repository name", () => {
    const [result] = normalizeCodeNodes([
      node("a", { type: "CodeSymbol", repo: "cognee-website", belongs_to_set: "crm" as unknown as string[] }),
    ]);
    expect(result.belongs_to_set).toEqual(["crm"]);
  });

  it("keeps a code node's real belongs_to_set instead of overwriting it", () => {
    const [result] = normalizeCodeNodes([
      node("a", { type: "CodeModule", belongs_to_set: ["github"] }),
    ]);
    expect(result.belongs_to_set).toEqual(["github"]);
  });

  it("keeps a code node's real source_node_set instead of tagging CODE_SOURCE", () => {
    const [result] = normalizeCodeNodes([
      node("a", { type: "CodeModule", source_node_set: "github" }),
    ]);
    expect(result.belongs_to_set).toBeUndefined();
  });

  it("leaves an already-entity-staged node untouched", () => {
    const original = node("a", { type: "CodeSymbol", stage: "entity", belongs_to_set: ["custom"] });
    const [result] = normalizeCodeNodes([original]);
    expect(result).toBe(original);
  });

  it("covers every documented code node type", () => {
    const types = [
      "CodeSymbol", "ExternalDependency", "CodeFileReference", "CodeTestReference",
      "CodeModule", "ApiEndpoint", "CodeInsight", "CodeRepository",
      "CodeExtractionAccount", "StorageResource",
    ];
    const results = normalizeCodeNodes(types.map((t, i) => node(String(i), { type: t })));
    results.forEach((r) => expect(r.stage).toBe("entity"));
  });

  it("returns an empty array for no nodes", () => {
    expect(normalizeCodeNodes([])).toEqual([]);
  });
});

describe("codeDisplayName", () => {
  it("shortens a dotted symbol to the part after the last dot", () => {
    const name = codeDisplayName(node("a", {
      type: "CodeSymbol",
      name: "cognee-saas-frontend/src/modules/tenant.isFreshlyCreatedTenant",
    }));
    expect(name).toBe("isFreshlyCreatedTenant");
  });

  it("shortens a nested dotted symbol under a scoped path", () => {
    const name = codeDisplayName(node("a", {
      type: "CodeSymbol",
      name: "src/@menu/utils.verticalNavClasses",
    }));
    expect(name).toBe("verticalNavClasses");
  });

  it("keeps a plain module name with no dot as-is", () => {
    const name = codeDisplayName(node("a", {
      type: "CodeModule",
      name: "cognee-saas-frontend/src/modules/tenant",
    }));
    expect(name).toBe("tenant");
  });

  it("keeps a filename with an extension intact when symbol_kind is document", () => {
    const name = codeDisplayName(node("a", {
      type: "CodeFileReference",
      name: ".claude/agents/pr-reviewer.md",
      symbol_kind: "document",
    }));
    expect(name).toBe("pr-reviewer.md");
  });

  it("keeps a filename with a known extension intact even without symbol_kind", () => {
    const name = codeDisplayName(node("a", {
      type: "CodeFileReference",
      name: ".github/pull_request_template.md",
    }));
    expect(name).toBe("pull_request_template.md");
  });

  it("passes a non-code node's name through unchanged", () => {
    const name = codeDisplayName(node("a", { type: "Entity", name: "Acme Corp" }));
    expect(name).toBe("Acme Corp");
  });

  it("returns an empty string for a node with no name", () => {
    expect(codeDisplayName(node("a", { type: "CodeSymbol" }))).toBe("");
  });

  it("falls back to the raw name when the last segment is empty", () => {
    const name = codeDisplayName(node("a", { type: "CodeModule", name: "src/modules/" }));
    expect(name).toBe("src/modules/");
  });

  it("does not strip a trailing dot with nothing meaningful after it", () => {
    const name = codeDisplayName(node("a", { type: "CodeSymbol", name: "src/tenant." }));
    expect(name).toBe("tenant.");
  });
});
