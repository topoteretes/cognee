import { codeIntentFor, lookupTermFor } from "../codeIntent";

const STRUCTURAL_FALLBACK = {
  operation: "query_facts",
  kinds: ["module", "route", "service", "storage"],
  limit: 200,
};

describe("codeIntentFor", () => {
  // The literal report this change answers: the sentence was sent as a symbol
  // name and came back "could not resolve explore seed name".
  it.each([
    "get the architecture",
    "Show the architecture",
    "give me an overview",
    "what is the structure of this repo",
    "how is this project organized",
    "How is the service built?",
  ])("reads %p as a request for the architecture", (text) => {
    expect(codeIntentFor(text).codeQuery).toEqual({ operation: "architecture" });
  });

  it("carries a listing of the structural nodes for pods that cannot draw the overview", () => {
    expect(codeIntentFor("get the architecture").fallback).toEqual(STRUCTURAL_FALLBACK);
  });

  it.each([
    ["List the API routes", "route"],
    ["show me the endpoints", "route"],
    ["List the modules", "module"],
    ["which services are there", "service"],
    ["what tables does it use", "storage"],
    ["list dependencies", "dependency"],
  ])("reads %p as a listing of %s nodes", (text, kind) => {
    expect(codeIntentFor(text)).toMatchObject({
      codeQuery: { operation: "query_facts", kinds: [kind], limit: 200 },
      listing: kind,
    });
  });

  it.each([
    ["retry", "retry"],
    ["CodeRetriever", "CodeRetriever"],
    ["where is the retry handled", "retry"],
    ["who calls resolve_seeds?", "resolve_seeds"],
    ["find parse_config()", "parse_config"],
    ["How does the UserService work?", "UserService"],
    ['where is "tokenizer" defined', "tokenizer"],
  ])("looks up the name in %p", (text, term) => {
    expect(codeIntentFor(text)).toEqual({
      codeQuery: { operation: "query_facts", name: term, limit: 50 },
      term,
    });
  });

  // A name is a name whatever words it contains.
  it.each([
    ["ArchitectureService.build", "ArchitectureService.build"],
    ["list_routes", "list_routes"],
    ["src/modules/overview", "src/modules/overview"],
  ])("does not mistake the identifier %p for a keyword", (text, term) => {
    expect(codeIntentFor(text).codeQuery).toEqual({ operation: "query_facts", name: term, limit: 50 });
  });

  // The pod only reads the query text when code_query holds nothing but the
  // operation, and drops it silently the moment a limit or a kind is added.
  // The name therefore always travels in code_query.
  it("never relies on the query text to carry the name", () => {
    for (const text of ["retry", "where is the retry handled", "List the modules", "get the architecture"]) {
      const { codeQuery } = codeIntentFor(text);
      const carriesItsOwnSubject = codeQuery.operation === "architecture" || "name" in codeQuery || "kinds" in codeQuery;
      expect(carriesItsOwnSubject).toBe(true);
    }
  });

  // The seed-based operations raise when the text does not resolve to a symbol.
  it.each(["get the architecture", "retry", "where is x", "List the modules", "??", "a the of"])(
    "only ever picks an operation that cannot fail on free text (%p)",
    (text) => {
      expect(["architecture", "query_facts"]).toContain(codeIntentFor(text).codeQuery.operation);
    },
  );
});

describe("lookupTermFor", () => {
  it("falls back to the whole text when every word is grammar", () => {
    expect(lookupTermFor("what is it")).toBe("what is it");
  });

  it("prefers the first identifier over a longer plain word", () => {
    expect(lookupTermFor("implementation of get_user somewhere")).toBe("get_user");
  });
});

describe("codeIntentFor on a file", () => {
  // The report: asked what a file contains, the path was looked up as a name
  // and matched only the file's own node — nothing inside a file is named by
  // its path.
  it.each([
    ["What is in this file: src/components/billing/AddPaymentMethodModal.tsx", "src/components/billing/AddPaymentMethodModal.tsx"],
    ["src/components/billing/AddPaymentMethodModal.tsx", "src/components/billing/AddPaymentMethodModal.tsx"],
    ["what does recall.py define?", "recall.py"],
    ["show me `cognee/api/client.py`", "cognee/api/client.py"],
    ["what's in app/page.jsx", "app/page.jsx"],
    ["what does `recall.py` do", "recall.py"],
    // A keyword inside a path is part of the path, not a request for the overview.
    ["src/modules/overview.ts", "src/modules/overview.ts"],
  ])("asks what %p defines, by file", (text, file) => {
    expect(codeIntentFor(text)).toEqual({
      codeQuery: { operation: "query_facts", file, limit: 200 },
      term: file,
      file,
    });
  });

  // A dotted symbol is not a file, and a directory has no one file to open.
  it.each([
    ["pkg.Widget.run", "pkg.Widget.run"],
    ["src/components/billing", "src/components/billing"],
    ["CodeRetriever", "CodeRetriever"],
  ])("still looks %p up by name", (text, term) => {
    expect(codeIntentFor(text)).toEqual({ codeQuery: { operation: "query_facts", name: term, limit: 50 }, term });
  });
});
