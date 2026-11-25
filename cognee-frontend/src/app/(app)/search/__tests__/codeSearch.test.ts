const mockSearchCode = jest.fn();
jest.mock("@/modules/datasets/searchCode", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockSearchCode(...args),
}));

import { answerCodeQuestion, formatCodeResults, formatFileContents } from "../codeSearch";
import { codeIntentFor } from "../codeIntent";
import type { CogneeInstance } from "@/modules/instances/types";

const instance = { name: "test", instanceId: "tenant-1", fetch: jest.fn() } as unknown as CogneeInstance;

const STRUCTURAL_FALLBACK = {
  operation: "query_facts",
  kinds: ["module", "route", "service", "storage"],
  limit: 200,
};

function sentCodeQuery(call = 0): unknown {
  return (mockSearchCode.mock.calls[call][1] as { codeQuery: unknown }).codeQuery;
}

afterEach(() => {
  jest.clearAllMocks();
});

describe("formatCodeResults", () => {
  const lookup = codeIntentFor("retry");

  it("lists each match with its kind and where it lives", () => {
    const text = formatCodeResults(
      [{
        operation: "query_facts",
        total: 2,
        facts: [
          { id: "1", kind: "symbol", symbol_kind: "function", name: "retry_with_backoff", file: "src/net.py", line: 42, description: "Retries a call." },
          { id: "2", kind: "module", name: "net.retry", path: "src/net/retry.py" },
        ],
      }],
      lookup,
    );

    expect(text).toContain("**2 matches for `retry`**");
    expect(text).toContain("- `retry_with_backoff` — symbol · function — `src/net.py:42`\n  Retries a call.");
    expect(text).toContain("- `net.retry` — module — `src/net/retry.py`");
  });

  it("says how many there are when it shows only the first page", () => {
    const facts = [{ id: "1", kind: "symbol", name: "retry" }];
    expect(formatCodeResults([{ operation: "query_facts", total: 312, facts }], lookup)).toContain(
      "**312 matches for `retry`**, showing the first 1",
    );
  });

  // An empty list is the pod's whole answer to a sentence. Returned bare it
  // reads as an empty workspace; what it means is "that is not a name".
  it("explains a miss instead of reporting an empty workspace", () => {
    const text = formatCodeResults([{ operation: "query_facts", total: 0, facts: [] }], codeIntentFor("where is the flux capacitor"));

    expect(text).toContain("No symbol, file or module name contains `capacitor`.");
    expect(text).toMatch(/matches names rather than answering questions/);
  });

  it("names the kind when a listing comes back empty", () => {
    expect(formatCodeResults([{ operation: "query_facts", total: 0, facts: [] }], codeIntentFor("List the API routes"))).toBe(
      "This dataset's code graph has no route entries.",
    );
  });

  it("draws the architecture as components and the dependencies between them", () => {
    const text = formatCodeResults(
      [{
        operation: "architecture",
        repos: ["acme/shop"],
        nodes: [
          { id: "1", kind: "module", name: "api", path: "src/api" },
          { id: "2", kind: "module", name: "billing", path: "src/billing" },
          { id: "3", kind: "route", name: "GET /orders" },
        ],
        edges: [{ source: "api", target: "billing", type: "imports", count: 3 }],
        stats: { nodes_total: 3, truncated: false },
      }],
      codeIntentFor("get the architecture"),
    );

    expect(text).toContain("**Architecture of `acme/shop`** — 3 components");
    expect(text).toContain("**Modules** (2)\n- `api` — `src/api`\n- `billing` — `src/billing`");
    expect(text).toContain("**Routes** (1)\n- `GET /orders`");
    expect(text).toContain("- `api` → `billing` (imports ×3)");
  });

  it("says so when the overview was cut short", () => {
    const text = formatCodeResults(
      [{ operation: "architecture", nodes: [{ id: "1", kind: "module", name: "api" }], stats: { nodes_total: 400, truncated: true } }],
      codeIntentFor("get the architecture"),
    );
    expect(text).toContain("400 components, showing 1 of 400");
  });

  it("groups the structural listing an older pod answers the overview with", () => {
    const text = formatCodeResults(
      [{
        operation: "query_facts",
        total: 2,
        facts: [{ id: "1", kind: "module", name: "api" }, { id: "2", kind: "storage", name: "orders" }],
      }],
      codeIntentFor("get the architecture"),
    );

    expect(text).toContain("**Structure of this codebase** — 2 entries");
    expect(text).toContain("**Modules** (1)\n- `api`");
    expect(text).toContain("**Storage** (1)\n- `orders`");
  });

  // One payload comes back per dataset searched, and most hold no code graph.
  it("leaves out the datasets that matched nothing when another one did", () => {
    const text = formatCodeResults(
      [
        { operation: "query_facts", total: 0, facts: [] },
        { operation: "query_facts", total: 1, facts: [{ id: "1", kind: "symbol", name: "retry" }] },
      ],
      lookup,
    );

    expect(text).toContain("**1 match for `retry`**");
    expect(text).not.toContain("No symbol");
  });

  it("reports the miss once when no dataset matched", () => {
    const miss = { operation: "query_facts" as const, total: 0, facts: [] };
    const text = formatCodeResults([miss, miss, miss], lookup);

    expect(text.match(/No symbol, file or module name contains/g)).toHaveLength(1);
  });

  it.each([
    ["no payload at all", []],
    ["an overview of nothing", [{ operation: "architecture" as const, nodes: [] }]],
  ])("points at the dataset when there is %s", (_case, results) => {
    expect(formatCodeResults(results, codeIntentFor("get the architecture"))).toMatch(
      /Code search works on datasets built from a repository/,
    );
  });
});

describe("answerCodeQuestion", () => {
  it("sends the operation the text was read as, for the dataset in scope", async () => {
    mockSearchCode.mockResolvedValue([{ operation: "query_facts", total: 0, facts: [] }]);

    await answerCodeQuestion(instance, "where is the retry handled", ["ds-1"]);

    expect(mockSearchCode).toHaveBeenCalledTimes(1);
    expect(mockSearchCode).toHaveBeenCalledWith(instance, {
      query: "where is the retry handled",
      codeQuery: { operation: "query_facts", name: "retry", limit: 50 },
      datasetIds: ["ds-1"],
    });
  });

  // "architecture" reached the pod seven weeks after code search did. A
  // workspace on the older build still holds every node the overview shows.
  it("lists the structure when the pod does not know the architecture operation", async () => {
    mockSearchCode
      .mockRejectedValueOnce(new Error("Unsupported CODE operation 'architecture'; expected one of explore, find_path, impact_analysis, query_facts, traverse."))
      .mockResolvedValueOnce([{ operation: "query_facts", total: 1, facts: [{ id: "1", kind: "module", name: "api" }] }]);

    const answer = await answerCodeQuestion(instance, "get the architecture", ["ds-1"]);

    expect(sentCodeQuery(0)).toEqual({ operation: "architecture" });
    expect(sentCodeQuery(1)).toEqual(STRUCTURAL_FALLBACK);
    expect(answer).toContain("**Structure of this codebase**");
  });

  it("does not retry a failure that is not about the operation", async () => {
    mockSearchCode.mockRejectedValue(new Error("Dataset not found."));

    await expect(answerCodeQuestion(instance, "get the architecture", ["ds-1"])).rejects.toThrow("Dataset not found.");
    expect(mockSearchCode).toHaveBeenCalledTimes(1);
  });

  it("does not retry a lookup, which has nothing older to fall back to", async () => {
    mockSearchCode.mockRejectedValue(new Error("Unsupported CODE operation 'query_facts'"));

    await expect(answerCodeQuestion(instance, "retry", ["ds-1"])).rejects.toThrow(/Unsupported CODE operation/);
    expect(mockSearchCode).toHaveBeenCalledTimes(1);
  });
});

const FILE = "src/components/billing/AddPaymentMethodModal.tsx";
const FILE_REF = {
  id: "f1",
  kind: "file_ref",
  name: FILE,
  file: FILE,
  line: 1,
  description: "file_ref: language=typescript",
  properties: { language: "typescript" },
};
const COMPONENT = {
  id: "s1",
  kind: "symbol",
  symbol_kind: "function",
  name: "src/components/billing/AddPaymentMethodModal.AddPaymentMethodModal",
  file: FILE,
  line: 14,
  relations: [
    { type: "calls", target: "src/lib/stripe.createSetupIntent" },
    { type: "calls", target: "useStripe" },
    { type: "imports", target: "@stripe/react-stripe-js" },
    { type: "part_of", target: "src/components/billing/AddPaymentMethodModal" },
  ],
};
const PROPS = {
  id: "s2",
  kind: "symbol",
  symbol_kind: "interface",
  name: "src/components/billing/AddPaymentMethodModal.AddPaymentMethodModalProps",
  file: FILE,
  line: 6,
};

function facts(list: object[]): unknown[] {
  return [{ operation: "query_facts", total: list.length, facts: list }];
}

describe("formatFileContents", () => {
  it("lists what the file defines, in source order, by the names used inside it", () => {
    const text = formatFileContents(FILE, [FILE_REF, COMPONENT, PROPS], 3);

    expect(text).toContain(`**\`${FILE}\`** — typescript — 2 entries`);
    expect(text).toContain(
      "**Symbols** (2)\n" +
        "- `AddPaymentMethodModalProps` — interface — line 6\n" +
        "- `AddPaymentMethodModal` — function — line 14\n" +
        "  calls `src/lib/stripe.createSetupIntent`, `useStripe` · imports `@stripe/react-stripe-js`",
    );
  });

  // The file's own node is the heading, and "part of this module" is true of
  // everything in it.
  it("leaves out the file itself and the containment edges", () => {
    const text = formatFileContents(FILE, [FILE_REF, COMPONENT], 2);

    expect(text).not.toContain("file_ref");
    expect(text).not.toContain("part_of");
  });

  it("caps the targets of one relation", () => {
    const busy = { ...COMPONENT, relations: Array.from({ length: 9 }, (_, i) => ({ type: "calls", target: `f${i}` })) };

    expect(formatFileContents(FILE, [busy], 1)).toContain("calls `f0`, `f1`, `f2`, `f3`, `f4`, `f5` +3 more");
  });

  it("says when it shows only part of a large file", () => {
    expect(formatFileContents(FILE, [FILE_REF, COMPONENT], 250)).toContain("1 entry, showing the first 2");
  });

  it("says so when the graph holds the file but nothing from inside it", () => {
    expect(formatFileContents(FILE, [FILE_REF], 1)).toMatch(/nothing was extracted from it/);
  });
});

describe("answerCodeQuestion on a file", () => {
  const QUESTION_TEXT = `What is in this file: ${FILE}`;

  it("asks for everything in the file by its path, in one request when the path matches", async () => {
    mockSearchCode.mockResolvedValueOnce(facts([FILE_REF, COMPONENT]));

    const answer = await answerCodeQuestion(instance, QUESTION_TEXT, ["ds-1"]);

    expect(mockSearchCode).toHaveBeenCalledTimes(1);
    expect(mockSearchCode).toHaveBeenCalledWith(instance, {
      query: QUESTION_TEXT,
      codeQuery: { operation: "query_facts", file: FILE, limit: 200 },
      datasetIds: ["ds-1"],
    });
    expect(answer).toContain("`AddPaymentMethodModal` — function — line 14");
  });

  // The exact-path index needs the path the graph stores; typed without its
  // directories, the file is found by name and asked about again.
  it("finds a file typed without its directories, then asks what it defines", async () => {
    mockSearchCode
      .mockResolvedValueOnce(facts([]))
      .mockResolvedValueOnce(facts([FILE_REF]))
      .mockResolvedValueOnce(facts([FILE_REF, COMPONENT]));

    const answer = await answerCodeQuestion(instance, "AddPaymentMethodModal.tsx", ["ds-1"]);

    expect(sentCodeQuery(0)).toEqual({ operation: "query_facts", file: "AddPaymentMethodModal.tsx", limit: 200 });
    expect(sentCodeQuery(1)).toEqual({ operation: "query_facts", name: "AddPaymentMethodModal.tsx", kinds: ["file_ref"], limit: 10 });
    expect(sentCodeQuery(2)).toEqual({ operation: "query_facts", file: FILE, limit: 200 });
    expect(answer).toContain(`**\`${FILE}\`** — typescript`);
  });

  it("asks which file was meant when the name fits several", async () => {
    mockSearchCode
      .mockResolvedValueOnce(facts([]))
      .mockResolvedValueOnce(facts([
        { id: "a", kind: "file_ref", name: "a/helper.py", file: "a/helper.py" },
        { id: "b", kind: "file_ref", name: "b/helper.py", file: "b/helper.py" },
      ]));

    const answer = await answerCodeQuestion(instance, "helper.py", ["ds-1"]);

    expect(mockSearchCode).toHaveBeenCalledTimes(2);
    expect(answer).toContain("**Several files match `helper.py`** — ask about one by its full path:");
    expect(answer).toContain("- `a/helper.py`\n- `b/helper.py`");
  });

  it("says so when no file matches", async () => {
    mockSearchCode.mockResolvedValue(facts([]));

    const answer = await answerCodeQuestion(instance, "missing.ts", ["ds-1"]);

    expect(mockSearchCode).toHaveBeenCalledTimes(2);
    expect(answer).toBe("No file in this dataset's code graph matches `missing.ts`.");
  });
});
