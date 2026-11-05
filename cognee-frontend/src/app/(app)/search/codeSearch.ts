import searchCode from "@/modules/datasets/searchCode";
import type {
  CodeArchitectureResult,
  CodeFact,
  CodeFactsResult,
  CodeSearchResult,
} from "@/modules/datasets/searchCode";
import type { CogneeInstance } from "@/modules/instances/types";
import { codeIntentFor, FILE_LIMIT, type CodeIntent } from "@/app/(app)/search/codeIntent";

const MAX_EDGES_SHOWN = 40;

const KIND_LABELS: Record<string, { one: string; many: string }> = {
  module: { one: "module", many: "Modules" },
  route: { one: "route", many: "Routes" },
  service: { one: "service", many: "Services" },
  storage: { one: "storage resource", many: "Storage" },
  dependency: { one: "dependency", many: "Dependencies" },
  symbol: { one: "symbol", many: "Symbols" },
  test_ref: { one: "test", many: "Tests" },
  insight: { one: "insight", many: "Insights" },
  lint: { one: "lint finding", many: "Lint findings" },
};

// Where a node sits rather than what it does — and a file view already says
// where everything in it sits.
const CONTAINMENT_RELATIONS = new Set(["part_of"]);
const MAX_TARGETS_SHOWN = 6;
// How many files a partial path may resolve to before asking which one.
const FILE_CANDIDATES = 10;

function location(fact: CodeFact): string {
  const file = fact.file ?? fact.path;
  if (!file) return "";
  return typeof fact.line === "number" ? `${file}:${fact.line}` : file;
}

function factLine(fact: CodeFact, showKind: boolean): string {
  const parts = [`\`${fact.name ?? fact.id}\``];
  if (showKind) parts.push(fact.symbol_kind ? `${fact.kind} · ${fact.symbol_kind}` : fact.kind);
  const where = location(fact);
  if (where) parts.push(`\`${where}\``);
  const line = `- ${parts.join(" — ")}`;
  return fact.description ? `${line}\n  ${fact.description}` : line;
}

function groupByKind(facts: CodeFact[]): Map<string, CodeFact[]> {
  const groups = new Map<string, CodeFact[]>();
  for (const fact of facts) {
    groups.set(fact.kind, [...(groups.get(fact.kind) ?? []), fact]);
  }
  return groups;
}

function kindSections(facts: CodeFact[]): string[] {
  return [...groupByKind(facts)].map(([kind, members]) => {
    const heading = KIND_LABELS[kind]?.many ?? kind;
    return `**${heading}** (${members.length})\n${members.map((f) => factLine(f, false)).join("\n")}`;
  });
}

function formatFacts(result: CodeFactsResult, intent: CodeIntent): string {
  const shown = result.facts.length;
  if (shown === 0) {
    if (intent.listing) {
      return `This dataset's code graph has no ${KIND_LABELS[intent.listing]?.one ?? intent.listing} entries.`;
    }
    return [
      `No symbol, file or module name contains \`${intent.term ?? ""}\`.`,
      "Code search matches names rather than answering questions — try a function, class or file name, or ask for the architecture.",
    ].join("\n\n");
  }

  const more = result.total > shown ? `, showing the first ${shown}` : "";
  // A structural listing stands in for the overview on a pod that cannot draw one.
  if (!intent.term && !intent.listing) {
    return [`**Structure of this codebase** — ${result.total} entries${more}`, ...kindSections(result.facts)].join("\n\n");
  }
  if (intent.listing) {
    const label = KIND_LABELS[intent.listing]?.many ?? intent.listing;
    return [`**${label}** — ${result.total}${more}`, result.facts.map((f) => factLine(f, false)).join("\n")].join("\n\n");
  }
  const noun = result.total === 1 ? "match" : "matches";
  return [
    `**${result.total} ${noun} for \`${intent.term ?? ""}\`**${more}`,
    result.facts.map((f) => factLine(f, true)).join("\n"),
  ].join("\n\n");
}

function formatArchitecture(result: CodeArchitectureResult): string {
  if (result.nodes.length === 0) {
    return "This dataset has no code graph to draw. Code search works on datasets built from a repository.";
  }
  const repos = (result.repos ?? []).map((r) => `\`${r}\``).join(", ");
  const total = result.stats?.nodes_total ?? result.nodes.length;
  const truncated = result.stats?.truncated ? `, showing ${result.nodes.length} of ${total}` : "";
  const sections = [`**Architecture${repos ? ` of ${repos}` : ""}** — ${total} components${truncated}`, ...kindSections(result.nodes)];

  const edges = (result.edges ?? []).filter((e) => e.source && e.target);
  if (edges.length > 0) {
    const lines = edges.slice(0, MAX_EDGES_SHOWN).map((e) => {
      const weight = e.count && e.count > 1 ? ` ×${e.count}` : "";
      return `- \`${e.source}\` → \`${e.target}\` (${e.type}${weight})`;
    });
    const rest = edges.length > MAX_EDGES_SHOWN ? `\n- … and ${edges.length - MAX_EDGES_SHOWN} more` : "";
    sections.push(`**How they depend on each other**\n${lines.join("\n")}${rest}`);
  }
  return sections.join("\n\n");
}

function isEmpty(result: CodeSearchResult): boolean {
  return result.operation === "architecture" ? result.nodes.length === 0 : result.facts.length === 0;
}

export function formatCodeResults(results: CodeSearchResult[], intent: CodeIntent): string {
  if (results.length === 0) {
    return "The workspace returned nothing for this dataset. Code search works on datasets built from a repository.";
  }
  // One payload per dataset searched. A dataset with no code graph answers
  // with an empty one, which is worth saying once if nothing matched anywhere
  // and not at all next to a dataset that did.
  const found = results.filter((result) => !isEmpty(result));
  return (found.length > 0 ? found : results.slice(0, 1))
    .map((result) => (result.operation === "architecture" ? formatArchitecture(result) : formatFacts(result, intent)))
    .join("\n\n---\n\n");
}

/** "inventory/pricing.py" defines "inventory/pricing.line_total"; inside the file it is "line_total". */
function nameInFile(fact: CodeFact, file: string): string {
  const name = fact.name ?? fact.id;
  const modulePath = file.replace(/\.[^./]+$/, "");
  return name.startsWith(`${modulePath}.`) ? name.slice(modulePath.length + 1) : name;
}

function relationsLine(fact: CodeFact): string {
  const byType = new Map<string, string[]>();
  for (const relation of fact.relations ?? []) {
    if (!relation.target || CONTAINMENT_RELATIONS.has(relation.type)) continue;
    byType.set(relation.type, [...(byType.get(relation.type) ?? []), relation.target]);
  }
  return [...byType]
    .map(([type, targets]) => {
      const shown = targets.slice(0, MAX_TARGETS_SHOWN).map((t) => `\`${t}\``).join(", ");
      const rest = targets.length > MAX_TARGETS_SHOWN ? ` +${targets.length - MAX_TARGETS_SHOWN} more` : "";
      return `${type} ${shown}${rest}`;
    })
    .join(" · ");
}

function fileEntryLine(fact: CodeFact, file: string): string {
  const parts = [`\`${nameInFile(fact, file)}\``];
  if (fact.symbol_kind) parts.push(fact.symbol_kind);
  if (typeof fact.line === "number") parts.push(`line ${fact.line}`);
  const lines = [`- ${parts.join(" — ")}`];
  if (fact.description) lines.push(`  ${fact.description}`);
  const relations = relationsLine(fact);
  if (relations) lines.push(`  ${relations}`);
  return lines.join("\n");
}

/** What one file defines, in source order, grouped by kind. */
export function formatFileContents(file: string, facts: CodeFact[], total: number): string {
  const own = facts.find((f) => f.kind === "file_ref");
  const language = own?.properties?.language;
  const heading = `**\`${file}\`**${typeof language === "string" ? ` — ${language}` : ""}`;
  const entries = facts.filter((f) => f.kind !== "file_ref").sort((a, b) => (a.line ?? 0) - (b.line ?? 0));
  if (entries.length === 0) {
    return `${heading}\n\nThe code graph has this file, but nothing was extracted from it — no functions, classes or other symbols.`;
  }
  const more = total > facts.length ? `, showing the first ${facts.length}` : "";
  const sections = [...groupByKind(entries)].map(
    ([kind, members]) =>
      `**${KIND_LABELS[kind]?.many ?? kind}** (${members.length})\n${members.map((f) => fileEntryLine(f, file)).join("\n")}`,
  );
  return [`${heading} — ${entries.length} ${entries.length === 1 ? "entry" : "entries"}${more}`, ...sections].join("\n\n");
}

function factsIn(results: CodeSearchResult[]): { facts: CodeFact[]; total: number } {
  const lists = results.filter((r): r is CodeFactsResult => r.operation === "query_facts");
  return { facts: lists.flatMap((r) => r.facts), total: lists.reduce((sum, r) => sum + r.total, 0) };
}

/**
 * What a file contains. Asked by the path the graph stores, one request
 * answers it. The path typed is often not that one — no directories, or
 * relative to somewhere else — so a miss looks the file up by name (its own
 * node is named by its path) and, when that finds exactly one, asks again.
 */
async function answerFileQuestion(
  instance: CogneeInstance,
  text: string,
  file: string,
  datasetIds: string[],
): Promise<string> {
  const contentsOf = async (path: string) =>
    factsIn(await searchCode(instance, { query: text, codeQuery: { operation: "query_facts", file: path, limit: FILE_LIMIT }, datasetIds }));

  const direct = await contentsOf(file);
  if (direct.facts.length > 0) return formatFileContents(file, direct.facts, direct.total);

  const found = factsIn(
    await searchCode(instance, {
      query: text,
      codeQuery: { operation: "query_facts", name: file, kinds: ["file_ref"], limit: FILE_CANDIDATES },
      datasetIds,
    }),
  );
  const paths = [...new Set(found.facts.map((f) => f.file ?? f.name ?? "").filter((path) => path.length > 0))];
  if (paths.length === 1) {
    const resolved = await contentsOf(paths[0]);
    return formatFileContents(paths[0], resolved.facts, resolved.total);
  }
  if (paths.length > 1) {
    const more = found.total > paths.length ? ` (showing ${paths.length} of ${found.total})` : "";
    return [`**Several files match \`${file}\`**${more} — ask about one by its full path:`, paths.map((path) => `- \`${path}\``).join("\n")].join("\n\n");
  }
  return `No file in this dataset's code graph matches \`${file}\`.`;
}

/**
 * The pod names the operation it does not know; everything else it raises
 * about a code search is a real failure and is left to surface.
 */
function isUnsupportedOperation(err: unknown): boolean {
  return err instanceof Error && /unsupported code operation/i.test(err.message);
}

/** One Code-mode turn, from the typed text to the text of the answer bubble. */
export async function answerCodeQuestion(
  instance: CogneeInstance,
  text: string,
  datasetIds: string[],
): Promise<string> {
  const intent = codeIntentFor(text);
  if (intent.file) return answerFileQuestion(instance, text, intent.file, datasetIds);
  try {
    const results = await searchCode(instance, { query: text, codeQuery: intent.codeQuery, datasetIds });
    return formatCodeResults(results, intent);
  } catch (err) {
    // "architecture" is newer than code search itself. A pod without it still
    // has every node the overview is drawn from, so list those instead of
    // failing a question the workspace can answer.
    if (!intent.fallback || !isUnsupportedOperation(err)) throw err;
    const results = await searchCode(instance, { query: text, codeQuery: intent.fallback, datasetIds });
    return formatCodeResults(results, intent);
  }
}
