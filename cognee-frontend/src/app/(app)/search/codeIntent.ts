import type { CodeFactKind, CodeQuery } from "@/modules/datasets/searchCode";

/**
 * Turns what was typed into the one code-graph operation it most plausibly
 * meant. This has to happen here: the pod's code retriever runs exactly the
 * operation it is handed and nothing on the server infers one from text, so a
 * search box that forwards the sentence gets the sentence resolved as a symbol
 * name and an error back.
 *
 * Only operations that cannot fail on free text are reachable from the box.
 * The seed-based ones (explore, traverse, find_path, impact_analysis) need an
 * exact symbol and raise when it does not resolve.
 */
export interface CodeIntent {
  codeQuery: CodeQuery;
  /** Used when the pod predates `codeQuery`'s operation. */
  fallback?: CodeQuery;
  /** The name looked up, when the intent is a lookup — shown back in the answer. */
  term?: string;
  /** The kind listed, when the intent is a listing. */
  listing?: CodeFactKind;
  /** The file asked about, when the intent is what that file contains. */
  file?: string;
}

const LOOKUP_LIMIT = 50;
const LISTING_LIMIT = 200;
/** Everything one file defines fits comfortably; the pod caps a page at 500. */
export const FILE_LIMIT = 200;

// A name that ends like a source file is a file, whatever else surrounds it.
// Anchored on the extension rather than a slash so "recall.py" counts, and a
// dotted symbol such as "pkg.Widget.run" does not.
const SOURCE_FILE = /\.(tsx?|jsx?|mjs|cjs|py|go|rs|java|kt|rb|php|cs|cc|cpp|c|h|hpp|swift|scala|vue|svelte|sql)$/i;

// What an architecture overview is made of, for pods that cannot draw one.
const STRUCTURAL_KINDS: CodeFactKind[] = ["module", "route", "service", "storage"];

const ARCHITECTURE_PATTERN =
  /\b(architecture|overview|structure|diagram)\b|\bhow\b.*\b(built|organi[sz]ed|structured|laid out)\b/i;

const LISTING_PATTERNS: ReadonlyArray<{ kind: CodeFactKind; pattern: RegExp }> = [
  { kind: "route", pattern: /\b(routes?|endpoints?|apis?)\b/i },
  { kind: "module", pattern: /\bmodules?\b/i },
  { kind: "service", pattern: /\bservices?\b/i },
  { kind: "storage", pattern: /\b(storage|tables?|databases?)\b/i },
  { kind: "dependency", pattern: /\b(dependenc(y|ies)|librar(y|ies))\b/i },
];

// Words a question is built from rather than about. Dropped before picking the
// name to look up, so "where is the retry handled" searches for "retry".
const STOPWORDS = new Set([
  "a", "about", "all", "an", "and", "any", "are", "at", "by", "call", "called", "calls",
  "can", "class", "code", "defined", "do", "does", "file", "find", "for", "from",
  "function", "get", "handled", "has", "have", "how", "i", "implemented", "in", "is",
  "it", "list", "me", "method", "my", "of", "on", "or", "show", "that", "the", "there",
  "this", "to", "used", "uses", "what", "where", "which", "who", "why", "with",
]);

function stripPunctuation(token: string): string {
  // Order matters: the call suffix has to go while its parentheses are still a
  // pair, or "parse_config()" loses one and keeps the other.
  return token
    .replace(/^[`'"“”‘’([{<]+/, "")
    .replace(/[`'"“”‘’\]}>?!,;:.]+$/, "")
    .replace(/\([^()]*\)$/, "")
    .replace(/[)`'"“”‘’\]}>?!,;:.]+$/, "");
}

/** Reads as an identifier or a path rather than a word. */
function isCodeShaped(token: string): boolean {
  return /[_./:]/.test(token) || /[a-z0-9][A-Z]/.test(token);
}

function tokensOf(text: string): string[] {
  return text.split(/\s+/).map(stripPunctuation).filter((t) => t.length > 0);
}

/**
 * The one name to look up. The pod matches a single case-insensitive substring
 * against node names, so a question has to be reduced to its most
 * identifier-like word: anything code-shaped wins, otherwise the longest word
 * that is not part of the question's grammar.
 */
export function lookupTermFor(text: string): string {
  const tokens = tokensOf(text);
  const codeShaped = tokens.find(isCodeShaped);
  if (codeShaped) return codeShaped;
  if (tokens.length === 1) return tokens[0];
  const content = tokens.filter((t) => !STOPWORDS.has(t.toLowerCase()));
  if (content.length === 0) return text.trim();
  return content.reduce((best, t) => (t.length > best.length ? t : best));
}

export function codeIntentFor(text: string): CodeIntent {
  const hasIdentifier = tokensOf(text).some(isCodeShaped);

  // An identifier outranks every keyword: "UserService.list_routes" is a name
  // to look up, whatever words it happens to contain.
  if (!hasIdentifier) {
    if (ARCHITECTURE_PATTERN.test(text)) {
      return {
        codeQuery: { operation: "architecture" },
        fallback: { operation: "query_facts", kinds: STRUCTURAL_KINDS, limit: LISTING_LIMIT },
      };
    }
    const listing = LISTING_PATTERNS.find(({ pattern }) => pattern.test(text));
    if (listing) {
      return {
        codeQuery: { operation: "query_facts", kinds: [listing.kind], limit: LISTING_LIMIT },
        listing: listing.kind,
      };
    }
  }

  const term = lookupTermFor(text);
  // Asked about a file, the answer is what it defines — its functions,
  // components and types — which the pod indexes by file. A name lookup finds
  // only the file's own node, since nothing inside it is named by its path.
  if (SOURCE_FILE.test(term)) {
    return { codeQuery: { operation: "query_facts", file: term, limit: FILE_LIMIT }, term, file: term };
  }
  return { codeQuery: { operation: "query_facts", name: term, limit: LOOKUP_LIMIT }, term };
}
