/**
 * Question search: tokenized, lightly stemmed and prefix-tolerant, so
 * "brain" finds "brains", "connect" finds "connecting", and half-typed
 * words match as you type. Every query word must match somewhere in the
 * question's searchable text (question, answer, topic label, reference).
 */

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

/**
 * Light English stemmer — just enough to unify singular/plural and the
 * common verb endings without dragging in a library for fixture search.
 */
function stem(word: string): string {
  if (word.length > 4 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.length > 4 && word.endsWith("ing")) return word.slice(0, -3);
  if (word.length > 3 && word.endsWith("ed")) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("es")) return word.slice(0, -2);
  if (word.length > 2 && word.endsWith("s") && !word.endsWith("ss")) return word.slice(0, -1);
  return word;
}

function tokenMatches(docToken: string, queryToken: string): boolean {
  if (docToken.startsWith(queryToken)) return true;
  const docStem = stem(docToken);
  const queryStem = stem(queryToken);
  return docStem === queryStem || docStem.startsWith(queryStem);
}

/** True when every word of `query` matches some word of `haystack`. */
export function matchesQuery(haystack: string, query: string): boolean {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return true;
  const docTokens = tokenize(haystack);
  return queryTokens.every((qt) => docTokens.some((dt) => tokenMatches(dt, qt)));
}
