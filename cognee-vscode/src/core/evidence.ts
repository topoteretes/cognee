/**
 * Parsing of Cognee's answer-grounded "Evidence" block into structured citations.
 *
 * When recall is called with `include_references=true`, completion strategies
 * append a block to the answer text in this shape:
 *
 *   <answer>
 *
 *   Evidence:
 *   - chunk 1 of document report.pdf (data_id: <uuid>, chunk_id: <uuid>): "<snippet>"
 *   - chunk 5 of document notes.md (chunk_id: chunk-2): <snippet>
 *
 * The parenthetical carries one or more comma-separated `key: value` pairs. The
 * current server emits `(data_id: …, chunk_id: …)`; older builds emitted just
 * `(chunk_id: …)`, and the snippet may or may not be wrapped in quotes — so we
 * capture the whole parenthetical, pull ids out of it by name, and unwrap the
 * snippet. This keeps parsing resilient to key order and additive metadata.
 *
 * Citations are chunk/document-level today because `DocumentChunk` carries a
 * `chunk_index` and `document_name` but no line offsets. The VS Code layer
 * resolves a citation to a location by opening the document and revealing the
 * best text match for the snippet.
 */

export interface Citation {
  /** The chunk ordinal as printed by the server (e.g. "3"). */
  chunkLabel: string;
  /** The source document name (basename), used to resolve a file in the workspace. */
  documentName: string;
  /** The chunk id, when present. */
  chunkId?: string;
  /** The source data id, when present. */
  dataId?: string;
  /** A short snippet of the cited chunk, used to reveal the region in-file. */
  snippet?: string;
  /** The raw citation line, preserved verbatim. */
  raw: string;
}

export interface ParsedAnswer {
  /** The answer text with the Evidence block removed. */
  answer: string;
  /** The raw Evidence block, when present. */
  evidenceText?: string;
  /** Structured citations parsed from the Evidence block. */
  citations: Citation[];
}

const EVIDENCE_MARKER = "\n\nEvidence:\n";

/**
 * A single citation line. The metadata parenthetical is optional and captured as
 * a whole (`[^)]*`) so ids can be extracted by name regardless of order; the
 * snippet is everything after the trailing colon.
 */
const CITATION_RE =
  /^-\s+chunk\s+(\S+)\s+of\s+document\s+(.+?)(?:\s+\(([^)]*)\))?:\s*([\s\S]*)$/i;

const CHUNK_ID_RE = /chunk_id:\s*([^,)\s]+)/i;
const DATA_ID_RE = /data_id:\s*([^,)\s]+)/i;

/** Split an answer string into its prose answer and structured citations. */
export function parseAnswer(text: string): ParsedAnswer {
  if (typeof text !== "string") {
    return { answer: text == null ? "" : String(text), citations: [] };
  }

  const markerIndex = text.indexOf(EVIDENCE_MARKER);
  if (markerIndex === -1) {
    return { answer: text, citations: [] };
  }

  const answer = text.slice(0, markerIndex);
  const evidenceText = text.slice(markerIndex + EVIDENCE_MARKER.length);
  return { answer, evidenceText, citations: parseEvidenceBlock(evidenceText) };
}

/** Parse the citation lines of an Evidence block. Non-matching lines are ignored. */
export function parseEvidenceBlock(block: string): Citation[] {
  const citations: Citation[] = [];
  let buffer: string[] = [];

  const flush = (): void => {
    if (buffer.length === 0) {
      return;
    }
    const raw = buffer.join("\n").trim();
    buffer = [];
    const match = CITATION_RE.exec(raw);
    if (!match) {
      return;
    }
    const [, chunkLabel, documentName, metadata, snippet] = match;
    const citation: Citation = {
      chunkLabel: chunkLabel.trim(),
      documentName: documentName.trim(),
      raw,
    };
    const chunkId = extractId(metadata, CHUNK_ID_RE);
    if (chunkId) {
      citation.chunkId = chunkId;
    }
    const dataId = extractId(metadata, DATA_ID_RE);
    if (dataId) {
      citation.dataId = dataId;
    }
    const trimmedSnippet = unwrapQuotes(snippet);
    if (trimmedSnippet) {
      citation.snippet = trimmedSnippet;
    }
    citations.push(citation);
  };

  for (const line of block.split("\n")) {
    if (/^-\s+/.test(line)) {
      flush();
    }
    buffer.push(line);
  }
  flush();

  return citations;
}

/** Pull a single id value out of the metadata parenthetical, if present. */
function extractId(metadata: string | undefined, pattern: RegExp): string | undefined {
  if (!metadata) {
    return undefined;
  }
  const match = pattern.exec(metadata);
  return match ? match[1].trim() : undefined;
}

/** Trim a snippet and remove a single pair of enclosing quotes, if any. */
function unwrapQuotes(snippet: string | undefined): string {
  const text = (snippet ?? "").trim();
  if (text.length >= 2) {
    const first = text[0];
    const last = text[text.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return text.slice(1, -1).trim();
    }
  }
  return text;
}
