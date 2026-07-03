/**
 * Parsing of Cognee's answer-grounded "Evidence" block into structured citations.
 *
 * When recall is called with `include_references=true`, completion strategies
 * append a block to the answer text in this exact shape (see the server's
 * `include_references` wiring and its unit tests):
 *
 *   <answer>
 *
 *   Evidence:
 *   - chunk 3 of document report.pdf (chunk_id: chunk-1): <snippet>
 *   - chunk 5 of document notes.md (chunk_id: chunk-2): <snippet>
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

const CITATION_RE =
  /^-\s+chunk\s+(\S+)\s+of\s+document\s+(.+?)\s+\(chunk_id:\s*(.*?)\):\s*([\s\S]*)$/i;

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
    const [, chunkLabel, documentName, chunkId, snippet] = match;
    const citation: Citation = {
      chunkLabel: chunkLabel.trim(),
      documentName: documentName.trim(),
      raw,
    };
    const trimmedChunkId = chunkId.trim();
    if (trimmedChunkId) {
      citation.chunkId = trimmedChunkId;
    }
    const trimmedSnippet = snippet.trim();
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
