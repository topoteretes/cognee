/**
 * Text handed to the pod without a filename is stored as "text_<md5>.txt".
 *
 * That used to mean one thing — a memory written through the memory API — and
 * the UI labelled every such blob "Memory" on that basis. It no longer does:
 * connector content (Drive, GitHub) arrives the same way, so the hash alone
 * says only "this came in as text", never where from. What the blob actually
 * is has to be read out of its contents.
 */
export function isTextBlobName(name: string): boolean {
  return /^text_[0-9a-f]{16,}(\.txt)?$/i.test(name);
}

export type TextBlobDescription =
  | { kind: "memory"; sessionId: string }
  | { kind: "text"; title: string };

/** Shown when a blob cannot be read, or has nothing usable to show. */
export const TEXT_BLOB_FALLBACK = "Text";

const TITLE_MAX = 80;
/** Markdown ornaments on the first line describe the text, they do not name it. */
const LEADING_ORNAMENT = /^[#>\-*\s]+/;

/** A title short enough for one table cell; the full text belongs in a tooltip. */
export function clipTitle(text: string): string {
  return text.length > TITLE_MAX ? `${text.slice(0, TITLE_MAX).trimEnd()}…` : text;
}

function firstLineTitle(raw: string): string {
  for (const line of raw.split("\n", 40)) {
    const cleaned = line.replace(LEADING_ORNAMENT, "").trim().replace(/\s+/g, " ");
    if (!cleaned) continue;
    return clipTitle(cleaned);
  }
  return "";
}

const REPOSITORY_LABEL_PREFIX = "code_repo:";

/** The slice of a dataset data row that says what it is. */
export interface NamedRow {
  id: string;
  name: string;
  label?: unknown;
}

/**
 * The repository a connector-synced row stands for, read from its `label`.
 *
 * A synced repository is stored as a hashed text blob too, but unlike an
 * anonymous one it arrives already knowing what it is, so there is nothing to
 * recover from its content. Only a hashed name qualifies: a row with a real
 * filename keeps it whatever its label says. The label is the directory the
 * repository was cloned into (`github.com-<owner>-<repo>`), and both owner and
 * repo can hold hyphens, so it is shown as given rather than split. Returned
 * whole: a cell clips it with `clipTitle`, but search, the delete prompt and
 * analytics need the name that tells two long repositories apart.
 * `null` for any other row.
 */
export function connectorRepository(row: Pick<NamedRow, "name" | "label">): string | null {
  if (!isTextBlobName(row.name)) return null;
  const { label } = row;
  if (typeof label !== "string" || !label.startsWith(REPOSITORY_LABEL_PREFIX)) return null;
  const repository = label.slice(REPOSITORY_LABEL_PREFIX.length).trim();
  return repository || null;
}

/** A hashed blob that has to be opened to find out what it is. */
export function needsContentRead(row: Pick<NamedRow, "name" | "label">): boolean {
  return isTextBlobName(row.name) && connectorRepository(row) === null;
}

/** The rows whose content still has to be read: unnamed, and not read yet. */
export function blobsToRead<T extends NamedRow>(rows: T[], described: Record<string, unknown>): T[] {
  return rows.filter((row) => needsContentRead(row) && !(row.id in described));
}

/**
 * What a hashed text blob is, read from its own first lines.
 *
 * The `Session ID:` header is written by the session→graph bridge, so it is
 * the one positive proof that a blob is a memory. Anything else is text from
 * somewhere this app cannot name — and saying "Text" plus its opening line is
 * honest, where "Memory" would be a claim about an origin nobody checked.
 */
export default function describeTextBlob(raw: string): TextBlobDescription {
  const session = raw.match(/^Session ID:\s*(\S+)/);
  if (session) return { kind: "memory", sessionId: session[1] };
  return { kind: "text", title: firstLineTitle(raw) };
}

/** The row title for a hashed blob; `null` is a read that failed. */
export function textBlobLabel(description: TextBlobDescription | null | undefined): string {
  if (!description) return TEXT_BLOB_FALLBACK;
  if (description.kind === "memory") return `Memory · ${description.sessionId}`;
  return description.title || TEXT_BLOB_FALLBACK;
}
