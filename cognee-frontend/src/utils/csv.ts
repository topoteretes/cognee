/**
 * Characters that make a spreadsheet treat a cell as a formula rather than
 * text: the OWASP CSV-injection lead set, plus a newline.
 *
 * Tab and carriage return are on that list because a spreadsheet strips them
 * before deciding. A leading SPACE is deliberately not, here or there, because
 * it genuinely prevents evaluation. This comment used to give that reason
 * backwards and claim " =cmd" was still a formula, which invites adding `\s` to
 * the class — and that would defuse legitimate values that merely begin with a
 * space, turning real numbers into text in the export.
 *
 * Newline is not part of OWASP's own lead set but belongs here regardless: CSV
 * quoting is what lets a cell hold a literal newline at all, so "\n=cmd..."
 * reaches this function verbatim and a spreadsheet still evaluates what
 * follows the newline as a formula. Without it here that payload passed
 * through undefused.
 */
const FORMULA_LEAD = /^[=+\-@\t\r\n]/;

/**
 * Quotes one CSV cell, and neutralizes a leading formula character.
 *
 * RFC 4180 quoting is not protection: the quotes are stripped at parse and
 * Excel, Sheets and LibreOffice then evaluate a cell starting `=`, `+`, `-` or
 * `@` (CWE-1236). The values here are not ours — dataset names, agent names and
 * session ids are all caller-supplied text, and `session_id` in particular is
 * validated by the pod only as non-empty. So a workspace member, or an agent
 * holding a scoped key, could plant a formula that runs on the machine of
 * whoever opens the export.
 *
 * A leading apostrophe is the standard defusal: spreadsheets read the rest as
 * literal text and do not display the quote. `String()` first because these
 * cells come from typed-but-unvalidated wire data, and `undefined.replace`
 * threw inside a click handler where nothing catches it.
 */
export function escapeCsvCell(value: string): string {
  const text = String(value ?? "");
  const safe = FORMULA_LEAD.test(text) ? `'${text}` : text;
  return `"${safe.replace(/"/g, '""')}"`;
}

/** One CSV document from a header row plus body rows, every cell quoted. */
export function toCsv(headers: readonly string[], rows: readonly (readonly string[])[]): string {
  return [headers, ...rows].map((cells) => cells.map(escapeCsvCell).join(",")).join("\n");
}

/**
 * Hands the file to the browser.
 *
 * Returns whether it got that far rather than throwing: a throw inside a click
 * handler is not caught by an error boundary in React 19, it goes to the global
 * error handler, so the button would appear to do nothing at all. The anchor is
 * attached to the document and the URL revoked on the next tick because some
 * browsers cancel a download whose blob URL is revoked in the same task as the
 * click.
 */
export function downloadCsv(filename: string, csv: string): boolean {
  try {
    // Prepend a BOM so Excel reads the UTF-8 text correctly.
    const blob = new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    return true;
  } catch (e) {
    console.error("[csv] download failed:", e instanceof Error ? e.message : e);
    return false;
  }
}
