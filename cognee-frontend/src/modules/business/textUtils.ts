// Shared by anything capping a string with an ellipsis — the Q&A list in
// NodePanel and SessionMemoryCard, the live-events quiet chip and narration
// in BusinessView. businessDraw.ts's truncateLabel is intentionally separate
// (fixed cap, no `max` param) — it caps canvas labels, a different shape of
// problem than truncating a line of UI text.
export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}
