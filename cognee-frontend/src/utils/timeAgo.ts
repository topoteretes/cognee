/**
 * Coarse "how long ago" for timestamps shown next to live state.
 *
 * Lived in AgentActivityTerminal, a 991-line client component. Importing eight
 * lines of arithmetic from there pulled the whole terminal into any page that
 * wanted a relative time, so it moved out; that module re-exports it and its
 * callers are unchanged.
 */
export function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
