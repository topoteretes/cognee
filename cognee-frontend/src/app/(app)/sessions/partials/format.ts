// Server returns some timestamps as naive ISO (no timezone designator), but
// they are actually UTC. Appending "Z" forces JS to parse them as UTC instead
// of local — without this, CEST users see everything offset by 2 hours.
export function parseServerIso(iso: string): Date {
  const hasTz = /Z$|[+-]\d{2}:?\d{2}$/.test(iso);
  return new Date(hasTz ? iso : iso + "Z");
}

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  return parseServerIso(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function formatRelativeTime(iso?: string | null): string {
  if (!iso) return "—";
  const d = parseServerIso(iso);
  const t = d.getTime();
  if (Number.isNaN(t)) return "—";
  const diffMs = Date.now() - t;
  const diffSec = Math.round(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function durationSeconds(s: { started_at: string | null; ended_at: string | null; last_activity_at: string | null }): number {
  if (!s.started_at) return 0;
  const start = new Date(s.started_at).getTime();
  const endIso = s.ended_at || s.last_activity_at;
  if (!endIso) return 0;
  return Math.max(0, (new Date(endIso).getTime() - start) / 1000);
}

export function formatDuration(sec: number): string {
  if (sec < 1) return `${sec.toFixed(1)}s`;
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
