import type { ReactNode } from "react";
import SkeletonBar from "@/ui/elements/SkeletonBar";

// Fixed, uneven widths so the placeholders read as conversation titles rather
// than a progress bar. Deterministic on purpose — a random width per render
// would shift on every re-render while history is still loading.
const TITLE_WIDTHS = ["78%", "58%", "88%", "64%", "72%", "50%"];

export default function HistorySkeleton({
  rows,
  showGroupLabel,
}: {
  rows: number;
  showGroupLabel: boolean;
}): ReactNode {
  return (
    <div role="status" aria-label="Loading past conversations">
      {/* Stands in for a date heading, but only when no real group is above it
          — otherwise it reads as a second heading for the group it follows. */}
      {showGroupLabel && (
        <span style={{ display: "block", padding: "10px 8px 4px" }}>
          <SkeletonBar width={54} height={9} />
        </span>
      )}
      {/* A placeholder with no rows is just an empty box that claims to be
          loading — always draw at least one. */}
      {Array.from({ length: Math.max(1, rows) }, (_, i) => (
        <span key={i} style={{ display: "block", padding: "8px 10px" }}>
          <SkeletonBar width={TITLE_WIDTHS[i % TITLE_WIDTHS.length]} height={13} />
        </span>
      ))}
    </div>
  );
}
