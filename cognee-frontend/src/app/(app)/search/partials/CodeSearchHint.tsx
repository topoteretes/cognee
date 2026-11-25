"use client";

import { GITHUB_HINT } from "@/app/(app)/search/searchTypes";

function InfoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.45)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

/**
 * Says which type a repository wants, because nothing else does. A GitHub
 * repository is ingested into the code graph, which only the "code" scope
 * reads — so a question asked under Text finds nothing and looks like an empty
 * workspace rather than the wrong lane.
 */
export default function CodeSearchHint(): React.ReactElement {
  return (
    <div
      role="note"
      data-testid="code-search-hint"
      style={{
        display: "flex", alignItems: "center", gap: 6,
        // The dataset picker's chrome, like everything else on this row — the
        // icon and the muted label are what mark it as a note rather than a
        // control, not a different box.
        background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8, padding: "6px 10px",
        fontSize: 13, color: "rgba(237,236,234,0.65)",
      }}
    >
      <InfoIcon />
      <span>{GITHUB_HINT}</span>
    </div>
  );
}
