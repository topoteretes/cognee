"use client";

import type { ReactElement } from "react";

/**
 * Page controls for a document list.
 *
 * The list endpoint is paginated, so a rendered list is one page of a dataset
 * that may hold six figures of documents. Without this the first page looks
 * like the whole dataset — which is worse than being slow, because it is wrong
 * and says nothing about being wrong.
 */
export default function Pager({
  page,
  pageSize,
  total,
  busy,
  onGoTo,
  compact,
}: {
  page: number;
  pageSize: number;
  total: number;
  busy?: boolean;
  onGoTo: (page: number) => void;
  compact?: boolean;
}): ReactElement | null {
  const lastPage = Math.max(0, Math.ceil(total / pageSize) - 1);

  // One page or fewer: the list already is the dataset, so say nothing.
  if (total <= pageSize) return null;

  const first = page * pageSize + 1;
  const last = Math.min((page + 1) * pageSize, total);
  const fontSize = compact ? 11 : 12;

  const button = (label: string, target: number, disabled: boolean) => (
    <button
      onClick={() => onGoTo(target)}
      disabled={disabled || busy}
      className={disabled || busy ? undefined : "cursor-pointer hover:bg-white/10"}
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 6,
        padding: compact ? "2px 8px" : "4px 10px",
        fontSize,
        fontWeight: 500,
        color: disabled || busy ? "rgba(237,236,234,0.25)" : "rgba(237,236,234,0.7)",
        cursor: disabled || busy ? "default" : "pointer",
      }}
    >
      {label}
    </button>
  );

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        padding: compact ? "8px 16px" : "12px 20px",
        borderTop: "1px solid rgba(255,255,255,0.07)",
      }}
    >
      <span style={{ fontSize, color: "rgba(237,236,234,0.45)", whiteSpace: "nowrap" }}>
        {first.toLocaleString()}&ndash;{last.toLocaleString()} of {total.toLocaleString()}
      </span>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {button("First", 0, page === 0)}
        {button("Prev", page - 1, page === 0)}
        <span style={{ fontSize, color: "rgba(237,236,234,0.45)", padding: "0 4px", whiteSpace: "nowrap" }}>
          {(page + 1).toLocaleString()} / {(lastPage + 1).toLocaleString()}
        </span>
        {button("Next", page + 1, page >= lastPage)}
        {button("Last", lastPage, page >= lastPage)}
      </div>
    </div>
  );
}
