"use client";

import { type ReactElement, type RefObject, useEffect, useRef } from "react";

/**
 * Loads the next page when it scrolls into view.
 *
 * The list endpoint is paginated, so a rendered list is one page of a dataset
 * that may hold six figures of documents. Page buttons made every document
 * reachable but put a click between the reader and the next hundred rows;
 * scrolling is how a file list is expected to behave.
 *
 * Loading is still bounded. Rows are real DOM — the whole reason the list is
 * paged is that rendering all of them froze the tab — so appending stops at
 * `maxLoaded` and says so, rather than degrading the page the further someone
 * scrolls.
 *
 * `rootRef` must be given when the list scrolls inside its own element:
 * IntersectionObserver watches the viewport unless told otherwise, and a
 * sentinel inside an inner scroller never intersects it.
 */
export default function ScrollLoader({
  loaded,
  total,
  maxLoaded,
  busy,
  onLoadMore,
  rootRef,
  noun = "documents",
  compact,
}: {
  loaded: number;
  total: number;
  maxLoaded: number;
  busy?: boolean;
  onLoadMore: () => void;
  rootRef?: RefObject<HTMLElement | null>;
  noun?: string;
  compact?: boolean;
}): ReactElement | null {
  const sentinel = useRef<HTMLDivElement | null>(null);

  const atCap = loaded >= maxLoaded;
  const hasMore = loaded < total && !atCap;

  // onLoadMore is typically a fresh closure each render; a ref keeps the
  // observer from being torn down and rebuilt on every one of them.
  const loadMore = useRef(onLoadMore);
  loadMore.current = onLoadMore;

  useEffect(() => {
    if (!hasMore || busy) return;

    const target = sentinel.current;
    if (!target || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) loadMore.current();
      },
      {
        root: rootRef?.current ?? null,
        // Start fetching before the sentinel is actually on screen, so the
        // next rows are usually there by the time the reader reaches them.
        rootMargin: "300px",
      },
    );

    observer.observe(target);
    return () => observer.disconnect();
  }, [hasMore, busy, rootRef]);

  // Nothing to say when the list already is the dataset.
  if (total <= loaded && !atCap) return null;

  const fontSize = compact ? 11 : 12;
  const padding = compact ? "10px 16px" : "14px 20px";
  const muted = "rgba(237,236,234,0.4)";

  return (
    <div
      ref={sentinel}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        padding,
        fontSize,
        color: muted,
        borderTop: "1px solid rgba(255,255,255,0.07)",
      }}
    >
      {busy && hasMore ? (
        <span>Loading more…</span>
      ) : (
        <span>
          {loaded.toLocaleString()} of {total.toLocaleString()} {noun}
        </span>
      )}

      {atCap && loaded < total && (
        <span style={{ color: "rgba(237,236,234,0.3)", textAlign: "center" }}>
          Showing the first {maxLoaded.toLocaleString()} — search to narrow the list.
        </span>
      )}

      {/* Scrolling is the normal path; the button is the fallback for anyone
          whose browser has no IntersectionObserver, and for keyboard users. */}
      {hasMore && !busy && (
        <button
          onClick={() => loadMore.current()}
          className="cursor-pointer hover:bg-white/10"
          style={{
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 6,
            padding: compact ? "2px 10px" : "4px 12px",
            fontSize,
            fontWeight: 500,
            color: "rgba(237,236,234,0.7)",
            cursor: "pointer",
          }}
        >
          Load more
        </button>
      )}
    </div>
  );
}
