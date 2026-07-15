"use client";

import { useState, useRef, useEffect } from "react";

const USE_CASES = [
  "A second brain",
  "Sales & deal intelligence",
  "Investment & research",
  "Docs & manuals",
  "Memory for coding agents",
] as const;

/**
 * Horizontally-scrollable use-case card strip with drag-to-scroll, edge fades,
 * and arrow button affordances. Self-contained — no external props needed.
 */
export function UseCaseSlider(): React.ReactElement {
  const sliderRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const startX = useRef(0);
  const scrollLeft = useRef(0);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;
    const update = () => {
      setCanScrollLeft(el.scrollLeft > 4);
      setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, []);

  function scrollBy(delta: number) {
    sliderRef.current?.scrollBy({ left: delta, behavior: "smooth" });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", lineHeight: "24px" }}>
          What you can build
        </h2>
        <p style={{ margin: "3px 0 0", fontSize: 13, color: "rgba(237,236,234,0.65)" }}>
          Persistent memory and knowledge graphs for any domain
        </p>
      </div>

      <style>{`
        .usecase-slider { overflow-x: auto; scrollbar-width: none; -ms-overflow-style: none; cursor: grab; user-select: none; }
        .usecase-slider::-webkit-scrollbar { display: none; }
        .usecase-slider.is-dragging { cursor: grabbing; }
        .usecase-card { transition: border-color 200ms, box-shadow 200ms, background 200ms; flex: 0 0 280px; height: 160px; border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); text-decoration: none; display: flex; align-items: center; justify-content: center; padding: 24px; background: rgba(0,0,0,0.45); backdrop-filter: blur(12px); text-align: center; }
        .usecase-card:hover { border-color: var(--color-cognee-lavender-tint-35); box-shadow: 0 8px 32px var(--color-cognee-lavender-tint-20); background: rgba(0,0,0,0.6); }
      `}</style>

      {/* Drag-to-scroll strip with conditional edge fades + arrow buttons.
          Fades only appear when there's content in that direction — otherwise
          the leftmost card's border would be occluded by a constant fade. */}
      <div style={{ position: "relative" }}>
        {canScrollLeft && (
          <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 64, background: "linear-gradient(to right, #000000, rgba(0,0,0,0))", zIndex: 2, pointerEvents: "none" }} />
        )}
        {canScrollRight && (
          <div style={{ position: "absolute", right: 0, top: 0, bottom: 0, width: 64, background: "linear-gradient(to left, #000000, rgba(0,0,0,0))", zIndex: 2, pointerEvents: "none" }} />
        )}

        {canScrollLeft && (
          <button
            onClick={() => scrollBy(-320)}
            aria-label="Scroll use cases left"
            style={{ position: "absolute", left: 8, top: "50%", transform: "translateY(-50%)", zIndex: 3, width: 36, height: 36, borderRadius: "50%", border: "1px solid rgba(255,255,255,0.15)", background: "rgba(20,20,22,0.85)", backdropFilter: "blur(8px)", color: "#EDECEA", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        )}
        {canScrollRight && (
          <button
            onClick={() => scrollBy(320)}
            aria-label="Scroll use cases right"
            style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", zIndex: 3, width: 36, height: 36, borderRadius: "50%", border: "1px solid rgba(255,255,255,0.15)", background: "rgba(20,20,22,0.85)", backdropFilter: "blur(8px)", color: "#EDECEA", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        )}

        <div
          ref={sliderRef}
          className="usecase-slider"
          onMouseDown={(e) => {
            dragging.current = true;
            startX.current = e.pageX - (sliderRef.current?.offsetLeft ?? 0);
            scrollLeft.current = sliderRef.current?.scrollLeft ?? 0;
            sliderRef.current?.classList.add("is-dragging");
          }}
          onMouseMove={(e) => {
            if (!dragging.current || !sliderRef.current) return;
            e.preventDefault();
            const x = e.pageX - sliderRef.current.offsetLeft;
            sliderRef.current.scrollLeft = scrollLeft.current - (x - startX.current) * 1.2;
          }}
          onMouseUp={() => { dragging.current = false; sliderRef.current?.classList.remove("is-dragging"); }}
          onMouseLeave={() => { dragging.current = false; sliderRef.current?.classList.remove("is-dragging"); }}
          style={{ display: "flex", gap: 16, padding: "4px 0 8px" }}
        >
          {USE_CASES.map((title) => (
            <a
              key={title}
              href="https://docs.cognee.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="usecase-card"
              draggable={false}
              onClick={(e) => {
                // Block navigation when the user was dragging (scroll ≠ pre-drag position).
                if (sliderRef.current && sliderRef.current.scrollLeft !== scrollLeft.current) {
                  e.preventDefault();
                }
              }}
            >
              <span style={{
                fontSize: 22,
                fontWeight: 500,
                color: "#EDECEA",
                fontFamily: '"TWKLausanne", sans-serif',
                letterSpacing: "0.08em",
                lineHeight: 1.25,
                textTransform: "uppercase",
              }}>
                {title}
              </span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
