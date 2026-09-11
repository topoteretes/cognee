"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

interface ScrollFadeContainerProps {
  className?: string;
  children: ReactNode;
}

// Reviewer feedback: a source list long enough to overflow gave no signal
// that there was more to scroll to. Tracks scroll position and only renders
// the "more below" fade while there's actually unscrolled content below, so
// it never lingers once the list is fully scrolled or short enough to fit.
export default function ScrollFadeContainer({ className, children }: ScrollFadeContainerProps): React.JSX.Element {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [canScrollMore, setCanScrollMore] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = (): void => {
      setCanScrollMore(el.scrollHeight - el.scrollTop - el.clientHeight > 4);
    };
    update();
    el.addEventListener("scroll", update);
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [children]);

  return (
    <div className="relative h-full">
      <div ref={scrollRef} className={className}>
        {children}
      </div>
      {canScrollMore && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex h-8 items-end justify-center bg-gradient-to-t from-[#0E1526] to-transparent pb-1 text-[10px] text-[#7E8CA6]">
          ▾
        </div>
      )}
    </div>
  );
}
