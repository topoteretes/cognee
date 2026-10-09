"use client";

import type { ReactElement } from "react";
import classNames from "classnames";
import SkeletonBar from "@/ui/elements/SkeletonBar";

export type CtaVariant = "primary" | "neutral";

interface DataSourceCardCtaProps {
  /** Omitted while the connection status is still being read. */
  label?: string;
  variant?: CtaVariant;
}

// A span, not a button: the card itself is the button, and nesting one inside
// another is invalid. Same treatment as the Agents grid's cta chips.
const CHIP = "inline-flex items-center self-start rounded-lg px-3.5 py-1.5 text-[13px] font-semibold";

export default function DataSourceCardCta({ label, variant = "neutral" }: DataSourceCardCtaProps): ReactElement {
  if (label === undefined) return <SkeletonBar width={92} height={30} />;

  return (
    <span
      className={classNames(
        CHIP,
        variant === "primary"
          ? "bg-cognee-purple text-white"
          : "border border-white/[0.18] bg-white/[0.06] text-[var(--color-cognee-fg,#EDECEA)]",
      )}
    >
      {label}
    </span>
  );
}
