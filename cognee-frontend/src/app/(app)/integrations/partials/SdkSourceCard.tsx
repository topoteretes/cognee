"use client";

import type { ReactElement } from "react";
import classNames from "classnames";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import ConnectorLogo from "./ConnectorLogo";

export type SdkSourceStatus = "connected" | "needs_attention" | "disconnected" | "unavailable";

const PILL = "flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold";
const CHIP = "inline-flex items-center self-start rounded-lg px-3.5 py-1.5 text-[13px] font-semibold";

const STATUS: Record<SdkSourceStatus, { label: string; pill: string; border: string }> = {
  connected: {
    label: "Connected",
    pill: "bg-[var(--color-cognee-success,#22C55E)]/15 text-[var(--color-cognee-success,#22C55E)]",
    border: "border-[var(--color-cognee-success,#22C55E)]/30",
  },
  needs_attention: {
    label: "Sync issue",
    pill: "bg-[var(--color-cognee-warning,#F59E0B)]/15 text-[var(--color-cognee-warning,#F59E0B)]",
    border: "border-[var(--color-cognee-warning,#F59E0B)]/40",
  },
  disconnected: {
    label: "Not connected",
    pill: "bg-[var(--color-cognee-danger,#EF4444)]/15 text-[var(--color-cognee-danger-fg,#FF8A8A)]",
    border: "border-white/10",
  },
  unavailable: {
    label: "Status unknown",
    pill: "bg-white/[0.06] font-medium text-[var(--color-cognee-fg,#EDECEA)]/55",
    border: "border-white/10",
  },
};

interface SdkSourceCardProps {
  name: string;
  description: string;
  logo: string;
  initials: string;
  color: string;
  /** Undefined while the connection is still being read. */
  status: SdkSourceStatus | undefined;
  healthText: string;
  actionLabel: string;
  onAction: () => void;
}

// A card for a data source the SDK's integrations router serves. Hosted
// connector cards (accounts, workspaces, per-tenant health) are not synced to
// this repo, so this one carries only what a single-user install has.
export default function SdkSourceCard({
  name, description, logo, initials, color, status, healthText, actionLabel, onAction,
}: SdkSourceCardProps): ReactElement {
  const look = status ? STATUS[status] : undefined;
  return (
    <button
      type="button"
      onClick={onAction}
      aria-label={`${actionLabel} ${name}`}
      className={classNames(
        "flex cursor-pointer flex-col gap-3 rounded-xl border bg-white/[0.06] p-4 text-left backdrop-blur-md transition-colors hover:border-white/[0.18] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70",
        look?.border ?? "border-white/10",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <ConnectorLogo logo={logo} initials={initials} color={color} size={40} />
          <span className="truncate text-[16px] font-medium text-[var(--color-cognee-fg,#EDECEA)]">{name}</span>
        </div>
        {look ? <span className={classNames(PILL, look.pill)}>{look.label}</span> : <SkeletonBar width={78} height={17} />}
      </div>
      <p className="m-0 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55">{description}</p>
      {status === "connected" || status === "needs_attention" ? (
        <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">{healthText}</p>
      ) : null}
      {status === "unavailable" && <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Couldn&apos;t read the connection status.</p>}
      <span className={classNames(CHIP, status === "connected" ? "border border-white/[0.18] bg-white/[0.06] text-[var(--color-cognee-fg,#EDECEA)]" : "bg-cognee-purple text-white")}>
        {actionLabel}
      </span>
    </button>
  );
}
