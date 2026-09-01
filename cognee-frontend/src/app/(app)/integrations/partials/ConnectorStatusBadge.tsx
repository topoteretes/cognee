"use client";

import type { ReactElement } from "react";
import classNames from "classnames";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import type { ConnectionStatus } from "@/modules/integrations/types";

interface ConnectorStatusBadgeProps {
  /** Undefined while the control-plane call is still in flight. */
  status: ConnectionStatus | undefined;
  /**
   * Health of a *connected* workspace (CLO-389): "degraded" means authorized
   * but not working — an expired token, or Slack refusing the bot.
   */
  syncStatus?: "ok" | "degraded";
}

const PILL = "flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px]";

export default function ConnectorStatusBadge({
  status,
  syncStatus,
}: ConnectorStatusBadgeProps): ReactElement | null {
  // A placeholder rather than the disconnected state: rendering "Connect" on a
  // workspace that is already connected, then flipping it a beat later, reads
  // as the app losing the connection.
  if (status === undefined) {
    return (
      <span className="shrink-0">
        <SkeletonBar width={78} height={17} />
      </span>
    );
  }

  if (status === "unavailable") {
    return <span className={classNames(PILL, "bg-white/[0.06] font-medium text-[var(--color-cognee-fg,#EDECEA)]/55")}>Status unknown</span>;
  }

  if (status !== "connected") return null;

  // Still connected, just not working. Saying "Disconnected" would be wrong
  // and actively unhelpful: the install exists and nobody removed it, so the
  // action is reconnect, not connect — and green here is a lie that costs
  // weeks, because the first real signal is stale search results.
  if (syncStatus === "degraded") {
    return (
      <span className={classNames(PILL, "bg-[var(--color-cognee-warning,#F59E0B)]/15 font-semibold text-[var(--color-cognee-warning,#F59E0B)]")}>
        <span className="size-1.5 rounded-full bg-[var(--color-cognee-warning,#F59E0B)]" />
        Needs reconnect
      </span>
    );
  }

  return (
    <span className={classNames(PILL, "bg-[var(--color-cognee-success,#22C55E)]/15 font-semibold text-[var(--color-cognee-success,#22C55E)]")}>
      <span className="size-1.5 rounded-full bg-[var(--color-cognee-success,#22C55E)]" />
      Connected
    </span>
  );
}
