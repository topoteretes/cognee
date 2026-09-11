"use client";

import type { ReactElement } from "react";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import { timeAgo } from "@/utils/timeAgo";
import type { ChannelSummary } from "./useConnectorChannels";

interface ChannelHealthLineProps {
  /** Undefined while the channel list is still being read. */
  summary: ChannelSummary | undefined;
  /**
   * When content was last handed to the tenant for ingestion (CLO-389).
   * "Sent", not "ingested": the tenant accepts the work and never reports
   * back, so this is the strongest claim the control plane can make.
   */
  lastSyncedAt?: string;
}

/**
 * Whether anything is actually reaching Cognee. "Connected" alone is not that
 * answer, and neither is the number of channels the bot can see: ingestion is
 * opt-in per channel (CLO-387), so a fully authorized workspace with nothing
 * selected reads as healthy while no content moves at all.
 */
export default function ChannelHealthLine({
  summary,
  lastSyncedAt,
}: ChannelHealthLineProps): ReactElement {
  if (!summary) return <SkeletonBar width={110} height={11} />;

  if (summary.error) {
    return <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Channel list unavailable</p>;
  }

  // Nothing shared with the bot at all, so there is nothing to select yet.
  if (summary.visible === 0) {
    return <p className="m-0 text-[12px] text-[var(--color-cognee-warning,#F59E0B)]">Not in any channel yet</p>;
  }

  if (summary.syncingHere === 0) {
    return <p className="m-0 text-[12px] text-[var(--color-cognee-warning,#F59E0B)]">No channels selected yet</p>;
  }

  return (
    <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
      {summary.syncingHere} of {summary.visible} channel{summary.visible === 1 ? "" : "s"} syncing
      {summary.routedAway > 0 && ` · ${summary.routedAway} routed elsewhere`}
      {lastSyncedAt && ` · last sent ${timeAgo(lastSyncedAt)}`}
    </p>
  );
}
