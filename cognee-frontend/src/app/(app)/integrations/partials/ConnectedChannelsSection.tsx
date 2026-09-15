"use client";

import { useEffect, useState, type ReactElement } from "react";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import setAllowedChannels from "@/modules/integrations/setAllowedChannels";
import type { AvailableTenant } from "@/modules/users/UserContext";
import type { ConnectorChannels } from "./useConnectorChannels";
import ConnectedChannelRow from "./ConnectedChannelRow";

interface ConnectedChannelsSectionProps {
  provider: string;
  /** Display name, e.g. "Slack". */
  providerName: string;
  tenantId: string;
  /** Owner of this workspace; only they see the channel names. */
  isOwner: boolean;
  ownedTenants: AvailableTenant[];
  /** Undefined while the channel list is still being read. */
  data: ConnectorChannels | undefined;
  onRouteApplied: (channelId: string, targetTenantId: string | null) => void;
  onAllowedApplied: (channelId: string, allowed: boolean) => void;
  /** Re-read the channel list from the provider. */
  onRefresh: () => Promise<void>;
}

/**
 * Font-size/weight can't live in this className: Tailwind's own Preflight
 * (and Mantine's) reset `button { font: inherit }`, and that reset's CSS
 * layer sits above Tailwind's utilities layer — so on a raw <button>, any
 * text-size/font-weight utility loses regardless of its own specificity.
 * BULK_ACTION_STYLE below carries those two properties as inline style,
 * which layers can't touch; everything else here is unaffected and stays
 * as classes.
 */
const BULK_ACTION =
  "cursor-pointer border-none bg-transparent p-0 text-cognee-lavender/80 " +
  "hover:text-cognee-lavender hover:underline " +
  "disabled:cursor-default disabled:text-[var(--color-cognee-fg,#EDECEA)]/30 disabled:no-underline disabled:hover:text-[var(--color-cognee-fg,#EDECEA)]/30 " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70";

const BULK_ACTION_STYLE = { fontSize: 12, fontWeight: 500 };

// Same font-size caveat as BULK_ACTION_STYLE above, but for <input>: the
// button/input/select/textarea reset hits this element too.
const SEARCH_INPUT_STYLE = { fontSize: 12 };
const SEARCH_INPUT =
  "mb-2 w-full rounded-md border border-white/[0.14] bg-white/[0.06] px-2.5 py-1.5 text-[var(--color-cognee-fg,#EDECEA)] placeholder:text-[var(--color-cognee-fg,#EDECEA)]/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70";

/**
 * What is actually flowing into this workspace. A "connected" badge says
 * nothing about whether any content reaches Cognee, so the manage modal leads
 * with the channel list and its count; per-channel routing (CLO-377) is layered
 * on for owners of more than one workspace, who are the only ones who can use it.
 *
 * The count is workspace-wide information, but the names are not: a private
 * channel the bot was invited into can be identifying on its own. Everyone gets
 * the health signal, only the owner gets the inventory. Enforced here rather
 * than relying on the fact that only owners can currently open this modal.
 */
export default function ConnectedChannelsSection({
  provider,
  providerName,
  tenantId,
  isOwner,
  ownedTenants,
  data,
  onRouteApplied,
  onAllowedApplied,
  onRefresh,
}: ConnectedChannelsSectionProps): ReactElement {
  const [saveError, setSaveError] = useState<string | null>(null);
  const [bulkSaving, setBulkSaving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [confirmingSelectAll, setConfirmingSelectAll] = useState(false);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const routeTargets = ownedTenants.filter((t) => t.id !== tenantId);

  // Filtering itself is cheap even at a few hundred rows, but debouncing
  // still avoids re-sorting/re-filtering (and re-rendering every row) on
  // every keystroke while someone is still mid-word.
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search), 150);
    return () => clearTimeout(id);
  }, [search]);

  async function refresh(): Promise<void> {
    setRefreshing(true);
    setSaveError(null);
    try {
      await onRefresh();
    } finally {
      setRefreshing(false);
    }
  }

  // The endpoint takes the whole selection, not a delta, so this has to live
  // where every channel is in scope rather than on the row being toggled.
  async function toggleAllowed(channelId: string, allowed: boolean): Promise<boolean> {
    if (!data) return false;
    const next = data.channels
      .filter((c) => (c.id === channelId ? allowed : c.allowed))
      .map((c) => c.id);

    setSaveError(null);
    const result = await setAllowedChannels(provider, tenantId, next);
    if (!result.success) {
      // Surfaced, not swallowed: the checkbox springs back on its own, which
      // on its own looks like a glitch rather than a refused change.
      setSaveError(result.error ?? "Could not save your channel selection.");
      return false;
    }
    onAllowedApplied(channelId, allowed);
    return true;
  }

  // One request for the whole set, not one per channel: a workspace can easily
  // have a hundred channels, and the endpoint replaces the selection anyway.
  async function setAll(allowed: boolean): Promise<void> {
    if (!data) return;
    const changed = data.channels.filter((c) => c.allowed !== allowed);
    if (changed.length === 0) return;

    setBulkSaving(true);
    setSaveError(null);
    setConfirmingSelectAll(false);
    const result = await setAllowedChannels(
      provider,
      tenantId,
      allowed ? data.channels.map((c) => c.id) : [],
    );
    setBulkSaving(false);

    if (!result.success) {
      setSaveError(result.error ?? "Could not save your channel selection.");
      return;
    }
    for (const channel of changed) onAllowedApplied(channel.id, allowed);
  }
  // Visible and syncing are different numbers: ingestion is opt-in per channel
  // (CLO-387), so counting everything the bot can see would overstate what
  // Cognee actually reads.
  const optedIn = data ? data.channels.filter((c) => c.allowed) : [];
  const routedAway = optedIn.filter(
    (c) => data && data.routes[c.id] !== undefined && data.routes[c.id] !== tenantId,
  );
  const syncingHere = optedIn.length - routedAway.length;
  // Selected channels first so the ones already feeding Cognee are the ones
  // in view, rather than mixed in wherever the provider happened to list them.
  const sortedChannels = data ? [...data.channels].sort((a, b) => Number(b.allowed) - Number(a.allowed)) : [];
  const query = debouncedSearch.trim().toLowerCase();
  const filteredChannels = query ? sortedChannels.filter((c) => c.name.toLowerCase().includes(query)) : sortedChannels;

  return (
    <div className="mt-4 border-t border-white/10 pt-3.5">
      <div className="mb-2.5 flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <p className="m-0 text-[13px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]">Channels</p>
          <button
            onClick={() => void refresh()}
            disabled={refreshing}
            className={BULK_ACTION}
            style={BULK_ACTION_STYLE}
            title="Re-read the channel list"
          >
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        {data && !data.error && (
          <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            {syncingHere} of {data.channels.length} syncing here
            {routedAway.length > 0 && ` · ${routedAway.length} routed elsewhere`}
          </p>
        )}
      </div>

      {!data ? (
        <div className="flex flex-col gap-2">
          <SkeletonBar width="70%" height={12} />
          {isOwner && <SkeletonBar width="55%" height={12} />}
          {isOwner && <SkeletonBar width="62%" height={12} />}
        </div>
      ) : data.error ? (
        <p className="m-0 text-[12px] leading-[1.45] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{data.error}</p>
      ) : data.channels.length === 0 ? (
        <p className="m-0 text-[12px] leading-[1.45] text-[var(--color-cognee-warning,#F59E0B)]">
          Cognee isn&apos;t in any channel yet. Invite it to one in {providerName}, then hit
          Refresh.
        </p>
      ) : !isOwner ? (
        <p className="m-0 text-[12px] leading-[1.45] text-[var(--color-cognee-fg,#EDECEA)]/55">
          Only workspace owners can see which channels.
        </p>
      ) : (
        <>
          {/* Kept above the list rather than replacing it: the owner still needs
              to see what is available to choose from. */}
          {optedIn.length === 0 && (
            <p className="m-0 mb-2 text-[12px] leading-[1.45] text-[var(--color-cognee-warning,#F59E0B)]">
              None of these are selected yet, so nothing is being read.
            </p>
          )}

          {data.channels.length > 5 && (
            <>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search channels"
                aria-label="Search channels"
                className={SEARCH_INPUT}
                style={SEARCH_INPUT_STYLE}
              />
              {query && filteredChannels.length > 0 && (
                <p className="m-0 mb-2 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/45">
                  {filteredChannels.length} match{filteredChannels.length === 1 ? "" : "es"}
                </p>
              )}
            </>
          )}

          {data.channels.length > 1 &&
            (confirmingSelectAll ? (
              // Asked, not just done: this reads every channel's history through
              // an LLM and makes all of it answerable for the whole workspace,
              // which is not something to hand to a single misplaced click.
              <div className="mb-2">
                <p className="m-0 mb-1.5 text-[12px] leading-[1.45] text-[var(--color-cognee-fg,#EDECEA)]/55">
                  Read all {data.channels.length} channels, history included? That uses credits, and
                  every message in them becomes answerable for everyone in this workspace.
                </p>
                <div className="flex items-center gap-2">
                  <button onClick={() => void setAll(true)} className={BULK_ACTION} style={BULK_ACTION_STYLE}>
                    Yes, select all
                  </button>
                  <span aria-hidden className="text-[11px] text-[var(--color-cognee-fg,#EDECEA)]/25">
                    ·
                  </span>
                  <button onClick={() => setConfirmingSelectAll(false)} className={BULK_ACTION} style={BULK_ACTION_STYLE}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="mb-2 flex items-center gap-2">
                <button
                  onClick={() => setConfirmingSelectAll(true)}
                  disabled={bulkSaving || optedIn.length === data.channels.length}
                  className={BULK_ACTION}
                  style={BULK_ACTION_STYLE}
                >
                  Select all
                </button>
                <span aria-hidden className="text-[11px] text-[var(--color-cognee-fg,#EDECEA)]/25">
                  ·
                </span>
                <button
                  onClick={() => void setAll(false)}
                  disabled={bulkSaving || optedIn.length === 0}
                  className={BULK_ACTION}
                  style={BULK_ACTION_STYLE}
                >
                  Clear all
                </button>
                {bulkSaving && <span className="text-[11px] text-[var(--color-cognee-fg,#EDECEA)]/45">Saving…</span>}
              </div>
            ))}
          {/* overscroll-contain: without it the trackpad's rubber-band at either
              end of this 320px list propagates to the ancestors, which reads as
              the whole dialog bouncing. [contain:paint] keeps the scroll from
              re-rasterizing the blurred panel it sits inside on every frame. */}
          {query && filteredChannels.length === 0 ? (
            <p className="m-0 text-[12px] leading-[1.45] text-[var(--color-cognee-fg,#EDECEA)]/45">
              No channels match &quot;{debouncedSearch.trim()}&quot;.
            </p>
          ) : (
            <div className="flex max-h-[320px] flex-col gap-1.5 overflow-y-auto overscroll-contain [contain:paint]">
              {filteredChannels.map((channel) => (
                <ConnectedChannelRow
                  key={channel.id}
                  provider={provider}
                  tenantId={tenantId}
                  channel={channel}
                  routedTo={data.routes[channel.id]}
                  routeTargets={routeTargets}
                  onRouteApplied={onRouteApplied}
                  onToggleAllowed={toggleAllowed}
                />
              ))}
            </div>
          )}

          {saveError && (
            <p className="m-0 mt-2 text-[12px] leading-[1.45] text-[var(--color-cognee-danger-fg,#FF8A8A)]">
              {saveError}
            </p>
          )}
        </>
      )}

      {routeTargets.length > 0 && data && data.channels.length > 0 && (
        <p className="m-0 mt-2.5 text-[12px] leading-[1.4] text-[var(--color-cognee-fg,#EDECEA)]/55">
          Pick another workspace to send a single channel there instead of this one.
        </p>
      )}
    </div>
  );
}
