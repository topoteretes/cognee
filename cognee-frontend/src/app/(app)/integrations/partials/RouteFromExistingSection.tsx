"use client";

import { useEffect, useMemo, useState, type ReactElement } from "react";
import classNames from "classnames";
import getChannels, { type IntegrationChannel } from "@/modules/integrations/getChannels";
import setChannelRoute from "@/modules/integrations/setChannelRoute";
import type { AvailableTenant } from "@/modules/users/UserContext";

interface RouteFromExistingSectionProps {
  provider: string;
  /** The (currently disconnected) workspace that should receive the routed channel. */
  targetTenantId: string;
  /** Workspaces the current user owns — one of these already has the integration connected. */
  ownedTenants: AvailableTenant[];
  onRouted: () => void;
  onCancel: () => void;
}

const LABEL = "mb-1.5 block text-[12px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]";
const SELECT =
  "mb-3.5 w-full rounded-lg border border-white/[0.14] bg-white/[0.06] px-2.5 py-2 text-[13px] text-[var(--color-cognee-fg,#EDECEA)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70";
const NOTE = "m-0 mb-3.5 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55";
const BUTTON =
  "flex-1 cursor-pointer rounded-lg py-2.5 text-[13px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black";

/**
 * Lets an owner with more than one workspace point a channel from an
 * already-connected workspace at this (currently disconnected) one, instead
 * of going through a second OAuth install (CLO-377).
 */
export default function RouteFromExistingSection({
  provider,
  targetTenantId,
  ownedTenants,
  onRouted,
  onCancel,
}: RouteFromExistingSectionProps): ReactElement {
  const sourceCandidates = useMemo(
    () => ownedTenants.filter((t) => t.id !== targetTenantId),
    [ownedTenants, targetTenantId],
  );
  // Keyed on the joined id list, not the array itself — `ownedTenants` is a
  // new array reference on every parent render, so depending on it directly
  // re-ran this effect (and re-fetched) every render instead of only when
  // the actual tenant set changed.
  const ownedTenantsKey = ownedTenants.map((t) => t.id).join(",");

  const [sourceTenantId, setSourceTenantId] = useState<string>(sourceCandidates[0]?.id ?? "");
  const [channels, setChannels] = useState<IntegrationChannel[]>([]);
  const [channelId, setChannelId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sourceTenantId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setChannelId("");
    getChannels(provider, sourceTenantId).then((result) => {
      if (cancelled) return;
      setChannels(result.channels);
      setChannelId(result.channels[0]?.id ?? "");
      setError(result.error);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [provider, sourceTenantId, ownedTenantsKey]);

  const handleRoute = async (): Promise<void> => {
    const channel = channels.find((c) => c.id === channelId);
    if (!channel) return;
    setSaving(true);
    setError(null);
    const result = await setChannelRoute(provider, sourceTenantId, channel.id, channel.name, targetTenantId);
    setSaving(false);
    if (result.success) {
      onRouted();
    } else {
      setError(result.error ?? "Could not route this channel. Please try again.");
    }
  };

  if (sourceCandidates.length === 0) {
    return <p className="m-0 text-[13px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/55">You don&apos;t own another workspace to route a channel from.</p>;
  }

  return (
    <div>
      <p className="m-0 mb-3.5 text-[13px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/55">
        Point a channel from a workspace you already connected at this one — no new authorization needed.
      </p>

      <label htmlFor="route-source" className={LABEL}>
        From workspace
      </label>
      <select
        id="route-source"
        value={sourceTenantId}
        onChange={(e) => setSourceTenantId(e.target.value)}
        className={SELECT}
      >
        {sourceCandidates.map((t) => (
          <option key={t.id} value={t.id}>
            {t.name}
          </option>
        ))}
      </select>

      <label htmlFor="route-channel" className={LABEL}>
        Channel
      </label>
      {loading ? (
        <p className={NOTE}>Loading channels…</p>
      ) : channels.length === 0 ? (
        <p className={NOTE}>No channels found for this workspace.</p>
      ) : (
        <select
          id="route-channel"
          value={channelId}
          onChange={(e) => setChannelId(e.target.value)}
          className={SELECT}
        >
          {channels.map((c) => (
            <option key={c.id} value={c.id}>
              #{c.name}
            </option>
          ))}
        </select>
      )}

      {error && <p className="m-0 mb-3 text-[12px] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{error}</p>}

      <div className="flex gap-2">
        <button onClick={onCancel} className={classNames(BUTTON, "border border-white/[0.18] bg-transparent text-[var(--color-cognee-fg,#EDECEA)]/70 hover:bg-white/[0.06]")}>
          Cancel
        </button>
        <button
          onClick={() => void handleRoute()}
          disabled={!channelId || saving}
          className={classNames(
            BUTTON,
            "border-none bg-cognee-purple text-white hover:bg-cognee-purple-hover",
            (!channelId || saving) && "cursor-default opacity-60",
          )}
        >
          {saving ? "Routing…" : "Route channel"}
        </button>
      </div>
    </div>
  );
}
