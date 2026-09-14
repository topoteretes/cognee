"use client";

import { useCallback, useEffect, useMemo, useState, type ReactElement } from "react";
import { useTenant } from "@/modules/tenant/TenantContext";
import { useUser } from "@/modules/users/UserContext";
import getConnectionStatus from "@/modules/integrations/getConnectionStatus";
import { describeOAuthFailure, stashOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import { DATA_SOURCE_CARDS } from "@/modules/integrations/dataSourceCards";
import type { TeamConnectorCfg, TeamConnectionState } from "@/modules/integrations/types";
import DataSourceCard from "./DataSourceCard";
import ConnectModal from "./ConnectModal";
import { useConnectorConnect } from "./useConnectorConnect";
import { useConnectorChannels } from "./useConnectorChannels";

const DISCONNECTED: TeamConnectionState = { status: "disconnected" };

// Below this many connectors the filter is pure noise above the grid — the
// whole catalog is already visible in one glance.
const SEARCH_MIN_CONNECTORS = 8;

const SEARCH_INPUT =
  "min-w-[200px] rounded-lg border border-white/10 bg-white/[0.06] px-3 py-2 text-[13px] text-[var(--color-cognee-fg,#EDECEA)] outline-none transition-colors placeholder:text-[var(--color-cognee-fg,#EDECEA)]/55 focus:border-cognee-lavender/50 focus:bg-white/[0.09]";

const PROVIDERS = DATA_SOURCE_CARDS.map((c) => c.key);

export default function DataSourceSection(): ReactElement {
  const { tenant, isOwner } = useTenant();
  const tenantId = tenant?.tenant_id ?? null;
  const { availableTenants } = useUser();
  // Memoized: a fresh array here on every render would re-trigger any child
  // effect keyed on it (e.g. RouteFromExistingSection's connection-status
  // fetch) on every unrelated re-render, not just when ownership actually changes.
  const ownedTenants = useMemo(
    () => availableTenants.filter((t) => t.isOwner),
    [availableTenants],
  );

  const [statuses, setStatuses] = useState<Record<string, TeamConnectionState>>({});
  const [activeCfg, setActiveCfg] = useState<TeamConnectorCfg | null>(null);
  const [search, setSearch] = useState("");

  const refreshStatus = useCallback(
    async (provider: string) => {
      if (!tenantId) return;
      const status = await getConnectionStatus(provider, tenantId);
      setStatuses((prev) => ({
        ...prev,
        [provider]: {
          status: status.failed ? "unavailable" : status.connected ? "connected" : "disconnected",
          workspaceName: status.teamName,
          viaRouting: status.viaRouting,
          routedTeamName: status.routedTeamName,
          routedChannelCount: status.routedChannelCount,
          // Left undefined when the control plane says nothing: the column is
          // nullable and nothing wrote it before CLO-389, so every install
          // predating it would otherwise be reported as verified-healthy.
          // Unknown renders the same as ok today, but it does not claim to be it.
          syncStatus:
            status.syncStatus === "degraded" || status.syncStatus === "ok"
              ? status.syncStatus
              : undefined,
          lastSyncedAt: status.lastSyncedAt,
        },
      }));
    },
    [tenantId],
  );

  // Clearing first puts the card back into its loading state, so a retry looks
  // like a retry instead of a frozen error.
  const retryStatus = useCallback(
    (provider: string) => {
      setStatuses((prev) => {
        const next = { ...prev };
        delete next[provider];
        return next;
      });
      void refreshStatus(provider);
    },
    [refreshStatus],
  );

  useEffect(() => {
    if (!tenantId) return;
    PROVIDERS.forEach((provider) => void refreshStatus(provider));
  }, [tenantId, refreshStatus]);

  // The OAuth popup lands back on this page (?slack=<outcome>); when we're that
  // popup, hand the outcome to the opener and close so its poll fires and
  // refetches. Runs first, before any of the section's own work.
  //
  // Recording the outcome is what lets the opener say *why* an install did not
  // happen: this window is about to be destroyed, taking the only copy of the
  // callback's verdict with it.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const provider = PROVIDERS.find((candidate) => params.has(candidate));
    if (provider && window.opener && window.opener !== window) {
      stashOAuthOutcome(provider, params.get(provider));
      window.close();
    }
  }, []);

  const { connectingKey, startConnect, disconnect, failure, clearFailure } = useConnectorConnect(
    tenantId,
    refreshStatus,
  );

  const connectedProviders = useMemo(
    () => PROVIDERS.filter((provider) => statuses[provider]?.status === "connected"),
    [statuses],
  );
  const { channelsFor, summaryFor, applyRoute, applyAllowed, refresh } = useConnectorChannels(
    tenantId,
    connectedProviders,
  );

  /** Undefined means the status read is still in flight. */
  const stateFor = (key: string): TeamConnectionState | undefined => statuses[key];

  // Dismissing the modal dismisses the failure with it — reopening the card
  // should show its current state, not the last attempt's error.
  const closeModal = useCallback(() => {
    setActiveCfg(null);
    clearFailure();
  }, [clearFailure]);

  /** The failure to show, only while its own connector's modal is open. */
  const connectError =
    activeCfg && failure?.provider === activeCfg.key
      ? describeOAuthFailure(failure.outcome, activeCfg.name)
      : null;

  const handleAuthorize = useCallback(() => {
    if (activeCfg) void startConnect(activeCfg.key);
  }, [activeCfg, startConnect]);

  const handleDisconnect = useCallback(async (): Promise<boolean> => {
    if (!activeCfg) return false;
    const disconnected = await disconnect(activeCfg.key);
    if (disconnected) setActiveCfg(null);
    return disconnected;
  }, [activeCfg, disconnect]);

  const filteredCards = DATA_SOURCE_CARDS.filter((cfg) =>
    cfg.name.toLowerCase().includes(search.trim().toLowerCase()),
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="m-0 mb-1 text-[18px] font-bold tracking-[-0.01em] text-[var(--color-cognee-fg,#EDECEA)]">Data sources</h2>
          <p className="m-0 text-[14px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            Connect the tools your team already uses to your brains. One connection per workspace, shared with everyone.
          </p>
        </div>
        {DATA_SOURCE_CARDS.length >= SEARCH_MIN_CONNECTORS && (
          <input
            type="text"
            placeholder="Search connectors…"
            aria-label="Search connectors"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className={SEARCH_INPUT}
          />
        )}
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
        {filteredCards.map((cfg) => (
          <DataSourceCard
            key={cfg.key}
            cfg={cfg}
            state={stateFor(cfg.key)}
            channels={summaryFor(cfg.key)}
            isOwner={isOwner}
            onManageClick={() => {
              setActiveCfg(cfg);
              // Channels are invited in Slack, so the list read at page load is
              // already stale by the time someone opens this. Opening it is the
              // clearest signal they want the current state.
              void refresh(cfg.key);
            }}
            onRetry={() => retryStatus(cfg.key)}
          />
        ))}
      </div>

      {filteredCards.length === 0 && (
        <p className="m-0 text-center text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55">No connectors match &quot;{search}&quot;.</p>
      )}

      {activeCfg && tenantId && (
        <ConnectModal
          cfg={activeCfg}
          state={stateFor(activeCfg.key) ?? DISCONNECTED}
          isConnecting={connectingKey === activeCfg.key}
          connectError={connectError}
          onClose={closeModal}
          onAuthorize={handleAuthorize}
          onDisconnect={handleDisconnect}
          tenantId={tenantId}
          isOwner={isOwner}
          ownedTenants={ownedTenants}
          channels={channelsFor(activeCfg.key)}
          onRouteApplied={(channelId, targetTenantId) => applyRoute(activeCfg.key, channelId, targetTenantId)}
          onAllowedApplied={(channelId, allowed) => applyAllowed(activeCfg.key, channelId, allowed)}
          onRefresh={() => refresh(activeCfg.key)}
          onRoutedFromExisting={() => {
            void refreshStatus(activeCfg.key);
            closeModal();
          }}
        />
      )}
    </div>
  );
}
