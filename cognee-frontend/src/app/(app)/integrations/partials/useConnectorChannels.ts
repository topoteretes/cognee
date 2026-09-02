"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import getChannels, { type IntegrationChannel } from "@/modules/integrations/getChannels";
import getChannelRoutes from "@/modules/integrations/getChannelRoutes";

export interface ConnectorChannels {
  channels: IntegrationChannel[];
  /** channel id → tenant its content is routed to; absent means this workspace. */
  routes: Record<string, string>;
  error: string | null;
}

export interface ChannelSummary {
  /** Channels the bot can see, whether or not they are opted in. */
  visible: number;
  /** Opted-in channels whose content lands in this workspace. */
  syncingHere: number;
  /** Opted-in channels whose content goes to a different workspace (CLO-377). */
  routedAway: number;
  error: string | null;
}

interface ConnectorChannelsApi {
  /** Undefined while the provider's channels are still being read. */
  channelsFor: (provider: string) => ConnectorChannels | undefined;
  summaryFor: (provider: string) => ChannelSummary | undefined;
  /** Reflect a saved route locally; `targetTenantId` null means "back to this workspace". */
  applyRoute: (provider: string, channelId: string, targetTenantId: string | null) => void;
  /** Reflect a saved opt-in change locally (CLO-387). */
  applyAllowed: (provider: string, channelId: string, allowed: boolean) => void;
  /**
   * Re-read one provider's channels. Inviting the bot to a channel happens in
   * Slack, not here, so nothing in this app can know it happened — without a
   * way to ask again, a newly invited channel stays invisible until a full
   * page reload.
   */
  refresh: (provider: string) => Promise<void>;
}

/**
 * Opted-in channels first, provider order preserved inside each group.
 *
 * Applied once per read rather than derived on render on purpose: sorting on
 * `allowed` every render would make a row jump out from under the cursor the
 * moment it is unchecked. The order is decided when the list arrives and stays
 * put while the owner works through it, and `applyAllowed` maps in place.
 */
function syncingFirst(channels: IntegrationChannel[]): IntegrationChannel[] {
  return [...channels].sort((a, b) => Number(b.allowed) - Number(a.allowed));
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// The provider list right after an OAuth connect is frequently not caught up
// yet (the install lands before the bot's membership is queryable), which
// reads as a real failure if surfaced immediately. A couple of quick retries
// covers that window without ever entering it for a genuinely broken token.
const CHANNEL_READ_RETRY_DELAYS_MS = [1200, 1800];

/**
 * Channels (and their CLO-377 routes) for every connected connector, read once
 * per provider and shared by the card's health line and the manage modal's
 * list. Fetched here rather than in each consumer because both answer the same
 * question ("what is actually flowing into this workspace?") and the provider
 * API behind it is rate-limited.
 */
export function useConnectorChannels(
  tenantId: string | null,
  connectedProviders: string[],
): ConnectorChannelsApi {
  const [byProvider, setByProvider] = useState<Record<string, ConnectorChannels>>({});
  // Joined, not the array itself: `connectedProviders` is derived from state
  // upstream and is a fresh reference on every render, which would re-fetch on
  // every unrelated re-render. The effect reads the providers back out of the
  // key so its dependency list stays honest.
  const providerKey = connectedProviders.join(",");

  // Read at write time, not captured: a response that lands after the user
  // switched workspaces must not paint another tenant's channels into this one.
  const tenantIdRef = useRef(tenantId);
  tenantIdRef.current = tenantId;

  // `retryOnError` is only for the automatic read below, not the public
  // `refresh` — a real broken connection should say so immediately when an
  // owner explicitly asks by hitting Refresh or opening the manage modal.
  const load = useCallback(
    async (provider: string, retryOnError: boolean): Promise<void> => {
      const forTenant = tenantIdRef.current;
      if (!forTenant) return;

      const routesPromise = getChannelRoutes(provider, forTenant);

      let channelsResult = await getChannels(provider, forTenant);
      if (retryOnError) {
        for (const delay of CHANNEL_READ_RETRY_DELAYS_MS) {
          if (!channelsResult.error) break;
          if (tenantIdRef.current !== forTenant) return;
          await wait(delay);
          if (tenantIdRef.current !== forTenant) return;
          channelsResult = await getChannels(provider, forTenant);
        }
      }

      const routes = await routesPromise;
      if (tenantIdRef.current !== forTenant) return;

      setByProvider((prev) => ({
        ...prev,
        [provider]: {
          channels: syncingFirst(channelsResult.channels),
          routes: Object.fromEntries(routes.map((r) => [r.resourceId, r.tenantId])),
          error: channelsResult.error,
        },
      }));
    },
    [],
  );

  const refresh = useCallback((provider: string) => load(provider, false), [load]);

  useEffect(() => {
    if (!tenantId || !providerKey) return;
    for (const provider of providerKey.split(",")) void load(provider, true);
  }, [tenantId, providerKey, load]);

  const channelsFor = useCallback(
    (provider: string): ConnectorChannels | undefined => byProvider[provider],
    [byProvider],
  );

  const summaryFor = useCallback(
    (provider: string): ChannelSummary | undefined => {
      const data = byProvider[provider];
      if (!data || !tenantId) return undefined;
      // Only opted-in channels count as syncing anywhere: a channel that is
      // visible but not selected produces no content, so routing it somewhere
      // else changes nothing.
      const optedIn = data.channels.filter((c) => c.allowed);
      const routedAway = optedIn.filter((c) => {
        const target = data.routes[c.id];
        return target !== undefined && target !== tenantId;
      }).length;
      return {
        visible: data.channels.length,
        syncingHere: optedIn.length - routedAway,
        routedAway,
        error: data.error,
      };
    },
    [byProvider, tenantId],
  );

  const applyRoute = useCallback(
    (provider: string, channelId: string, targetTenantId: string | null) => {
      setByProvider((prev) => {
        const current = prev[provider];
        if (!current) return prev;
        const routes = { ...current.routes };
        if (targetTenantId === null) delete routes[channelId];
        else routes[channelId] = targetTenantId;
        return { ...prev, [provider]: { ...current, routes } };
      });
    },
    [],
  );

  const applyAllowed = useCallback(
    (provider: string, channelId: string, allowed: boolean) => {
      setByProvider((prev) => {
        const current = prev[provider];
        if (!current) return prev;
        return {
          ...prev,
          [provider]: {
            ...current,
            channels: current.channels.map((c) =>
              c.id === channelId ? { ...c, allowed } : c,
            ),
          },
        };
      });
    },
    [],
  );

  return { channelsFor, summaryFor, applyRoute, applyAllowed, refresh };
}
