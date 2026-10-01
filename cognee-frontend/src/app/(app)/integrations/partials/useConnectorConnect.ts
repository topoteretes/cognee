"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import disconnectConnection from "@/modules/integrations/disconnectConnection";
import startConnection from "@/modules/integrations/startConnection";
import { isFailureOutcome, takeOAuthOutcome, type OAuthFailure } from "@/modules/integrations/oauthOutcome";

const POPUP_POLL_INTERVAL_MS = 600;
const POPUP_FEATURES = "width=600,height=760";

interface ConnectorConnect {
  connectingKey: string | null;
  startConnect: (provider: string) => Promise<void>;
  /** Resolves false when the backend refused, so the caller can stay put. */
  disconnect: (provider: string) => Promise<boolean>;
  /**
   * Why the last install round-trip did not connect, or null. Set only when
   * the callback actually reported an outcome — a popup the user closed by
   * hand leaves none, and silence is the right answer there.
   */
  failure: OAuthFailure | null;
  clearFailure: () => void;
}

/**
 * Drives the OAuth popup for team-scoped connectors. The popup is opened
 * synchronously on the click gesture (so the browser doesn't block it) and
 * only navigated once the backend returns the authorize URL. Completion is
 * detected by polling for the popup closing — which the callback page does
 * itself — rather than postMessage, avoiding cross-origin message plumbing.
 */
export function useConnectorConnect(
  tenantId: string | null,
  onChanged: (provider: string) => void,
): ConnectorConnect {
  const [connectingKey, setConnectingKey] = useState<string | null>(null);
  const [failure, setFailure] = useState<OAuthFailure | null>(null);
  const pollRef = useRef<number | null>(null);

  const clearFailure = useCallback(() => setFailure(null), []);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  const startConnect = useCallback(
    async (provider: string) => {
      if (!tenantId) return;

      const popup = window.open("", "cognee-oauth", POPUP_FEATURES);
      setConnectingKey(provider);
      // A retry must not read as still-failing while it is in flight.
      setFailure(null);
      // Drop anything a previous, unconsumed round-trip left behind, so it
      // cannot be mistaken for this attempt's result.
      takeOAuthOutcome(provider);

      const result = await startConnection(provider, tenantId);
      if (result.error || !result.authorizeUrl) {
        popup?.close();
        setConnectingKey(null);
        notifications.show({ color: "red", message: result.error ?? "Could not start the connection." });
        return;
      }

      if (popup) popup.location.href = result.authorizeUrl;

      pollRef.current = window.setInterval(() => {
        if (popup && !popup.closed) return;
        if (pollRef.current !== null) window.clearInterval(pollRef.current);
        pollRef.current = null;
        setConnectingKey(null);

        // Read before refetching: the callback's own verdict is more precise
        // than "the status still says disconnected", and it is the only thing
        // that can distinguish a refusal from a workspace that simply is not
        // connected yet.
        const outcome = takeOAuthOutcome(provider);
        if (isFailureOutcome(outcome)) setFailure({ provider, outcome });

        onChanged(provider);
      }, POPUP_POLL_INTERVAL_MS);
    },
    [tenantId, onChanged],
  );

  const disconnect = useCallback(
    async (provider: string): Promise<boolean> => {
      if (!tenantId) return false;
      const result = await disconnectConnection(provider, tenantId);
      if (!result.success) {
        notifications.show({ color: "red", message: result.error ?? "Could not disconnect." });
        return false;
      }
      onChanged(provider);
      return true;
    },
    [tenantId, onChanged],
  );

  return { connectingKey, startConnect, disconnect, failure, clearFailure };
}
