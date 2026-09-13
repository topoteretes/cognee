"use client";

import { useState, type ReactElement } from "react";
import Link from "next/link";
import { Loader } from "@mantine/core";
import ModalShell from "@/ui/elements/ModalShell";
import type { TeamConnectorCfg, TeamConnectionState } from "@/modules/integrations/types";
import type { AvailableTenant } from "@/modules/users/UserContext";
import ConnectedChannelsSection from "./ConnectedChannelsSection";
import RouteFromExistingSection from "./RouteFromExistingSection";
import ConnectorLogo from "./ConnectorLogo";
import DisconnectZone from "./DisconnectZone";
import type { ConnectorChannels } from "./useConnectorChannels";

interface ConnectModalProps {
  cfg: TeamConnectorCfg;
  state: TeamConnectionState;
  /** True while the OAuth popup is open and we're awaiting its completion. */
  isConnecting: boolean;
  /**
   * Why the last authorization did not connect, or null. Rendered in the modal
   * rather than a toast: the modal is already open and stays open, and this is
   * the only place the reader can act on what it says.
   */
  connectError: string | null;
  onClose: () => void;
  onAuthorize: () => void;
  /** Resolves false when the backend refused, so the modal stays open. */
  onDisconnect: () => Promise<boolean>;
  /** This workspace's id — needed to fetch/save channel routes (CLO-377). */
  tenantId: string;
  /** Owner of this workspace; gates the channel names and the disconnect. */
  isOwner: boolean;
  /** Workspaces the current user owns — channels can only route to one of these. */
  ownedTenants: AvailableTenant[];
  /** Undefined while this connector's channel list is still being read. */
  channels: ConnectorChannels | undefined;
  onRouteApplied: (channelId: string, targetTenantId: string | null) => void;
  onAllowedApplied: (channelId: string, allowed: boolean) => void;
  onRefresh: () => Promise<void>;
  /** A channel from another owned workspace was just routed here — refresh and close. */
  onRoutedFromExisting: () => void;
}

const GUIDE_LINK = "text-cognee-lavender underline underline-offset-2";

// Font-size/weight/line-height as inline style, not classes, on every
// <button> in this file: Preflight/Mantine reset `button { font: inherit }`
// in a CSS layer above Tailwind's utilities, so those three properties lose
// to that reset on a raw <button> regardless of the utility's own
// specificity. Inline style sits outside layers entirely and always wins.
const CLOSE_BUTTON_STYLE = { fontSize: 18, lineHeight: 1 };
const BUTTON_TEXT_STYLE = { fontSize: 13, fontWeight: 600 };
const ROUTE_FROM_EXISTING_STYLE = { fontSize: 12 };

const RECONNECT_BUTTON =
  "cursor-pointer rounded-lg border-none bg-cognee-purple px-3.5 py-1.5 text-white transition-colors hover:bg-cognee-purple-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black";

const PRIMARY_BUTTON =
  "w-full cursor-pointer rounded-lg border-none bg-cognee-purple py-2.5 text-white transition-colors hover:bg-cognee-purple-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black";

export default function ConnectModal({
  cfg,
  state,
  isConnecting,
  connectError,
  onClose,
  onAuthorize,
  onDisconnect,
  tenantId,
  isOwner,
  ownedTenants,
  channels,
  onRouteApplied,
  onAllowedApplied,
  onRefresh,
  onRoutedFromExisting,
}: ConnectModalProps): ReactElement {
  const isConnected = state.status === "connected";
  const [showRouteFromExisting, setShowRouteFromExisting] = useState(false);
  const canRouteFromExisting = cfg.supportsChannelRouting && ownedTenants.some((t) => t.id !== tenantId);
  const title = isConnected ? `${cfg.name} connected` : `Connect ${cfg.name}`;

  return (
    <ModalShell onClose={onClose} width={420} label={title}>
      <div className="flex items-center gap-3">
        <ConnectorLogo logo={cfg.logo} initials={cfg.initials} color={cfg.color} size={38} />
        <h2 className="m-0 flex-1 text-[16px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]">{title}</h2>
        <button
          onClick={onClose}
          aria-label="Close"
          className="cursor-pointer rounded border-none bg-transparent p-0.5 text-[var(--color-cognee-fg,#EDECEA)]/55 transition-colors hover:text-[var(--color-cognee-fg,#EDECEA)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70"
          style={CLOSE_BUTTON_STYLE}
        >
          ×
        </button>
      </div>

      {state.viaRouting && (
        <p className="m-0 text-[12px] leading-[1.4] text-[var(--color-cognee-fg,#EDECEA)]/55 italic">
          Also receiving {state.routedChannelCount ?? 1} channel{(state.routedChannelCount ?? 1) === 1 ? "" : "s"} routed
          here from {state.routedTeamName ?? "another workspace"} you own — manage that routing from its tenant.
        </p>
      )}

      {/* Before the body, in both the connected and disconnected branches: a
          failed reconnect matters just as much as a failed first connect, and
          in neither case is anything below it worth reading first. */}
      {connectError && (
        <div
          role="alert"
          className="rounded-lg border border-[var(--color-cognee-danger,#EF4444)]/30 bg-[var(--color-cognee-danger,#EF4444)]/10 px-3.5 py-3"
        >
          <p className="m-0 text-[12px] leading-[1.5] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{connectError}</p>
        </div>
      )}

      {isConnected ? (
        <div>
          {/* Above the workspace identity, because it is the only thing worth
              doing here until it is fixed. Reconnecting is the same OAuth trip
              as connecting; the backend upserts the existing install and
              carries the channel selection across, which is the bit an owner
              will not assume. */}
          {state.syncStatus === "degraded" && isOwner && (
            <div className="mb-3 rounded-lg border border-[var(--color-cognee-warning,#F59E0B)]/30 bg-[var(--color-cognee-warning,#F59E0B)]/10 px-3.5 py-3">
              <p className="m-0 mb-2 text-[12px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/70">
                {cfg.name} stopped accepting this connection, so nothing new is being read.
                Reconnecting restores it and keeps the channels you selected.
              </p>
              {isConnecting ? (
                <p className="m-0 flex items-center gap-2 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
                  <Loader size={12} color="#BC9BFF" />
                  Waiting for authorization in the {cfg.name} window…
                </p>
              ) : (
                <button onClick={onAuthorize} className={RECONNECT_BUTTON} style={BUTTON_TEXT_STYLE}>
                  Reconnect {cfg.name}
                </button>
              )}
            </div>
          )}

          <div className="rounded-lg border border-white/10 bg-white/[0.04] px-3.5 py-3">
            <div className="text-[11px] tracking-[0.04em] text-[var(--color-cognee-fg,#EDECEA)]/55 uppercase">{cfg.name} workspace</div>
            <div className="mt-0.5 truncate text-[13px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]">
              {state.workspaceName ?? cfg.name}
            </div>
          </div>

          {/* The two facts that change what an owner selects, next to the
              selection itself rather than a click away. */}
          <p className="m-0 mt-3 text-[12px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/55">
            Cognee reads the channels selected below, including the history they already have, and
            anyone in this workspace can then ask about them.{" "}
            <Link href="/integrations/slack" className={GUIDE_LINK}>
              How {cfg.name} memory works
            </Link>
          </p>

          {cfg.supportsChannelRouting && (
            <ConnectedChannelsSection
              provider={cfg.key}
              providerName={cfg.name}
              tenantId={tenantId}
              isOwner={isOwner}
              ownedTenants={ownedTenants}
              data={channels}
              onRouteApplied={onRouteApplied}
              onAllowedApplied={onAllowedApplied}
              onRefresh={onRefresh}
            />
          )}

          {isOwner && (
            <DisconnectZone name={state.workspaceName ?? cfg.name} provider={cfg.name} onDisconnect={onDisconnect} />
          )}
        </div>
      ) : isConnecting ? (
        <p className="m-0 flex items-center gap-2 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55">
          <Loader size={14} color="#BC9BFF" />
          Waiting for authorization in the {cfg.name} window…
        </p>
      ) : showRouteFromExisting ? (
        <RouteFromExistingSection
          provider={cfg.key}
          targetTenantId={tenantId}
          ownedTenants={ownedTenants}
          onRouted={onRoutedFromExisting}
          onCancel={() => setShowRouteFromExisting(false)}
        />
      ) : (
        <div>
          <p className="m-0 mb-3.5 text-[13px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/55">Cognee will be able to:</p>
          <ul className="m-0 mb-[18px] flex list-none flex-col gap-[7px] p-0">
            {cfg.permissions.map((permission) => (
              <li key={permission} className="relative pl-[18px] text-[13px] text-[var(--color-cognee-fg,#EDECEA)]">
                <span className="absolute left-0 text-[var(--color-cognee-success,#22C55E)]">✓</span>
                {permission}
              </li>
            ))}
          </ul>
          <button onClick={onAuthorize} className={PRIMARY_BUTTON} style={BUTTON_TEXT_STYLE}>
            Continue with {cfg.name}
          </button>
          <p className="m-0 mt-2.5 text-center text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            Opens {cfg.name}&apos;s authorization window — this page stays open
          </p>
          <p className="m-0 mt-2.5 text-center text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            <Link href="/integrations/slack" className={GUIDE_LINK}>
              How {cfg.name} memory works
            </Link>
          </p>
          {canRouteFromExisting && (
            <button
              onClick={() => setShowRouteFromExisting(true)}
              className="mt-2.5 w-full cursor-pointer border-none bg-transparent p-0 text-[var(--color-cognee-fg,#EDECEA)]/55 underline transition-colors hover:text-[var(--color-cognee-fg,#EDECEA)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70"
              style={ROUTE_FROM_EXISTING_STYLE}
            >
              Or route channels from a workspace you already own
            </button>
          )}
        </div>
      )}
    </ModalShell>
  );
}
