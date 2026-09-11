"use client";

import type { ReactElement } from "react";
import classNames from "classnames";
import type { TeamConnectorCfg, TeamConnectionState } from "@/modules/integrations/types";
import ConnectorLogo from "./ConnectorLogo";
import ConnectorStatusBadge from "./ConnectorStatusBadge";
import ChannelHealthLine from "./ChannelHealthLine";
import DataSourceCardCta, { type CtaVariant } from "./DataSourceCardCta";
import type { ChannelSummary } from "./useConnectorChannels";

interface DataSourceCardProps {
  cfg: TeamConnectorCfg;
  /** Undefined until the connection status has been fetched. */
  state: TeamConnectionState | undefined;
  /** Undefined until the channel list has been read; only set when connected. */
  channels: ChannelSummary | undefined;
  isOwner: boolean;
  onManageClick: () => void;
  /** Re-read the connection status after a failed read. */
  onRetry: () => void;
}

// height:100% + the grid's default align-items:stretch make every card in a row
// equal height; the flex column then pins the footer to the bottom via mt-auto
// so it lines up across cards regardless of how much text sits above it.
const CARD =
  "flex h-full flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.06] p-5 text-left backdrop-blur-[12px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black";

interface Cta {
  label: string;
  variant: CtaVariant;
  onClick: () => void;
}

// The whole card is the click target (matching the Agents and More-data-sources
// grids), so it needs exactly one action. Null means there is nothing this user
// can do here and the card renders inert.
function ctaFor(props: DataSourceCardProps, state: TeamConnectionState): Cta | null {
  if (state.status === "unavailable") return { label: "Retry", variant: "neutral", onClick: props.onRetry };
  if (!props.isOwner) return null;
  if (state.status === "connected") {
    // A degraded connection (CLO-389) is authorized but not working, so the
    // card leads with the fix instead of "Manage" — the actual reconnect lives
    // in the modal, next to the explanation of what it does and does not touch.
    return state.syncStatus === "degraded"
      ? { label: "Reconnect", variant: "primary", onClick: props.onManageClick }
      : { label: "Manage", variant: "neutral", onClick: props.onManageClick };
  }
  return { label: "Connect", variant: "primary", onClick: props.onManageClick };
}

export default function DataSourceCard(props: DataSourceCardProps): ReactElement {
  const { cfg, state, channels } = props;
  const isConnected = state?.status === "connected";
  const cta = state ? ctaFor(props, state) : null;

  return (
    <button
      type="button"
      disabled={!cta}
      onClick={cta?.onClick}
      aria-label={cta ? `${cta.label} ${cfg.name}` : undefined}
      className={classNames(CARD, cta ? "cursor-pointer hover:border-white/[0.18] hover:bg-white/[0.09]" : "cursor-default")}
    >
      <div className="flex w-full items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <ConnectorLogo logo={cfg.logo} initials={cfg.initials} color={cfg.color} size={40} />
          <span className="truncate font-sans text-[16px] font-medium text-[var(--color-cognee-fg,#EDECEA)]">{cfg.name}</span>
        </div>
        <ConnectorStatusBadge status={state?.status} syncStatus={state?.syncStatus} />
      </div>

      <p className="m-0 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55">{cfg.description}</p>

      {isConnected && (
        <div className="flex flex-col gap-1">
          {state?.workspaceName && (
            <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">
              Workspace <strong className="font-semibold text-[var(--color-cognee-fg,#EDECEA)]">{state.workspaceName}</strong>
            </p>
          )}
          <ChannelHealthLine summary={channels} lastSyncedAt={state?.lastSyncedAt} />
        </div>
      )}

      {state?.status === "unavailable" && (
        <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Couldn&apos;t read the connection status.</p>
      )}

      {/* "Needs reconnect" on its own says what to press, not what happened. */}
      {isConnected && state?.syncStatus === "degraded" && (
        <p className="m-0 text-[12px] text-[var(--color-cognee-warning,#F59E0B)]">
          {cfg.name} stopped accepting this connection.
        </p>
      )}

      <div className="mt-auto">
        {!state ? (
          <DataSourceCardCta />
        ) : cta ? (
          <DataSourceCardCta label={cta.label} variant={cta.variant} />
        ) : (
          // cta is null only for a non-owner on a settled status.
          <p className="m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Only workspace owners can connect {cfg.name}.</p>
        )}
      </div>
    </button>
  );
}
