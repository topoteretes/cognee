"use client";

import { useState, type ReactElement } from "react";
import { Loader } from "@mantine/core";
import classNames from "classnames";

interface DisconnectZoneProps {
  /** Connected account/workspace name, named in the confirmation copy. */
  name: string;
  /** Connector name, e.g. "Slack". */
  provider: string;
  /** Resolves false when the backend refused, which restores the confirm state. */
  onDisconnect: () => Promise<boolean>;
}

// font-size/weight as inline style, not classes: Preflight/Mantine reset
// `button { font: inherit }` in a CSS layer above Tailwind's utilities, so a
// text-size/font-weight utility on a raw <button> loses regardless of its
// own specificity. Inline style sits outside layers entirely and always wins.
const SMALL_BUTTON = "flex cursor-pointer items-center gap-1.5 rounded-lg px-3.5 py-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black";
const SMALL_BUTTON_STYLE = { fontSize: 13, fontWeight: 600 };
const DISCONNECT_TRIGGER_STYLE = { fontSize: 12, fontWeight: 500 };

/**
 * Two-step disconnect, kept visually quiet: the destructive action shouldn't
 * be the largest and most inviting thing in a "connected" modal, and it kills
 * an ingestion pipeline the whole workspace depends on, so it asks first.
 */
export default function DisconnectZone({ name, provider, onDisconnect }: DisconnectZoneProps): ReactElement {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function confirm(): Promise<void> {
    setBusy(true);
    const disconnected = await onDisconnect();
    // On success the parent unmounts this modal; only a refusal returns here,
    // and it already surfaced its own error notification.
    if (!disconnected) {
      setBusy(false);
      setConfirming(false);
    }
  }

  if (!confirming) {
    return (
      <div className="mt-4 border-t border-white/10 pt-3.5">
        <button
          onClick={() => setConfirming(true)}
          className="cursor-pointer border-none bg-transparent p-0 text-[var(--color-cognee-danger-fg,#FF8A8A)] underline-offset-2 transition-colors hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70"
          style={DISCONNECT_TRIGGER_STYLE}
        >
          Disconnect {provider}
        </button>
      </div>
    );
  }

  return (
    <div className="mt-4 border-t border-white/10 pt-3.5">
      <p className="m-0 mb-2.5 text-[13px] leading-[1.45] text-[var(--color-cognee-fg,#EDECEA)]/55">
        Disconnect <strong className="font-semibold text-[var(--color-cognee-fg,#EDECEA)]">{name}</strong>? New messages stop syncing. What
        Cognee already learned from this workspace stays in memory.
      </p>
      <div className="flex justify-end gap-2">
        <button
          onClick={() => setConfirming(false)}
          className={classNames(SMALL_BUTTON, "border border-white/10 bg-white/[0.06] text-[var(--color-cognee-fg,#EDECEA)]/70 hover:bg-white/[0.1]")}
          style={SMALL_BUTTON_STYLE}
        >
          Keep connected
        </button>
        <button
          onClick={() => void confirm()}
          disabled={busy}
          className={classNames(SMALL_BUTTON, "border-none bg-[var(--color-cognee-danger,#EF4444)] text-white hover:bg-[var(--color-cognee-danger,#EF4444)]/85")}
          style={SMALL_BUTTON_STYLE}
        >
          {busy && <Loader size={12} color="#fff" />}
          {busy ? "Disconnecting…" : "Disconnect"}
        </button>
      </div>
    </div>
  );
}
