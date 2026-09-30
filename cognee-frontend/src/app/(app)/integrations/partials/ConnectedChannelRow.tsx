"use client";

import { useState, type ReactElement } from "react";
import { Select } from "@mantine/core";
import classNames from "classnames";
import setChannelRoute from "@/modules/integrations/setChannelRoute";
import deleteChannelRoute from "@/modules/integrations/deleteChannelRoute";
import type { IntegrationChannel } from "@/modules/integrations/getChannels";
import type { AvailableTenant } from "@/modules/users/UserContext";

/** Select value standing for "leave this channel with the workspace that connected it". */
export const PRIMARY_VALUE = "__primary__";

interface ConnectedChannelRowProps {
  provider: string;
  tenantId: string;
  channel: IntegrationChannel;
  /** Tenant this channel is routed to, or undefined when it stays here. */
  routedTo: string | undefined;
  /** Other workspaces the user owns; empty means routing isn't offered. */
  routeTargets: AvailableTenant[];
  onRouteApplied: (channelId: string, targetTenantId: string | null) => void;
  /** Opt this channel in or out of ingestion (CLO-387). Resolves false on failure. */
  onToggleAllowed: (channelId: string, allowed: boolean) => Promise<boolean>;
}

/**
 * Mantine's Select, not a native <select> — a native select's open option
 * list is the browser/OS's own popup, which CSS (including `color-scheme`)
 * cannot reliably restyle. On this dark modal that showed up as a plain
 * white system dropdown fighting the rest of the UI. Mantine's dropdown is a
 * real DOM popover, so the dark theme below actually applies to it, not just
 * to the closed control.
 *
 * Ghost by default (ROOT_STYLES) — a border/background only on hover or
 * focus — so a row with routing available doesn't look heavier than the
 * plain "Not selected" / "Routed elsewhere" text the other rows show in the
 * same column. Same width, right-aligned text, and font size as those so the
 * column reads as one consistent list rather than some rows getting a
 * button and others a label.
 */
const ROOT_STYLES = { width: 152 };

// The input itself: Select's own styles selectors reach this part.
const SELECT_STYLES = {
  input: {
    minHeight: 22,
    height: 22,
    border: "1px solid transparent",
    backgroundColor: "transparent",
    color: "var(--color-cognee-fg, #EDECEA)",
    opacity: 0.6,
    fontSize: 11,
    textAlign: "right" as const,
    paddingRight: 20,
  },
  section: { width: 16 },
};

/**
 * The popup itself lives outside Select's own styles selectors — it's a
 * separate <Combobox>, and only comboboxProps reaches its "dropdown" and
 * "option" selectors. Passing dropdown/option colors to Select's own
 * `styles` (as before) silently did nothing, which is why the popup kept
 * MantineProvider's global *light* theme background regardless.
 * `!important` via Tailwind's `!` prefix because the CSS module's own
 * background-color declaration and this one have equal specificity, and
 * source order between them isn't something to depend on.
 */
const COMBOBOX_CLASSNAMES = {
  dropdown: "!border !border-white/[0.14] !bg-[#141414]",
  option: "!text-[11px] !text-[var(--color-cognee-fg,#EDECEA)] hover:!bg-white/[0.08] data-[combobox-selected]:!bg-cognee-lavender/25",
};

// A single down chevron, same glyph used elsewhere in this modal's controls —
// Mantine's default up/down selector icon reads as a distinct, unfamiliar
// affordance dropped into an otherwise consistent set of dropdowns.
const CHEVRON = (
  <svg width="8" height="8" viewBox="0 0 12 12" fill="none" aria-hidden>
    <path d="M3 4.5L6 7.5L9 4.5" stroke="#EDECEA" strokeOpacity={0.5} strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export default function ConnectedChannelRow({
  provider,
  tenantId,
  channel,
  routedTo,
  routeTargets,
  onRouteApplied,
  onToggleAllowed,
}: ConnectedChannelRowProps): ReactElement {
  const [saving, setSaving] = useState(false);
  const [toggling, setToggling] = useState(false);
  const isRoutedAway = routedTo !== undefined && routedTo !== tenantId;

  async function toggleAllowed(next: boolean): Promise<void> {
    setToggling(true);
    try {
      await onToggleAllowed(channel.id, next);
    } finally {
      // In a finally: an unexpected throw would otherwise leave the checkbox
      // disabled for good, with no way back other than reopening the modal.
      setToggling(false);
    }
  }

  async function changeRoute(value: string): Promise<void> {
    setSaving(true);
    const result =
      value === PRIMARY_VALUE
        ? await deleteChannelRoute(provider, tenantId, channel.id)
        : await setChannelRoute(provider, tenantId, channel.id, channel.name, value);
    setSaving(false);
    if (result.success) onRouteApplied(channel.id, value === PRIMARY_VALUE ? null : value);
  }

  return (
    // shrink-0 + fixed height: the parent list is max-h-[320px] with
    // overflow-y-auto, and once the channels overflow that box, a flex
    // column's default flex-shrink:1 squeezes every row down toward its own
    // min-content floor instead of actually scrolling — and a <Select> and a
    // plain <span> have different floors, so rows with routing available
    // rendered visibly taller than "Not selected" rows in the same list.
    // shrink-0 makes the list scroll (its stated job) instead of compressing.
    <div className="flex shrink-0 items-center justify-between gap-2" style={{ height: 28 }}>
      <label className="flex min-w-0 cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={channel.allowed}
          disabled={toggling}
          aria-label={`Read #${channel.name} into memory`}
          onChange={(e) => void toggleAllowed(e.target.checked)}
          className="shrink-0 cursor-pointer accent-cognee-lavender"
        />
        <span
          className={classNames(
            "truncate text-[13px]",
            channel.allowed ? "text-[var(--color-cognee-fg,#EDECEA)]" : "text-[var(--color-cognee-fg,#EDECEA)]/55",
          )}
        >
          {channel.isPrivate ? "🔒 " : "#"}
          {channel.name}
        </span>
      </label>
      {/* Routing a channel that isn't opted in moves nothing, so the picker is
          replaced by the reason it wouldn't do anything. */}
      {!channel.allowed ? (
        <span className="w-[152px] shrink-0 text-right text-[11px] text-[var(--color-cognee-fg,#EDECEA)]/45">
          Not selected
        </span>
      ) : routeTargets.length > 0 ? (
        <Select
          size="xs"
          value={routedTo ?? PRIMARY_VALUE}
          disabled={saving}
          aria-label={`Route #${channel.name} to a workspace`}
          onChange={(value) => value && void changeRoute(value)}
          data={[{ value: PRIMARY_VALUE, label: "This workspace" }, ...routeTargets.map((t) => ({ value: t.id, label: t.name }))]}
          allowDeselect={false}
          checkIconPosition="right"
          rightSection={CHEVRON}
          comboboxProps={{ withinPortal: true, width: 180, position: "bottom-end", classNames: COMBOBOX_CLASSNAMES }}
          style={ROOT_STYLES}
          styles={SELECT_STYLES}
        />
      ) : (
        isRoutedAway && (
          <span className="w-[152px] shrink-0 text-right text-[11px] text-[var(--color-cognee-fg,#EDECEA)]/45">
            Routed elsewhere
          </span>
        )
      )}
    </div>
  );
}
