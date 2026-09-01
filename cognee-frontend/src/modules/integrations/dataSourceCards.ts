// TeamConnectorCfg is inlined rather than imported from ./types: types.ts is
// excluded from the public frontend sync (its only other consumers are under
// the excluded app/(app)/integrations/ and app/(app)/link-slack/ routes,
// CLO-686), but this file (and the dashboard OverviewPage that imports it)
// stays synced.

/** A connector that is shared by the whole workspace (OAuth), not per-user. */
export interface TeamConnectorCfg {
  key: string;
  name: string;
  description: string;
  initials: string;
  /**
   * Monochrome glyph filename (no extension) under
   * /visuals/logos/datasources, rendered white on the brand tile. Falls back
   * to `initials` when absent.
   */
  logo?: string;
  color: string;
  /**
   * What Cognee gets access to, listed in the connect modal before the user
   * authorizes. Written in the provider's own vocabulary (channels, pages,
   * repos), so the modal itself stays connector-agnostic.
   */
  permissions: string[];
  /**
   * Provider exposes named sub-resources whose questions can be pointed at a
   * different workspace the same owner controls (CLO-377). Only channel-based
   * connectors do.
   */
  supportsChannelRouting: boolean;
}

// Team-scoped: one connection per workspace, managed by the workspace owner —
// unlike Agents/Automation, which are per-user setup wizards.
//
// Live connectors only. Not-yet-built sources belong in MoreDataSourcesSection,
// the false-door list with its own tracked notify-me capture; putting one here
// would render it twice on the page and offer no way to register interest.
export const DATA_SOURCE_CARDS: TeamConnectorCfg[] = [
  {
    key: "slack",
    name: "Slack",
    description: "Turn channels and threads into searchable memory.",
    initials: "Sl",
    logo: "slack",
    color: "#4A154B",
    permissions: ["Read messages from channels you select", "See workspace and member names"],
    supportsChannelRouting: true,
  },
];
