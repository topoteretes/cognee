import type { TeamConnectorCfg } from "./types";

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
