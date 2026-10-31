import { isRolloutEnabled } from "@/utils/isRolloutEnabled";
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
    api: "slack",
    resourceNoun: { singular: "channel", plural: "channels" },
    selectionNote:
      "Cognee reads the channels selected below, including the history they already have, and anyone in this workspace can then ask about them. Unticking a channel later only stops reading it; what was already read stays.",
    guidePath: "/integrations/slack",
  },
  {
    key: "github",
    name: "GitHub",
    description: "Turn repository code and docs into searchable memory.",
    initials: "GH",
    logo: "github",
    color: "#181717",
    permissions: [
      "Read code and documentation in the repositories you select",
      "See the organization and repository names it is installed on",
    ],
    // The framework's cross-tenant routing has no GitHub equivalent: a
    // repository is not addressable the way a channel is.
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "repository", plural: "repositories" },
    selectionNote:
      "Cognee reads the repositories selected below, and anyone in this workspace can then ask about their code and docs. Unticking one later only stops reading it; what was already read stays.",
    // Only the "Install on another organization" path can still dead-end
    // this way (CLO-627) — every other Connect/Reconnect goes through
    // link_existing, which always calls back.
    connectingHint:
      "If GitHub's window stalls on the app's settings page, close it and try again — that can only happen when installing fresh on an organization that already has Cognee.",
    disconnectNote:
      "The Cognee GitHub App stays installed on the organization — removing it is a GitHub-side action. Reconnecting lets you pick the account or organization again: the same one picks up where it left off, a different one starts its own.",
    linkExisting: true,
  },
  {
    key: "linear",
    name: "Linear",
    description: "Turn issues, comments and projects from the teams you select into searchable memory.",
    initials: "Li",
    logo: "linear",
    color: "#5E6AD2",
    permissions: [
      "Read issues, comments and projects in the teams you select",
      "See your workspace and team names, including private teams the account can see",
    ],
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "team", plural: "teams" },
    // Private teams are listed too: the grant covers every team the account
    // can see, and what is read becomes visible to the whole workspace.
    selectionNote:
      "Cognee reads the issues, comments and projects of the teams selected below, private teams included, and anyone in this workspace can then ask about them. Unticking a team later only stops reading it; what was already read stays, and so does anything deleted in Linear afterwards.",
    supportsDataDeletion: true,
  },
  {
    key: "notion",
    name: "Notion",
    description: "Turn the pages and databases you select into searchable memory.",
    initials: "No",
    logo: "notion",
    color: "#000000",
    permissions: [
      "Read the pages and databases you share with it, including their content",
      "See your workspace name",
    ],
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "page", plural: "pages" },
    // Notion has no "what can this see" call of its own: access is granted
    // page by page from Notion's side, and the workspace-wide selection below
    // only narrows what Cognee reads out of what was already shared with it.
    selectionNote:
      "Cognee reads the pages and databases selected below, and anyone in this workspace can then ask about them — Notion's own page-level permissions are not carried over. Attachments are kept as a name and caption only, not their contents, and a mention of another page stays as it read until that page is itself edited. Unticking a page or database later only stops reading it; what was already read stays.",
    // Notion has no manage page: sharing more pages with the integration only
    // happens through its own consent-page picker, reached by connecting again.
    addResourcesViaReconnect: true,
  },
  {
    key: "google_drive",
    name: "Google Drive",
    description: "Turn the Drive folders and shared drives you select into searchable memory.",
    initials: "Dr",
    logo: "googledrive",
    color: "#4285F4",
    // Exactly the grant (drive.readonly plus the account's email), and
    // nothing about following changes: polling is off everywhere, so content
    // is read on connect and each time a selection is saved.
    permissions: [
      "Read files in the folders and shared drives you select, including nested folders",
      "See your folder and shared drive names, and the email address of the account",
    ],
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "folder", plural: "folders" },
    selectionNote:
      "Cognee reads the folders and shared drives selected below, including everything nested in them, when you connect and each time you save a selection. Anyone in this workspace can then ask about their files.",
    // A null selection is not "every folder" here: it is the whole of My
    // Drive plus every shared drive the account can see.
    readEverythingLabel: "Read everything, now and later",
    // My Drive and each shared drive are listed alongside the folders and are
    // selectable, but selecting one reads everything inside it forever — so
    // counts and copy have to be able to name them apart from folders.
    containerNoun: { singular: "drive", plural: "drives" },
    // cloud-backend#495 takes `delete_data` on disconnect. GitHub's backend
    // will too once that ships; until then its disconnect must not offer a
    // delete the backend silently ignores.
    supportsDataDeletion: true,
  },
  {
    key: "gmail",
    name: "Gmail",
    description: "Turn the Gmail labels you select into searchable memory.",
    initials: "Gm",
    logo: "gmail",
    color: "#EA4335",
    // Exactly the grant: gmail.readonly plus the account's email.
    permissions: [
      "Read email in the labels you select",
      "See your label names, and the email address of the account",
    ],
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "label", plural: "labels" },
    // Mail is private, so nothing is read until a label is ticked.
    selectionNote:
      "Cognee reads the emails in the labels selected below when you connect and each time you save a selection, and anyone in this workspace can then ask about them. Unticking a label later only stops reading it; what was already read stays.",
    supportsDataDeletion: true,
  },
  {
    key: "google_calendar",
    name: "Google Calendar",
    description: "Turn the events in the calendars you select into searchable memory.",
    initials: "Ca",
    logo: "google_calendar",
    color: "#4285F4",
    // Exactly the grant: calendar.readonly plus the account's email.
    permissions: [
      "Read events in the calendars you select, including attendees and meeting details",
      "See your calendar names, and the email address of the account",
    ],
    supportsChannelRouting: false,
    api: "framework",
    resourceNoun: { singular: "calendar", plural: "calendars" },
    // Events name people and meetings, so nothing is read until a calendar is ticked.
    selectionNote:
      "Cognee reads the events in the calendars selected below when you connect and each time you save a selection, and anyone in this workspace can then ask about them. Unticking a calendar later forgets its events, and so does an event deleted in Google Calendar afterwards.",
    requiresSelection: true,
    supportsDataDeletion: true,
  },
];

/**
 * Connectors still in rollout (CLO-1100): listed on dev.cognee.ai, Vercel
 * previews and local dev, but not on the hosts below. The card stays in
 * DATA_SOURCE_CARDS either way, so status reads and lookups by key are
 * untouched — only listing it and starting a connection are gated.
 */
const DEV_ONLY_PROVIDERS: ReadonlySet<string> = new Set([
  "google_drive",
  "gmail",
  "google_calendar",
]);
const RELEASED_HOSTS: ReadonlySet<string> = new Set(["staging.cognee.ai", "platform.cognee.ai"]);

export function isProviderShownOnHost(provider: string, host: string | null): boolean {
  if (!DEV_ONLY_PROVIDERS.has(provider)) return true;
  return isRolloutEnabled(RELEASED_HOSTS, host);
}
