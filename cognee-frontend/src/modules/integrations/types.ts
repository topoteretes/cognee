import type { ReactNode } from "react";
import type { PreferredOs } from "@/ui/layout/OsPreferenceContext";
import type { ResourceHealthSummary } from "./connectorTypes";

export interface StepDef {
  title: string;
  description: ReactNode;
  code?: string;
  codeToCopy?: string;
  loading?: boolean;
  /** When set, renders multiple separately-copyable code blocks (commands run one at a time). */
  codeBlocks?: { code: string; codeToCopy?: string; label?: string; loading?: boolean }[];
  /** Custom rendered content (e.g. an annotated config preview) shown below the description. */
  content?: ReactNode;
  /** Numbered terminal block with one "copy the lot" button — the shape the
   *  onboarding flow uses. Takes precedence over `code`/`codeBlocks`. */
  lines?: string[];
  copyLabel?: string;
}

/** A connector whose setup is a self-serve wizard (plugin install, MCP config, API key). */
export interface SetupConnectorCfg {
  key: string;
  name: string;
  cta: string;
  description: string;
  /** Icon shown in the card badge and the modal header (24px) */
  icon: ReactNode;
  /**
   * `os` drives the shell-based connectors (credentials, skill installs): the
   * commands genuinely differ between bash and PowerShell. MCP connectors
   * configure themselves through JSON and can ignore it.
   */
  buildSteps: (baseUrl: string, apiKey: string, isInitializing: boolean, os: PreferredOs) => StepDef[];
  /**
   * Whether this connector's steps actually change with `os`. Drives the
   * Mac/Windows toggle in the wizard header — showing it on a connector whose
   * instructions are identical either way just invites pointless clicking.
   */
  osAware?: boolean;
}

/**
 * Which backend contract a connector speaks.
 *
 * "slack" is Slack's own eleven hand-written routes; "framework" is the
 * provider-generic connector API (CLO-628/637). This is not a migration
 * waiting to happen — Slack staying off the framework is a decision recorded
 * in cloud-backend's `connectors.md`, because its connect flow carries an API
 * key mint, allowlist preservation across reinstall and an account lock the
 * generic flow has no equivalent for.
 */
export type ConnectorApi = "slack" | "framework";

/**
 * What this provider calls the things it ingests, so the shared picker can
 * say "3 repositories" or "3 channels" without knowing which provider it is
 * rendering.
 */
export interface ResourceNoun {
  singular: string;
  plural: string;
}

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
  api: ConnectorApi;
  resourceNoun: ResourceNoun;
  /**
   * What selecting resources means for this provider, shown above the picker.
   * Written per connector rather than generated from the noun: what Cognee
   * does with a channel's backlog and with a repository's contents are
   * different promises, and a sentence true of both says nothing.
   */
  selectionNote: string;
  /**
   * In-app guide for this connector, when one exists. Absent means no link is
   * rendered — pointing at a page nobody wrote is worse than saying nothing.
   */
  guidePath?: string;
  /**
   * What to try when the provider's authorization window leads nowhere, shown
   * beside the spinner. GitHub is the case that needs one: it redirects an
   * already-installed app to its settings page, which never calls back.
   */
  connectingHint?: string;
  /**
   * What disconnecting does *not* do, when that is not obvious. GitHub's
   * disconnect revokes the credential and leaves the App installed on the
   * org, which is a GitHub-side action an owner will not assume.
   */
  disconnectNote?: string;
  /**
   * This connector's backend supports linking to an existing installation
   * over OAuth (`authorize?link_existing=true`) instead of a plain OAuth
   * connect: every Connect, Reconnect and "connect another account" opens an
   * account/org picker rather than the control plane silently reusing
   * whichever one it last saw (CLO-627). GitHub is the only connector this
   * exists for — it is also the only one where a second tenant can otherwise
   * be locked out of an org that already has the App installed.
   */
  linkExisting?: boolean;
  /**
   * The button that un-narrows a selection, when the default "Read every
   * {noun}, including new ones" would understate it. A null selection means
   * "everything this connection covers", and for Drive that is the whole of
   * My Drive plus every shared drive — a wider scope than "every folder"
   * describes (review-reported), so Drive says "everything" instead.
   */
  readEverythingLabel?: string;
  /**
   * What this provider calls a resource that *contains* others — Drive's
   * account root and shared drives. Undefined for a provider whose resources
   * are all leaves, which is every connector but Drive.
   *
   * Selecting one is an open-ended promise (everything in it, including what
   * is added later), so a count or a sentence that lumps it in with the
   * leaves says something untrue. Written per connector for the same reason
   * `resourceNoun` is: "drive" and "workspace" are not interchangeable.
   */
  containerNoun?: ResourceNoun;
  /**
   * Offer to delete what the connection imported when it is disconnected
   * (`DELETE …/connection?delete_data=true`). Per connector rather than for
   * the whole framework: a backend that predates the flag ignores it and
   * reports success, so only a connector whose backend is known to take it
   * may offer a delete that would otherwise silently not happen.
   */
  supportsDataDeletion?: boolean;
  /**
   * This provider has no manage URL (`manageUrl` is always undefined) — the
   * only way to add more resources is the provider's own consent-page picker,
   * reached by running the connect flow again. Notion is the case: re-running
   * it is the same OAuth trip as a first connect or a degraded reconnect, the
   * backend upserts the existing credential and carries the current selection
   * across. A provider with a real manage URL (GitHub) ignores this.
   */
  addResourcesViaReconnect?: boolean;
  /**
   * The backend refuses a null selection ("everything") and an empty one for
   * this provider, because its source cannot sync either: Calendar's forgets
   * the events of every calendar left out of the list. Mirrors cloud-backend's
   * `CloudConnector.selection_allows_all = False`, which the catalog does not
   * carry. Without it the picker offers a "read everything" button that can
   * only answer 422, and a Save that can only answer 422.
   */
  requiresSelection?: boolean;
}

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  /** The status read failed; whether this workspace is connected is unknown. */
  | "unavailable";

export interface TeamConnectionState {
  status: ConnectionStatus;
  workspaceName?: string;
  /**
   * Independent of `status` — this tenant also receives channel(s) routed
   * here from a *different* workspace's install (CLO-377). Can be true
   * whether this tenant has its own separate connection or none at all;
   * routing is managed from the tenant that owns that other connection.
   */
  viaRouting?: boolean;
  routedTeamName?: string;
  routedChannelCount?: number;
  /**
   * Health of a connection that *is* connected (CLO-389). Deliberately not a
   * member of `ConnectionStatus`, which describes the read rather than the
   * connection: a degraded workspace is still connected — it keeps its channel
   * list and its routing, and callers gating on `status === "connected"` must
   * go on treating it that way.
   */
  syncStatus?: "ok" | "degraded";
  /** When content was last handed to the tenant for ingestion. */
  lastSyncedAt?: string;
  connectedAt?: string;
  /**
   * Per-resource outcome counts (framework connectors only). `syncStatus`
   * reports the connection as ok while one of forty repositories has never
   * cloned; this is what tells those apart without reading the whole list.
   */
  health?: ResourceHealthSummary;
  /**
   * Where this connection's content lands. Named after an id the provider
   * cannot re-issue rather than the account's label, so a person has no way
   * to work it out from the account name in front of them — the connection
   * has to say it.
   */
  datasetName?: string;
  /**
   * The id behind `datasetName` (CLO-1014), needed to pass this connection's
   * current target back to `POST .../connection/dataset` when the picker
   * switches away from it. Undefined for a connector's own default - the
   * backend sends `dataset_id: null` there even though `datasetName` is set
   * - and defined for every chosen target; `datasetIsDefault` is what tells
   * the two apart, not whether this is present.
   */
  datasetId?: string;
  /** Whether `datasetName` is this connector's own default, rather than a chosen target (CLO-1014). */
  datasetIsDefault?: boolean;
  /**
   * Where to widen what this connection can see, if the connector has a
   * provider-side surface for that at all (CLO-626) — undefined for every
   * connector but GitHub. Backend-computed, the same way `authorize_url` is:
   * the URL shape is provider knowledge (which page needs which GitHub
   * grant), not something this app should rebuild from an account label.
   */
  manageUrl?: string;
}
