/**
 * The connector framework's wire shapes (CLO-628/637/638).
 *
 * Slack is deliberately not on this contract: it keeps its own eleven
 * hand-written routes and a different response shape, by a decision recorded
 * in cloud-backend's `connectors.md` ("Why Slack stays off the framework").
 * The two families coexist here for the same reason they coexist there, and
 * a connector declares which one it speaks through `TeamConnectorCfg.api`.
 */

/** One entry of `GET /integrations/connectors`. Display facts, never secrets. */
export interface ConnectorCatalogEntry {
  provider: string;
  displayName: string;
  /**
   * Whether this deployment holds the provider's secrets. When false, every
   * connect attempt can only answer 503, so the card must not offer one.
   */
  configured: boolean;
  /** Whether this provider has anything to pick between. */
  supportsResourceSelection: boolean;
  /**
   * The connector failed to describe itself. Broken, which needs a different
   * fix from unconfigured and must not be reported as the same thing.
   */
  degraded: boolean;
}

export type ResourceSyncStatus = "pending" | "ok" | "error";

export interface ResourceSync {
  status: ResourceSyncStatus;
  /** The provider's own reason for a failed read; null on any other status. */
  error: string | null;
  /**
   * When this resource last ingested. Carried across a later failure on
   * purpose, so "failing now, last succeeded yesterday" stays expressible.
   */
  syncedAt?: string;
}

/**
 * Provider-specific display hints. Free-form on the wire; the fields GitHub
 * and Google Drive send are typed here because the picker renders them, and
 * everything else is ignored rather than guessed at.
 */
export interface ConnectorResourceAttributes {
  private?: boolean;
  /** Linear's short team key ("ENG"), shown beside the team's name. */
  key?: string;
  archived?: boolean;
  defaultBranch?: string;
  language?: string;
  /**
   * Drive lists the account root, every shared drive and every folder as one
   * flat list (cloud-backend#495). `root` marks the first two — selecting
   * one reads the whole drive — and `sharedDrive` tells those two apart.
   */
  root?: boolean;
  sharedDrive?: boolean;
  /** The shared drive a folder sits in; absent for one under My Drive. */
  driveId?: string;
  /** Notion's "page" or "data_source" (its name for a database). Display only. */
  kind?: string;
  /** Calendar's own calendar of the account, listed by its address. */
  primary?: boolean;
  /** What the account may do with a Calendar calendar ("owner", "reader", …). Display only. */
  accessRole?: string;
}

export interface ConnectorResource {
  id: string;
  name: string;
  description: string | null;
  attributes: ConnectorResourceAttributes;
  /**
   * This resource's own last outcome, or null when nothing is known — never
   * dispatched, or connected before per-resource outcomes were recorded.
   * Null is *not* `pending`: one means "never asked", the other "asked and
   * waiting", and rendering them alike reports a resource as in flight
   * forever.
   */
  sync: ResourceSync | null;
}

/**
 * What a connection could ingest, and what it is currently narrowed to.
 *
 * `selected` is tri-state and every state means something different:
 * `null` is "everything this connection covers, including resources added
 * later", `[]` is "narrowed to nothing", and a list is exactly those. A UI
 * that renders `null` as "none selected" reports a healthy connection as
 * switched off.
 */
export interface ConnectorResources {
  resources: ConnectorResource[];
  selected: string[] | null;
  /**
   * Leaving a resource out of the selection makes the source forget what it
   * read from it (Calendar). Absent from a backend that predates the field,
   * which reads as "unticking only stops reading", as for every other
   * connector.
   */
  unselectForgetsData?: boolean;
}

/**
 * Why a resource list could not be read. A reason rather than a sentence: the
 * noun differs per connector ("repository", "channel"), so the wording is the
 * UI's to build and this layer stays provider-agnostic.
 */
export type ResourceReadFailure = "not_connected" | "provider_unreachable" | "unavailable";

/**
 * A resource read, where "could not read" is its own answer rather than an
 * empty list. `data` is null exactly when `failure` is set: a failed read must
 * never be able to produce `selected: null`, which would read as the widest
 * possible selection.
 */
export interface ConnectorResourcesResult {
  data: ConnectorResources | null;
  failure: ResourceReadFailure | null;
}

/** Why a selection was not saved. Same reasoning as `ResourceReadFailure`. */
export type ResourceSaveFailure = "forbidden" | "not_connected" | "rejected" | "unavailable";

/** Per-status counts from `GET /{provider}/connection`. */
export interface ResourceHealthSummary {
  ok: number;
  error: number;
  pending: number;
}

/**
 * One entry of `GET /{provider}/connections` (CLO-528/890) — a tenant can
 * hold more than one account for the same framework connector (a personal
 * GitHub account alongside a company org), each with its own independent
 * repository selection, since the backend keys that selection on the
 * account's own credential row, not on the tenant.
 */
export interface ConnectorAccount {
  accountId: string;
  accountLabel: string | undefined;
  connectedAt: string | undefined;
  lastSyncedAt: string | undefined;
  syncStatus: string | undefined;
  health: ResourceHealthSummary | undefined;
}

/**
 * One installation the signed-in account can link (CLO-627's
 * `GET /{provider}/link-choices`) — an owner picks one of these rather than
 * the control plane silently reusing whatever the OAuth session last saw.
 */
export interface LinkChoice {
  installationId: string;
  accountLogin: string;
  /** Null when the provider does not say — never guessed at. */
  accountType: "Organization" | "User" | null;
  /** Already linked to a Cognee workspace; the picker shows but never lets this be chosen. */
  alreadyConnected: boolean;
}

/**
 * Why a link-choices read or a link confirmation failed. Shared by both calls
 * — a token this expired or wrong-user for `link-choices` is exactly as
 * expired or wrong-user for the `link` that would have used it next.
 */
export type LinkFailure = "expired" | "forbidden" | "failed";

/** One dataset a connection could be pointed at instead of its default (CLO-1014). */
export interface DatasetOption {
  id: string;
  name: string;
}

/**
 * `GET /{provider}/dataset-options`'s payload: every writable dataset this
 * connection could target, plus the name its own default already carries.
 */
export interface DatasetOptions {
  options: DatasetOption[];
  defaultDatasetName: string;
}

/**
 * Why a dataset-options read failed. `not_available` is its own state
 * (CLO-1014's 502): a tenant pod that predates the feature, not a workspace
 * with nothing to pick from, and the two must not read the same in the UI.
 */
export type DatasetOptionsFailure = "forbidden" | "not_connected" | "not_available" | "unavailable";

/**
 * A dataset-options read, where "could not read" is its own answer rather
 * than an empty list, same shape as `ConnectorResourcesResult`, for the
 * identical reason.
 */
export interface DatasetOptionsResult {
  data: DatasetOptions | null;
  failure: DatasetOptionsFailure | null;
}

/**
 * Why `POST .../connection/dataset` refused, one entry per distinct backend
 * answer (CLO-1014); each gets its own copy in the picker rather than a
 * shared "could not save", since each names a different thing the owner
 * can actually do about it.
 */
export type SetDatasetTargetFailure =
  | "forbidden"
  | "not_connected"
  | "reserved_default_name"
  | "sync_in_progress"
  | "dataset_conflict"
  | "delete_unsupported"
  | "concurrent_change"
  /**
   * The dataset picked is no longer one this account can write to - someone
   * else's permissions on it changed between the picker loading and Save.
   * The picker's list is stale, not just this one submission.
   */
  | "dataset_not_writable"
  /** A disconnect is already in flight for this connection; retryable once it finishes. */
  | "disconnect_in_progress"
  | "not_available"
  /**
   * The switch itself committed, but the requested delete of the old target
   * failed afterward (a 502 distinct from `not_available`'s pre-write one -
   * see `setConnectionDataset.ts`). The caller must still treat this as a
   * changed target, not as a no-op.
   */
  | "switched_delete_failed"
  /**
   * A default target with nothing a delete could remove (the connector's own
   * `dataset_name()` is `None` - true of every framework connector but
   * Google Drive and Gmail). Its own reason rather than `delete_unsupported`,
   * which is about a *shared* target's own inability to be scoped safely.
   */
  | "default_not_deletable"
  | "rejected"
  | "unavailable";

export interface SetDatasetTargetResult {
  success: boolean;
  failure?: SetDatasetTargetFailure;
  datasetName?: string;
  datasetId?: string;
  datasetIsDefault?: boolean;
}
