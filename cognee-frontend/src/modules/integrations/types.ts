import type { ReactNode } from "react";
import type { PreferredOs } from "@/ui/layout/OsPreferenceContext";

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
}
