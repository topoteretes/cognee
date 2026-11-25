import { CogneeInstance } from "../instances/types";
import { channelDisplayName } from "../sessions/getSessions";

/** One row of the tenant roster: a human member or a registered agent. */
export interface Agent {
  id: string;
  email: string;
  /** De-slugged agent name, lowercase on the wire — see agentDisplayName. */
  agent_type: string;
  agent_short_id: string;
  is_agent: boolean;
  is_default: boolean;
  status: string;
}

/**
 * `agent_type` as it should be shown to a person.
 *
 * The endpoint de-slugs the agent's address without restoring case, so a
 * client that identifies itself as Claude Code arrives as `"claude code"`.
 * Rendering it raw both looks wrong next to every other label in the
 * dashboard and fails to match the channel names in CHANNEL_DEFS
 * (getSessions.ts), which the Via column derives from session prefixes — the
 * same agent would then read "claude code" in one column and "Claude Code"
 * in the next.
 */
export function agentDisplayName(agentType: string): string {
  // A known client wins, so the roster spells it exactly as the Via column
  // does. Title-casing alone produces "Openclaw"/"Vs Code"/"Gemini Cli".
  const channel = channelDisplayName(agentType);
  if (channel) return channel;
  // `\b` is ASCII-only even under the `u` flag, so `\b\p{Ll}` skipped a
  // leading "é" and then capitalised the letter after it: "élodie.dupont"
  // came out "éLodie.Dupont". Match the separators explicitly instead.
  return agentType.replace(/(^|[\s\-_.])(\p{Ll})/gu, (_m, sep: string, ch: string) => sep + ch.toUpperCase());
}

/**
 * Registered agents/users for the current tenant: id, email, agent_type,
 * agent_short_id, is_agent/is_default flags and a
 * LIVE/INACTIVE/NEVER_CONNECTED status. Mirrors getDatasets.ts.
 *
 * `agent_type` arrives lowercase — the endpoint builds it by de-slugging the
 * local part of the agent's `{slug}+{parent_uuid}@cognee.agent` address
 * (`display_name.replace("-", " ").replace("_", " ")`), so an agent that
 * calls itself Claude Code is `"claude code"` here. Use agentDisplayName
 * before rendering it.
 *
 * Throws on a payload that is not an array rather than coercing it to []: a
 * 200 carrying an error envelope would otherwise be indistinguishable from a
 * workspace with nobody in it, which is exactly the failure the activity feed
 * was just fixed for.
 */
export default function getAgents(
  instance: CogneeInstance,
  signal?: AbortSignal,
  timeoutMs?: number,
): Promise<Agent[]> {
  const init: RequestInit & { timeoutMs?: number } = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    timeoutMs,
  };
  return instance.fetch("/v1/activity/agents", init).then((response) => response.json()).then((payload) => {
    if (!Array.isArray(payload)) {
      throw new Error("activity/agents returned a non-array payload");
    }
    return payload as Agent[];
  });
}
