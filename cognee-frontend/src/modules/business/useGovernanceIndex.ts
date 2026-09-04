"use client";

import { useMemo } from "react";
import type { BusinessGraphNode, VisualizationPayload } from "./types";

export interface AccessSlot {
  owns: boolean;
  perms: Set<string>;
}

export interface GovernanceIndex {
  tenants: BusinessGraphNode[];
  users: BusinessGraphNode[];
  agents: BusinessGraphNode[];
  datasets: BusinessGraphNode[];
  datasetById: Record<string, BusinessGraphNode>;
  // principalId ("user:<id>" / "agent:<id>") -> datasetId -> what it can do.
  access: Record<string, Record<string, AccessSlot>>;
  agentOwnerId: Record<string, string>;
}

const EMPTY: GovernanceIndex = {
  tenants: [], users: [], agents: [], datasets: [], datasetById: {}, access: {}, agentOwnerId: {},
};

function grantAccess(access: GovernanceIndex["access"], principalId: string, datasetId: string): AccessSlot {
  const slots = (access[principalId] = access[principalId] || {});
  return (slots[datasetId] = slots[datasetId] || { owns: false, perms: new Set() });
}

// build_provenance_graph namespaces actor-graph node ids ("dataset:<uuid>")
// so they never collide with raw memory-node ids — but /visualize/brains
// (content layer) keys its payload by the bare dataset UUID. Every dataset
// id this index exposes downstream (rail cards, access lookups) must be
// bare, so a click here can be used directly as contentLayer's
// activeDatasetId without every caller needing to know about the prefix.
function bareDatasetId(id: string): string {
  const prefix = "dataset:";
  return id.startsWith(prefix) ? id.slice(prefix.length) : id;
}

// Ports the governance-overlay indexes built once from the actor/access
// edges (customer_tutorial.html:5655-5692) — who owns/can touch what, which
// user operates which agent. The Business rails read this directly; the
// canvas never does (access is a rail highlight, never a drawn edge — see
// wireUserHover in the ticket).
export function useGovernanceIndex(payload: VisualizationPayload | null): GovernanceIndex {
  return useMemo(() => {
    if (!payload) return EMPTY;
    const byId: Record<string, BusinessGraphNode> = {};
    payload.nodes.forEach((n) => { byId[n.id] = n; });

    const access: GovernanceIndex["access"] = {};
    const agentOwnerId: Record<string, string> = {};
    payload.links.forEach((l) => {
      const s = byId[l.source], t = byId[l.target];
      if (!s || !t) return;
      if (s.type === "User" && t.type === "Agent" && l.relation === "operates") {
        agentOwnerId[t.id] = s.id;
      }
      if (s.type === "Agent" && t.type === "Dataset" && (l.relation === "reads" || l.relation === "writes")) {
        grantAccess(access, s.id, bareDatasetId(t.id)).perms.add(l.relation === "reads" ? "read" : "write");
      }
      if (s.type === "User" && t.type === "Dataset") {
        if (l.relation === "owns") grantAccess(access, s.id, bareDatasetId(t.id)).owns = true;
        else if (typeof l.relation === "string" && l.relation.startsWith("can_")) {
          grantAccess(access, s.id, bareDatasetId(t.id)).perms.add(l.relation.slice(4));
        } else if (l.relation === "reads" || l.relation === "writes") {
          grantAccess(access, s.id, bareDatasetId(t.id)).perms.add(l.relation === "reads" ? "read" : "write");
        }
      }
    });

    const datasets = payload.nodes
      .filter((n) => n.type === "Dataset")
      .map((d) => ({ ...d, id: bareDatasetId(d.id) }));
    const datasetById: Record<string, BusinessGraphNode> = {};
    datasets.forEach((d) => { datasetById[d.id] = d; });

    return {
      tenants: payload.nodes.filter((n) => n.type === "Tenant"),
      users: payload.nodes.filter((n) => n.type === "User"),
      agents: payload.nodes.filter((n) => n.type === "Agent"),
      datasets,
      datasetById,
      access,
      agentOwnerId,
    };
  }, [payload]);
}

export function accessibleDatasetIds(index: GovernanceIndex, principalId: string): Set<string> {
  return new Set(Object.keys(index.access[principalId] || {}));
}

export function permissionCode(slot: AccessSlot): string {
  if (slot.owns) return "owner";
  const p = slot.perms;
  const code = (p.has("read") ? "R" : "") + (p.has("write") ? "W" : "") +
    (p.has("share") ? "S" : "") + (p.has("delete") ? "D" : "");
  return code || "R";
}

// The governance payload (memory_provenance) carries no is_current flag —
// that came from the classic Graph view's own session data, which this port
// doesn't have. In local/OSS mode there's only ever one user, so "the one
// user" and "you" are the same fact; multi-user (cloud) mode would need the
// backend to actually mark the caller, which this doesn't attempt.
// A raw graph node id some principals carry instead of a real name/email
// (e.g. "user:920fd9eb") — a short prefix, a colon, then a hex-looking
// fragment. Distinct from a real display name ("Jane Doe") or email.
const RAW_PRINCIPAL_ID_RE = /^\w+:[0-9a-f-]+$/i;

export function userLabel(index: GovernanceIndex, user: BusinessGraphNode): string {
  if (index.users.length === 1) return "you";
  const name = String(user.name || "");
  // Only the raw-id case has nothing real to show — fall back to a stable
  // "member N" (by position in the users list) rather than echoing an
  // internal id. Anything else (a full email, a display name) is shown as-is.
  if (!name || RAW_PRINCIPAL_ID_RE.test(name)) {
    const position = index.users.findIndex((u) => u.id === user.id);
    return `member ${position + 1}`;
  }
  return name;
}
