"use client";

import { useEffect, useMemo, useRef } from "react";
import { trackEvent } from "@/modules/analytics";
import type { Agent } from "@/ui/layout/FilterContext";
import type { PipelineRun, SessionRow } from "@/modules/sessions/getSessions";

export interface ChecklistCompletion {
  upload_document: boolean;
  first_query: boolean;
  connect_agent: boolean;
}

interface ChecklistCompletionInput {
  runs: PipelineRun[];
  sessions: SessionRow[];
  agents: Agent[];
  /** True while telemetry (runs/sessions) has not yet produced a trustworthy
   *  answer — still loading, or errored. Gates `upload_document`/`first_query`
   *  only: those two never read the agent roster. */
  telemetryNotReady: boolean;
  /** Same idea for the agent roster. A pod that has not allowlisted
   *  `GET /v1/activity/agents` answers 404 forever — no retry, no poll — so
   *  this can stay true for the life of the page, and it must not also block
   *  the other two items. */
  agentsNotReady: boolean;
}

/**
 * Fires `checklist_item_completed` once per item, the moment it first
 * becomes true — never on a render where it is already true, and never for
 * an item whose own source has not actually answered yet.
 *
 * Each item is seeded and tracked independently rather than as one atomic
 * snapshot: `upload_document`/`first_query` read runs/sessions, `connect_agent`
 * reads the agent roster, and gating all three on whichever source is slowest
 * (or permanently erroring) either goes dark for events that had nothing to
 * do with the failure or seeds the wrong one off an empty array — a workspace
 * that finished an item long ago then fires a spurious completion on the next
 * successful poll of the OTHER source. See OverviewPage for the two concrete
 * failure modes this was written against.
 */
export function useChecklistCompletionTracking(input: ChecklistCompletionInput): void {
  const { runs, sessions, agents, telemetryNotReady, agentsNotReady } = input;

  const checklistDone: ChecklistCompletion = useMemo(() => ({
    // Only a real pipeline run (upload/cognify) counts here — operation rows
    // (recall/search/remember/etc., kind: "operation") mean the agent talked
    // to memory, not that anything was uploaded.
    upload_document: runs.some((r) => r.kind === "pipeline"),
    first_query: sessions.length > 0,
    connect_agent: agents.some((a) => a.is_agent && !a.is_default),
  }), [runs, sessions, agents]);

  const prev = useRef<Partial<ChecklistCompletion>>({});

  useEffect(() => {
    if (!telemetryNotReady) {
      if (prev.current.upload_document === undefined) {
        prev.current.upload_document = checklistDone.upload_document;
        prev.current.first_query = checklistDone.first_query;
      } else {
        if (checklistDone.upload_document && !prev.current.upload_document) {
          trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "upload_document" } });
        }
        if (checklistDone.first_query && !prev.current.first_query) {
          trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "first_query" } });
        }
        prev.current.upload_document = checklistDone.upload_document;
        prev.current.first_query = checklistDone.first_query;
      }
    }
    if (!agentsNotReady) {
      if (prev.current.connect_agent === undefined) {
        prev.current.connect_agent = checklistDone.connect_agent;
      } else {
        if (checklistDone.connect_agent && !prev.current.connect_agent) {
          trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "connect_agent" } });
        }
        prev.current.connect_agent = checklistDone.connect_agent;
      }
    }
  }, [checklistDone, telemetryNotReady, agentsNotReady]);
}
