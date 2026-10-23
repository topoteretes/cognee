"use client";

import { useCallback, useMemo, useState, type ReactElement } from "react";
import ModalShell from "@/ui/elements/ModalShell";
import { ACCESS_LEVELS, levelFor, type AccessLevel } from "@/modules/permissions/accessLevels";
import changeDatasetAccess from "@/modules/permissions/changeDatasetAccess";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { BORDER_SOFT, TEXT, TEXT_FAINT, ghostButtonStyle, primaryButtonStyle, selectStyle } from "@/ui/elements/panelTokens";
import AccessLevelSelect from "@/ui/elements/AccessLevelSelect";
import { notifyFailure } from "@/utils/memberNotifications";
import { modalHintStyle, modalTitleStyle } from "@/ui/elements/modalStyles";
import useShareCandidates, { WORKSPACE_LABEL, type ShareCandidate } from "@/app/(app)/datasets/partials/useShareCandidates";
import type { DatasetAccess } from "@/modules/permissions/types";

interface Grant {
  principalId: string;
  label: string;
  level: AccessLevel | null;
}

interface ShareAccessModalProps {
  datasetId: string;
  datasetName: string;
  access: DatasetAccess;
  tenantId: string | null;
  onClose: () => void;
  /** Called once a change has gone through, so the lists behind this can re-read. */
  onChanged: () => void;
}

function grantsOf(access: DatasetAccess, labels: ReadonlyMap<string, string>): Grant[] {
  return (access.sharedWith ?? [])
    // The owner and their agents are not "shared with"; they are the owner.
    .filter((grant) => grant.principalId !== access.ownerId && grant.parentUserId === null)
    .map((grant) => ({
      principalId: grant.principalId,
      label: grant.principalType === "tenant" ? WORKSPACE_LABEL : (grant.name ?? labels.get(grant.principalId) ?? "Unknown"),
      level: levelFor(grant.permissions),
    }));
}

/**
 * Who has a dataset and at which level, and the way to give it to someone else:
 * a person in the workspace, a group, or everyone. Each change shows at once and is
 * sent to the pod; a refusal puts the list back to what the pod holds.
 */
export default function ShareAccessModal({
  datasetId, datasetName, access, tenantId, onClose, onChanged,
}: ShareAccessModalProps): ReactElement {
  const { cogniInstance } = useCogniInstance();
  const candidates = useShareCandidates();
  const labels = useMemo(() => new Map((candidates ?? []).map((c) => [c.id, c.label])), [candidates]);
  const [local, setLocal] = useState<Grant[] | null>(null);
  const grants = local ?? grantsOf(access, labels);
  const [pick, setPick] = useState("");
  const [level, setLevel] = useState<AccessLevel>("view");

  const change = useCallback(async (principalId: string, label: string, target: AccessLevel | null): Promise<void> => {
    if (!cogniInstance) return;
    const others = grants.filter((grant) => grant.principalId !== principalId);
    setLocal(target ? [...others, { principalId, label, level: target }] : others);
    try {
      await changeDatasetAccess(principalId, datasetId, target, cogniInstance);
    } catch (e) {
      notifyFailure(target ? "Access not changed" : "Access not removed", e);
      setLocal(null);
    }
    onChanged();
  }, [cogniInstance, datasetId, grants, onChanged]);

  const options: ShareCandidate[] = (candidates ?? []).filter(
    (candidate) => candidate.id !== access.ownerId && !grants.some((grant) => grant.principalId === candidate.id),
  );
  const chosen = options.find((candidate) => candidate.id === pick);

  return (
    <ModalShell onClose={onClose} label={`Share ${datasetName}`} width={520}>
      <div>
        <h2 style={modalTitleStyle}>Share {datasetName}</h2>
        <p style={modalHintStyle}>Private until you share it. Give a person, a group or the whole workspace the same access to it.</p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", border: `1px solid ${BORDER_SOFT}`, borderRadius: 10 }}>
        {grants.length === 0 && <div style={{ padding: 16, fontSize: 13, color: TEXT_FAINT }}>Only you can open this dataset.</div>}
        {grants.map((grant, index) => (
          <div key={grant.principalId} className="flex items-center gap-3"
            style={{ padding: "10px 14px", borderTop: index === 0 ? "none" : `1px solid ${BORDER_SOFT}` }}>
            <span className="flex-1 min-w-0 truncate" style={{ fontSize: 14, color: TEXT }}>{grant.label}</span>
            <AccessLevelSelect value={grant.level} disabled={!cogniInstance} label={`Access for ${grant.label}`}
              onChange={(next) => void change(grant.principalId, grant.label, next)} />
            <button style={ghostButtonStyle} aria-label={`Remove ${grant.label}`}
              onClick={() => void change(grant.principalId, grant.label, null)}>Remove</button>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <select aria-label="Share with" value={pick} onChange={(event) => setPick(event.target.value)}
          disabled={candidates === null} style={{ ...selectStyle, flex: 1, minWidth: 0 }}>
          <option value="">{candidates === null ? "Loading…" : "Add a person, group or everyone…"}</option>
          {options.map((candidate) => (
            <option key={candidate.id} value={candidate.id}>
              {candidate.kind === "role" ? `Group: ${candidate.label}` : candidate.label}
            </option>
          ))}
        </select>
        <AccessLevelSelect value={level} disabled={false} label="Level for the new access" onChange={setLevel} />
        <button style={primaryButtonStyle} disabled={!chosen || !tenantId}
          onClick={() => { if (chosen) { void change(chosen.id, chosen.label, level); setPick(""); } }}>
          Share
        </button>
      </div>
      <p style={{ ...modalHintStyle, margin: 0 }}>
        {ACCESS_LEVELS.map((entry) => `${entry.label}: ${entry.hint.toLowerCase()}`).join(" · ")}
      </p>
    </ModalShell>
  );
}
