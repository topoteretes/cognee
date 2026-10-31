import getMyUserId from "@/modules/apiKeys/getMyUserId";
import type { CogneeInstance } from "@/modules/instances/types";

/**
 * The pod's own id for the authenticated user.
 *
 * Needed because session rows are visible to more people than they belong to:
 * the pod admits a session when `user_id IN visible_ids OR dataset_id IN
 * permitted_dataset_ids`, and UI-created datasets carry tenant-wide read, so a
 * teammate's rows come back too. Telling them apart needs this id.
 *
 * The control plane's id IS the pod's id, despite the separate databases: when
 * a user is synced into a tenant, the pod inserts `User(id=user_id)` with the
 * id the control plane sent, and the pod's Auth0 strategy only ever looks users
 * up by email — it never mints one. So the control plane can answer this in a
 * single call, and the pod is not asked at all.
 *
 * Returns null when it cannot be established — callers must treat that as
 * "cannot tell whose rows these are", never as "no filter needed".
 */
// A pod's id for a user never changes, so resolve it once per instance per tab.
// Keyed on instanceId (COG-6233), and a Promise so concurrent callers share one
// in-flight lookup. Only a successful result is kept — a failure must be
// retried, not remembered.
const _podUserIds = new Map<string, Promise<string | null>>();

export default async function getPodUserId(instance: CogneeInstance): Promise<string | null> {
  const cached = _podUserIds.get(instance.instanceId);
  if (cached) return cached;
  const lookup = getMyUserId().catch((err) => {
    console.warn("[getPodUserId] could not resolve the user id:", err instanceof Error ? err.message : err);
    return null;
  });
  _podUserIds.set(instance.instanceId, lookup);
  lookup.then((id) => {
    if (!id) _podUserIds.delete(instance.instanceId);
  });
  return lookup;
}
