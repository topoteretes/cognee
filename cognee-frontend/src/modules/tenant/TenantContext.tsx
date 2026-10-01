"use client";

import { createContext, useContext } from "react";
import { CogneeInstance } from "@/modules/instances/types";
import { Tenant } from "./types";
import localFetch from "@/modules/instances/localFetch";
import { useUser, type AvailableTenant } from "@/modules/users/UserContext";

// Open-source override — identical to the SaaS TenantContext except PlanType,
// whose home (the cloud payment infrastructure module) is private. Keep the
// interface in lockstep with the SaaS original: synced shared UI and the synced
// LocalProvider both type against it, and the sync build gate fails on drift.
export type PlanType = string | null;

// Re-exported for backward compatibility — the type now lives in UserContext
// since the tenant LIST is a User-domain fact ("what am I a member of").
// TenantContext only owns the ACTIVE tenant's operational state.
export type { AvailableTenant };

// The auto-created first workspace every user gets on sign-up. It can never be
// deleted, and is identified purely by this name (kept in sync with the name
// used when the workspace is created in TenantProvider).
export const PERSONAL_WORKSPACE_NAME = "Personal Workspace";

export interface TenantContextValue {
  tenant: Tenant | null;
  cogniInstance: CogneeInstance | null;
  localInstance: CogneeInstance;
  serviceUrl: string | null;
  apiKey: string;
  isInitializing: boolean;
  // False while a freshly-created workspace is still provisioning in the
  // background. The UI is rendered, but tenant API calls will fail until
  // the pod is reachable.
  tenantReady: boolean;
  // True once pod-readiness polling has genuinely given up — distinct from
  // tenantReady, which optimistically flips true regardless of outcome so
  // query gates and nav locks don't hold forever on a dead pod. Consumers
  // that need a real terminal error state (not just "still connecting")
  // read this instead.
  podUnreachable: boolean;
  error: string | null;
  statusMessage: { title: string; subtitle: string } | null;
  switchTenant: (tenantId: string, tenantName?: string, navigateTo?: string) => void;
  planType: PlanType;
  hasAccess: boolean;
  requestCreateWorkspace: () => void;
  nameModalOpen: boolean;
  releaseLoader: () => void;
}

export const localInstance: CogneeInstance = {
  name: "LocalCognee",
  instanceId: "local",
  fetch: localFetch,
};

export const TenantContext = createContext<TenantContextValue>({
  tenant: null,
  cogniInstance: null,
  localInstance,
  serviceUrl: null,
  apiKey: "",
  isInitializing: true,
  tenantReady: true,
  podUnreachable: false,
  error: null,
  statusMessage: null,
  switchTenant: () => {},
  planType: null,
  hasAccess: true,
  requestCreateWorkspace: () => {},
  nameModalOpen: false,
  releaseLoader: () => {},
});

export function useTenant() {
  const context = useContext(TenantContext);
  // The tenant list is a User-domain fact — composed in here from UserContext
  // so every existing caller of useTenant().availableTenants keeps working.
  const { availableTenants } = useUser();
  // Whether the signed-in user OWNS the currently-selected tenant. Used to gate
  // owner-only surfaces (e.g. Billing). Defaults to false when the current
  // tenant isn't resolved yet, so owner-only UI stays hidden until we're sure.
  const currentAvailable = availableTenants.find(
    (t) => t.id === context.tenant?.tenant_id,
  );
  const isOwner = currentAvailable?.isOwner ?? false;
  // Whether the currently-selected tenant is the user's personal workspace —
  // matched by name, since the tenant list carries no creation date. Falls back
  // to the active tenant's own name when it isn't in availableTenants yet.
  const currentName = currentAvailable?.name ?? context.tenant?.tenant_name;
  const isPersonal = currentName === PERSONAL_WORKSPACE_NAME;
  return {
    tenant: context.tenant,
    isInitializing: context.isInitializing,
    tenantReady: context.tenantReady,
    podUnreachable: context.podUnreachable,
    error: context.error,
    availableTenants,
    isOwner,
    isPersonal,
    switchTenant: context.switchTenant,
    planType: context.planType,
    hasAccess: context.hasAccess,
    requestCreateWorkspace: context.requestCreateWorkspace,
    nameModalOpen: context.nameModalOpen,
    releaseLoader: context.releaseLoader,
  };
}

export function useCogniInstance() {
  const context = useContext(TenantContext);
  return {
    cogniInstance: context.cogniInstance,
    localInstance: context.localInstance,
    serviceUrl: context.serviceUrl,
    apiKey: context.apiKey,
    isInitializing: context.isInitializing,
    error: context.error,
    statusMessage: context.statusMessage,
  };
}
