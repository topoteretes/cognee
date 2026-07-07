"use client";

import { createContext, useContext } from "react";
import { CogneeInstance } from "@/modules/instances/types";
import { Tenant } from "./types";
import localFetch from "@/modules/instances/localFetch";

export type PlanType = string | null;

export interface AvailableTenant {
  id: string;
  name: string;
  isOwner: boolean;
  ownerHasSubscription: boolean;
}

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
  // background. Always true in the open-source build — there is no pod to
  // wait for.
  tenantReady: boolean;
  error: string | null;
  statusMessage: { title: string; subtitle: string } | null;
  availableTenants: AvailableTenant[];
  switchTenant: (tenantId: string, tenantName?: string, navigateTo?: string) => void;
  planType: PlanType;
  hasAccess: boolean;
  requestCreateWorkspace: () => void;
  nameModalOpen: boolean;
  releaseLoader: () => void;
}

export const localInstance: CogneeInstance = {
  name: "LocalCognee",
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
  error: null,
  statusMessage: null,
  availableTenants: [],
  switchTenant: () => {},
  planType: null,
  hasAccess: true,
  requestCreateWorkspace: () => {},
  nameModalOpen: false,
  releaseLoader: () => {},
});

export function useTenant() {
  const context = useContext(TenantContext);
  const currentAvailable = context.availableTenants.find(
    (t) => t.id === context.tenant?.tenant_id,
  );
  const isOwner = currentAvailable?.isOwner ?? false;
  const currentName = currentAvailable?.name ?? context.tenant?.tenant_name;
  const isPersonal = currentName === PERSONAL_WORKSPACE_NAME;
  return {
    tenant: context.tenant,
    isInitializing: context.isInitializing,
    tenantReady: context.tenantReady,
    error: context.error,
    availableTenants: context.availableTenants,
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
