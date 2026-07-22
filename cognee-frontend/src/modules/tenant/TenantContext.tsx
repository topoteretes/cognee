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

export interface TenantContextValue {
  tenant: Tenant | null;
  cogniInstance: CogneeInstance | null;
  localInstance: CogneeInstance;
  serviceUrl: string | null;
  apiKey: string;
  isInitializing: boolean;
  // Whether the current tenant's pod has finished provisioning and can serve
  // requests — distinct from isInitializing (which just means "we haven't
  // resolved a tenant yet"). Consumers gate on this before firing the first
  // pod-bound request (see FilterContext's datasetsQuery, OverviewPage).
  tenantReady: boolean;
  // True only when a tenant is resolved but its pod is confirmed unreachable
  // (as opposed to still starting up) — checked before tenantReady-gated
  // "still setting up" UI so a genuinely dead pod doesn't show a perpetual
  // loading state.
  podUnreachable: boolean;
  error: string | null;
  statusMessage: { title: string; subtitle: string } | null;
  availableTenants: AvailableTenant[];
  switchTenant: (tenantId: string, tenantName?: string, navigateTo?: string) => void;
  planType: PlanType;
  hasAccess: boolean;
  requestCreateWorkspace: () => void;
  // Whether the current user owns the current tenant (vs. a member/guest).
  isOwner: boolean;
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
  tenantReady: false,
  podUnreachable: false,
  error: null,
  statusMessage: null,
  availableTenants: [],
  switchTenant: () => {},
  planType: null,
  hasAccess: true,
  requestCreateWorkspace: () => {},
  isOwner: true,
  nameModalOpen: false,
  releaseLoader: () => {},
});

export function useTenant() {
  const context = useContext(TenantContext);
  return {
    tenant: context.tenant,
    isInitializing: context.isInitializing,
    tenantReady: context.tenantReady,
    podUnreachable: context.podUnreachable,
    error: context.error,
    availableTenants: context.availableTenants,
    switchTenant: context.switchTenant,
    planType: context.planType,
    hasAccess: context.hasAccess,
    requestCreateWorkspace: context.requestCreateWorkspace,
    isOwner: context.isOwner,
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
