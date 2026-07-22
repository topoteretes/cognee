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
  isInitializing: boolean;
  tenantReady: boolean;
  podUnreachable: boolean;
  isOwner: boolean;
  error: string | null;
  statusMessage: { title: string; subtitle: string } | null;
  availableTenants: AvailableTenant[];
  switchTenant: (tenantId: string, tenantName?: string, navigateTo?: string) => void;
  planType: PlanType;
  hasAccess: boolean;
  requestCreateWorkspace: () => void;
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
  isInitializing: true,
  tenantReady: false,
  podUnreachable: false,
  isOwner: false,
  error: null,
  statusMessage: null,
  availableTenants: [],
  switchTenant: () => {},
  planType: null,
  hasAccess: true,
  requestCreateWorkspace: () => {},
  releaseLoader: () => {},
});

export function useTenant() {
  const context = useContext(TenantContext);
  return {
    tenant: context.tenant,
    isInitializing: context.isInitializing,
    tenantReady: context.tenantReady,
    podUnreachable: context.podUnreachable,
    isOwner: context.isOwner,
    error: context.error,
    availableTenants: context.availableTenants,
    switchTenant: context.switchTenant,
    planType: context.planType,
    hasAccess: context.hasAccess,
    requestCreateWorkspace: context.requestCreateWorkspace,
    releaseLoader: context.releaseLoader,
  };
}

export function useCogniInstance() {
  const context = useContext(TenantContext);
  return {
    cogniInstance: context.cogniInstance,
    localInstance: context.localInstance,
    serviceUrl: context.serviceUrl,
    isInitializing: context.isInitializing,
    error: context.error,
    statusMessage: context.statusMessage,
  };
}
