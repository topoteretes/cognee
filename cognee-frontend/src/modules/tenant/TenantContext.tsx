"use client";

import { createContext, useContext } from "react";
import { CogneeInstance } from "@/modules/instances/types";
import { Tenant } from "./types";
import localFetch from "@/modules/instances/localFetch";

export type PlanType = string | null;

export interface AvailableTenant {
  id: string;
  name: string;
}

export interface TenantContextValue {
  tenant: Tenant | null;
  cogniInstance: CogneeInstance | null;
  localInstance: CogneeInstance;
  serviceUrl: string | null;
  isInitializing: boolean;
  error: string | null;
  statusMessage: { title: string; subtitle: string } | null;
  availableTenants: AvailableTenant[];
  switchTenant: (tenantId: string) => void;
  planType: PlanType;
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
  error: null,
  statusMessage: null,
  availableTenants: [],
  switchTenant: () => {},
  planType: null,
});

export function useTenant() {
  const context = useContext(TenantContext);
  return {
    tenant: context.tenant,
    isInitializing: context.isInitializing,
    error: context.error,
    availableTenants: context.availableTenants,
    switchTenant: context.switchTenant,
    planType: context.planType,
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
