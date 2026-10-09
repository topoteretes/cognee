"use client";

import { ErrorBoundary } from "react-error-boundary";
import "./business.css";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import BusinessView from "./BusinessView";
import BusinessErrorFallback from "@/modules/business/panels/BusinessErrorFallback";
import { captureException } from "@/utils/monitoring";

export default function BusinessPage() {
  const { cogniInstance } = useCogniInstance();
  // CustomAppShell already renders the pod-provisioning state for this route
  // (POD_DEPENDENT_PATHS includes "/business") before this ever mounts.
  if (!cogniInstance) return null;
  return (
    <ErrorBoundary
      FallbackComponent={BusinessErrorFallback}
      onError={(error, info) => captureException(error, { componentStack: info.componentStack })}
    >
      <BusinessView cogniInstance={cogniInstance} />
    </ErrorBoundary>
  );
}
