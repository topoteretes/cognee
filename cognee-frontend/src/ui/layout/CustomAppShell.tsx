"use client";

import { PropsWithChildren } from "react";
import { usePathname } from "next/navigation";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import TopBar from "./TopBar";
import CustomAppShellNavbar from "./Navbar/CustomAppShellNavbar";
import PageLoading from "@/ui/elements/PageLoading";
import PodUnreachableCard from "@/ui/elements/PodUnreachableCard";
import ProvisioningBanner from "./ProvisioningBanner";
import WorkspaceProvisioning from "./WorkspaceProvisioning";

const SHELL_HIDDEN_PATHS = [
  "/account",
  "/plan",
  "/setup",
  "/onboarding",
  "/welcome",
  "/sign-in",
  "/sign-up",
  "/forgot-password",
];

// Pages that talk to the tenant pod — they must wait for the pod to be ready
// (and the API client to exist) before rendering, otherwise they hang or show
// empty/stale data. Dashboard is excluded (it has its own skeleton); API Keys is
// excluded (it shows the real tenant URL early with a "warming up" badge).
export const POD_DEPENDENT_PATHS = [
  "/search",
  "/datasets",
  "/skills",
  "/knowledge-graph",
  "/schema",
  "/sessions",
];

function isPodDependent(pathname: string): boolean {
  return POD_DEPENDENT_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export default function CustomAppShell({ children }: PropsWithChildren) {
  const pathname = usePathname();
  const hideShell = SHELL_HIDDEN_PATHS.includes(pathname);
  const { cogniInstance, statusMessage, isInitializing } = useCogniInstance();
  const { tenantReady, podUnreachable } = useTenant();

  if (hideShell) {
    return <>{children}</>;
  }

  // Pod-dependent routes can't render real content until the pod is reachable.
  // Split the two states: podUnreachable is terminal (readiness polling gave
  // up) and gets the error card with Try again / Sign out; podPending is still
  // connecting and gets the provisioning spinner. Without this split, a dead
  // pod left every non-dashboard pod-dependent route stuck on an eternal
  // "Setting up your workspace" spinner with no error and no way out.
  const onPodDependent = isPodDependent(pathname);
  const podUnreachableHere = onPodDependent && podUnreachable;
  const podPending = onPodDependent && !podUnreachable && (!tenantReady || !cogniInstance);
  const isDashboard = pathname === "/dashboard";

  return (
    <div
      className="flex flex-col h-screen"
      style={{
        backgroundColor: "#000000",
        backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
      }}
    >
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        <CustomAppShellNavbar />
        <main className="flex-1 overflow-auto flex flex-col" style={{ background: "transparent" }}>
          {/* App-wide provisioning banner — shown only while still connecting.
              Suppressed once podUnreachable is terminal, otherwise the
              "Setting up your workspace" banner contradicted the "trouble
              reaching your workspace" error card by rendering alongside it. */}
          {!isInitializing && !tenantReady && !podUnreachable && <ProvisioningBanner />}
          {isInitializing && !isDashboard ? (
            <PageLoading name={statusMessage?.title ?? ""} />
          ) : podUnreachableHere ? (
            <PodUnreachableCard pageName="Workspace" />
          ) : podPending ? (
            <WorkspaceProvisioning />
          ) : children}
        </main>
      </div>
    </div>
  );
}
