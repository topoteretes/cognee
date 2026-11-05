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
import WorkspaceWaking from "./WorkspaceWaking";
import { NEW_ONBOARDING_PATH } from "@/utils/onboardingPaths";

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
  "/sessions",
  "/business",
  // `useDashboardTelemetry`'s query is `enabled: tenantReady && ...`, so on a
  // suspended or unreachable workspace it never runs: `isPending` stays true
  // and `isError` stays false forever, and the page rendered nothing but
  // `AnalyticsSkeleton` with no error and no way out. Gating it here the same
  // way as the other pod-dependent routes means the shell shows
  // `PodUnreachableCard`/`WorkspaceWaking` before the page (and its query) ever
  // mounts, instead of the page having to duplicate that guard itself.
  "/analytics",
  // Loads its model straight off the pod, so refreshing it on a sleeping
  // workspace failed the load while the wake was still finishing. Gating it
  // here keeps the editor from mounting, and loading, until the pod answers.
  "/graph-models",
  // Its whole content is two pod endpoints. Left out, a dead pod left the
  // route on a permanent skeleton: the query is disabled while tenantReady is
  // false, a disabled react-query stays `isPending` forever, so `loading` never
  // cleared and no error ever surfaced — no banner, no PodUnreachableCard, no
  // way out. Exactly what the split below exists to prevent.
  "/activity",
];

function isPodDependent(pathname: string): boolean {
  return POD_DEPENDENT_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export default function CustomAppShell({ children }: PropsWithChildren) {
  const pathname = usePathname();
  const hideShell = SHELL_HIDDEN_PATHS.includes(pathname);
  const { cogniInstance, statusMessage, isInitializing } = useCogniInstance();
  const { tenantReady, podUnreachable, wakeState, wakeStartedAt, currentName } = useTenant();

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
  // The new onboarding window renders the same dashboard tree, so it brings the
  // dashboard's skeleton and waking screen too.
  const isDashboard = pathname === "/dashboard" || pathname === NEW_ONBOARDING_PATH;
  // `/` re-exports OverviewPage, so it renders the dashboard tree and brings
  // its own waking screen with it.
  const rendersDashboard = isDashboard || pathname === "/";
  // Only the routes that actually put a waking screen on screen may drop the
  // banner. Everywhere else it is the sole indication anything is happening,
  // and suppressing it there left eleven routes silently waiting out a wake.
  const showsWakingScreen = wakeState === "waking" && (podPending || rendersDashboard);

  return (
    <div
      className="flex flex-col h-screen"
      style={{
        // Lighter and finer than the auth/onboarding pages' pure-black 33px
        // grid, which stays as is: their Tetris blocks are aligned to it.
        backgroundColor: "#111111",
        backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "16px 16px",
      }}
    >
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        <CustomAppShellNavbar />
        {/* scrollbar-gutter:stable reserves the scrollbar's width even when no
            scrollbar is showing, so pages whose height crosses the scroll
            threshold (e.g. expanding a panel) don't reflow every element's
            width when the classic scrollbar appears/disappears. */}
        <main className="flex-1 overflow-auto flex flex-col [scrollbar-gutter:stable]" style={{ background: "transparent" }}>
          {/* App-wide provisioning banner — shown only while still connecting.
              Suppressed once podUnreachable is terminal, otherwise the
              "Setting up your workspace" banner contradicted the "trouble
              reaching your workspace" error card by rendering alongside it. */}
          {/* Suppressed only where WorkspaceWaking is on screen saying the same
              thing, since two live regions with one message talk over each
              other. On every other route the banner is the whole signal, so it
              stays and switches to the wake wording instead. */}
          {!isInitializing && !tenantReady && !podUnreachable && !showsWakingScreen && (
            <ProvisioningBanner waking={wakeState === "waking"} />
          )}
          {isInitializing && !isDashboard ? (
            <PageLoading name={statusMessage?.title ?? ""} />
          ) : podUnreachableHere ? (
            <PodUnreachableCard pageName="Workspace" wakeFailed={wakeState === "failed"} workspaceName={currentName} />
          ) : podPending && wakeState === "waking" ? (
            <WorkspaceWaking workspaceName={currentName} startedAt={wakeStartedAt} />
          ) : podPending ? (
            <WorkspaceProvisioning />
          ) : children}
        </main>
      </div>
    </div>
  );
}
