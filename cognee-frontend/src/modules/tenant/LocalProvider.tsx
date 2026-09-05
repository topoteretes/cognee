"use client";

import { useEffect, useMemo, useState } from "react";
import { Tenant } from "./types";
import { TenantContext, localInstance } from "./TenantContext";
import { tokens } from "@/ui/theme/tokens";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";
import { UserContext, type AvailableTenant } from "@/modules/users/UserContext";
import { request, type WorkspaceContext } from "@/modules/workspace/api";

// Local servers can have several users and teams. Ownership must come from
// the authenticated SDK account, just as it does for a hosted workspace.

export function LocalProvider({ children }: { children: React.ReactNode }) {
  const localApiUrl = getLocalApiUrl();
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [workspace, setWorkspace] = useState<WorkspaceContext | null>(null);

  // Memoized: consumers key effects off availableTenants, so a fresh array on
  // every render would re-fire them on unrelated re-renders.
  const userContextValue = useMemo(
    () => ({
      userMe: null,
      isLoading: false,
      isUserMeError: false,
      markWelcomeSeen: async () => {},
      markOnboardingComplete: async () => {},
      dismissFeatureAnnouncement: async () => {},
      availableTenants: (workspace?.teams ?? []).map((team): AvailableTenant => ({
        id: team.id, name: team.name, isOwner: team.is_owner, ownerHasSubscription: false,
      })),
      isLoadingTenants: false,
      isTenantsError: false,
      refetchTenants: () => {},
      setAvailableTenantsOptimistic: () => {},
    }),
    [workspace],
  );

  useEffect(() => {
    let cancelled = false;

    async function init() {
      // Guard: don't check auth if we're already on the login page (avoids redirect loop)
      if (typeof window !== "undefined" && window.location.pathname === "/local-login") {
        setIsInitializing(false);
        return;
      }

      try {
        // Check if we're authenticated with the local backend
        const meResponse = await global.fetch(`${localApiUrl}/api/v1/users/me`, {
          credentials: "include",
        });

        if (meResponse.status === 401 || meResponse.status === 403) {
          // Not authenticated — redirect to local login
          window.location.href = "/local-login";
          return;
        }

        if (!meResponse.ok) {
          throw new Error(`Local backend returned ${meResponse.status}: ${meResponse.statusText}`);
        }

        if (cancelled) return;

        // Authenticated — set up the local instance
        const current = await request<WorkspaceContext>("/v1/workspace/context");
        if (cancelled) return;
        setWorkspace(current);
        const selected = current.teams.find((team) => team.id === current.user.tenant_id);
        setTenant({ tenant_id: selected?.id ?? "", tenant_name: selected?.name ?? "Personal" });
      } catch (err) {
        if (cancelled) return;

        // Network error — backend probably not running
        if (err instanceof TypeError) {
          setError("Cannot connect to local Cognee backend at " + localApiUrl + ". Is it running?");
        } else {
          const message = err instanceof Error ? err.message : "Failed to connect to local backend";
          setError(message);
        }
      } finally {
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    }

    init();

    return () => {
      cancelled = true;
    };
  }, [localApiUrl]);

  if (error && !isInitializing) {
    return (
      <ErrorScreen message={error} />
    );
  }

  return (
    // UserContext sits above TenantContext because useTenant() reads
    // availableTenants out of it to derive isOwner/isPersonal. The tenant list
    // is the only field carrying real information here; the rest stay inert,
    // since local mode has no cloud user service to load them from.
    <UserContext.Provider value={userContextValue}>
      <TenantContext.Provider value={{
        tenant,
        cogniInstance: localInstance,
        localInstance,
        serviceUrl: localApiUrl,
        apiKey: "",
        isInitializing,
        tenantReady: true,
        podUnreachable: false,
        error,
        statusMessage: null,
        switchTenant: async (tenantId) => {
          try {
            await request("/v1/permissions/tenants/select", "POST", { tenant_id: tenantId || null });
            window.location.reload();
          } catch (error) { setError(error instanceof Error ? error.message : "Could not switch team"); }
        },
        planType: null,
        hasAccess: true,
        requestCreateWorkspace: () => { window.location.href = "/workspace"; },
        nameModalOpen: false,
        releaseLoader: () => {},
      }}>
        {children}
      </TenantContext.Provider>
    </UserContext.Provider>
  );
}

function ErrorScreen({ message }: { message: string }) {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      padding: "2rem",
      textAlign: "center",
    }}>
      <div style={{
        backgroundColor: "#ffffff",
        borderRadius: "0.75rem",
        padding: "2.5rem",
        maxWidth: "28rem",
        width: "100%",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
      }}>
        <h2 style={{ margin: "0 0 0.75rem", fontSize: "1.25rem", fontWeight: 700, color: tokens.textDark }}>
          Connection Error
        </h2>
        <p style={{ margin: "0 0 1.5rem", fontSize: "0.875rem", color: tokens.textSecondary }}>
          {message}
        </p>
        <button
          onClick={() => window.location.reload()}
          style={{
            padding: "0.5rem 1.5rem",
            borderRadius: "0.5rem",
            border: "1px solid #d1d5db",
            backgroundColor: "#ffffff",
            cursor: "pointer",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
        >
          Try again
        </button>
      </div>
    </div>
  );
}
