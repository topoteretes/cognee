"use client";

import { useEffect, useMemo, useState } from "react";
import { Tenant } from "./types";
import { TenantContext, localInstance } from "./TenantContext";
import { tokens } from "@/ui/theme/tokens";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";
import { UserContext, type AvailableTenant } from "@/modules/users/UserContext";

// Single-user local mode: the only user is always the owner of the only
// workspace. `useTenant()` derives isOwner by looking the active tenant up in
// UserContext's availableTenants, and the cloud UserProvider that would fill
// that list is not mounted here — so without this, isOwner is permanently
// false and every owner-gated surface (the Connect buttons on the Data
// Sources cards, for one) renders inert.
const LOCAL_TENANT_ID = "local";
const LOCAL_AVAILABLE_TENANTS: AvailableTenant[] = [
  { id: LOCAL_TENANT_ID, name: LOCAL_TENANT_ID, isOwner: true, ownerHasSubscription: false },
];

export function LocalProvider({ children }: { children: React.ReactNode }) {
  const localApiUrl = getLocalApiUrl();
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      availableTenants: LOCAL_AVAILABLE_TENANTS,
      isLoadingTenants: false,
      isTenantsError: false,
      refetchTenants: () => {},
      setAvailableTenantsOptimistic: () => {},
    }),
    [],
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
        setTenant({ tenant_id: LOCAL_TENANT_ID, tenant_name: LOCAL_TENANT_ID });
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
  }, []);

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
        switchTenant: () => {},
        planType: null,
        hasAccess: true,
        requestCreateWorkspace: () => {},
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
