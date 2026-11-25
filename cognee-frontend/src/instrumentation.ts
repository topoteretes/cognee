import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

// Docker validates in its entrypoint; this also reports bad configuration
// when launched through next start or cognee-cli -ui.
export async function register(): Promise<void> {
  try {
    getServerBackendUrl();
  } catch (error) {
    console.error("[cognee-ui] Invalid backend configuration:", error);
  }
}
