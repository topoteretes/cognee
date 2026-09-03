import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

/**
 * Runs once when the server starts.
 *
 * Surfaces a broken COGNEE_BACKEND_URL in the first lines of the log rather
 * than only in the 500 of whichever request happens to arrive first. The
 * published image additionally refuses to start at all (see
 * docker-entrypoint.sh); this covers the plain `next start` and
 * `cognee-cli -ui` paths, where exiting the process is not ours to decide.
 */
export async function register() {
  try {
    getServerBackendUrl();
  } catch (error) {
    console.error(`[cognee-ui] ${(error as Error).message}`);
  }
}
