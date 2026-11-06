import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function register() {
  try {
    getServerBackendUrl();
  } catch (error) {
    console.error(`[cognee-ui] ${(error as Error).message}`);
  }
}
