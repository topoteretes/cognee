import { collectRuntimeConfig } from "@/modules/config/serverRuntimeConfig";

/**
 * Diagnostics endpoint: reports the backend this container resolved.
 *
 * The app does not depend on it. Pages read their config from the inert JSON
 * the root layout renders into <head>. This route exists so an operator can
 * answer "what backend does this container think it has?" with a single curl,
 * and so the image's HEALTHCHECK has something cheap to probe that also proves
 * the configuration resolved rather than merely that a port is open.
 */
export const dynamic = "force-dynamic";

export function GET() {
  return Response.json(collectRuntimeConfig(), {
    headers: { "Cache-Control": "no-store" },
  });
}
