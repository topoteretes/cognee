import { RUNTIME_CONFIG_ELEMENT_ID, serializeRuntimeConfig } from "./runtimeConfig";
import { collectRuntimeConfig } from "./serverRuntimeConfig";

/**
 * Renders the runtime config into <head> as inert JSON.
 *
 * Rendered from the root layout, which is marked force-dynamic so this is
 * evaluated per request. Prerendering it would freeze the value into the
 * build, which is precisely the problem this exists to solve.
 */
export default function RuntimeConfigScript() {
  const config = serializeRuntimeConfig(collectRuntimeConfig());

  return (
    <script
      id={RUNTIME_CONFIG_ELEMENT_ID}
      type="application/json"
      dangerouslySetInnerHTML={{ __html: config }}
    />
  );
}
