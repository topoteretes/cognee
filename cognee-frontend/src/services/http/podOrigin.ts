/**
 * Tells the tenant pod that a request came from this dashboard.
 *
 * The pod stamps the value onto every `pipeline_runs` row it writes for the
 * request (see `_stamp_operation_origin` in cognee-saas-pod/api/server.py),
 * which is what fills the Activity log's Via column. Without it an upload done
 * here is indistinguishable from an agent calling the same HTTP API with the
 * same kind of key, and the row reads "—".
 *
 * Both pod request paths have to send it — the shared fetch client and the XHR
 * upload path — so the name and value live here rather than being typed twice.
 * The pod allowlists the value and treats it as telemetry, never as an
 * authorization input.
 */
export const POD_ORIGIN_HEADER = "X-Cognee-Origin";
export const POD_ORIGIN_CLOUD = "cloud";
