import { CogneeInstance } from "../instances/types";

export interface SchemaRelationship {
  to_type: string;
  relation: string;
  count: number;
}

/** One semantic type in the data-derived schema inventory (from PR #2939). */
export interface SchemaTypeEntry {
  type: string;
  count: number;
  samples: string[];
  sample_size: number;
  relationships: SchemaRelationship[];
}

/**
 * Fetch the data-derived schema inventory for a dataset — what the knowledge
 * graph actually contains, summarized by semantic type. Backed by the cognee
 * `/v1/schema/inventory` endpoint (get_schema_inventory SDK + schema router).
 */
export default function getSchemaInventory(
  instance: CogneeInstance,
  datasetId: string,
  samplesPerType = 5,
): Promise<SchemaTypeEntry[]> {
  const path = `/v1/schema/inventory?dataset_id=${encodeURIComponent(datasetId)}&samples_per_type=${samplesPerType}`;
  console.log("[getSchemaInventory] fetching", path, "instance:", instance);
  return instance
    .fetch(path, { method: "GET", headers: { "Content-Type": "application/json" } })
    .then(async (response) => {
      console.log("[getSchemaInventory] response status:", response.status, response.url);
      if (!response.ok) {
        const body = await response.text().catch(() => "(unreadable)");
        console.error("[getSchemaInventory] non-ok response body:", body);
        throw new Error(`Schema inventory returned ${response.status}: ${body}`);
      }
      const json = await response.json();
      console.log("[getSchemaInventory] parsed JSON:", json);
      return json;
    });
}
