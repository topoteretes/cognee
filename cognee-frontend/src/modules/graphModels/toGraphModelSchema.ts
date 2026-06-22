import type { CleanEntitySchema, CleanGraphSchema } from "./types";

type JsonObject = Record<string, unknown>;

const PRIMITIVE_TYPE_MAP: Record<string, string> = {
  string: "string",
  number: "number",
  boolean: "boolean",
  date: "string", // ISO date string
};

/**
 * Builds the `{EntityName}Type` definition — the Cognify type-marker object
 * that every entity instance must carry as its `is_type` field.
 *
 * Mirrors the Pydantic pattern:
 *   class PersonType(BaseModel):
 *     name: str = "Person"
 *     metadata: dict = {"index_fields": ["name"]}
 */
function buildEntityTypeDef(entity: CleanEntitySchema): JsonObject {
  const indexFields =
    entity.indexFields && entity.indexFields.length > 0
      ? entity.indexFields
      : ["name"];
  return {
    properties: {
      name: { default: entity.name, type: "string" },
      metadata: {
        additionalProperties: true,
        default: { index_fields: indexFields },
        type: "object",
      },
    },
    title: `${entity.name}Type`,
    type: "object",
  };
}

/**
 * Builds the JSON Schema body for one entity (the `properties`, `required`,
 * `title`, `type` block) — usable both as the top-level root schema and as an
 * entry inside `$defs`.
 *
 * The output mirrors `EntityClass.model_json_schema()` minus the nested
 * `$defs` (all defs live at the root level of the combined schema).
 */
function buildEntitySchema(entity: CleanEntitySchema): JsonObject {
  const indexFields =
    entity.indexFields && entity.indexFields.length > 0
      ? entity.indexFields
      : ["name"];

  const properties: Record<string, unknown> = {
    // `name` is the primary node identifier — always present
    name: { type: "string" },
    // `is_type` carries the entity type marker (required by Cognify)
    is_type: { $ref: `#/$defs/${entity.name}Type` },
    // `metadata` tells Cognify which fields to index
    metadata: {
      additionalProperties: true,
      default: { index_fields: indexFields },
      type: "object",
    },
  };

  const required: string[] = ["name", "is_type"];

  for (const field of entity.fields) {
    if (field.name === "name") continue; // already added above

    if (field.kind === "primitive") {
      properties[field.name] = {
        type: PRIMITIVE_TYPE_MAP[field.primitiveType] ?? "string",
      };
      if (field.required) required.push(field.name);
    } else if (field.kind === "enum") {
      properties[field.name] = { enum: field.enumValues, type: "string" };
      if (field.required) required.push(field.name);
    } else if (field.kind === "relation") {
      const target = field.relation.targetEntityName;
      properties[field.name] =
        field.relation.cardinality === "many"
          ? { default: [], items: { $ref: `#/$defs/${target}` }, type: "array" }
          : { $ref: `#/$defs/${target}` };
    }
  }

  return { properties, required, title: entity.name, type: "object" };
}

/**
 * Converts our internal `CleanGraphSchema` to the JSON Schema format that the
 * Cognify `/cognify` endpoint expects for the `graphModel` field.
 *
 * This mirrors `SomeModel.model_json_schema()` from Pydantic:
 *
 * - The **first entity** in the schema becomes the top-level (root) object.
 * - **All** entities and their `*Type` definitions are placed in `$defs` so
 *   Cognify is aware of every node type in the graph, including those only
 *   reachable via relations.
 * - Self-referential relations (e.g. `Concept → related_to → Concept`) are
 *   handled safely via `$ref` pointers with no JS recursion.
 *
 * Example output structure (mirrors the ProgrammingLanguage test template):
 * {
 *   "$defs": {
 *     "PersonType": { ... },
 *     "Organization": { ... },
 *     "OrganizationType": { ... },
 *     ...
 *   },
 *   "properties": { "name": ..., "is_type": ..., "metadata": ..., ... },
 *   "required": ["name", "is_type"],
 *   "title": "Person",
 *   "type": "object"
 * }
 */
export function toGraphModelSchema(schema: CleanGraphSchema): object {
  if (schema.entities.length === 0) return {};

  const defs: Record<string, JsonObject> = {};

  // Register every entity and its type definition in $defs upfront.
  // This handles self-referential relations without recursion — the $ref
  // pointer is resolved by JSON Schema consumers, not by our JS code.
  for (const entity of schema.entities) {
    defs[`${entity.name}Type`] = buildEntityTypeDef(entity);
    defs[entity.name] = buildEntitySchema(entity);
  }

  // The first entity is the root — its schema is spread at the top level.
  const root = schema.entities[0];

  return {
    $defs: defs,
    ...buildEntitySchema(root),
  };
}
