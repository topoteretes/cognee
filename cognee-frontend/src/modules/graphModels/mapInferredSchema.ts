import { v4 as uuid } from "uuid";
import type { FieldSchema, GraphSchema, PrimitiveType } from "@/modules/graphModels/types";

// Shape of a single JSON-Schema property as produced by the schema-inference
// LLM ($ref for a single relation, array+items.$ref for a many relation,
// otherwise a primitive `type`).
interface JsonSchemaProperty {
  $ref?: string;
  type?: string;
  items?: { $ref?: string };
}

interface JsonSchemaDef {
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
}

// Converts the JSON Schema returned by inferSchema (a `$defs` map of entity
// definitions) into our internal GraphSchema. Entity defs whose key ends in
// "Type" are enum wrappers and skipped; the "is_type"/"metadata" bookkeeping
// properties are dropped.
export default function mapInferredSchema(graphSchema: Record<string, unknown>): GraphSchema {
  const defs = (graphSchema.$defs as Record<string, JsonSchemaDef> | undefined) ?? {};

  return {
    options: {},
    entities: Object.entries(defs)
      .filter(([key]) => !key.endsWith("Type"))
      .map(([name, def]) => ({
        _id: uuid(),
        name: def.title || name,
        description: def.description || "",
        fields: Object.entries(def.properties ?? {})
          .filter(([fieldName]) => fieldName !== "is_type" && fieldName !== "metadata")
          .map(([fieldName, fieldDef]): FieldSchema => {
            if (fieldDef.$ref) {
              const target = fieldDef.$ref.replace("#/$defs/", "");
              return { _id: uuid(), name: fieldName, kind: "relation", relation: { targetEntityName: target, cardinality: "one" } };
            }
            if (fieldDef.type === "array" && fieldDef.items?.$ref) {
              const target = fieldDef.items.$ref.replace("#/$defs/", "");
              return { _id: uuid(), name: fieldName, kind: "relation", relation: { targetEntityName: target, cardinality: "many" } };
            }
            const primitiveType: PrimitiveType =
              fieldDef.type === "number" || fieldDef.type === "integer"
                ? "number"
                : fieldDef.type === "boolean"
                  ? "boolean"
                  : "string";
            return { _id: uuid(), name: fieldName, kind: "primitive", primitiveType, required: (def.required ?? []).includes(fieldName) };
          }),
        indexFields: [],
      })),
  };
}
