// ── Canonical schema types ──────────────────────────────────────────────────

export type PrimitiveType = "string" | "number" | "boolean" | "date";

export type PrimitiveField = {
  _id: string; // UI-only, stripped on export
  name: string;
  kind: "primitive";
  primitiveType: PrimitiveType;
  required?: boolean;
  description?: string;
};

export type EnumField = {
  _id: string;
  name: string;
  kind: "enum";
  enumValues: string[];
  required?: boolean;
  description?: string;
};

export type RelationField = {
  _id: string;
  name: string;
  kind: "relation";
  relation: {
    targetEntityName: string;
    cardinality: "one" | "many";
    inverse?: {
      enabled: boolean;
      name?: string;
      cardinality?: "one" | "many";
    };
  };
  required?: boolean;
  description?: string;
};

export type FieldSchema = PrimitiveField | EnumField | RelationField;

export type EntitySchema = {
  _id: string; // UI-only stable identity, stripped on export
  name: string;
  description?: string;
  primaryLabelField?: string;
  indexFields?: string[];
  fields: FieldSchema[];
};

export type GraphSchema = {
  options: { autoTypeNodes?: boolean };
  entities: EntitySchema[];
};

// ── Persisted model record ────────────────────────────────────────────────────

export type GraphModel = {
  id: string;
  name: string;
  schema: GraphSchema;
  createdAt: string;
  updatedAt: string;
  status: "draft" | "published";
};

// ── Clean export types (no _id) ───────────────────────────────────────────────

export type CleanFieldSchema =
  | {
      name: string;
      kind: "primitive";
      primitiveType: PrimitiveType;
      required?: boolean;
      description?: string;
    }
  | {
      name: string;
      kind: "enum";
      enumValues: string[];
      required?: boolean;
      description?: string;
    }
  | {
      name: string;
      kind: "relation";
      relation: {
        targetEntityName: string;
        cardinality: "one" | "many";
        inverse?: {
          enabled: boolean;
          name?: string;
          cardinality?: "one" | "many";
        };
      };
      required?: boolean;
      description?: string;
    };

export type CleanEntitySchema = {
  name: string;
  description?: string;
  primaryLabelField?: string;
  indexFields?: string[];
  fields: CleanFieldSchema[];
};

export type CleanGraphSchema = {
  options: { autoTypeNodes?: boolean };
  entities: CleanEntitySchema[];
};

// ── Field input type (distributed Omit so union members keep their properties) ─

export type FieldInput =
  | Omit<PrimitiveField, "_id">
  | Omit<EnumField, "_id">
  | Omit<RelationField, "_id">;

// ── Helpers ────────────────────────────────────────────────────────────────────

export function toCleanSchema(schema: GraphSchema): CleanGraphSchema {
  return {
    options: { ...schema.options },
    entities: schema.entities.map((e) => ({
      name: e.name,
      ...(e.description ? { description: e.description } : {}),
      ...(e.primaryLabelField ? { primaryLabelField: e.primaryLabelField } : {}),
      ...(e.indexFields?.length ? { indexFields: e.indexFields } : {}),
      fields: e.fields.map(({ _id: _unused, ...rest }) => rest as CleanFieldSchema),
    })),
  };
}

export function fieldTypeLabel(field: FieldSchema): string {
  if (field.kind === "primitive") return field.primitiveType;
  if (field.kind === "enum") return "enum";
  return field.relation.cardinality === "many"
    ? `→ ${field.relation.targetEntityName} (many)`
    : `→ ${field.relation.targetEntityName}`;
}
