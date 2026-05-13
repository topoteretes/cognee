import type { GraphSchema, EntitySchema, FieldSchema, EnumField, RelationField } from "./types";

export type ValidationSeverity = "error" | "warn";

export type ValidationIssue = {
  severity: ValidationSeverity;
  path: string; // human-readable dot path, e.g. "Entity.field_name"
  entityId?: string;
  fieldId?: string;
  message: string;
};

const PASCAL_CASE = /^[A-Z][a-zA-Z0-9]*$/;
const SNAKE_CASE = /^[a-z][a-z0-9_]*$/;

export function validateSchema(schema: GraphSchema): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const entityNames = new Set<string>();

  for (const entity of schema.entities) {
    // ── Entity-level ────────────────────────────────────────────────────────
    if (!entity.name) {
      issues.push({
        severity: "error",
        path: `(unnamed entity)`,
        entityId: entity._id,
        message: "Entity name must not be empty.",
      });
    } else {
      if (!PASCAL_CASE.test(entity.name)) {
        issues.push({
          severity: "error",
          path: entity.name,
          entityId: entity._id,
          message: `Entity name "${entity.name}" must be PascalCase.`,
        });
      }
      if (entityNames.has(entity.name)) {
        issues.push({
          severity: "error",
          path: entity.name,
          entityId: entity._id,
          message: `Duplicate entity name "${entity.name}".`,
        });
      }
      entityNames.add(entity.name);
    }

    // Warn if no primary label and no "name" field
    const primitiveFieldNames = entity.fields
      .filter((f) => f.kind === "primitive")
      .map((f) => f.name);

    if (!entity.primaryLabelField && !primitiveFieldNames.includes("name")) {
      issues.push({
        severity: "warn",
        path: entity.name || "(unnamed)",
        entityId: entity._id,
        message: `Entity "${entity.name || "(unnamed)"}" has no primary label and no "name" field.`,
      });
    }

    // indexFields must be subset of primitive fields
    const indexFields = entity.indexFields ?? [];
    for (const idxField of indexFields) {
      if (!primitiveFieldNames.includes(idxField)) {
        issues.push({
          severity: "error",
          path: `${entity.name}.indexFields`,
          entityId: entity._id,
          message: `Index field "${idxField}" is not a primitive field of "${entity.name}".`,
        });
      }
    }

    // ── Field-level ──────────────────────────────────────────────────────────
    const fieldNames = new Set<string>();

    for (const field of entity.fields) {
      const fPath = `${entity.name || "(unnamed)"}.${field.name || "(unnamed field)"}`;

      if (!field.name) {
        issues.push({
          severity: "error",
          path: fPath,
          entityId: entity._id,
          fieldId: field._id,
          message: "Field name must not be empty.",
        });
      } else {
        if (!SNAKE_CASE.test(field.name)) {
          issues.push({
            severity: "error",
            path: fPath,
            entityId: entity._id,
            fieldId: field._id,
            message: `Field name "${field.name}" must be snake_case.`,
          });
        }
        if (fieldNames.has(field.name)) {
          issues.push({
            severity: "error",
            path: fPath,
            entityId: entity._id,
            fieldId: field._id,
            message: `Duplicate field name "${field.name}" in entity "${entity.name}".`,
          });
        }
        fieldNames.add(field.name);
      }

      if (field.kind === "enum") {
        const enumField = field as EnumField;
        if (!enumField.enumValues || enumField.enumValues.length === 0) {
          issues.push({
            severity: "error",
            path: fPath,
            entityId: entity._id,
            fieldId: field._id,
            message: `Enum field "${field.name}" must have at least one value.`,
          });
        }
      }

      if (field.kind === "relation") {
        const relField = field as RelationField;
        if (!relField.relation.targetEntityName) {
          issues.push({
            severity: "error",
            path: fPath,
            entityId: entity._id,
            fieldId: field._id,
            message: `Relation field "${field.name}" must have a target entity.`,
          });
        }
      }
    }
  }

  // ── Cross-entity: warn on missing relation targets ─────────────────────────
  for (const entity of schema.entities) {
    for (const field of entity.fields) {
      if (field.kind === "relation") {
        const relField = field as RelationField;
        const target = relField.relation.targetEntityName;
        if (target && !entityNames.has(target)) {
          issues.push({
            severity: "warn",
            path: `${entity.name}.${field.name}`,
            entityId: entity._id,
            fieldId: field._id,
            message: `Relation target entity "${target}" does not exist yet.`,
          });
        }
      }
    }
  }

  return issues;
}

export function issuesForEntity(
  issues: ValidationIssue[],
  entityId: string
): ValidationIssue[] {
  return issues.filter((i) => i.entityId === entityId);
}

export function issueCountForEntity(
  issues: ValidationIssue[],
  entityId: string
): { errors: number; warnings: number } {
  const entityIssues = issuesForEntity(issues, entityId);
  return {
    errors: entityIssues.filter((i) => i.severity === "error").length,
    warnings: entityIssues.filter((i) => i.severity === "warn").length,
  };
}
