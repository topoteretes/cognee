import { v4 as uuid } from "uuid";
import type { EntitySchema, FieldInput, FieldSchema, GraphSchema } from "./types";

// ── Action types ───────────────────────────────────────────────────────────────

export type SchemaAction =
  | { type: "SET_SCHEMA"; schema: GraphSchema }
  | { type: "ADD_ENTITY"; name: string; description?: string }
  | {
      type: "UPDATE_ENTITY";
      entityId: string;
      updates: Partial<Pick<EntitySchema, "name" | "description" | "primaryLabelField" | "indexFields">>;
    }
  | { type: "DELETE_ENTITY"; entityId: string }
  | { type: "DUPLICATE_ENTITY"; entityId: string }
  | { type: "ADD_FIELD"; entityId: string; field: FieldInput }
  | { type: "UPDATE_FIELD"; entityId: string; fieldId: string; field: FieldInput }
  | { type: "DELETE_FIELD"; entityId: string; fieldId: string }
  | { type: "DUPLICATE_FIELD"; entityId: string; fieldId: string }
  | { type: "SET_OPTION"; key: "autoTypeNodes"; value: boolean };

// ── Reducer ────────────────────────────────────────────────────────────────────

export function schemaReducer(state: GraphSchema, action: SchemaAction): GraphSchema {
  switch (action.type) {
    case "SET_SCHEMA":
      return action.schema;

    case "ADD_ENTITY": {
      const newEntity: EntitySchema = {
        _id: uuid(),
        name: action.name,
        description: action.description,
        fields: [],
        indexFields: [],
      };
      return { ...state, entities: [...state.entities, newEntity] };
    }

    case "UPDATE_ENTITY":
      return {
        ...state,
        entities: state.entities.map((e) =>
          e._id === action.entityId ? { ...e, ...action.updates } : e
        ),
      };

    case "DELETE_ENTITY":
      return {
        ...state,
        entities: state.entities.filter((e) => e._id !== action.entityId),
      };

    case "DUPLICATE_ENTITY": {
      const src = state.entities.find((e) => e._id === action.entityId);
      if (!src) return state;
      const copy: EntitySchema = {
        ...src,
        _id: uuid(),
        name: `${src.name}Copy`,
        fields: src.fields.map((f) => ({ ...f, _id: uuid() })),
      };
      return { ...state, entities: [...state.entities, copy] };
    }

    case "ADD_FIELD":
      return {
        ...state,
        entities: state.entities.map((e) =>
          e._id === action.entityId
            ? { ...e, fields: [...e.fields, { ...action.field, _id: uuid() } as FieldSchema] }
            : e
        ),
      };

    case "UPDATE_FIELD":
      return {
        ...state,
        entities: state.entities.map((e) =>
          e._id === action.entityId
            ? {
                ...e,
                fields: e.fields.map((f) =>
                  f._id === action.fieldId
                    ? ({ ...action.field, _id: f._id } as FieldSchema)
                    : f
                ),
              }
            : e
        ),
      };

    case "DELETE_FIELD":
      return {
        ...state,
        entities: state.entities.map((e) =>
          e._id === action.entityId
            ? { ...e, fields: e.fields.filter((f) => f._id !== action.fieldId) }
            : e
        ),
      };

    case "DUPLICATE_FIELD": {
      return {
        ...state,
        entities: state.entities.map((e) => {
          if (e._id !== action.entityId) return e;
          const src = e.fields.find((f) => f._id === action.fieldId);
          if (!src) return e;
          const copy = { ...src, _id: uuid(), name: `${src.name}_copy` } as FieldSchema;
          const idx = e.fields.indexOf(src);
          const fields = [...e.fields];
          fields.splice(idx + 1, 0, copy);
          return { ...e, fields };
        }),
      };
    }

    case "SET_OPTION":
      return { ...state, options: { ...state.options, [action.key]: action.value } };

    default:
      return state;
  }
}

export function emptySchema(): GraphSchema {
  return { options: {}, entities: [] };
}
