import { v4 as uuid } from "uuid";
import type { GraphModel, GraphSchema } from "./types";
import { emptySchema } from "./schemaReducer";

const STORAGE_KEY = "cognee-graph-models";

export function loadModels(): GraphModel[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as GraphModel[]) : [];
  } catch {
    return [];
  }
}

export function saveModels(models: GraphModel[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(models));
}

export function createModel(name: string): GraphModel {
  const now = new Date().toISOString();
  return {
    id: uuid(),
    name,
    schema: emptySchema(),
    createdAt: now,
    updatedAt: now,
    status: "draft",
  };
}

export function getModel(id: string): GraphModel | undefined {
  return loadModels().find((m) => m.id === id);
}

export function upsertModel(model: GraphModel): void {
  const models = loadModels();
  const idx = models.findIndex((m) => m.id === model.id);
  const updated = { ...model, updatedAt: new Date().toISOString() };
  if (idx >= 0) {
    models[idx] = updated;
  } else {
    models.push(updated);
  }
  saveModels(models);
}

export function deleteModel(id: string): void {
  saveModels(loadModels().filter((m) => m.id !== id));
}

export function duplicateModel(id: string): GraphModel | undefined {
  const src = getModel(id);
  if (!src) return undefined;
  const copy = createModel(`${src.name} (copy)`);
  copy.schema = deepCloneSchema(src.schema);
  upsertModel(copy);
  return copy;
}

function deepCloneSchema(schema: GraphSchema): GraphSchema {
  return JSON.parse(JSON.stringify(schema));
}

// ── Default graph model ───────────────────────────────────────────────────────

export const DEFAULT_GRAPH_MODEL_ID = "00000000-0000-0000-0000-000000000001";

const ACTIVE_MODEL_KEY = "cognee-active-graph-model-id";

export function buildDefaultGraphModel(): GraphModel {
  const mkId = () => uuid();
  const now = new Date().toISOString();
  return {
    id: DEFAULT_GRAPH_MODEL_ID,
    name: "General Knowledge Graph",
    status: "published",
    createdAt: now,
    updatedAt: now,
    schema: {
      options: { autoTypeNodes: true },
      entities: [
        {
          _id: mkId(),
          name: "Person",
          description: "A human being mentioned in the text",
          primaryLabelField: "name",
          indexFields: ["name"],
          fields: [
            { _id: mkId(), name: "name", kind: "primitive", primitiveType: "string", required: true },
            { _id: mkId(), name: "role", kind: "primitive", primitiveType: "string" },
            { _id: mkId(), name: "description", kind: "primitive", primitiveType: "string" },
            { _id: mkId(), name: "affiliated_with", kind: "relation", relation: { targetEntityName: "Organization", cardinality: "many" } },
          ],
        },
        {
          _id: mkId(),
          name: "Organization",
          description: "A company, institution, or group",
          primaryLabelField: "name",
          indexFields: ["name"],
          fields: [
            { _id: mkId(), name: "name", kind: "primitive", primitiveType: "string", required: true },
            { _id: mkId(), name: "industry", kind: "primitive", primitiveType: "string" },
            { _id: mkId(), name: "description", kind: "primitive", primitiveType: "string" },
          ],
        },
        {
          _id: mkId(),
          name: "Concept",
          description: "An idea, technology, or topic discussed",
          primaryLabelField: "name",
          indexFields: ["name"],
          fields: [
            { _id: mkId(), name: "name", kind: "primitive", primitiveType: "string", required: true },
            { _id: mkId(), name: "description", kind: "primitive", primitiveType: "string" },
            { _id: mkId(), name: "related_to", kind: "relation", relation: { targetEntityName: "Concept", cardinality: "many" } },
          ],
        },
        {
          _id: mkId(),
          name: "Event",
          description: "A happening or occurrence with a timeline",
          primaryLabelField: "name",
          indexFields: ["name"],
          fields: [
            { _id: mkId(), name: "name", kind: "primitive", primitiveType: "string", required: true },
            { _id: mkId(), name: "date", kind: "primitive", primitiveType: "date" },
            { _id: mkId(), name: "description", kind: "primitive", primitiveType: "string" },
            { _id: mkId(), name: "participants", kind: "relation", relation: { targetEntityName: "Person", cardinality: "many" } },
          ],
        },
        {
          _id: mkId(),
          name: "Location",
          description: "A geographical place",
          primaryLabelField: "name",
          indexFields: ["name"],
          fields: [
            { _id: mkId(), name: "name", kind: "primitive", primitiveType: "string", required: true },
            { _id: mkId(), name: "country", kind: "primitive", primitiveType: "string" },
          ],
        },
      ],
    },
  };
}

/** Ensures the default model exists in localStorage. Returns its ID. */
export function ensureDefaultModel(): string {
  if (typeof window === "undefined") return DEFAULT_GRAPH_MODEL_ID;
  const models = loadModels();
  if (!models.find((m) => m.id === DEFAULT_GRAPH_MODEL_ID)) {
    saveModels([buildDefaultGraphModel(), ...models]);
  }
  return DEFAULT_GRAPH_MODEL_ID;
}

export function getActiveGraphModelId(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(ACTIVE_MODEL_KEY) ?? "";
}

export function setActiveGraphModelId(id: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(ACTIVE_MODEL_KEY, id);
}
