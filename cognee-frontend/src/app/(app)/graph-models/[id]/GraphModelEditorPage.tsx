"use client";

import { useCallback, useEffect, useReducer, useState } from "react";
import { notifications } from "@mantine/notifications";
import { useRouter } from "next/navigation";
import type { GraphModel } from "@/modules/graphModels/types";
import { schemaReducer, emptySchema } from "@/modules/graphModels/schemaReducer";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { syncGraphModels, loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import { inferSchema, downloadRawData } from "@/modules/llm/managementLlmApi";
import getDatasetData from "@/modules/datasets/getDatasetData";
import { v4 as uuid } from "uuid";
import { TrackPageView, trackEvent } from "@/modules/analytics";

import AddEntityModal from "./components/AddEntityModal";
import SchemaGraphPreview from "./components/SchemaGraphPreview";

interface GraphModelEditorPageProps {
  modelId: string;
}

export default function GraphModelEditorPage({ modelId }: GraphModelEditorPageProps) {
  const router = useRouter();
  const { cogniInstance } = useCogniInstance();
  const { datasets: contextDatasets } = useFilter();

  // ── Model metadata ──────────────────────────────────────────────────────────
  const [modelName, setModelName] = useState("Untitled Model");
  const [savedModelId, setSavedModelId] = useState<string>(modelId);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [assignedDatasetIds, setAssignedDatasetIds] = useState<string[]>([]);

  // ── Schema state ────────────────────────────────────────────────────────────
  const [schema, dispatch] = useReducer(schemaReducer, emptySchema());

  const dirtyDispatch: typeof dispatch = useCallback(
    (action) => {
      dispatch(action);
      setIsDirty(true);
    },
    [dispatch]
  );

  // ── UI state ─────────────────────────────────────────────────────────────────
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [addEntityModalOpen, setAddEntityModalOpen] = useState(false);
  const [addingField, setAddingField] = useState(false);
  const [newFieldName, setNewFieldName] = useState("");
  const [newFieldType, setNewFieldType] = useState<"string" | "number" | "boolean" | "date" | "relation">("string");
  const [newFieldTarget, setNewFieldTarget] = useState("");
  const [editingName, setEditingName] = useState(false);

  // ── Regenerate modal state ─────────────────────────────────────────────────
  const [showRegenerateModal, setShowRegenerateModal] = useState(false);
  const [regenDatasets, setRegenDatasets] = useState<{ id: string; name: string }[]>([]);
  const [regenSelectedDataset, setRegenSelectedDataset] = useState<string | null>(null);
  const [regenFiles, setRegenFiles] = useState<{ id: string; name: string }[]>([]);
  const [regenSelectedFiles, setRegenSelectedFiles] = useState<Set<string>>(new Set());
  const [regenLoadingFiles, setRegenLoadingFiles] = useState(false);

  // ── Load model from sessionStorage (fast) or backend config ─────────────────
  useEffect(() => {
    if (modelId === "new") {
      const newId = uuid();
      setSavedModelId(newId);
      window.history.replaceState({}, "", `/graph-models/${newId}`);
      return;
    }

    // Try sessionStorage first (set by dataset page when creating a model)
    const cached = sessionStorage.getItem(`graph-model-${modelId}`);
    if (cached) {
      try {
        const model = JSON.parse(cached) as GraphModel;
        setModelName(model.name);
        dispatch({ type: "SET_SCHEMA", schema: model.schema });
        setSavedModelId(model.id);
        setAssignedDatasetIds(model.assignedDatasets ?? []);
        setIsDirty(false);
        sessionStorage.removeItem(`graph-model-${modelId}`);
        return;
      } catch {}
    }

    if (!cogniInstance) return;
    loadGraphModelsConfig(cogniInstance).then((cfg) => {
      const model = cfg.models.find((m) => m.id === modelId);
      if (!model) {
        router.replace("/datasets");
        return;
      }
      setModelName(model.name);
      dispatch({ type: "SET_SCHEMA", schema: model.schema });
      setSavedModelId(model.id);
      setAssignedDatasetIds(model.assignedDatasets ?? []);
      setIsDirty(false);
    }).catch(() => {
      router.replace("/datasets");
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cogniInstance]);

  // ── Clear selection if entity was deleted ────────────────────────────────────
  useEffect(() => {
    if (
      selectedEntityId &&
      !schema.entities.find((e) => e._id === selectedEntityId)
    ) {
      setSelectedEntityId(null);
      setSidebarOpen(false);
    }
  }, [schema.entities, selectedEntityId]);

  // Reset add-field form when switching entities
  useEffect(() => {
    setAddingField(false);
    setNewFieldName("");
    setNewFieldType("string");
    setNewFieldTarget("");
  }, [selectedEntityId]);

  // ── Warn on close ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // ── Stats ──────────────────────────────────────────────────────────────────

  // ── Stats ──────────────────────────────────────────────────────────────────
  const entityCount = schema.entities.length;
  const relationCount = schema.entities.reduce(
    (sum, e) => sum + e.fields.filter((f) => f.kind === "relation").length,
    0
  );

  // ── Save to backend config ──────────────────────────────────────────────────
  async function handleSave() {
    if (!cogniInstance) {
      notifications.show({ title: "Not connected", message: "Connect to a Cognee instance to save.", color: "yellow" });
      return;
    }

    setIsSaving(true);
    try {
      const cfg = await loadGraphModelsConfig(cogniInstance);
      const now = new Date().toISOString();
      const currentModel: GraphModel = {
        id: savedModelId,
        name: modelName,
        schema,
        createdAt: cfg.models.find((m) => m.id === savedModelId)?.createdAt ?? now,
        updatedAt: now,
        status: "draft",
        assignedDatasets: cfg.models.find((m) => m.id === savedModelId)?.assignedDatasets,
      };
      // Upsert into the models list
      const idx = cfg.models.findIndex((m) => m.id === savedModelId);
      const updatedModels = [...cfg.models];
      if (idx >= 0) {
        updatedModels[idx] = currentModel;
      } else {
        updatedModels.push(currentModel);
      }
      await syncGraphModels(cogniInstance, updatedModels);
      setIsDirty(false);
      trackEvent({ pageName: "Graph Model Editor", eventName: "model_saved", additionalProperties: { model_id: savedModelId, model_name: modelName, entity_count: String(schema.entities.length) } });
      notifications.show({ title: "Saved", message: `"${modelName}" saved.`, color: "green" });
    } catch (err) {
      console.error("Save failed:", err);
      notifications.show({ title: "Save failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setIsSaving(false);
    }
  }

  // ── Regenerate schema via LLM ───────────────────────────────────────────────
  async function openRegenerateModal() {
    if (!cogniInstance) {
      notifications.show({ title: "Not connected", message: "Connect to a Cognee instance first.", color: "yellow" });
      return;
    }
    setShowRegenerateModal(true);
    setRegenSelectedDataset(null);
    setRegenFiles([]);
    setRegenSelectedFiles(new Set());
    try {
      const datasetList = contextDatasets.map((d) => ({ id: d.id, name: d.name }));
      setRegenDatasets(datasetList);

      // Auto-select the first assigned dataset if available
      const autoSelect = assignedDatasetIds.find((id) => datasetList.some((d) => d.id === id));
      if (autoSelect) {
        handleRegenDatasetSelect(autoSelect);
      }
    } catch {
      setRegenDatasets([]);
    }
  }

  async function handleRegenDatasetSelect(datasetId: string) {
    if (!cogniInstance) return;
    setRegenSelectedDataset(datasetId);
    setRegenSelectedFiles(new Set());
    setRegenLoadingFiles(true);
    try {
      const data = await getDatasetData(datasetId, cogniInstance);
      const files = Array.isArray(data) ? data.map((d: any) => ({
        id: d.id,
        name: d.name || d.rawDataLocation?.split("/").pop() || d.id,
      })) : [];
      setRegenFiles(files);
      setRegenSelectedFiles(new Set(files.map((f: { id: string }) => f.id)));
    } catch {
      setRegenFiles([]);
    } finally {
      setRegenLoadingFiles(false);
    }
  }

  async function handleRegenerate() {
    if (!cogniInstance || regenSelectedFiles.size === 0 || !regenSelectedDataset) return;

    const selectedFileEntries = regenFiles.filter((f) => regenSelectedFiles.has(f.id));

    setShowRegenerateModal(false);
    setIsRegenerating(true);
    try {
      // Download the already-processed text from the dataset's raw storage.
      // The /raw endpoint serves extracted text (not the original file), so
      // we send it as the `text` parameter rather than as file uploads.
      const textParts = await Promise.all(
        selectedFileEntries.map(async (f) => {
          const { blob } = await downloadRawData(cogniInstance, regenSelectedDataset!, f.id);
          return blob.text();
        }),
      );
      const combinedText = textParts.join("\n\n");

      console.log("[InferSchema] Files selected:", selectedFileEntries.map(f => f.name));
      console.log("[InferSchema] Text parts lengths:", textParts.map((t, i) => `${selectedFileEntries[i]?.name}: ${t.length} chars`));
      console.log("[InferSchema] Combined text length:", combinedText.length);
      console.log("[InferSchema] Combined text preview (first 500 chars):", combinedText.slice(0, 500));
      console.log("[InferSchema] Combined text preview (last 200 chars):", combinedText.slice(-200));

      // Retry up to 3 times — LLM output can be non-deterministic
      const MAX_ATTEMPTS = 3;
      let result: Awaited<ReturnType<typeof inferSchema>> | null = null;
      let lastError: unknown = null;
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        try {
          result = await inferSchema(cogniInstance, combinedText);
          break;
        } catch (err) {
          lastError = err;
          if (attempt < MAX_ATTEMPTS) {
            notifications.show({ title: "Retrying...", message: `Attempt ${attempt} failed, trying again (${attempt}/${MAX_ATTEMPTS})`, color: "yellow", autoClose: 3000 });
          }
        }
      }
      if (!result) throw lastError;

      if (result.graphSchema) {
        const defs = (result.graphSchema as Record<string, any>).$defs || {};
        const newSchema = {
          options: {},
          entities: Object.entries(defs)
            .filter(([key]) => !key.endsWith("Type"))
            .map(([name, def]: [string, any]) => ({
              _id: uuid(),
              name: def.title || name,
              description: def.description || "",
              fields: Object.entries(def.properties || {})
                .filter(([fieldName]) => fieldName !== "is_type" && fieldName !== "metadata")
                .map(([fieldName, fieldDef]: [string, any]) => {
                  if (fieldDef.$ref) {
                    const target = fieldDef.$ref.replace("#/$defs/", "");
                    return { _id: uuid(), name: fieldName, kind: "relation" as const, relation: { targetEntityName: target, cardinality: "one" as const } };
                  }
                  if (fieldDef.type === "array" && fieldDef.items?.$ref) {
                    const target = fieldDef.items.$ref.replace("#/$defs/", "");
                    return { _id: uuid(), name: fieldName, kind: "relation" as const, relation: { targetEntityName: target, cardinality: "many" as const } };
                  }
                  const primitiveType = (fieldDef.type === "number" || fieldDef.type === "integer") ? "number" : fieldDef.type === "boolean" ? "boolean" : "string";
                  return { _id: uuid(), name: fieldName, kind: "primitive" as const, primitiveType: primitiveType as "string" | "number" | "boolean" | "date", required: (def.required || []).includes(fieldName) };
                }),
              indexFields: [],
            })),
        };
        dispatch({ type: "SET_SCHEMA", schema: newSchema });
        setIsDirty(true);
        notifications.show({ title: "Schema regenerated", message: `Detected ${newSchema.entities.length} entity types from ${selectedFileEntries.length} file${selectedFileEntries.length !== 1 ? "s" : ""}.`, color: "green", autoClose: 4000 });
      }
    } catch (err) {
      console.error("Regenerate failed:", err);
      notifications.show({ title: "Regeneration failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setIsRegenerating(false);
    }
  }

  // ── Entity actions ──────────────────────────────────────────────────────────
  function handleAddEntity(name: string, description?: string) {
    dirtyDispatch({ type: "ADD_ENTITY", name, description });
    trackEvent({ pageName: "Graph Model Editor", eventName: "entity_added", additionalProperties: { model_id: savedModelId, entity_name: name } });
  }

  function handleCreateEntityFromRelation(name: string) {
    dirtyDispatch({ type: "ADD_ENTITY", name });
    setTimeout(() => {
      const newEntity = schema.entities.find((e) => e.name === name);
      if (newEntity) setSelectedEntityId(newEntity._id);
    }, 50);
  }

  function handleDeleteEntity(entityId: string) {
    dirtyDispatch({ type: "DELETE_ENTITY", entityId });
    trackEvent({ pageName: "Graph Model Editor", eventName: "entity_deleted", additionalProperties: { model_id: savedModelId } });
  }

  const selectedEntity = schema.entities.find((e) => e._id === selectedEntityId) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <TrackPageView page="Graph Model Editor" />

      {/* ── Header bar (Paper-style) ──────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBlock: "0.75rem", paddingInline: "1.5rem", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
          {/* Back button */}
          <button
            onClick={() => {
              if (isDirty && !window.confirm("You have unsaved changes. Leave anyway?")) return;
              router.push(assignedDatasetIds.length > 0 ? `/datasets/${assignedDatasetIds[0]}` : "/datasets");
            }}
            className="cursor-pointer hover:bg-white/10"
            style={{ background: "none", border: "none", padding: "2px 6px", borderRadius: 4, fontSize: 16, color: "rgba(237,236,234,0.55)", fontFamily: "inherit" }}
            title="Back to models"
          >
            &larr;
          </button>
          {editingName ? (
            <input
              autoFocus
              type="text"
              value={modelName}
              onChange={(e) => { setModelName(e.target.value); setIsDirty(true); }}
              onBlur={() => setEditingName(false)}
              onKeyDown={(e) => { if (e.key === "Enter") setEditingName(false); if (e.key === "Escape") setEditingName(false); }}
              style={{ fontSize: "0.875rem", fontWeight: 700, color: "#EDECEA", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 4, padding: "1px 6px", fontFamily: "inherit", outline: "none", minWidth: 120 }}
            />
          ) : (
            <span
              onDoubleClick={() => setEditingName(true)}
              title="Double-click to rename"
              style={{ fontSize: "0.875rem", fontWeight: 700, color: "#EDECEA", cursor: "default" }}
            >
              {modelName}
            </span>
          )}
          <span style={{ color: "rgba(237,236,234,0.35)", fontSize: "0.6875rem" }}>·</span>
          <span style={{ color: "#BC9BFF", fontSize: "0.6875rem" }}>
            {entityCount} type{entityCount !== 1 ? "s" : ""}, {relationCount} relationship{relationCount !== 1 ? "s" : ""}
          </span>
          {isDirty && (
            <>
              <span style={{ color: "rgba(237,236,234,0.35)", fontSize: "0.6875rem" }}>·</span>
              <span style={{ color: "#FBBF24", fontSize: "0.6875rem" }}>Unsaved</span>
            </>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginLeft: "auto" }}>
          <button
            onClick={async () => {
              if (!cogniInstance) return;
              if (!window.confirm(`Delete "${modelName}"? This cannot be undone.`)) return;
              try {
                const cfg = await loadGraphModelsConfig(cogniInstance);
                const updated = cfg.models.filter((m) => m.id !== savedModelId);
                await syncGraphModels(cogniInstance, updated);
                notifications.show({ title: "Deleted", message: `"${modelName}" has been deleted.`, color: "green", autoClose: 4000 });
                router.push(assignedDatasetIds.length > 0 ? `/datasets/${assignedDatasetIds[0]}` : "/datasets");
              } catch (err) {
                notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
              }
            }}
            className="cursor-pointer hover:opacity-100"
            style={{ display: "flex", alignItems: "center", justifyContent: "center", background: "none", borderRadius: "0.375rem", border: "1px solid rgba(239,68,68,0.4)", paddingBlock: "0.4375rem", paddingInline: "0.625rem", opacity: 0.7, transition: "opacity 150ms" }}
            title="Delete graph model"
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
          <button
            onClick={isRegenerating ? undefined : openRegenerateModal}
            disabled={isRegenerating}
            className="cursor-pointer"
            style={{ display: "flex", alignItems: "center", gap: "0.375rem", background: "rgba(255,255,255,0.06)", borderRadius: "0.375rem", border: "1px solid rgba(255,255,255,0.12)", paddingBlock: "0.4375rem", paddingInline: "1rem", color: "#EDECEA", fontSize: "0.8125rem", fontWeight: 500, lineHeight: "20px", opacity: isRegenerating ? 0.6 : 1 }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ animation: isRegenerating ? "spin 1s linear infinite" : "none" }}>
              <path d="M1.5 8a6.5 6.5 0 0111.48-4.16M14.5 8a6.5 6.5 0 01-11.48 4.16" stroke="#BC9BFF" strokeWidth="1.3" strokeLinecap="round" />
              <path d="M13 1v3h-3M3 15v-3h3" stroke="#BC9BFF" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {isRegenerating ? "Regenerating..." : "Regenerate"}
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="cursor-pointer"
            style={{ background: "#6510F4", borderRadius: "0.375rem", border: "none", paddingBlock: "0.4375rem", paddingInline: "1.25rem", color: "#fff", fontSize: "0.8125rem", fontWeight: 500, lineHeight: "20px" }}
          >
            {isSaving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      {/* ── Main area: graph canvas + right sidebar ───────────────────────── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* ── Graph canvas (center) ───────────────────────────────────── */}
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          <SchemaGraphPreview
            schema={schema}
            selectedEntityId={selectedEntityId}
            onEntitySelect={(id) => { setSelectedEntityId(id); setSidebarOpen(true); }}
          />

          {/* Floating toolbar (top-left) */}
          <div style={{ position: "absolute", top: 12, left: 12, display: "flex", gap: "0.375rem", zIndex: 10 }}>
            <button
              onClick={() => setAddEntityModalOpen(true)}
              className="cursor-pointer hover:bg-white/10"
              style={{ display: "flex", alignItems: "center", gap: "0.375rem", background: "rgba(15,15,15,0.85)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "0.375rem", paddingBlock: "0.375rem", paddingInline: "0.625rem", fontSize: "0.6875rem", fontWeight: 500, color: "#EDECEA" }}
            >
              <span style={{ color: "#BC9BFF", fontSize: "1rem", lineHeight: "20px" }}>+</span>
              Add entity type
            </button>
          </div>

          {/* Bottom hint bar */}
          <div style={{ position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)", background: "rgba(15,15,15,0.85)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "0.375rem", paddingBlock: "0.375rem", paddingInline: "0.75rem", zIndex: 10 }}>
            <span style={{ color: "rgba(237,236,234,0.7)", fontSize: "0.6875rem", lineHeight: "20px", whiteSpace: "nowrap" }}>
              Click a type to edit · Drag to reposition
            </span>
          </div>
        </div>

        {/* ── Right sidebar (entity editor) ───────────────────────────── */}
        {sidebarOpen && selectedEntity && (
        <div style={{ width: 280, flexShrink: 0, borderLeft: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", backdropFilter: "blur(12px)", display: "flex", flexDirection: "column", overflow: "auto" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", padding: "1.25rem", flex: 1 }}>
              {/* Header */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: "0.8125rem", fontWeight: 700, color: "#EDECEA" }}>
                  {selectedEntity.name}
                </span>
                <button
                  onClick={() => { setSelectedEntityId(null); setSidebarOpen(false); }}
                  className="cursor-pointer"
                  style={{ background: "none", border: "none", padding: 0, color: "rgba(237,236,234,0.4)", fontSize: 14 }}
                >
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="rgba(237,236,234,0.4)" strokeWidth="1.3" strokeLinecap="round" /></svg>
                </button>
              </div>

              {/* Name field */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                <span style={{ color: "rgba(237,236,234,0.45)", fontSize: "0.625rem", fontWeight: 700, lineHeight: "20px" }}>NAME</span>
                <input
                  type="text"
                  value={selectedEntity.name}
                  onChange={(e) => dirtyDispatch({ type: "UPDATE_ENTITY", entityId: selectedEntity._id, updates: { name: e.target.value } })}
                  style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", borderRadius: "0.375rem", padding: "0.4375rem 0.625rem", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }}
                />
              </div>

              {/* Description field */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                <span style={{ color: "rgba(237,236,234,0.45)", fontSize: "0.625rem", fontWeight: 700, lineHeight: "20px" }}>DESCRIPTION</span>
                <input
                  type="text"
                  value={selectedEntity.description ?? ""}
                  onChange={(e) => dirtyDispatch({ type: "UPDATE_ENTITY", entityId: selectedEntity._id, updates: { description: e.target.value } })}
                  placeholder="Describe this entity type..."
                  style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", borderRadius: "0.375rem", padding: "0.4375rem 0.625rem", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }}
                />
              </div>

              {/* Fields */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                <span style={{ color: "rgba(237,236,234,0.45)", fontSize: "0.625rem", fontWeight: 700, lineHeight: "20px" }}>
                  FIELDS ({selectedEntity.fields.length})
                </span>
                {selectedEntity.fields.map((field) => (
                  <div key={field._id} style={{ display: "flex", alignItems: "center", gap: "0.375rem", background: "rgba(255,255,255,0.06)", borderRadius: "0.25rem", padding: "0.375rem 0.5rem" }}>
                    <span style={{ fontSize: "0.75rem", color: "#EDECEA", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {field.name}
                    </span>
                    <span style={{ fontSize: "0.625rem", color: "rgba(237,236,234,0.45)", flexShrink: 0 }}>
                      {field.kind === "relation"
                        ? `→ ${field.relation.targetEntityName}`
                        : field.kind === "enum"
                        ? "enum"
                        : field.primitiveType}
                    </span>
                    <button
                      onClick={() => dirtyDispatch({ type: "DELETE_FIELD", entityId: selectedEntity._id, fieldId: field._id })}
                      className="cursor-pointer hover:opacity-100"
                      style={{ background: "none", border: "none", padding: 0, opacity: 0.3, transition: "opacity 150ms", flexShrink: 0, lineHeight: 1 }}
                      title="Remove field"
                    >
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 2l6 6M8 2l-6 6" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" /></svg>
                    </button>
                  </div>
                ))}

                {/* Add field */}
                {addingField ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "0.375rem", padding: "0.5rem" }}>
                    <input
                      type="text"
                      value={newFieldName}
                      onChange={(e) => setNewFieldName(e.target.value)}
                      placeholder="Field name"
                      autoFocus
                      style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "#EDECEA", borderRadius: "0.25rem", padding: "0.3rem 0.5rem", fontSize: "0.75rem", fontFamily: "inherit", outline: "none" }}
                    />
                    <select
                      value={newFieldType}
                      onChange={(e) => setNewFieldType(e.target.value as typeof newFieldType)}
                      style={{ border: "1px solid rgba(255,255,255,0.12)", borderRadius: "0.25rem", padding: "0.3rem 0.5rem", fontSize: "0.75rem", fontFamily: "inherit", outline: "none", background: "rgba(255,255,255,0.06)", color: "#EDECEA" }}
                    >
                      <option value="string">String</option>
                      <option value="number">Number</option>
                      <option value="boolean">Boolean</option>
                      <option value="date">Date</option>
                      <option value="relation">Relation →</option>
                    </select>
                    {newFieldType === "relation" && (
                      <select
                        value={newFieldTarget}
                        onChange={(e) => setNewFieldTarget(e.target.value)}
                        style={{ border: "1px solid rgba(255,255,255,0.12)", borderRadius: "0.25rem", padding: "0.3rem 0.5rem", fontSize: "0.75rem", fontFamily: "inherit", outline: "none", background: "rgba(255,255,255,0.06)", color: "#EDECEA" }}
                      >
                        <option value="">Select target entity...</option>
                        {schema.entities.map((e) => (
                          <option key={e._id} value={e.name}>{e.name}</option>
                        ))}
                      </select>
                    )}
                    <div style={{ display: "flex", gap: "0.25rem" }}>
                      <button
                        onClick={() => { setAddingField(false); setNewFieldName(""); setNewFieldType("string"); setNewFieldTarget(""); }}
                        className="cursor-pointer"
                        style={{ flex: 1, background: "none", border: "1px solid rgba(255,255,255,0.15)", borderRadius: "0.25rem", padding: "0.25rem", fontSize: "0.6875rem", color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => {
                          if (!newFieldName.trim()) return;
                          if (newFieldType === "relation") {
                            if (!newFieldTarget) return;
                            dirtyDispatch({ type: "ADD_FIELD", entityId: selectedEntity._id, field: { name: newFieldName.trim(), kind: "relation", relation: { targetEntityName: newFieldTarget, cardinality: "many" } } });
                          } else {
                            dirtyDispatch({ type: "ADD_FIELD", entityId: selectedEntity._id, field: { name: newFieldName.trim(), kind: "primitive", primitiveType: newFieldType } });
                          }
                          setNewFieldName(""); setNewFieldType("string"); setNewFieldTarget(""); setAddingField(false);
                        }}
                        className="cursor-pointer"
                        style={{ flex: 1, background: "#6510F4", border: "none", borderRadius: "0.25rem", padding: "0.25rem", fontSize: "0.6875rem", color: "#fff", fontWeight: 500, fontFamily: "inherit" }}
                      >
                        Add
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => setAddingField(true)}
                    className="cursor-pointer hover:bg-white/10"
                    style={{ display: "flex", alignItems: "center", gap: "0.25rem", background: "none", border: "1px dashed rgba(255,255,255,0.2)", borderRadius: "0.25rem", padding: "0.375rem 0.5rem", fontSize: "0.6875rem", color: "#BC9BFF", fontWeight: 500, fontFamily: "inherit" }}
                  >
                    <span style={{ fontSize: "0.875rem", lineHeight: 1 }}>+</span> Add field
                  </button>
                )}
              </div>

              {/* Relationships */}
              {(() => {
                const relations = selectedEntity.fields.filter((f) => f.kind === "relation");
                if (relations.length === 0) return null;
                return (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                    <span style={{ color: "rgba(237,236,234,0.45)", fontSize: "0.625rem", fontWeight: 700, lineHeight: "20px" }}>RELATIONSHIPS</span>
                    {relations.map((field) => (
                      <div key={field._id} style={{ display: "flex", alignItems: "center", gap: "0.375rem", background: "rgba(255,255,255,0.06)", borderRadius: "0.25rem", padding: "0.375rem 0.5rem" }}>
                        <span style={{ color: "#BC9BFF", fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', fontSize: "0.6875rem" }}>
                          {field.name}
                        </span>
                        <span style={{ color: "rgba(237,236,234,0.55)", fontSize: "0.6875rem" }}>
                          → {field.kind === "relation" ? field.relation.targetEntityName : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* Bottom actions */}
              <div style={{ display: "flex", gap: "0.5rem", marginTop: "auto" }}>
                <button
                  onClick={() => handleDeleteEntity(selectedEntity._id)}
                  className="cursor-pointer hover:bg-red-500/10"
                  style={{ flex: 1, background: "transparent", border: "1px solid rgba(239,68,68,0.5)", borderRadius: "0.375rem", padding: "0.4375rem", color: "#EF4444", fontSize: "0.6875rem", fontFamily: "inherit" }}
                >
                  Delete type
                </button>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="cursor-pointer"
                  style={{ flex: 1, background: "#6510F4", border: "none", borderRadius: "0.375rem", padding: "0.4375rem", color: "#fff", fontSize: "0.6875rem", fontWeight: 500, fontFamily: "inherit" }}
                >
                  Save changes
                </button>
              </div>
            </div>
        </div>
        )}
      </div>

      {/* ── Modals ──────────────────────────────────────────────────────────── */}
      <AddEntityModal
        opened={addEntityModalOpen}
        onClose={() => setAddEntityModalOpen(false)}
        onSubmit={(name, description) => {
          handleAddEntity(name, description);
          setTimeout(() => {
            const found = schema.entities.find((e) => e.name === name);
            if (found) setSelectedEntityId(found._id);
          }, 50);
        }}
      />
      {/* ── Regenerate modal ────────────────────────────────────────────── */}
      {showRegenerateModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowRegenerateModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 480, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Regenerate Schema</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
              Select a dataset and files to analyze. Cognee will infer entity types and relationships from the selected files.
            </p>

            {schema.entities.length > 0 && (
              <div style={{ display: "flex", alignItems: "flex-start", gap: 8, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 8, padding: "10px 12px" }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0, marginTop: 1 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(239,68,68,0.25)" stroke="#EF4444" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#FCA5A5">!</text></svg>
                <span style={{ fontSize: 13, color: "#FCA5A5", lineHeight: "20px" }}>This will overwrite your existing graph model. The current schema will be replaced entirely.</span>
              </div>
            )}

            {/* Dataset selector */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.45)", textTransform: "uppercase", letterSpacing: 0.3 }}>Dataset</label>
              <select
                value={regenSelectedDataset ?? ""}
                onChange={(e) => e.target.value && handleRegenDatasetSelect(e.target.value)}
                style={{ border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none", background: "rgba(255,255,255,0.06)" }}
              >
                <option value="">Select a dataset...</option>
                {regenDatasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>

            {/* File list */}
            {regenSelectedDataset && (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.45)", textTransform: "uppercase", letterSpacing: 0.3 }}>
                    Files {!regenLoadingFiles && regenFiles.length > 0 && `(${regenSelectedFiles.size}/${regenFiles.length})`}
                  </label>
                  {regenFiles.length > 0 && (
                    <button
                      onClick={() => setRegenSelectedFiles(regenSelectedFiles.size === regenFiles.length ? new Set() : new Set(regenFiles.map((f) => f.id)))}
                      className="cursor-pointer"
                      style={{ background: "none", border: "none", fontSize: 11, color: "#6510F4", fontWeight: 500, fontFamily: "inherit", padding: 0 }}
                    >
                      {regenSelectedFiles.size === regenFiles.length ? "Deselect all" : "Select all"}
                    </button>
                  )}
                </div>
                {regenLoadingFiles ? (
                  <div style={{ padding: "12px 0", textAlign: "center", fontSize: 13, color: "rgba(237,236,234,0.55)" }}>Loading files...</div>
                ) : regenFiles.length === 0 ? (
                  <div style={{ padding: "12px 0", textAlign: "center", fontSize: 13, color: "rgba(237,236,234,0.55)" }}>No files in this dataset</div>
                ) : (
                  <div style={{ border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, maxHeight: 220, overflowY: "auto" }}>
                    {regenFiles.map((file) => (
                      <label key={file.id} className="hover:bg-white/10" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", cursor: "pointer", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                        <input
                          type="checkbox"
                          checked={regenSelectedFiles.has(file.id)}
                          onChange={() => {
                            setRegenSelectedFiles((prev) => {
                              const next = new Set(prev);
                              if (next.has(file.id)) next.delete(file.id); else next.add(file.id);
                              return next;
                            });
                          }}
                          style={{ accentColor: "#6510F4" }}
                        />
                        <span style={{ fontSize: 13, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 4 }}>
              <button onClick={() => setShowRegenerateModal(false)} className="cursor-pointer" style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.8)", fontFamily: "inherit" }}>Cancel</button>
              <button
                onClick={handleRegenerate}
                disabled={regenSelectedFiles.size === 0}
                className="cursor-pointer"
                style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit", opacity: regenSelectedFiles.size === 0 ? 0.5 : 1 }}
              >
                Regenerate from {regenSelectedFiles.size} file{regenSelectedFiles.size !== 1 ? "s" : ""}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
