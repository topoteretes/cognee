"use client";

import { useCallback, useEffect, useMemo, useReducer, useState } from "react";
import {
  ActionIcon,
  Button,
  CopyButton,
  Flex,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useRouter } from "next/navigation";
import { tokens } from "@/ui/theme/tokens";
import type { GraphSchema } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { schemaReducer, emptySchema } from "@/modules/graphModels/schemaReducer";
import { validateSchema } from "@/modules/graphModels/validator";
import { createModel, getModel, upsertModel } from "@/modules/graphModels/storage";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { syncGraphModels } from "@/modules/configuration/userConfiguration";
import { TrackPageView, trackEvent } from "@/modules/analytics";

import EntitiesPanel from "./components/EntitiesPanel";
import EntityEditor from "./components/EntityEditor";
import AddEntityModal from "./components/AddEntityModal";
import SchemaGraphPreview from "./components/SchemaGraphPreview";
import TestExtractionPanel from "./components/TestExtractionPanel";
import ValidationIssuesPanel from "./components/ValidationIssuesPanel";

interface GraphModelEditorPageProps {
  modelId: string;
}

export default function GraphModelEditorPage({ modelId }: GraphModelEditorPageProps) {
  const router = useRouter();
  const { cogniInstance } = useCogniInstance();

  // ── Model metadata ──────────────────────────────────────────────────────────
  const [modelName, setModelName] = useState("Untitled Model");
  const [savedModelId, setSavedModelId] = useState<string>(modelId);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // ── Schema state ────────────────────────────────────────────────────────────
  const [schema, dispatch] = useReducer(schemaReducer, emptySchema());

  // Mark dirty on every dispatch
  const dirtyDispatch: typeof dispatch = useCallback(
    (action) => {
      dispatch(action);
      setIsDirty(true);
    },
    [dispatch]
  );

  // ── UI state ─────────────────────────────────────────────────────────────────
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [rightPanel, setRightPanel] = useState<"graph" | "test">("graph");
  const [addEntityModalOpen, setAddEntityModalOpen] = useState(false);
  const [validationOpen, setValidationOpen] = useState(false);

  // ── Load model ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (modelId === "new") {
      // brand-new model: create in storage now so it has an ID
      const model = createModel("Untitled Model");
      upsertModel(model);
      setSavedModelId(model.id);
      // replace URL without adding to history
      window.history.replaceState({}, "", `/graph-models/${model.id}`);
      return;
    }
    const model = getModel(modelId);
    if (!model) {
      router.replace("/graph-models");
      return;
    }
    setModelName(model.name);
    dispatch({ type: "SET_SCHEMA", schema: model.schema });
    setSavedModelId(model.id);
    setIsDirty(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Auto-select first entity ────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedEntityId && schema.entities.length > 0) {
      setSelectedEntityId(schema.entities[0]._id);
    }
    // If the selected entity was deleted, clear selection
    if (
      selectedEntityId &&
      !schema.entities.find((e) => e._id === selectedEntityId)
    ) {
      setSelectedEntityId(schema.entities[0]?._id ?? null);
    }
  }, [schema.entities, selectedEntityId]);

  // ── Warn browser on close/refresh when unsaved ──────────────────────────────
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // ── Auto-save to localStorage on every dirty change ───────────────────────
  useEffect(() => {
    if (!isDirty) return;
    const existing = getModel(savedModelId);
    upsertModel({
      id: savedModelId,
      name: modelName,
      schema,
      createdAt: existing?.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: existing?.status ?? "draft",
    });
  }, [schema, modelName, savedModelId, isDirty]);

  // ── Validation ──────────────────────────────────────────────────────────────
  const issues = useMemo(() => validateSchema(schema), [schema]);
  const errorCount = issues.filter((i) => i.severity === "error").length;
  const warnCount = issues.filter((i) => i.severity === "warn").length;

  // ── Save (localStorage + backend) ───────────────────────────────────────────
  async function handleSave() {
    const existing = getModel(savedModelId);
    upsertModel({
      id: savedModelId,
      name: modelName,
      schema,
      createdAt: existing?.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: existing?.status ?? "draft",
    });
    setIsDirty(false);
    trackEvent({ pageName: "Graph Model Editor", eventName: "model_saved", additionalProperties: { model_id: savedModelId, model_name: modelName, entity_count: String(schema.entities.length) } });

    if (!cogniInstance) {
      notifications.show({
        title: "Saved locally",
        message: `"${modelName}" saved. Connect a Cognee instance to sync to cloud.`,
        color: "green",
      });
      return;
    }

    setIsSaving(true);
    try {
      await syncGraphModels(cogniInstance);
      notifications.show({
        title: "Saved",
        message: `"${modelName}" saved and synced to cloud.`,
        color: "green",
      });
    } catch {
      notifications.show({
        title: "Local save succeeded",
        message: `"${modelName}" saved locally. Cloud sync failed — try again.`,
        color: "yellow",
      });
    } finally {
      setIsSaving(false);
    }
  }

  // ── Export ──────────────────────────────────────────────────────────────────
  function handleExport() {
    const clean = toCleanSchema(schema);
    const payload = JSON.stringify({ name: modelName, schema: clean }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${modelName.replace(/\s+/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function getExportString(): string {
    return JSON.stringify({ name: modelName, schema: toCleanSchema(schema) }, null, 2);
  }

  // ── Entity actions ──────────────────────────────────────────────────────────
  function handleAddEntity(name: string, description?: string) {
    dirtyDispatch({ type: "ADD_ENTITY", name, description });
    trackEvent({ pageName: "Graph Model Editor", eventName: "entity_added", additionalProperties: { model_id: savedModelId, entity_name: name } });
  }

  function handleCreateEntityFromRelation(name: string) {
    dirtyDispatch({ type: "ADD_ENTITY", name });
    // Select the new entity after state update
    setTimeout(() => {
      const newEntity = schema.entities.find((e) => e.name === name);
      if (newEntity) setSelectedEntityId(newEntity._id);
    }, 50);
  }

  function handleDeleteEntity(entityId: string) {
    const entityName = schema.entities.find((e) => e._id === entityId)?.name;
    dirtyDispatch({ type: "DELETE_ENTITY", entityId });
    trackEvent({ pageName: "Graph Model Editor", eventName: "entity_deleted", additionalProperties: { model_id: savedModelId, entity_name: entityName ?? entityId } });
  }

  function handleDuplicateEntity(entityId: string) {
    dirtyDispatch({ type: "DUPLICATE_ENTITY", entityId });
  }

  // ── Jump to field (from validation / edge click) ───────────────────────────
  function handleJumpTo(entityId?: string, fieldId?: string) {
    if (entityId) {
      setSelectedEntityId(entityId);
    }
    // fieldId highlight could be implemented with more state if needed
    setValidationOpen(false);
  }

  const selectedEntity = schema.entities.find((e) => e._id === selectedEntityId) ?? null;

  return (
    <Stack className="!gap-[0.625rem] h-full overflow-hidden">
      <TrackPageView page="Graph Model Editor" />
      {/* ── Header bar ──────────────────────────────────────────────────────── */}
      <Flex
        className="items-center gap-[0.75rem] px-[1rem] py-[0.625rem] rounded-[0.5rem] flex-shrink-0"
        bg="white"
        style={{ border: `1px solid ${tokens.border}` }}
      >
        {/* Back */}
        <Tooltip label="Back to list" withArrow>
          <ActionIcon
            variant="subtle"
            size="sm"
            onClick={() => {
              if (isDirty && !window.confirm("You have changes not yet saved to the cloud. Leave anyway?")) return;
              router.push("/graph-models");
            }}
          >
            ←
          </ActionIcon>
        </Tooltip>

        {/* Inline model name */}
        <TextInput
          value={modelName}
          onChange={(e) => {
            setModelName(e.currentTarget.value);
            setIsDirty(true);
          }}
          variant="unstyled"
          fw={600}
          size="md"
          className="flex-1 min-w-0"
          styles={{ input: { fontSize: "1rem", fontWeight: 600 } }}
          placeholder="Model name"
        />

        {isDirty && (
          <Text size="xs" c={tokens.textMuted}>
            Unsaved changes
          </Text>
        )}

        {/* Validate */}
        <Tooltip label="Run validation" withArrow>
          <Button
            variant="default"
            size="xs"
            radius="0.25rem"
            onClick={() => {
              setValidationOpen(true);
            }}
            leftSection={
              errorCount > 0 ? "✕" : warnCount > 0 ? "⚠" : "✓"
            }
            color={errorCount > 0 ? "red" : warnCount > 0 ? "yellow" : "green"}
          >
            Validate{" "}
            {errorCount > 0 && (
              <Text component="span" size="xs" c="red" ml="0.25rem">
                ({errorCount})
              </Text>
            )}
          </Button>
        </Tooltip>

        {/* Export */}
        <Button
          variant="default"
          size="xs"
          radius="0.25rem"
          onClick={handleExport}
        >
          Export JSON
        </Button>

        {/* Copy JSON */}
        <CopyButton value={getExportString()}>
          {({ copied, copy }) => (
            <Tooltip label={copied ? "Copied!" : "Copy JSON"} withArrow>
              <Button
                variant="default"
                size="xs"
                radius="0.25rem"
                onClick={copy}
                color={copied ? "teal" : undefined}
              >
                {copied ? "✓ Copied" : "Copy JSON"}
              </Button>
            </Tooltip>
          )}
        </CopyButton>

        {/* Save */}
        <Button
          size="xs"
          radius="0.25rem"
          onClick={handleSave}
          loading={isSaving}
          style={{ backgroundColor: tokens.purple }}
        >
          Save
        </Button>
      </Flex>

      {/* ── Split layout ─────────────────────────────────────────────────────── */}
      <Flex gap="0.625rem" className="flex-1 overflow-hidden min-h-0">
        {/* ── Left column ──────────────────────────────────────────────────── */}
        <Flex direction="column" gap="0.625rem" style={{ width: "380px", flexShrink: 0, overflow: "hidden" }}>
          <EntitiesPanel
            entities={schema.entities}
            selectedEntityId={selectedEntityId}
            issues={issues}
            onSelect={setSelectedEntityId}
            onAdd={() => setAddEntityModalOpen(true)}
            onDuplicate={handleDuplicateEntity}
            onDelete={handleDeleteEntity}
          />

          {selectedEntity ? (
            <EntityEditor
              entity={selectedEntity}
              allEntities={schema.entities}
              dispatch={dirtyDispatch}
              issues={issues}
              onCreateEntity={handleCreateEntityFromRelation}
            />
          ) : (
            <Flex
              direction="column"
              align="center"
              justify="center"
              className="flex-1 rounded-[0.5rem]"
              bg="white"
              style={{ border: `1px solid ${tokens.border}` }}
            >
              <Text size="sm" c={tokens.textMuted} ta="center" px="2rem">
                {schema.entities.length === 0
                  ? 'Click "+ New Entity" to get started'
                  : "Select an entity to edit"}
              </Text>
            </Flex>
          )}
        </Flex>

        {/* ── Right column ─────────────────────────────────────────────────── */}
        <Flex direction="column" gap="0.625rem" className="flex-1 overflow-auto min-w-0">
          {/* Panel toggle — sits above the self-contained card */}
          <Flex
            className="items-center justify-between px-[1rem] py-[0.5rem] rounded-[0.5rem] flex-shrink-0"
            bg="white"
            style={{ border: `1px solid ${tokens.border}` }}
          >
            <SegmentedControl
              size="xs"
              value={rightPanel}
              onChange={(v) => setRightPanel(v as "graph" | "test")}
              data={[
                { value: "graph", label: "Schema Graph" },
                { value: "test", label: "Test Extraction" },
              ]}
              radius="0.5rem"
            />
            <Text size="xs" c={tokens.textMuted}>
              {schema.entities.length} entit{schema.entities.length === 1 ? "y" : "ies"}
            </Text>
          </Flex>

          {/* Panel content — each panel is its own self-contained card */}
          {rightPanel === "graph" ? (
            <SchemaGraphPreview
              schema={schema}
              selectedEntityId={selectedEntityId}
              onEntitySelect={setSelectedEntityId}
              onJumpToField={(entityId, fieldId) => handleJumpTo(entityId, fieldId)}
            />
          ) : (
            <Stack
              className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] flex-1 !gap-[0]"
              bg="white"
            >
              <TestExtractionPanel schema={schema} />
            </Stack>
          )}
        </Flex>
      </Flex>

      {/* ── Modals / Drawers ──────────────────────────────────────────────────── */}
      <AddEntityModal
        opened={addEntityModalOpen}
        onClose={() => setAddEntityModalOpen(false)}
        onSubmit={(name, description) => {
          handleAddEntity(name, description);
          // Select the newly added entity after dispatch
          setTimeout(() => {
            setSelectedEntityId(
              (prev) => {
                const found = schema.entities.find((e) => e.name === name);
                return found?._id ?? prev;
              }
            );
          }, 50);
        }}
      />

      <ValidationIssuesPanel
        opened={validationOpen}
        onClose={() => setValidationOpen(false)}
        issues={issues}
        onJumpTo={handleJumpTo}
      />
    </Stack>
  );
}
