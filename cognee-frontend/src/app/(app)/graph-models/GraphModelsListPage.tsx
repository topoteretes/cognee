"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ActionIcon,
  Badge,
  Button,
  Flex,
  Group,
  Modal,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from "@mantine/core";
import { useRouter } from "next/navigation";
import { tokens } from "@/ui/theme/tokens";
import type { GraphModel } from "@/modules/graphModels/types";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import {
  createModel,
  deleteModel,
  duplicateModel,
  loadModels,
  upsertModel,
  getActiveGraphModelId,
  setActiveGraphModelId,
  ensureDefaultModel,
} from "@/modules/graphModels/storage";
import { toCleanSchema } from "@/modules/graphModels/types";

export default function GraphModelsListPage() {
  const router = useRouter();
  const [models, setModels] = useState<GraphModel[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newModelName, setNewModelName] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [activeModelId, setActiveModelIdState] = useState<string>("");

  const refresh = useCallback(() => {
    const defaultId = ensureDefaultModel();
    setModels(loadModels());
    setActiveModelIdState(getActiveGraphModelId() ?? defaultId);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function handleSetActive(id: string) {
    setActiveGraphModelId(id);
    setActiveModelIdState(id);
    trackEvent({ pageName: "Graph Models", eventName: "model_set_active", additionalProperties: { model_id: id } });
  }

  function handleCreate() {
    if (!newModelName.trim()) return;
    const model = createModel(newModelName.trim());
    upsertModel(model);
    setCreateModalOpen(false);
    setNewModelName("");
    trackEvent({ pageName: "Graph Models", eventName: "model_created", additionalProperties: { model_name: newModelName.trim() } });
    router.push(`/graph-models/${model.id}`);
  }

  function handleDuplicate(id: string) {
    duplicateModel(id);
    trackEvent({ pageName: "Graph Models", eventName: "model_duplicated", additionalProperties: { model_id: id } });
    refresh();
  }

  function handleDelete(id: string) {
    deleteModel(id);
    trackEvent({ pageName: "Graph Models", eventName: "model_deleted", additionalProperties: { model_id: id } });
    setDeleteConfirmId(null);
    refresh();
  }

  function handleExport(model: GraphModel) {
    const clean = toCleanSchema(model.schema);
    const payload = JSON.stringify({ name: model.name, schema: clean }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model.name.replace(/\s+/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
    trackEvent({ pageName: "Graph Models", eventName: "model_exported", additionalProperties: { model_id: model.id, model_name: model.name } });
  }

  const modelToDelete = models.find((m) => m.id === deleteConfirmId);

  return (
    <Stack className="!gap-[0.625rem]">
      <TrackPageView page="Graph Models" />
      <Stack
        className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
        bg="white"
      >
        <Flex className="justify-between items-start mb-[1.375rem]">
          <Stack className="!gap-[0]">
            <Title size="h2" mb="0.125rem">
              Graph Models
            </Title>
            <Text c={tokens.textMuted} size="lg">
              Define graph-extraction schemas for your datasets
            </Text>
          </Stack>
          <Button
            onClick={() => setCreateModalOpen(true)}
            style={{ backgroundColor: tokens.purple }}
            radius="0.5rem"
          >
            + New Graph Model
          </Button>
        </Flex>

        {models.length === 0 ? (
          <Flex
            direction="column"
            align="center"
            justify="center"
            className="py-[3rem] border border-dashed rounded-[0.5rem] border-cognee-border"
          >
            <Text c={tokens.textMuted} size="lg" mb="1rem">
              No graph models yet
            </Text>
            <Button
              variant="outline"
              onClick={() => setCreateModalOpen(true)}
              radius="0.5rem"
            >
              Create your first model
            </Button>
          </Flex>
        ) : (
          <Table horizontalSpacing="md" verticalSpacing="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Entities</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Updated</Table.Th>
                <Table.Th>Actions</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {models.map((model) => (
                <Table.Tr
                  key={model.id}
                  style={{ cursor: "pointer", transition: "background-color 0.12s" }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "rgba(92,16,244,0.06)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ""; }}
                  onClick={() => router.push(`/graph-models/${model.id}`)}
                >
                  <Table.Td>
                    <Text fw={500}>{model.name}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c={tokens.textMuted}>
                      {model.schema.entities.length}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Group gap="0.375rem">
                      <Badge
                        color={model.status === "published" ? "green" : "gray"}
                        variant="light"
                        size="sm"
                      >
                        {model.status}
                      </Badge>
                      {model.id === activeModelId && (
                        <Badge color="violet" variant="filled" size="sm">
                          Active
                        </Badge>
                      )}
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c={tokens.textMuted}>
                      {new Date(model.updatedAt).toLocaleDateString()}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Group gap="0.25rem">
                      {model.id !== activeModelId && (
                        <Tooltip label="Set as active" withArrow>
                          <ActionIcon
                            variant="subtle"
                            size="sm"
                            color="violet"
                            onClick={(e) => { e.stopPropagation(); handleSetActive(model.id); }}
                          >
                            ✓
                          </ActionIcon>
                        </Tooltip>
                      )}
                      <Tooltip label="Duplicate" withArrow>
                        <ActionIcon
                          variant="subtle"
                          size="sm"
                          onClick={(e) => { e.stopPropagation(); handleDuplicate(model.id); }}
                        >
                          ⧉
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Export JSON" withArrow>
                        <ActionIcon
                          variant="subtle"
                          size="sm"
                          onClick={(e) => { e.stopPropagation(); handleExport(model); }}
                        >
                          ↓
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Delete" withArrow>
                        <ActionIcon
                          variant="subtle"
                          color="red"
                          size="sm"
                          onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(model.id); }}
                        >
                          ✕
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}
      </Stack>

      {/* Create modal */}
      <Modal
        opened={createModalOpen}
        onClose={() => {
          setCreateModalOpen(false);
          setNewModelName("");
        }}
        title="New Graph Model"
        centered
        radius="0.5rem"
      >
        <Stack gap="1rem">
          <TextInput
            label="Model name"
            placeholder="e.g. Knowledge Graph"
            value={newModelName}
            onChange={(e) => setNewModelName(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
            autoFocus
          />
          <Flex justify="flex-end" gap="0.5rem">
            <Button
              variant="default"
              onClick={() => {
                setCreateModalOpen(false);
                setNewModelName("");
              }}
              radius="0.5rem"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!newModelName.trim()}
              style={{ backgroundColor: tokens.purple }}
              radius="0.5rem"
            >
              Create &amp; Edit
            </Button>
          </Flex>
        </Stack>
      </Modal>

      {/* Delete confirm modal */}
      <Modal
        opened={!!deleteConfirmId}
        onClose={() => setDeleteConfirmId(null)}
        title="Delete Graph Model"
        centered
        radius="0.5rem"
      >
        <Stack gap="1rem">
          <Text>
            Are you sure you want to delete{" "}
            <Text component="span" fw={600}>
              {modelToDelete?.name}
            </Text>
            ? This cannot be undone.
          </Text>
          <Flex justify="flex-end" gap="0.5rem">
            <Button
              variant="default"
              onClick={() => setDeleteConfirmId(null)}
              radius="0.5rem"
            >
              Cancel
            </Button>
            <Button
              color="red"
              onClick={() => deleteConfirmId && handleDelete(deleteConfirmId)}
              radius="0.5rem"
            >
              Delete
            </Button>
          </Flex>
        </Stack>
      </Modal>
    </Stack>
  );
}
