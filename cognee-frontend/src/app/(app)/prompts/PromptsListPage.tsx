"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ActionIcon,
  Button,
  Flex,
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
import { TrackPageView, trackEvent } from "@/modules/analytics";
import type { Prompt } from "@/modules/prompts/storage";
import {
  createPrompt,
  deletePrompt,
  loadPrompts,
  upsertPrompt,
} from "@/modules/prompts/storage";

export default function PromptsListPage() {
  const router = useRouter();
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newPromptName, setNewPromptName] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setPrompts(loadPrompts());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function handleCreate() {
    if (!newPromptName.trim()) return;
    const prompt = createPrompt(newPromptName.trim());
    upsertPrompt(prompt);
    setCreateModalOpen(false);
    setNewPromptName("");
    trackEvent({ pageName: "Prompts", eventName: "prompt_created", additionalProperties: { prompt_name: newPromptName.trim() } });
    router.push(`/prompts/${prompt.id}`);
  }

  function handleDelete(id: string) {
    deletePrompt(id);
    trackEvent({ pageName: "Prompts", eventName: "prompt_deleted", additionalProperties: { prompt_id: id } });
    setDeleteConfirmId(null);
    refresh();
  }

  const promptToDelete = prompts.find((p) => p.id === deleteConfirmId);

  return (
    <Stack className="!gap-[0.625rem]">
      <TrackPageView page="Prompts" />
      <Stack
        className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
        bg="white"
      >
        <Flex className="justify-between items-start mb-[1.375rem]">
          <Stack className="!gap-[0]">
            <Title size="h2" mb="0.125rem">
              Prompts
            </Title>
            <Text c={tokens.textMuted} size="lg">
              Define custom extraction prompts for your datasets
            </Text>
          </Stack>
          <Button
            onClick={() => setCreateModalOpen(true)}
            style={{ backgroundColor: tokens.purple }}
            radius="0.5rem"
          >
            + New Prompt
          </Button>
        </Flex>

        {prompts.length === 0 ? (
          <Flex
            direction="column"
            align="center"
            justify="center"
            className="py-[3rem] border border-dashed rounded-[0.5rem] border-cognee-border"
          >
            <Text c={tokens.textMuted} size="lg" mb="1rem">
              No prompts yet
            </Text>
            <Button
              variant="outline"
              onClick={() => setCreateModalOpen(true)}
              radius="0.5rem"
            >
              Create your first prompt
            </Button>
          </Flex>
        ) : (
          <Table horizontalSpacing="md" verticalSpacing="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Updated</Table.Th>
                <Table.Th>Actions</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {prompts.map((prompt) => (
                <Table.Tr
                  key={prompt.id}
                  style={{ cursor: "pointer", transition: "background-color 0.12s" }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "rgba(92,16,244,0.06)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ""; }}
                  onClick={() => router.push(`/prompts/${prompt.id}`)}
                >
                  <Table.Td>
                    <Text fw={500}>{prompt.name}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c={tokens.textMuted}>
                      {new Date(prompt.updatedAt).toLocaleDateString()}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Tooltip label="Delete" withArrow>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        size="sm"
                        onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(prompt.id); }}
                      >
                        ✕
                      </ActionIcon>
                    </Tooltip>
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
        onClose={() => { setCreateModalOpen(false); setNewPromptName(""); }}
        title="New Prompt"
        centered
        radius="0.5rem"
      >
        <Stack gap="1rem">
          <TextInput
            label="Prompt name"
            placeholder="e.g. Technical Knowledge Graph"
            value={newPromptName}
            onChange={(e) => setNewPromptName(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
            autoFocus
          />
          <Flex justify="flex-end" gap="0.5rem">
            <Button
              variant="default"
              onClick={() => { setCreateModalOpen(false); setNewPromptName(""); }}
              radius="0.5rem"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!newPromptName.trim()}
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
        title="Delete Prompt"
        centered
        radius="0.5rem"
      >
        <Stack gap="1rem">
          <Text>
            Are you sure you want to delete{" "}
            <Text component="span" fw={600}>{promptToDelete?.name}</Text>? This cannot be undone.
          </Text>
          <Flex justify="flex-end" gap="0.5rem">
            <Button variant="default" onClick={() => setDeleteConfirmId(null)} radius="0.5rem">
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
