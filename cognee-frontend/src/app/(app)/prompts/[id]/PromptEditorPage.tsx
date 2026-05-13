"use client";

import { useEffect, useState } from "react";
import { notifications } from "@mantine/notifications";
import {
  ActionIcon,
  Button,
  Flex,
  Stack,
  Text,
  Textarea,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { useRouter } from "next/navigation";
import { tokens } from "@/ui/theme/tokens";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import { syncPrompts } from "@/modules/configuration/configurationActions";
import {
  createPrompt,
  getPrompt,
  loadPrompts,
  upsertPrompt,
} from "@/modules/prompts/storage";

interface PromptEditorPageProps {
  promptId: string;
}

export default function PromptEditorPage({ promptId }: PromptEditorPageProps) {
  const router = useRouter();
  const [promptName, setPromptName] = useState("Untitled Prompt");
  const [content, setContent] = useState("");
  const [savedId, setSavedId] = useState(promptId);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Load or create
  useEffect(() => {
    if (promptId === "new") {
      const prompt = createPrompt("Untitled Prompt");
      upsertPrompt(prompt);
      setSavedId(prompt.id);
      setPromptName(prompt.name);
      setContent(prompt.content);
      window.history.replaceState({}, "", `/prompts/${prompt.id}`);
      return;
    }
    const prompt = getPrompt(promptId);
    if (!prompt) {
      router.replace("/prompts");
      return;
    }
    setPromptName(prompt.name);
    setContent(prompt.content);
    setSavedId(prompt.id);
    setIsDirty(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Warn on close when unsaved
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // Auto-save to localStorage on every change
  useEffect(() => {
    if (!isDirty) return;
    const existing = getPrompt(savedId);
    upsertPrompt({
      id: savedId,
      name: promptName,
      content,
      createdAt: existing?.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
  }, [content, promptName, savedId, isDirty]);

  async function handleSave() {
    const existing = getPrompt(savedId);
    upsertPrompt({
      id: savedId,
      name: promptName,
      content,
      createdAt: existing?.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
    setIsDirty(false);
    trackEvent({ pageName: "Prompt Editor", eventName: "prompt_saved", additionalProperties: { prompt_id: savedId, prompt_name: promptName } });

    setIsSaving(true);
    try {
      await syncPrompts(loadPrompts());
      notifications.show({
        title: "Saved",
        message: `"${promptName}" saved and synced to cloud.`,
        color: "green",
      });
    } catch {
      notifications.show({
        title: "Local save succeeded",
        message: `"${promptName}" saved locally. Cloud sync failed — try again.`,
        color: "yellow",
      });
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <Stack className="!gap-[0.625rem] h-full overflow-hidden">
      <TrackPageView page="Prompt Editor" />

      {/* Header bar */}
      <Flex
        className="items-center gap-[0.75rem] px-[1rem] py-[0.625rem] rounded-[0.5rem] flex-shrink-0"
        bg="white"
        style={{ border: `1px solid ${tokens.border}` }}
      >
        <Tooltip label="Back to list" withArrow>
          <ActionIcon
            variant="subtle"
            size="sm"
            onClick={() => {
              if (isDirty && !window.confirm("You have unsaved changes. Leave anyway?")) return;
              router.push("/prompts");
            }}
          >
            ←
          </ActionIcon>
        </Tooltip>

        <TextInput
          value={promptName}
          onChange={(e) => { setPromptName(e.currentTarget.value); setIsDirty(true); }}
          variant="unstyled"
          fw={600}
          size="md"
          className="flex-1 min-w-0"
          styles={{ input: { fontSize: "1rem", fontWeight: 600 } }}
          placeholder="Prompt name"
        />

        {isDirty && (
          <Text size="xs" c={tokens.textMuted}>Unsaved changes</Text>
        )}

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

      {/* Editor */}
      <Stack
        className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] flex-1 !gap-[0.75rem] overflow-hidden"
        bg="white"
        style={{ border: `1px solid ${tokens.border}` }}
      >
        <Text size="sm" c={tokens.textMuted}>
          This prompt is injected as <Text component="span" ff="monospace" size="sm">custom_prompt</Text> during cognify. Use it to guide how the knowledge graph is extracted from your documents.
        </Text>
        <Textarea
          value={content}
          onChange={(e) => { setContent(e.currentTarget.value); setIsDirty(true); }}
          autosize
          minRows={20}
          classNames={{ input: "!font-mono !text-sm !leading-relaxed !border-cognee-border" }}
          radius="0.5rem"
          placeholder="Enter your extraction prompt..."
        />
      </Stack>
    </Stack>
  );
}
