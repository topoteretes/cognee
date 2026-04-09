"use client";

import { useState } from "react";
import { Button, Flex, Modal, Stack, Text, TextInput, Textarea } from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";

interface AddEntityModalProps {
  opened: boolean;
  onClose: () => void;
  onSubmit: (name: string, description?: string) => void;
}

export default function AddEntityModal({ opened, onClose, onSubmit }: AddEntityModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [nameError, setNameError] = useState("");

  function handleSubmit() {
    const trimmed = name.trim();
    if (!trimmed) {
      setNameError("Name is required.");
      return;
    }
    if (!/^[A-Z][a-zA-Z0-9]*$/.test(trimmed)) {
      setNameError("Must be PascalCase (e.g. PersonEntity).");
      return;
    }
    onSubmit(trimmed, description.trim() || undefined);
    handleClose();
  }

  function handleClose() {
    setName("");
    setDescription("");
    setNameError("");
    onClose();
  }

  return (
    <Modal opened={opened} onClose={handleClose} title="New Entity" centered radius="0.5rem">
      <Stack gap="0.75rem">
        <TextInput
          label="Entity name"
          placeholder="e.g. Person"
          description="Must be PascalCase"
          value={name}
          onChange={(e) => {
            setName(e.currentTarget.value);
            setNameError("");
          }}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          error={nameError}
          classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
          radius="0.5rem"
          autoFocus
        />
        <Textarea
          label="Description"
          placeholder="Optional description"
          value={description}
          onChange={(e) => setDescription(e.currentTarget.value)}
          classNames={{ input: "!border-cognee-border" }}
          radius="0.5rem"
          rows={2}
        />
        <Text size="xs" c={tokens.textMuted}>
          Tip: Entity names should be singular PascalCase nouns (Person, Document, Company).
        </Text>
        <Flex justify="flex-end" gap="0.5rem" mt="0.25rem">
          <Button variant="default" onClick={handleClose} radius="0.5rem">
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!name.trim()}
            style={{ backgroundColor: tokens.purple }}
            radius="0.5rem"
          >
            Add Entity
          </Button>
        </Flex>
      </Stack>
    </Modal>
  );
}
