"use client";

import { useState } from "react";
import {
  ActionIcon,
  Badge,
  Button,
  Flex,
  ScrollArea,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { EntitySchema } from "@/modules/graphModels/types";
import type { ValidationIssue } from "@/modules/graphModels/validator";
import { issueCountForEntity } from "@/modules/graphModels/validator";

interface EntitiesPanelProps {
  entities: EntitySchema[];
  selectedEntityId: string | null;
  issues: ValidationIssue[];
  onSelect: (entityId: string) => void;
  onAdd: () => void;
  onDuplicate: (entityId: string) => void;
  onDelete: (entityId: string) => void;
}

export default function EntitiesPanel({
  entities,
  selectedEntityId,
  issues,
  onSelect,
  onAdd,
  onDuplicate,
  onDelete,
}: EntitiesPanelProps) {
  const [search, setSearch] = useState("");

  const filtered = entities.filter((e) =>
    e.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <Stack
      className="rounded-[0.5rem] !gap-[0] overflow-hidden"
      bg="white"
      style={{ border: `1px solid ${tokens.border}` }}
    >
      {/* Header */}
      <Flex
        className="justify-between items-center px-[1rem] py-[0.75rem]"
        style={{ borderBottom: `1px solid ${tokens.borderLight}` }}
      >
        <Text fw={600} size="sm">
          Entities
          {entities.length > 0 && (
            <Text component="span" c={tokens.textMuted} size="xs" ml="0.25rem">
              ({entities.length})
            </Text>
          )}
        </Text>
        <Button
          size="xs"
          onClick={onAdd}
          style={{ backgroundColor: tokens.purple }}
          radius="0.25rem"
        >
          + New Entity
        </Button>
      </Flex>

      {/* Search */}
      <Flex className="px-[0.75rem] py-[0.5rem]" style={{ borderBottom: `1px solid ${tokens.borderLight}` }}>
        <TextInput
          size="xs"
          placeholder="Search entities…"
          value={search}
          onChange={(e) => setSearch(e.currentTarget.value)}
          className="w-full"
          classNames={{ input: "!border-cognee-border" }}
          radius="0.25rem"
        />
      </Flex>

      {/* Entity list */}
      <ScrollArea style={{ maxHeight: "240px" }}>
        {filtered.length === 0 ? (
          <Flex align="center" justify="center" className="py-[1.5rem]">
            <Text size="xs" c={tokens.textMuted}>
              {entities.length === 0 ? "No entities yet" : "No results"}
            </Text>
          </Flex>
        ) : (
          <Stack gap="0">
            {filtered.map((entity) => {
              const { errors, warnings } = issueCountForEntity(issues, entity._id);
              const isSelected = entity._id === selectedEntityId;

              return (
                <Flex
                  key={entity._id}
                  className={`items-center justify-between px-[0.75rem] py-[0.5rem] cursor-pointer hover:bg-cognee-hover ${
                    isSelected ? "bg-cognee-hover" : ""
                  }`}
                  style={
                    isSelected
                      ? { borderLeft: `3px solid ${tokens.purple}` }
                      : { borderLeft: "3px solid transparent" }
                  }
                  onClick={() => onSelect(entity._id)}
                >
                  <Flex align="center" gap="0.5rem" className="flex-1 min-w-0">
                    <Text size="sm" fw={isSelected ? 600 : 400} className="truncate">
                      {entity.name || "(unnamed)"}
                    </Text>
                    {errors > 0 && (
                      <Badge color="red" variant="light" size="xs">
                        {errors}
                      </Badge>
                    )}
                    {warnings > 0 && errors === 0 && (
                      <Badge color="yellow" variant="light" size="xs">
                        {warnings}
                      </Badge>
                    )}
                  </Flex>
                  <Flex gap="0.125rem" onClick={(e) => e.stopPropagation()}>
                    <Tooltip label="Duplicate" withArrow position="top">
                      <ActionIcon
                        variant="subtle"
                        size="xs"
                        onClick={() => onDuplicate(entity._id)}
                      >
                        ⧉
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Delete" withArrow position="top">
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        size="xs"
                        onClick={() => onDelete(entity._id)}
                      >
                        ✕
                      </ActionIcon>
                    </Tooltip>
                  </Flex>
                </Flex>
              );
            })}
          </Stack>
        )}
      </ScrollArea>
    </Stack>
  );
}
