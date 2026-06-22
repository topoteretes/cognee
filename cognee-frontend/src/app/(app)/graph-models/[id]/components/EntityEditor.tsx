"use client";

import {
  Checkbox,
  Flex,
  MultiSelect,
  ScrollArea,
  Select,
  Stack,
  Tabs,
  Text,
  TextInput,
  Textarea,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { EntitySchema } from "@/modules/graphModels/types";
import type { SchemaAction } from "@/modules/graphModels/schemaReducer";
import type { ValidationIssue } from "@/modules/graphModels/validator";
import FieldsTable from "./FieldsTable";

interface EntityEditorProps {
  entity: EntitySchema;
  allEntities: EntitySchema[];
  dispatch: React.Dispatch<SchemaAction>;
  issues: ValidationIssue[];
  onCreateEntity: (name: string) => void;
}

export default function EntityEditor({
  entity,
  allEntities,
  dispatch,
  issues,
  onCreateEntity,
}: EntityEditorProps) {
  const primitiveFields = entity.fields
    .filter((f) => f.kind === "primitive")
    .map((f) => ({ value: f.name, label: f.name }));

  return (
    <Stack
      className="rounded-[0.5rem] flex-1 overflow-hidden !gap-[0]"
      bg="white"
      style={{ border: `1px solid ${tokens.border}` }}
    >
      {/* Entity name + description header */}
      <Stack
        gap="0.5rem"
        className="px-[1rem] py-[0.75rem]"
        style={{ borderBottom: `1px solid ${tokens.borderLight}` }}
      >
        <Flex gap="0.5rem" align="flex-end">
          <TextInput
            label="Entity name"
            value={entity.name}
            onChange={(e) =>
              dispatch({
                type: "UPDATE_ENTITY",
                entityId: entity._id,
                updates: { name: e.currentTarget.value },
              })
            }
            classNames={{ input: "!h-[2.25rem] !border-cognee-border" }}
            radius="0.5rem"
            size="sm"
            className="flex-1"
          />
        </Flex>
        <Textarea
          label="Description"
          placeholder="Optional description"
          value={entity.description ?? ""}
          onChange={(e) =>
            dispatch({
              type: "UPDATE_ENTITY",
              entityId: entity._id,
              updates: { description: e.currentTarget.value || undefined },
            })
          }
          classNames={{ input: "!border-cognee-border" }}
          radius="0.5rem"
          size="sm"
          rows={2}
          autosize
          minRows={1}
          maxRows={3}
        />
      </Stack>

      <Tabs defaultValue="fields" className="flex flex-col flex-1 overflow-hidden">
        <Tabs.List px="1rem" style={{ borderBottom: `1px solid ${tokens.borderLight}` }}>
          <Tabs.Tab value="fields" size="sm">
            Fields
            <Text component="span" size="xs" c={tokens.textMuted} ml="0.25rem">
              ({entity.fields.length})
            </Text>
          </Tabs.Tab>
          <Tabs.Tab value="indexing" size="sm">
            Indexing
          </Tabs.Tab>
        </Tabs.List>

        {/* Fields tab */}
        <Tabs.Panel value="fields" className="flex-1 overflow-hidden">
          <ScrollArea className="h-full" p="0.75rem">
            <FieldsTable
              entity={entity}
              allEntities={allEntities}
              dispatch={dispatch}
              issues={issues}
              onCreateEntity={onCreateEntity}
            />
          </ScrollArea>
        </Tabs.Panel>

        {/* Indexing tab */}
        <Tabs.Panel value="indexing">
          <ScrollArea p="0.75rem">
            <Stack gap="1rem">
              <Select
                label="Primary label field"
                description='The field used as the node label. Defaults to "name" if present.'
                placeholder='Select field (default: "name")'
                data={primitiveFields}
                value={entity.primaryLabelField ?? null}
                onChange={(v) =>
                  dispatch({
                    type: "UPDATE_ENTITY",
                    entityId: entity._id,
                    updates: { primaryLabelField: v ?? undefined },
                  })
                }
                classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
                radius="0.5rem"
                clearable
              />

              <Stack gap="0.5rem">
                <Text size="sm" fw={500}>
                  Index fields
                </Text>
                <Text size="xs" c={tokens.textMuted}>
                  Primitive fields that should be indexed for fast lookups.
                </Text>
                {primitiveFields.length === 0 ? (
                  <Text size="xs" c={tokens.textMuted}>
                    Add primitive fields first.
                  </Text>
                ) : (
                  <MultiSelect
                    data={primitiveFields}
                    value={entity.indexFields ?? []}
                    onChange={(v) =>
                      dispatch({
                        type: "UPDATE_ENTITY",
                        entityId: entity._id,
                        updates: { indexFields: v },
                      })
                    }
                    placeholder="Select fields to index"
                    classNames={{ input: "!border-cognee-border" }}
                    radius="0.5rem"
                  />
                )}
              </Stack>

              <Stack gap="0.5rem">
                <Text size="sm" fw={500}>
                  Auto-type nodes
                </Text>
                <Checkbox
                  label="Auto-type nodes for this entity"
                  checked={false}
                  disabled
                  size="sm"
                />
                <Text size="xs" c={tokens.textMuted}>
                  Per-entity auto-type override (schema-level option in Advanced).
                </Text>
              </Stack>
            </Stack>
          </ScrollArea>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}
