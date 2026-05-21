"use client";

import { useState } from "react";
import {
  ActionIcon,
  Badge,
  Button,
  Flex,
  ScrollArea,
  Stack,
  Table,
  Text,
  Tooltip,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { EntitySchema, FieldInput, FieldSchema, RelationField } from "@/modules/graphModels/types";
import { fieldTypeLabel } from "@/modules/graphModels/types";
import type { SchemaAction } from "@/modules/graphModels/schemaReducer";
import type { ValidationIssue } from "@/modules/graphModels/validator";
import AddEditFieldModal from "./AddEditFieldModal";

interface FieldsTableProps {
  entity: EntitySchema;
  allEntities: EntitySchema[];
  dispatch: React.Dispatch<SchemaAction>;
  issues: ValidationIssue[];
  onCreateEntity: (name: string) => void;
}

function fieldKindBadgeColor(field: FieldSchema): string {
  if (field.kind === "primitive") return "gray";
  if (field.kind === "enum") return "blue";
  return "violet";
}

export default function FieldsTable({
  entity,
  allEntities,
  dispatch,
  issues,
  onCreateEntity,
}: FieldsTableProps) {
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [editField, setEditField] = useState<FieldSchema | null>(null);

  function handleAddField(field: FieldInput) {
    dispatch({ type: "ADD_FIELD", entityId: entity._id, field });
  }

  function handleUpdateField(field: FieldInput) {
    if (!editField) return;
    dispatch({
      type: "UPDATE_FIELD",
      entityId: entity._id,
      fieldId: editField._id,
      field,
    });
    setEditField(null);
  }

  function handleDeleteField(fieldId: string) {
    dispatch({ type: "DELETE_FIELD", entityId: entity._id, fieldId });
  }

  function handleDuplicateField(fieldId: string) {
    dispatch({ type: "DUPLICATE_FIELD", entityId: entity._id, fieldId });
  }

  const entityNames = new Set(allEntities.map((e) => e.name));

  return (
    <Stack gap="0.75rem">
      {entity.fields.length === 0 ? (
        <Flex
          direction="column"
          align="center"
          justify="center"
          className="py-[2rem] border border-dashed rounded-[0.5rem] border-cognee-border"
        >
          <Text size="sm" c={tokens.textMuted} mb="0.75rem">
            No fields yet
          </Text>
          <Button
            size="xs"
            variant="outline"
            onClick={() => setAddModalOpen(true)}
            radius="0.25rem"
          >
            + Add first field
          </Button>
        </Flex>
      ) : (
        <ScrollArea>
          <Table horizontalSpacing="sm" verticalSpacing="xs" highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>
                  <Text size="xs" c={tokens.textMuted} fw={600}>
                    Name
                  </Text>
                </Table.Th>
                <Table.Th>
                  <Text size="xs" c={tokens.textMuted} fw={600}>
                    Type
                  </Text>
                </Table.Th>
                <Table.Th>
                  <Text size="xs" c={tokens.textMuted} fw={600}>
                    Req.
                  </Text>
                </Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {entity.fields.map((field) => {
                const fieldIssues = issues.filter(
                  (i) => i.entityId === entity._id && i.fieldId === field._id
                );
                const isMissingTarget =
                  field.kind === "relation" &&
                  (field as RelationField).relation.targetEntityName &&
                  !entityNames.has((field as RelationField).relation.targetEntityName);

                return (
                  <Table.Tr key={field._id}>
                    <Table.Td>
                      <Flex align="center" gap="0.375rem">
                        <Text size="sm" fw={500}>
                          {field.name || "(unnamed)"}
                        </Text>
                        {fieldIssues.some((i) => i.severity === "error") && (
                          <Badge color="red" variant="light" size="xs">
                            !
                          </Badge>
                        )}
                        {fieldIssues.some((i) => i.severity === "warn") &&
                          !fieldIssues.some((i) => i.severity === "error") && (
                            <Badge color="yellow" variant="light" size="xs">
                              ⚠
                            </Badge>
                          )}
                      </Flex>
                    </Table.Td>
                    <Table.Td>
                      <Stack gap="0.25rem" align="flex-start">
                        <Badge
                          color={fieldKindBadgeColor(field)}
                          variant="light"
                          size="sm"
                        >
                          {fieldTypeLabel(field)}
                        </Badge>
                        {isMissingTarget && (
                          <Flex
                            gap="0.375rem"
                            align="center"
                            className="px-[0.5rem] py-[0.25rem] rounded-[0.25rem]"
                            style={{ backgroundColor: "#fff8e6", border: "1px solid #f0c36d" }}
                          >
                            <Text size="xs" c="#a16207">
                              ⚠ Missing entity
                            </Text>
                            <Button
                              size="xs"
                              variant="subtle"
                              color="orange"
                              onClick={() =>
                                onCreateEntity(
                                  (field as RelationField).relation.targetEntityName
                                )
                              }
                              style={{ padding: "0 4px", height: "1.5rem", fontSize: "11px" }}
                            >
                              Create &quot;{(field as RelationField).relation.targetEntityName}&quot;
                            </Button>
                          </Flex>
                        )}
                      </Stack>
                    </Table.Td>
                    <Table.Td>
                      {field.required ? (
                        <Badge color="red" variant="dot" size="sm">
                          yes
                        </Badge>
                      ) : (
                        <Text size="xs" c={tokens.textMuted}>
                          —
                        </Text>
                      )}
                    </Table.Td>
                    <Table.Td>
                      <Flex gap="0.125rem" justify="flex-end">
                        <Tooltip label="Edit" withArrow>
                          <ActionIcon
                            variant="subtle"
                            size="xs"
                            onClick={() => setEditField(field)}
                          >
                            ✏️
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Duplicate" withArrow>
                          <ActionIcon
                            variant="subtle"
                            size="xs"
                            onClick={() => handleDuplicateField(field._id)}
                          >
                            ⧉
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete" withArrow>
                          <ActionIcon
                            variant="subtle"
                            color="red"
                            size="xs"
                            onClick={() => handleDeleteField(field._id)}
                          >
                            ✕
                          </ActionIcon>
                        </Tooltip>
                      </Flex>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      )}

      <Button
        size="xs"
        variant="outline"
        onClick={() => setAddModalOpen(true)}
        radius="0.25rem"
        className="self-start"
      >
        + Add Field
      </Button>

      {/* Add modal */}
      <AddEditFieldModal
        opened={addModalOpen}
        onClose={() => setAddModalOpen(false)}
        onSubmit={handleAddField}
        allEntities={allEntities}
        currentEntityId={entity._id}
      />

      {/* Edit modal */}
      {editField && (
        <AddEditFieldModal
          opened={!!editField}
          onClose={() => setEditField(null)}
          onSubmit={handleUpdateField}
          existingField={editField}
          allEntities={allEntities}
          currentEntityId={entity._id}
        />
      )}
    </Stack>
  );
}
