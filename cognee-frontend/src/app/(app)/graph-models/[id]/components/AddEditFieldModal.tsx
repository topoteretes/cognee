"use client";

import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Checkbox,
  Flex,
  Modal,
  Select,
  Stack,
  Text,
  TextInput,
  Textarea,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { EntitySchema, FieldInput, FieldSchema, PrimitiveType } from "@/modules/graphModels/types";

type FieldKind = "primitive" | "enum" | "relation";

const PRIMITIVE_OPTIONS: { value: PrimitiveType; label: string }[] = [
  { value: "string", label: "Text (string)" },
  { value: "number", label: "Number" },
  { value: "boolean", label: "Boolean" },
  { value: "date", label: "Date" },
];

const KIND_OPTIONS = [
  { value: "primitive", label: "Primitive" },
  { value: "enum", label: "Enum" },
  { value: "relation_one", label: "Relation (one)" },
  { value: "relation_many", label: "Relation (many)" },
];

interface AddEditFieldModalProps {
  opened: boolean;
  onClose: () => void;
  onSubmit: (field: FieldInput) => void;
  existingField?: FieldSchema;
  allEntities: EntitySchema[];
  currentEntityId: string;
}

export default function AddEditFieldModal({
  opened,
  onClose,
  onSubmit,
  existingField,
  allEntities,
  currentEntityId,
}: AddEditFieldModalProps) {
  const isEdit = !!existingField;

  // ── Shared state ────────────────────────────────────────────────────────────
  const [name, setName] = useState("");
  const [nameError, setNameError] = useState("");
  const [description, setDescription] = useState("");
  const [required, setRequired] = useState(false);
  const [kindSelect, setKindSelect] = useState<string>("primitive");

  // Primitive
  const [primitiveType, setPrimitiveType] = useState<PrimitiveType>("string");

  // Enum
  const [enumValues, setEnumValues] = useState<string[]>([]);
  const [enumInput, setEnumInput] = useState("");

  // Relation
  const [targetEntityName, setTargetEntityName] = useState("");
  const [inverseEnabled, setInverseEnabled] = useState(false);
  const [inverseName, setInverseName] = useState("");
  const [inverseCardinality, setInverseCardinality] = useState<"one" | "many">("many");

  // Populate for edit
  useEffect(() => {
    if (!existingField || !opened) return;
    setName(existingField.name);
    setDescription(existingField.description ?? "");
    setRequired(existingField.required ?? false);

    if (existingField.kind === "primitive") {
      setKindSelect("primitive");
      setPrimitiveType(existingField.primitiveType);
    } else if (existingField.kind === "enum") {
      setKindSelect("enum");
      setEnumValues(existingField.enumValues);
    } else {
      setKindSelect(existingField.relation.cardinality === "many" ? "relation_many" : "relation_one");
      setTargetEntityName(existingField.relation.targetEntityName);
      const inv = existingField.relation.inverse;
      setInverseEnabled(inv?.enabled ?? false);
      setInverseName(inv?.name ?? "");
      setInverseCardinality(inv?.cardinality ?? "many");
    }
  }, [existingField, opened]);

  function reset() {
    setName("");
    setNameError("");
    setDescription("");
    setRequired(false);
    setKindSelect("primitive");
    setPrimitiveType("string");
    setEnumValues([]);
    setEnumInput("");
    setTargetEntityName("");
    setInverseEnabled(false);
    setInverseName("");
    setInverseCardinality("many");
  }

  function handleClose() {
    reset();
    onClose();
  }

  function handleSubmit() {
    const trimmedName = name.trim();
    if (!trimmedName) {
      setNameError("Field name is required.");
      return;
    }
    if (!/^[a-z][a-z0-9_]*$/.test(trimmedName)) {
      setNameError("Must be snake_case (e.g. full_name).");
      return;
    }

    let field: FieldInput;

    if (kindSelect === "primitive") {
      field = {
        name: trimmedName,
        kind: "primitive" as const,
        primitiveType,
        required: required || undefined,
        description: description.trim() || undefined,
      };
    } else if (kindSelect === "enum") {
      field = {
        name: trimmedName,
        kind: "enum" as const,
        enumValues,
        required: required || undefined,
        description: description.trim() || undefined,
      };
    } else {
      const cardinality: "one" | "many" = kindSelect === "relation_many" ? "many" : "one";
      field = {
        name: trimmedName,
        kind: "relation" as const,
        relation: {
          targetEntityName: targetEntityName.trim(),
          cardinality,
          inverse: inverseEnabled
            ? {
                enabled: true,
                name: inverseName.trim() || undefined,
                cardinality: inverseCardinality,
              }
            : { enabled: false },
        },
        required: required || undefined,
        description: description.trim() || undefined,
      };
    }

    onSubmit(field);
    handleClose();
  }

  // Entity autocomplete options (excluding self)
  const entityOptions = allEntities
    .filter((e) => e._id !== currentEntityId)
    .map((e) => ({ value: e.name, label: e.name }));

  const missingTarget =
    (kindSelect === "relation_one" || kindSelect === "relation_many") &&
    targetEntityName.trim() &&
    !allEntities.some((e) => e.name === targetEntityName.trim());

  function addEnumValue() {
    const val = enumInput.trim();
    if (val && !enumValues.includes(val)) {
      setEnumValues((prev) => [...prev, val]);
    }
    setEnumInput("");
  }

  return (
    <Modal
      opened={opened}
      onClose={handleClose}
      title={isEdit ? "Edit Field" : "Add Field"}
      centered
      radius="0.5rem"
      size="md"
    >
      <Stack gap="0.75rem">
        {/* Name + Kind row */}
        <Flex gap="0.75rem">
          <TextInput
            label="Field name"
            placeholder="e.g. full_name"
            description="snake_case"
            value={name}
            onChange={(e) => {
              setName(e.currentTarget.value);
              setNameError("");
            }}
            error={nameError}
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
            className="flex-1"
            autoFocus
          />
          <Select
            label="Type"
            data={KIND_OPTIONS}
            value={kindSelect}
            onChange={(v) => v && setKindSelect(v)}
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
            className="w-[180px]"
          />
        </Flex>

        {/* Primitive options */}
        {kindSelect === "primitive" && (
          <Select
            label="Primitive type"
            data={PRIMITIVE_OPTIONS}
            value={primitiveType}
            onChange={(v) => v && setPrimitiveType(v as PrimitiveType)}
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
          />
        )}

        {/* Enum options */}
        {kindSelect === "enum" && (
          <Stack gap="0.5rem">
            <Text size="sm" fw={500}>
              Enum values
            </Text>
            <Flex gap="0.5rem" wrap="wrap">
              {enumValues.map((v) => (
                <Badge
                  key={v}
                  variant="light"
                  color="blue"
                  style={{ cursor: "pointer" }}
                  onClick={() => setEnumValues((prev) => prev.filter((x) => x !== v))}
                >
                  {v} ✕
                </Badge>
              ))}
              {enumValues.length === 0 && (
                <Text size="xs" c={tokens.textMuted}>
                  No values yet
                </Text>
              )}
            </Flex>
            <Flex gap="0.5rem">
              <TextInput
                placeholder="Add value…"
                value={enumInput}
                onChange={(e) => setEnumInput(e.currentTarget.value)}
                onKeyDown={(e) => e.key === "Enter" && addEnumValue()}
                classNames={{ input: "!h-[2.25rem] !border-cognee-border" }}
                radius="0.5rem"
                className="flex-1"
                size="sm"
              />
              <Button
                size="sm"
                variant="default"
                onClick={addEnumValue}
                disabled={!enumInput.trim()}
                radius="0.5rem"
              >
                Add
              </Button>
            </Flex>
            <Text size="xs" c={tokens.textMuted}>
              Press Enter or click Add to add a value. Click a badge to remove it.
            </Text>
          </Stack>
        )}

        {/* Relation options */}
        {(kindSelect === "relation_one" || kindSelect === "relation_many") && (
          <Stack gap="0.75rem">
            {/* Free-text input with browser datalist for entity autocomplete */}
            <div>
              <Text size="sm" fw={500} mb="0.25rem">
                Target entity
              </Text>
              <input
                list="entity-autocomplete-options"
                value={targetEntityName}
                onChange={(e) => setTargetEntityName(e.currentTarget.value)}
                placeholder="Select or type entity name"
                className="w-full h-[2.75rem] px-[0.75rem] rounded-[0.5rem] text-sm border border-cognee-border outline-none focus:border-cognee-purple"
                style={{ border: `1px solid #D8D8D8` }}
              />
              <datalist id="entity-autocomplete-options">
                {entityOptions.map((o) => (
                  <option key={o.value} value={o.value} />
                ))}
              </datalist>
            </div>

            {missingTarget && (
              <Flex
                className="items-center gap-2 px-[0.75rem] py-[0.5rem] rounded-[0.5rem]"
                style={{ backgroundColor: "#fff8e6", border: "1px solid #f0c36d" }}
              >
                <Text size="xs" c="#a16207">
                  ⚠ Entity &quot;{targetEntityName}&quot; does not exist yet.
                </Text>
              </Flex>
            )}

            <Checkbox
              label="Enable inverse relation"
              checked={inverseEnabled}
              onChange={(e) => setInverseEnabled(e.currentTarget.checked)}
              size="sm"
            />

            {inverseEnabled && (
              <Flex gap="0.75rem">
                <TextInput
                  label="Inverse field name"
                  placeholder="e.g. authored_by"
                  value={inverseName}
                  onChange={(e) => setInverseName(e.currentTarget.value)}
                  classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
                  radius="0.5rem"
                  className="flex-1"
                />
                <Select
                  label="Inverse cardinality"
                  data={[
                    { value: "one", label: "One" },
                    { value: "many", label: "Many" },
                  ]}
                  value={inverseCardinality}
                  onChange={(v) => v && setInverseCardinality(v as "one" | "many")}
                  classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
                  radius="0.5rem"
                  className="w-[130px]"
                />
              </Flex>
            )}
          </Stack>
        )}

        {/* Shared: Description + Required */}
        <Textarea
          label="Description"
          placeholder="Optional description"
          value={description}
          onChange={(e) => setDescription(e.currentTarget.value)}
          classNames={{ input: "!border-cognee-border" }}
          radius="0.5rem"
          rows={2}
        />

        <Checkbox
          label="Required"
          checked={required}
          onChange={(e) => setRequired(e.currentTarget.checked)}
          size="sm"
        />

        <Flex justify="flex-end" gap="0.5rem" mt="0.25rem">
          <Button variant="default" onClick={handleClose} radius="0.5rem">
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            style={{ backgroundColor: tokens.purple }}
            radius="0.5rem"
          >
            {isEdit ? "Save Changes" : "Add Field"}
          </Button>
        </Flex>
      </Stack>
    </Modal>
  );
}
