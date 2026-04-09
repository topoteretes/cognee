"use client";

import { useRouter } from "next/navigation";
import {
  Divider,
  Flex,
  Select,
  Text,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { Dataset } from "@/modules/ingestion/useDatasets";
import type { Prompt } from "@/modules/prompts/storage";
import DatasetSelect from "@/ui/elements/Widgets/elements/DatasetSelect";

interface DashboardControlPanelProps {
  datasets: Dataset[];
  selectedDatasetId: string | null;
  onDatasetChange: (id: string | null) => void;
  addDataset: (name: string) => Promise<void>;
  refreshDatasets: () => Promise<Dataset[]>;
  prompts: Prompt[];
  activePromptId: string | null;
  onPromptChange: (id: string | null) => void;
  llmModel: string;
  onLlmModelChange: (model: string) => void;
}

const SELECT_STYLES = {
  input: {
    borderColor: "#d9d9d9",
    borderRadius: "0.5rem",
  },
  dropdown: {
    borderRadius: "0.5rem",
    boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
  },
};

export default function DashboardControlPanel({
  datasets,
  selectedDatasetId,
  onDatasetChange,
  addDataset,
  refreshDatasets,
  prompts,
  activePromptId,
  onPromptChange,
  llmModel,
  onLlmModelChange,
}: DashboardControlPanelProps) {
  const router = useRouter();

  return (
    <Flex
      align="center"
      gap="1.25rem"
      wrap="wrap"
      className="rounded-[0.5rem] px-[2rem] py-[1.125rem]"
      bg="white"
      style={{ border: `1px solid ${tokens.border}` }}
    >
      {/* ── Dataset ─────────────────────────────────────────────────────── */}
      <DatasetSelect
        label="Dataset"
        datasets={datasets}
        addDataset={addDataset}
        refreshDatasets={refreshDatasets}
        selectedValue={selectedDatasetId}
        onChange={onDatasetChange}
        hasAdd
      />

      <Divider orientation="vertical" style={{ height: 24 }} />

      {/* ── Graph Model ─────────────────────────────────────────────────── */}
      <Flex align="center" gap="0.5rem">
        <Text size="sm" c={tokens.textMuted} className="whitespace-nowrap">
          Graph Model
        </Text>
        <Select
          allowDeselect={false}
          data={[
            { value: "", label: "Automatic" },
            { value: "__new__", label: "+ Create New" },
          ]}
          value=""
          size="sm"
          w={200}
          styles={SELECT_STYLES}
          renderOption={({ option }) =>
            option.value === "__new__" ? (
              <div>
                <div style={{ color: tokens.purple, fontWeight: 500 }}>{option.label}</div>
                <div style={{ fontSize: "0.7rem", color: tokens.purple, lineHeight: 1.2 }}>
                  Coming soon
                </div>
              </div>
            ) : (
              <span>{option.label}</span>
            )
          }
          onChange={() => {}}
        />
      </Flex>

      <Divider orientation="vertical" style={{ height: 24 }} />

      {/* ── Model ───────────────────────────────────────────────────────── */}
      <Flex align="center" gap="0.5rem">
        <Text size="sm" c={tokens.textMuted} className="whitespace-nowrap">
          Model
        </Text>
        <Select
          allowDeselect={false}
          data={[
            { value: "gpt-5-mini", label: "OpenAI GPT-5-mini" },
            { value: "gpt-4o", label: "OpenAI GPT-4o" },
            { value: "gpt-4o-mini", label: "OpenAI GPT-4o-mini" },
            { value: "claude", label: "Anthropic Claude" },
            { value: "custom", label: "Custom" },
          ]}
          value={llmModel}
          size="sm"
          w={180}
          styles={SELECT_STYLES}
          renderOption={({ option }) =>
            option.value === "custom" ? (
              <div>
                <div>{option.label}</div>
                <div style={{ fontSize: "0.7rem", color: tokens.purple, lineHeight: 1.2 }}>
                  Coming soon
                </div>
              </div>
            ) : (
              <span>{option.label}</span>
            )
          }
          onChange={(val) => {
            if (val && val !== "custom") onLlmModelChange(val);
          }}
        />
      </Flex>

      <Divider orientation="vertical" style={{ height: 24 }} />

      {/* ── Prompt ──────────────────────────────────────────────────────── */}
      <Flex align="center" gap="0.5rem">
        <Text size="sm" c={tokens.textMuted} className="whitespace-nowrap">
          Prompt
        </Text>
        <Select
          placeholder="Default"
          allowDeselect={false}
          data={[
            { value: "", label: "Default" },
            ...prompts.map((p) => ({ value: p.id, label: p.name })),
            { value: "__new__", label: "+ Create New" },
          ]}
          value={activePromptId ?? ""}
          size="sm"
          w={180}
          styles={SELECT_STYLES}
          renderOption={({ option }) =>
            option.value === "__new__" ? (
              <span style={{ color: tokens.purple, fontWeight: 500 }}>{option.label}</span>
            ) : (
              <span>{option.label}</span>
            )
          }
          onChange={(val) => {
            if (val === "__new__") {
              router.push("/prompts");
              return;
            }
            onPromptChange(val || null);
          }}
        />
      </Flex>

      {/* ── Spacer ──────────────────────────────────────────────────────── */}
      <div style={{ flex: 1 }} />
    </Flex>
  );
}
