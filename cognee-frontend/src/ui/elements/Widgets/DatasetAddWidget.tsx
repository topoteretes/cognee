import {
  Stack,
  Title,
  Text,
  Flex,
  Button,
  TextInput,
} from "@mantine/core";

import LogSection from "./elements/LogSection";
import { CogneeInstance } from "@/modules/instances/types";
import { tokens } from "@/ui/theme/tokens";
import { useCallback, useState, KeyboardEvent } from "react";
import { Dropzone } from "@mantine/dropzone";
import addData, { addUrlData } from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import { useToggle } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { trackEvent } from "@/modules/analytics";

interface DatasetAddWidgetProps {
  selectedDatasetId: string | null;
  refreshDatasets: () => Promise<unknown>;
  instance: CogneeInstance;
  onCognifyStart?: () => void;
  onCognifyComplete?: () => void;
  customPrompt?: string;
  llmModel?: string;
  activeGraphModelSchema?: object;
  activePromptName?: string;
}

export default function DatasetAddWidget({
  selectedDatasetId,
  refreshDatasets,
  instance,
  onCognifyStart,
  onCognifyComplete,
  customPrompt,
  llmModel,
  activeGraphModelSchema,
  activePromptName,
}: DatasetAddWidgetProps) {
  const [uploadedFiles, setUploadedFiles] = useState<Map<string, Array<File>>>(
    new Map(),
  );
  const [uploadingInProgress, setUploadingInProgress] = useState<boolean>(false);

  const handleAddFiles = useCallback(
    (payload: File[] | null) => {
      setUploadingInProgress(true);
      if (payload === null || selectedDatasetId === null) {
        setUploadingInProgress(false);
        return;
      }
      setUploadedFiles((prevMap) => {
        const newFileMap = new Map(prevMap);
        const existingFiles = newFileMap.get(selectedDatasetId) || [];
        newFileMap.set(selectedDatasetId, [...existingFiles, ...payload]);
        return newFileMap;
      });
      trackEvent({
        pageName: "Dashboard",
        eventName: "upload_data",
        additionalProperties: {
          type: "file",
          file_count: String(payload.length),
          dataset_id: selectedDatasetId,
        },
      });
      setUploadingInProgress(false);
    },
    [selectedDatasetId],
  );

  const removeFile = (key: string) => {
    setUploadedFiles((prevMap) => {
      const newMap = new Map(prevMap);
      newMap.delete(key);
      return newMap;
    });
  };

  const [addingFiles, toggleAddingFiles] = useToggle();

  const addFilesToDataset = useCallback(() => {
    if (selectedDatasetId === null) return;
    trackEvent({
      pageName: "Dashboard",
      eventName: "add_data",
      additionalProperties: { type: "file", dataset_id: selectedDatasetId },
    });
    toggleAddingFiles();
    onCognifyStart?.();
    addData(
      { id: selectedDatasetId },
      uploadedFiles.get(selectedDatasetId) ?? [],
      instance,
    )
      .then(({ dataset_id, dataset_name }) => {
        trackEvent({ pageName: "Dashboard", eventName: "files_uploaded", additionalProperties: { dataset_id, dataset_name, file_count: String(uploadedFiles.get(selectedDatasetId)?.length ?? 0) } });
        refreshDatasets();
        removeFile(selectedDatasetId);
        toggleAddingFiles();
        notifications.show({
          title: "Files successfully added to dataset",
          message: "",
          color: "green",
        });
        cognifyDataset(
          { id: dataset_id, name: dataset_name, data: [], status: "" },
          instance,
          {
            graphModel: activeGraphModelSchema,
            customPrompt,
            llmModel,
          },
        )
          .then(() => {
            trackEvent({ pageName: "Dashboard", eventName: "cognify_completed", additionalProperties: { dataset_id, dataset_name, model: llmModel ?? "", prompt: activePromptName ?? "" } });
            onCognifyComplete?.();
          })
          .catch(() => {
            trackEvent({ pageName: "Dashboard", eventName: "cognify_failed", additionalProperties: { dataset_id, dataset_name, model: llmModel ?? "", prompt: activePromptName ?? "" } });
            onCognifyComplete?.();
          });
      })
      .catch((resp) => {
        toggleAddingFiles();
        onCognifyComplete?.();
        notifications.show({
          title: "Something went wrong while adding to dataset",
          message: resp instanceof Error ? resp.message : String(resp),
          color: "red",
        });
      });
  }, [instance, refreshDatasets, selectedDatasetId, toggleAddingFiles, uploadedFiles, onCognifyStart, onCognifyComplete, customPrompt, llmModel, activeGraphModelSchema, activePromptName]);

  const [showImportUrl, toggleShowImportUrl] = useToggle();
  const [importUrl, setImportUrl] = useState<string>("");

  const importUrlOnEnterKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") importUrlToDataset();
  };

  const importUrlToDataset = useCallback(() => {
    if (selectedDatasetId === null) return;
    trackEvent({
      pageName: "Dashboard",
      eventName: "upload_data",
      additionalProperties: { type: "text", dataset_id: selectedDatasetId },
    });
    trackEvent({
      pageName: "Dashboard",
      eventName: "add_data",
      additionalProperties: { type: "text", dataset_id: selectedDatasetId },
    });
    toggleAddingFiles();
    onCognifyStart?.();
    addUrlData({ id: selectedDatasetId }, importUrl, instance)
      .then(({ dataset_id, dataset_name }) => {
        trackEvent({ pageName: "Dashboard", eventName: "url_ingested", additionalProperties: { dataset_id, dataset_name } });
        refreshDatasets();
        setImportUrl("");
        toggleAddingFiles();
        notifications.show({
          title: "Text successfully added to dataset",
          message: "",
          color: "green",
        });
        cognifyDataset(
          { id: dataset_id, name: dataset_name, data: [], status: "" },
          instance,
          {
            graphModel: activeGraphModelSchema,
            customPrompt,
            llmModel,
          },
        )
          .then(() => {
            trackEvent({ pageName: "Dashboard", eventName: "cognify_completed", additionalProperties: { dataset_id, dataset_name, model: llmModel ?? "", prompt: activePromptName ?? "" } });
            onCognifyComplete?.();
          })
          .catch(() => {
            trackEvent({ pageName: "Dashboard", eventName: "cognify_failed", additionalProperties: { dataset_id, dataset_name, model: llmModel ?? "", prompt: activePromptName ?? "" } });
            onCognifyComplete?.();
          });
      })
      .catch((resp) => {
        toggleAddingFiles();
        onCognifyComplete?.();
        notifications.show({
          title: "Something went wrong while adding to dataset",
          message: resp instanceof Error ? resp.message : String(resp),
          color: "red",
        });
      });
  }, [importUrl, instance, refreshDatasets, selectedDatasetId, toggleAddingFiles, onCognifyStart, onCognifyComplete, customPrompt, llmModel, activeGraphModelSchema, activePromptName]);

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
      bg="white"
    >
      <Stack className="!gap-[0]" mb="1.25rem">
        <Title size="h2" mb="0.125rem">
          Add data
        </Title>
        <Text c={tokens.textMuted} size="lg">
          Upload documents to extract entities, relationships, and concepts
          into a searchable knowledge graph
        </Text>
      </Stack>

      <Stack mb="1rem">
        <Dropzone
          onDrop={handleAddFiles}
          loading={uploadingInProgress}
          disabled={selectedDatasetId === null}
          opacity={selectedDatasetId === null ? "0.5" : "1"}
        >
          <Flex
            className="justify-center items-center h-[5rem] rounded-[0.5rem]"
            style={{
              backgroundColor: "rgba(92, 16, 244, 0.08)",
              border: "1px dashed rgba(92, 16, 244, 0.4)",
              cursor: selectedDatasetId === null ? "not-allowed" : "pointer",
            }}
          >
            <Text size="md" c="primary2.6" fw={500}>
              ↑ Drop files here or browse
            </Text>
          </Flex>
        </Dropzone>

        <Flex gap="0.625rem">
          <Flex
            flex="1"
            className="justify-center items-center h-[2.5rem] border rounded-[0.5rem] border-[#757470]"
            onClick={() => toggleShowImportUrl()}
            opacity={selectedDatasetId === null ? "0.5" : "1"}
            style={{ cursor: selectedDatasetId === null ? "not-allowed" : "pointer" }}
          >
            <Text size="md" c="secondary3.7">
              Add text
            </Text>
          </Flex>
          <Flex
            flex="1"
            className="justify-center items-center h-[2.5rem] border rounded-[0.5rem] border-secondary-3"
            opacity="0.5"
            style={{ cursor: "not-allowed" }}
          >
            <Text size="md" c="secondary3.2">
              Connect storage
            </Text>
          </Flex>
        </Flex>
      </Stack>

      {selectedDatasetId !== null &&
        ((uploadedFiles.get(selectedDatasetId)?.length ?? 0) > 0 ||
          (showImportUrl && importUrl !== "")) && (
          <LogSection
            items={[
              ...(uploadedFiles.get(selectedDatasetId)?.map(({ name }) => name) ?? []),
              ...(showImportUrl && importUrl !== "" ? [importUrl] : []),
            ]}
          />
        )}

      {selectedDatasetId !== null && showImportUrl && (
        <TextInput
          disabled={addingFiles}
          placeholder="Paste or enter text"
          value={importUrl}
          onChange={(e) => setImportUrl(e.target.value)}
          onKeyDown={importUrlOnEnterKeyDown}
        />
      )}

      <Button
        disabled={
          (importUrl === "" &&
            (selectedDatasetId === null ||
              uploadedFiles.get(selectedDatasetId) === undefined)) ||
          addingFiles
        }
        color="primary2.6"
        mt="1rem"
        onClick={showImportUrl && importUrl !== "" ? importUrlToDataset : addFilesToDataset}
      >
        Add data
      </Button>
    </Stack>
  );
}
