"use client";

import { FormEvent, useCallback, useRef, useState } from "react";
import { Stack, Text, Flex, Box } from "@mantine/core";

import useDatasets, { Dataset } from "@/modules/ingestion/useDatasets";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { CogneeInstance } from "@/modules/instances/types";
import { Modal, useModal } from "@/ui/elements/Modal";
import CreateNewDatasetModal from "@/app/(app)/dashboard/elements/CreateNewDatasetAccordion";
import CTAButton from "@/ui/elements/CTAButton";
import GhostButton from "@/ui/elements/GhostButton";
import IconButton from "@/ui/elements/IconButton";
import { LoadingIndicator } from "@/ui/app";
import DeleteIcon from "@/ui/icons/DeleteIcon";
import CloseIcon from "@/ui/icons/CloseIcon";
import { trackEvent } from "@/modules/analytics";
import PlusIcon from "@/ui/icons/PlusIcon";
import addData from "@/modules/ingestion/addData";
import { notifications } from "@mantine/notifications";

interface DatasetFile {
  id: string;
  name: string;
}

function DatasetRow({
  dataset,
  instance,
  onDeleteDataset,
  onDeleteFile,
  getDatasetData,
  onFilesAdded,
}: {
  dataset: Dataset;
  instance: CogneeInstance;
  onDeleteDataset: (id: string, name: string) => void;
  onDeleteFile: (datasetId: string, fileId: string, fileName: string) => void;
  getDatasetData: (datasetId: string) => Promise<DatasetFile[]>;
  onFilesAdded?: (datasetId: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [files, setFiles] = useState<DatasetFile[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleToggle = useCallback(async () => {
    if (!isExpanded) {
      setIsExpanded(true);
      if (files.length === 0) {
        setIsLoadingFiles(true);
        try {
          const data = await getDatasetData(dataset.id);
          setFiles(data || []);
        } catch {
          setFiles([]);
        } finally {
          setIsLoadingFiles(false);
        }
      }
    } else {
      setIsExpanded(false);
    }
  }, [isExpanded, files.length, getDatasetData, dataset.id]);

  const handleDeleteFile = useCallback(
    (fileId: string, fileName: string) => {
      onDeleteFile(dataset.id, fileId, fileName);
      setFiles((prev) => prev.filter((f) => f.id !== fileId));
    },
    [dataset.id, onDeleteFile],
  );

  const handleFileInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files ?? []);
      if (selected.length === 0) return;
      // Reset input so the same file can be re-selected
      e.target.value = "";
      setIsUploading(true);
      try {
        await addData({ id: dataset.id }, selected, instance);
        trackEvent({
          pageName: "Datasets",
          eventName: "document_uploaded",
          additionalProperties: {
            type: "file",
            file_count: String(selected.length),
            dataset_id: dataset.id,
          },
        });
        notifications.show({
          title: "Files added",
          message: `${selected.length} file(s) added to "${dataset.name}"`,
          color: "green",
        });
        onFilesAdded?.(dataset.id);
        // Refresh file list if expanded
        if (isExpanded) {
          const data = await getDatasetData(dataset.id);
          setFiles(data || []);
        } else {
          // Force reload on next expand
          setFiles([]);
        }
      } catch (err) {
        notifications.show({
          title: "Upload failed",
          message: err instanceof Error ? err.message : String(err),
          color: "red",
        });
      } finally {
        setIsUploading(false);
      }
    },
    [dataset.id, dataset.name, instance, isExpanded, getDatasetData, onFilesAdded],
  );

  return (
    <div className="border-b border-gray-200 last:border-b-0">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: "none" }}
        onChange={handleFileInputChange}
      />
      <Flex
        className="px-4 py-3 cursor-pointer hover:bg-gray-50 items-center"
        justify="space-between"
        onClick={handleToggle}
      >
        <Flex gap="sm" align="center">
          <Text
            size="sm"
            style={{
              transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 0.15s ease",
            }}
          >
            &#9654;
          </Text>
          <Text size="sm" fw={500}>
            {dataset.name}
          </Text>
          {dataset.status && (
            <Text size="xs" c="dimmed">
              {dataset.status}
            </Text>
          )}
        </Flex>
        <Flex gap="xs" align="center">
          <IconButton
            disabled={isUploading}
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            title="Add files to dataset"
          >
            {isUploading ? <LoadingIndicator /> : <PlusIcon />}
          </IconButton>
          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              onDeleteDataset(dataset.id, dataset.name);
            }}
          >
            <DeleteIcon />
          </IconButton>
        </Flex>
      </Flex>
      {isExpanded && (
        <div className="pl-10 pr-4 pb-3">
          {isLoadingFiles ? (
            <Flex align="center" gap="xs" className="py-2">
              <LoadingIndicator />
              <Text size="xs" c="dimmed">
                Loading files...
              </Text>
            </Flex>
          ) : files.length === 0 ? (
            <Text size="xs" c="dimmed" className="py-2">
              No files in this dataset.
            </Text>
          ) : (
            <Stack gap="2px">
              {files.map((file) => (
                <Flex
                  key={file.id}
                  className="py-1.5 px-2 rounded hover:bg-gray-50 items-center"
                  justify="space-between"
                >
                  <Text size="xs">{decodeURIComponent(file.name)}</Text>
                  <IconButton
                    onClick={() => handleDeleteFile(file.id, file.name)}
                  >
                    <DeleteIcon width={10} height={12} />
                  </IconButton>
                </Flex>
              ))}
            </Stack>
          )}
        </div>
      )}
    </div>
  );
}

function DatasetsContent({ instance }: { instance: CogneeInstance }) {
  const {
    datasets,
    addDataset,
    removeDataset,
    removeDatasetData,
    refreshDatasets,
    getDatasetData,
  } = useDatasets(instance, "");

  const [newDatasetError, setNewDatasetError] = useState("");

  // Create dataset modal
  const {
    isModalOpen: isCreateModalOpen,
    openModal: openCreateModal,
    closeModal: closeCreateModal,
    isActionLoading: isCreatePending,
  } = useModal();

  // Delete dataset modal
  const {
    modalState: deleteDatasetState,
    isModalOpen: isDeleteDatasetOpen,
    openModal: openDeleteDatasetModal,
    closeModal: closeDeleteDatasetModal,
    confirmAction: confirmDeleteDataset,
    isActionLoading: isDeleteDatasetPending,
  } = useModal<{ id: string; name: string }>(
    false,
    async (state) => {
      await removeDataset(state.id);
      trackEvent({ pageName: "Datasets", eventName: "dataset_deleted", additionalProperties: { dataset_id: state.id, dataset_name: state.name } });
    },
  );

  // Delete file modal
  const {
    modalState: deleteFileState,
    isModalOpen: isDeleteFileOpen,
    openModal: openDeleteFileModal,
    closeModal: closeDeleteFileModal,
    confirmAction: confirmDeleteFile,
    isActionLoading: isDeleteFilePending,
  } = useModal<{ datasetId: string; fileId: string; fileName: string }>(
    false,
    async (state) => {
      await removeDatasetData(state.datasetId, state.fileId);
      trackEvent({ pageName: "Datasets", eventName: "file_deleted", additionalProperties: { dataset_id: state.datasetId, file_id: state.fileId, file_name: state.fileName } });
      await refreshDatasets();
    },
  );

  const handleNewDatasetSubmit = useCallback(
    async (event?: FormEvent<HTMLFormElement>) => {
      event?.preventDefault();
      setNewDatasetError("");

      const formData = new FormData(event?.currentTarget);
      const datasetName = (formData.get("datasetName") as string)?.trim();

      if (!datasetName) {
        setNewDatasetError("Dataset name cannot be empty.");
        return;
      }
      if (datasetName.includes(" ") || datasetName.includes(".")) {
        setNewDatasetError("Dataset name cannot contain spaces or periods.");
        return;
      }

      try {
        await addDataset(datasetName);
        trackEvent({ pageName: "Datasets", eventName: "dataset_created", additionalProperties: { dataset_name: datasetName } });
        closeCreateModal();
      } catch {
        setNewDatasetError("Failed to create dataset.");
      }
    },
    [addDataset, closeCreateModal],
  );

  const handleDeleteDataset = useCallback(
    (id: string, name: string) => {
      openDeleteDatasetModal({ id, name });
    },
    [openDeleteDatasetModal],
  );

  const handleDeleteFile = useCallback(
    (datasetId: string, fileId: string, fileName: string) => {
      openDeleteFileModal({ datasetId, fileId, fileName });
    },
    [openDeleteFileModal],
  );

  return (
    <>
      <Stack className="h-full overflow-auto" gap="0.625rem">
        <Stack
          className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem]"
          bg="#ffffff"
          gap="md"
        >
          <Flex justify="space-between" align="center">
            <Text size="xl" fw={600}>
              Datasets
            </Text>
            <CTAButton onClick={() => { setNewDatasetError(""); openCreateModal(); }}>
              + New Dataset
            </CTAButton>
          </Flex>

          <Box className="border border-gray-200 rounded-lg overflow-hidden">
            {datasets.length === 0 ? (
              <Text size="sm" c="dimmed" className="p-4 text-center">
                No datasets yet. Create one to get started.
              </Text>
            ) : (
              datasets.map((dataset) => (
                <DatasetRow
                  key={dataset.id}
                  dataset={dataset}
                  instance={instance}
                  onDeleteDataset={handleDeleteDataset}
                  onDeleteFile={handleDeleteFile}
                  getDatasetData={getDatasetData}
                />
              ))
            )}
          </Box>
        </Stack>
      </Stack>

      <CreateNewDatasetModal
        isOpen={isCreateModalOpen}
        isNewDatasetPending={isCreatePending}
        closeNewDatasetModal={closeCreateModal}
        handleNewDatasetSubmitConfirm={handleNewDatasetSubmit}
        newDatasetError={newDatasetError}
      />

      <Modal isOpen={isDeleteDatasetOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete dataset?</span>
            <IconButton
              disabled={isDeleteDatasetPending}
              onClick={closeDeleteDatasetModal}
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className="mt-8 mb-6">
            Are you sure you want to delete{" "}
            <strong>{deleteDatasetState?.name}</strong>? This action cannot be
            undone.
          </div>
          <div className="flex flex-row gap-4 justify-end">
            <GhostButton
              disabled={isDeleteDatasetPending}
              onClick={closeDeleteDatasetModal}
            >
              cancel
            </GhostButton>
            <CTAButton
              disabled={isDeleteDatasetPending}
              onClick={() => confirmDeleteDataset()}
            >
              {isDeleteDatasetPending && <LoadingIndicator color="white" />}
              delete
            </CTAButton>
          </div>
        </div>
      </Modal>

      <Modal isOpen={isDeleteFileOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete file?</span>
            <IconButton
              disabled={isDeleteFilePending}
              onClick={closeDeleteFileModal}
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className="mt-8 mb-6">
            Are you sure you want to delete{" "}
            <strong>{deleteFileState?.fileName}</strong>? This action cannot be
            undone.
          </div>
          <div className="flex flex-row gap-4 justify-end">
            <GhostButton
              disabled={isDeleteFilePending}
              onClick={closeDeleteFileModal}
            >
              cancel
            </GhostButton>
            <CTAButton
              disabled={isDeleteFilePending}
              onClick={() => confirmDeleteFile()}
            >
              {isDeleteFilePending && <LoadingIndicator color="white" />}
              delete
            </CTAButton>
          </div>
        </div>
      </Modal>
    </>
  );
}

export default function DatasetsBody() {
  const { cogniInstance, isInitializing } = useCogniInstance();

  if (isInitializing || !cogniInstance) {
    return (
      <Stack className="h-full items-center justify-center">
        <LoadingIndicator />
      </Stack>
    );
  }

  return <DatasetsContent instance={cogniInstance} />;
}
