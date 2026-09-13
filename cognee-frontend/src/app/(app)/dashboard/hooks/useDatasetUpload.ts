"use client";

import { useState, useCallback } from "react";
import { notifications } from "@mantine/notifications";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import type { Dataset } from "@/ui/layout/FilterContext";
import { trackEvent } from "@/modules/analytics";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import createDataset from "@/modules/datasets/createDataset";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import buildCognifyOptions from "@/modules/configuration/buildCognifyOptions";

export interface UploadDoneState {
  datasetName: string;
  datasetId: string;
}

export interface DatasetUploadState {
  isUploading: boolean;
  showDatasetPicker: boolean;
  pendingFiles: File[];
  showUploadDoneModal: UploadDoneState | null;
  setShowDatasetPicker: (open: boolean) => void;
  setPendingFiles: (files: File[]) => void;
  setShowUploadDoneModal: (state: UploadDoneState | null) => void;
  handleDashboardUpload: (e: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
  handlePickDataset: (ds: Dataset) => Promise<void>;
}

export function useDatasetUpload(): DatasetUploadState {
  const { cogniInstance } = useCogniInstance();
  const { datasets, setSelectedDataset, refreshDatasets } = useFilter();
  const { isUploading, upload } = useBrainUpload(cogniInstance);

  const [showDatasetPicker, setShowDatasetPicker] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [showUploadDoneModal, setShowUploadDoneModal] = useState<UploadDoneState | null>(null);

  async function uploadToDataset(ds: Dataset, files: File[]): Promise<void> {
    if (!cogniInstance) return;

    const showUploadFailed = (error: unknown): void => {
      console.error("Dashboard upload failed:", error);
      notifications.show({
        title: "Upload failed",
        message: error instanceof Error ? error.message : String(error),
        color: "red",
      });
    };

    // Graph-model / prompt / ontology assignments are best-effort enrichment;
    // if the config can't be read, fail the upload the same way the old inline
    // flow did (it ran this inside the upload try/catch).
    let options;
    try {
      options = buildCognifyOptions(await loadGraphModelsConfig(cogniInstance), ds.id);
    } catch (error) {
      showUploadFailed(error);
      return;
    }

    await upload({
      datasetId: ds.id,
      files,
      options,
      onLimitExceeded: (selected) =>
        notifications.show({
          title: "Too many files",
          message: `You selected ${selected.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`,
          color: "red",
        }),
      onUploadError: showUploadFailed,
      onUploaded: () => {
        trackEvent({
          pageName: "Dashboard",
          eventName: "dashboard_files_uploaded",
          additionalProperties: {
            dataset_id: ds.id,
            dataset_name: ds.name,
            file_count: String(files.length),
          },
        });
        notifications.show({
          title: `Files uploaded to "${ds.name}"`,
          message: `${files.length} file(s) added. Cognify running.`,
          color: "blue",
          autoClose: 5000,
        });
      },
      onProcessed: () => {
        refreshDatasets();
        setShowUploadDoneModal({ datasetName: ds.name, datasetId: ds.id });
      },
      onProcessingError: (error) => {
        console.error("Dataset processing failed:", error);
        refreshDatasets();
        notifications.show({
          title: "Knowledge graph build failed",
          message: `Files were added, but building the knowledge graph failed: ${error instanceof Error ? error.message : String(error)}`,
          color: "red",
        });
      },
    });
  }

  const handleDashboardUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!cogniInstance || !e.target.files?.length) return;
      const files = Array.from(e.target.files);
      e.target.value = "";

      if (datasets.length === 1) {
        await uploadToDataset(datasets[0], files);
        return;
      }
      if (datasets.length === 0) {
        const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
        refreshDatasets();
        await uploadToDataset(ds, files);
        return;
      }
      // Multiple datasets, none selected — let the user pick.
      setPendingFiles(files);
      setShowDatasetPicker(true);
    },
    // uploadToDataset is stable within the render; cogniInstance/datasets/refreshDatasets
    // are the real reactive inputs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cogniInstance, datasets, refreshDatasets],
  );

  const handlePickDataset = useCallback(
    async (ds: Dataset) => {
      setShowDatasetPicker(false);
      setSelectedDataset(ds);
      trackEvent({
        pageName: "Dashboard",
        eventName: "dashboard_dataset_picked",
        additionalProperties: { dataset_id: ds.id, dataset_name: ds.name },
      });
      await uploadToDataset(ds, pendingFiles);
      setPendingFiles([]);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pendingFiles, setSelectedDataset],
  );

  return {
    isUploading,
    showDatasetPicker,
    pendingFiles,
    showUploadDoneModal,
    setShowDatasetPicker,
    setPendingFiles,
    setShowUploadDoneModal,
    handleDashboardUpload,
    handlePickDataset,
  };
}
