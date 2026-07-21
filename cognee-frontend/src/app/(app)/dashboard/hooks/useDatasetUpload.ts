"use client";

import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { notifications } from "@mantine/notifications";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import type { Dataset } from "@/ui/layout/FilterContext";
import { trackEvent } from "@/modules/analytics";
import rememberData from "@/modules/ingestion/rememberData";
import createDataset from "@/modules/datasets/createDataset";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { datasetStatusQueryKey } from "@/modules/datasets/useDatasetStatuses";
import {
  loadGraphModelsConfig,
  findModelForDataset,
  findPromptForDataset,
  findOntologyForDataset,
} from "@/modules/configuration/userConfiguration";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";

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
  const { tenant } = useTenant();
  const { datasets, setSelectedDataset, refreshDatasets } = useFilter();
  const queryClient = useQueryClient();

  const [isUploading, setIsUploading] = useState(false);
  const [showDatasetPicker, setShowDatasetPicker] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [showUploadDoneModal, setShowUploadDoneModal] = useState<UploadDoneState | null>(null);

  async function uploadToDataset(ds: Dataset, files: File[]): Promise<void> {
    if (!cogniInstance) return;
    setIsUploading(true);
    try {
      const cfg = await loadGraphModelsConfig(cogniInstance);
      const rememberOpts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};

      const assignedModel = findModelForDataset(cfg.models, ds.id);
      if (assignedModel) {
        rememberOpts.graphModel = toGraphModelSchema(toCleanSchema(assignedModel.schema));
      }
      const promptName = findPromptForDataset(cfg.promptAssignments ?? {}, ds.id);
      if (promptName && cfg.customPrompts?.[promptName]) {
        rememberOpts.customPrompt = cfg.customPrompts[promptName];
      }
      const ontologyKey = findOntologyForDataset(cfg.ontologyAssignments ?? {}, ds.id);
      if (ontologyKey) rememberOpts.ontologyKey = [ontologyKey];

      await rememberData({ id: ds.id }, files, cogniInstance, rememberOpts);
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
      await pollDatasetStatus(ds.id, cogniInstance, { intervalMs: 5000 });
      // Any open Datasets/Detail/Knowledge Graph page shares this query key —
      // invalidate so they reflect the finished upload immediately instead of
      // waiting for their own next poll tick.
      queryClient.invalidateQueries({ queryKey: datasetStatusQueryKey(tenant?.tenant_id) });
      refreshDatasets();
      setShowUploadDoneModal({ datasetName: ds.name, datasetId: ds.id });
    } catch (err) {
      console.error("Dashboard upload failed:", err);
      notifications.show({
        title: "Upload failed",
        message: err instanceof Error ? err.message : String(err),
        color: "red",
      });
    } finally {
      setIsUploading(false);
    }
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
