"use client";

import { v4 as uuid4 } from "uuid";
import { ChangeEvent, useEffect } from "react";

import { LoadingIndicator } from "@/ui/App";
import { CTAButton, StatusIndicator } from "@/ui/elements";

import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import useDatasets, { Dataset } from "@/modules/ingestion/useDatasets";
import getDatasetGraph from "@/modules/datasets/getDatasetGraph";
import createDataset from "@/modules/datasets/createDataset";
import { useBoolean } from '@/utils';

export interface NodesAndEdges {
  nodes: { id: string; label: string }[];
  links: { source: string; target: string; label: string }[];
}

interface CogneeAddWidgetProps {
  onData: (data: NodesAndEdges) => void;
}

export default function CogneeAddWidget({ onData }: CogneeAddWidgetProps) {
  const {
    datasets,
    addDataset,
    removeDataset,
    refreshDatasets,
  } = useDatasets();

  useEffect(() => {
    refreshDatasets()
      .then((datasets) => {
        const dataset = datasets?.[0];

        if (dataset) {
          getDatasetGraph(dataset || { id: uuid4() })
            .then((graph) => onData({
              nodes: graph.nodes,
              links: graph.edges,
            }));
        }
      });
  }, [onData, refreshDatasets]);

  const {
    value: isProcessingFiles,
    setTrue: setProcessingFilesInProgress,
    setFalse: setProcessingFilesDone,
  } = useBoolean(false);
  
  const handleAddFiles = (dataset: Dataset, event: ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();

    if (isProcessingFiles) {
      return;
    }

    setProcessingFilesInProgress();

    if (!event.target.files) {
      throw new Error("Error: No files added to the uploader input.");
    }

    const files: File[] = Array.from(event.target.files);

    return addData(dataset, files)
      .then(() => {
        const onUpdate = (data: any) => {
          onData({
            nodes: data.nodes,
            links: data.edges,
          });
          setProcessingFilesDone();
        };

        return cognifyDataset(dataset, onUpdate);
      });
  };

  const handleAddFilesNoDataset = (event: ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();

    if (isProcessingFiles) {
      return;
    }

    setProcessingFilesInProgress();

    createDataset({ name: "main_dataset" })
      .then((newDataset: Dataset) => {
        return handleAddFiles(newDataset, event);
      });
  };

  return (
    <div className="flex flex-col gap-4 mb-4">
      {datasets.length ? datasets.map((dataset) => (
        <div key={dataset.id} className="flex gap-8 items-center justify-between">
          <div className="flex flex-row gap-4 items-center">
            <StatusIndicator status={dataset.status} />
            <span className="text-white">{dataset.name}</span>
          </div>
          <CTAButton type="button" className="relative">
            <input type="file" multiple onChange={handleAddFiles.bind(null, dataset)} className="absolute w-full h-full cursor-pointer opacity-0" />
            <span className="flex flex-row gap-2 items-center">
              + Add Data
              {isProcessingFiles && <LoadingIndicator />}
            </span>
          </CTAButton>
        </div>
      )) : (
        <CTAButton type="button" className="relative">
          <input type="file" multiple onChange={handleAddFilesNoDataset} className="absolute w-full h-full cursor-pointer opacity-0" />
          <span className="flex flex-row gap-2 items-center">
            + Add Data
            {isProcessingFiles && <LoadingIndicator />}
          </span>
        </CTAButton>
      )}
    </div>
  );
}
