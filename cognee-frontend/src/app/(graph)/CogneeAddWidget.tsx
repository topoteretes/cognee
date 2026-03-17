"use client";

import { ChangeEvent, useEffect } from "react";

import { SearchView } from "@/ui/Partials";
import { LoadingIndicator } from "@/ui/App";
import { AddIcon, SearchIcon } from "@/ui/Icons";
import { CTAButton, GhostButton, Modal, NeutralButton, StatusIndicator } from "@/ui/elements";

import { useBoolean } from "@/utils";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import createDataset from "@/modules/datasets/createDataset";
import getDatasetGraph from "@/modules/datasets/getDatasetGraph";
import useDatasets, { Dataset } from "@/modules/ingestion/useDatasets";

export interface NodesAndLinks {
  nodes: { id: string; label: string }[];
  links: { source: string; target: string; label: string }[];
}

export interface NodesAndEdges {
  nodes: { id: string; label: string }[];
  edges: { source: string; target: string; label: string }[];
}

interface CogneeAddWidgetProps {
  onData: (data: NodesAndLinks) => void;
  useCloud?: boolean;
}

export default function CogneeAddWidget({ onData, useCloud = false }: CogneeAddWidgetProps) {
  const {
    datasets,
    refreshDatasets,
  } = useDatasets();

  useEffect(() => {
    refreshDatasets()
      .then((datasets) => {
        const dataset = datasets?.[0];

        if (dataset) {
          getDatasetGraph(dataset)
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
      return;
    }

    const files: File[] = Array.from(event.target.files);

    if (!files.length) {
      return;
    }

    return addData(dataset, files)
      .then(() => {
        // const onUpdate = (data: NodesAndEdges) => {
        //   onData({
        //     nodes: data.nodes,
        //     links: data.edges,
        //   });
        //   setProcessingFilesDone();
        // };

        return cognifyDataset(dataset, useCloud)
          .then(() => {
            refreshDatasets();
            setProcessingFilesDone();
          });
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

  const {
    value: isSearchModalOpen,
    setTrue: openSearchModal,
    setFalse: closeSearchModal,
  } = useBoolean(false);

  const handleSearchClick = () => {
    openSearchModal();
  };

  return (
    <div className="flex flex-col gap-4">
      {datasets.length ? datasets.map((dataset) => (
        <div key={dataset.id} className="flex gap-8 items-center justify-between">
          <div className="flex flex-row gap-4 items-center">
            <StatusIndicator status={dataset.status} />
            <span className="text-white">{dataset.name}</span>
          </div>
          <div className="flex gap-2">
            <CTAButton type="button" className="relative">
              <input tabIndex={-1} type="file" multiple onChange={handleAddFiles.bind(null, dataset)} className="absolute w-full h-full cursor-pointer opacity-0" />
              <span className="flex flex-row gap-2 items-center">
                <AddIcon />
                {isProcessingFiles && <LoadingIndicator />}
              </span>
            </CTAButton>
            <NeutralButton onClick={handleSearchClick} type="button">
              <SearchIcon />
            </NeutralButton>
          </div>
        </div>
      )) : (
        <CTAButton type="button" className="relative" disabled={isProcessingFiles}>
          <input disabled={isProcessingFiles} tabIndex={-1} type="file" multiple onChange={handleAddFilesNoDataset} className="absolute w-full h-full cursor-pointer opacity-0" />
          <span className="flex flex-row gap-2 items-center">
            + Add Data
            {isProcessingFiles && <LoadingIndicator />}
          </span>
        </CTAButton>
      )}
      <Modal isOpen={isSearchModalOpen}>
        <div className="relative w-full max-w-3xl h-full max-h-5/6">
          <GhostButton onClick={closeSearchModal} className="absolute right-2 top-2">
            <AddIcon className="rotate-45" />
          </GhostButton>
          <SearchView />
        </div>
      </Modal>
    </div>
  );
}
