"use client";

import { v4 as uuid4 } from "uuid";
import { ChangeEvent, useEffect } from "react";
import { CTAButton, StatusIndicator } from "@/ui/elements";

import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import useDatasets from "@/modules/ingestion/useDatasets";
import getDatasetGraph from '@/modules/datasets/getDatasetGraph';

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

        // For CrewAI we don't have a dataset.
        // if (dataset) {
        getDatasetGraph(dataset || { id: uuid4() })
          .then((graph) => onData({
            nodes: graph.nodes,
            links: graph.edges,
          }));
        // }
      });
  }, [onData, refreshDatasets]);

  const handleAddFiles = (dataset: { id?: string, name?: string }, event: ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();

    if (!event.currentTarget.files) {
      throw new Error("Error: No files added to the uploader input.");
    }

    const files: File[] = Array.from(event.currentTarget.files);

    return addData(dataset, files)
      .then(() => {
        const onUpdate = (data: any) => {
          onData({
            nodes: data.payload.nodes,
            links: data.payload.edges,
          });
        };

        return cognifyDataset(dataset, onUpdate);
      });
  };

  return (
    <div className="flex flex-col gap-4 mb-4">
      {datasets.length ? datasets.map((dataset) => (
        <div key={dataset.id} className="flex gap-8 items-center">
          <div className="flex flex-row gap-4 items-center">
            <StatusIndicator status={dataset.status} />
            <span className="text-white">{dataset.name}</span>
          </div>
          <CTAButton type="button" className="relative">
            <input type="file" multiple onChange={handleAddFiles.bind(null, dataset)} className="absolute w-full h-full cursor-pointer opacity-0" />
            <span>+ Add Data</span>
          </CTAButton>
        </div>
      )) : (
        <CTAButton type="button" className="relative">
          <input type="file" multiple onChange={handleAddFiles.bind(null, { name: "main_dataset" })} className="absolute w-full h-full cursor-pointer opacity-0" />
          <span>+ Add Data</span>
        </CTAButton>
      )}
    </div>
  );
}
