import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import createDataset from "@/modules/datasets/createDataset";
import { CogneeInstance } from "@/modules/instances/types";
import { DataFile } from "./useData";
import deleteDataset from "../datasets/deleteDataset";
import getDatasets from "../datasets/getDatasets";
import getDatasetData from "../datasets/getDatasetData";
import deleteDatasetData from "../datasets/deleteDatasetData";
import searchDataset from "../datasets/searchDataset";
import getVisualization from "../datasets/visualizeDataset";

export interface Dataset {
  id: string;
  name: string;
  data: DataFile[];
  status: string;
}

function filterDatasets(datasets: Dataset[], searchValue: string) {
  if (searchValue.trim() === "") {
    return datasets;
  }

  const lowercaseSearchValue = searchValue.toLowerCase();

  return datasets.filter((dataset) =>
    dataset.name.toLowerCase().includes(lowercaseSearchValue)
  );
}

function useDatasets(instance: CogneeInstance, searchValue: string, onReady?: () => void) {
  const allDatasets = useRef<Dataset[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const onReadyCalledRef = useRef(false);

  useLayoutEffect(() => {
    setDatasets(filterDatasets(allDatasets.current, searchValue));
  }, [searchValue])

  const addDataset = useCallback((datasetName: string) => {
    return createDataset({ name: datasetName  }, instance)
      .then((dataset) => {
        const newDatasets = [
          ...allDatasets.current,
          dataset,
        ];
        allDatasets.current = newDatasets;
        setDatasets(filterDatasets(newDatasets, searchValue));
      });
  }, [searchValue, instance]);

  const removeDataset = useCallback((datasetId: string) => {
    return deleteDataset(datasetId, instance)
      .then(() => {
        const newDatasets = allDatasets.current.filter((dataset) => dataset.id !== datasetId)
        allDatasets.current = newDatasets;
        setDatasets(filterDatasets(newDatasets, searchValue));
      });
  }, [searchValue, instance]);

  const fetchDatasets = useCallback(() => {
    return getDatasets(instance)
      .then((datasets) => {
        allDatasets.current = datasets;
        setDatasets(filterDatasets(datasets, searchValue));

        if (!onReadyCalledRef.current && onReady) {
          onReadyCalledRef.current = true;
          onReady();
        }

        return datasets;
      })
      .catch((error) => {
        const message =
          typeof error === "object" && error !== null
            ? (error as { detail?: string; message?: string }).detail ??
              (error as Error).message
            : String(error);
        const isConnectionError = message === "No connection to the server.";
        if (isConnectionError) {
          console.warn("Datasets not available yet:", message);
        } else {
          console.error("Error fetching datasets:", message);
        }

        if (!onReadyCalledRef.current && onReady) {
          onReadyCalledRef.current = true;
          onReady();
        }

        return [] as Dataset[];
      });
  }, [searchValue, instance, onReady]);

  useEffect(() => {
    if (allDatasets.current.length === 0) {
      fetchDatasets();
    }
  }, [fetchDatasets]);

  const fetchDatasetData = useCallback((datasetId: string) => {
    return getDatasetData(datasetId, instance)
      .then((data) => {
        const datasetIndex = datasets.findIndex((dataset) => dataset.id === datasetId);

        if (datasetIndex >= 0) {
          const newDatasets = [
            ...allDatasets.current.slice(0, datasetIndex),
              {
              ...allDatasets.current[datasetIndex],
                data,
              },
            ...allDatasets.current.slice(datasetIndex + 1),
          ];

          allDatasets.current = newDatasets;

          setDatasets(filterDatasets(newDatasets, searchValue));
        }

        return data;
      });
  }, [datasets, instance, searchValue]);

  const removeDatasetData = useCallback((datasetId: string, dataId: string) => {
    return deleteDatasetData(datasetId, dataId, instance);
  }, [instance]);

  const visualizeDataset = useCallback((datasetId: string) => {
    return getVisualization(instance, datasetId);
  }, [instance]);

  const searchDatasetFiles = useCallback((datasetId: string, searchQuery: string) => {
    return searchDataset(instance, {datasetIds: [datasetId], query: searchQuery});
  }, [instance]);

  return {
    datasets,
    addDataset,
    removeDataset,
    removeDatasetData,
    visualizeDataset,
    searchDataset: searchDatasetFiles,
    refreshDatasets: fetchDatasets,
    getDatasetData: fetchDatasetData,
  };
};

export default useDatasets;
