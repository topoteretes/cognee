import { useCallback, useState } from 'react';

import { fetch } from '@/utils';
import { DataFile } from './useData';
import createDataset from "../datasets/createDataset";

export interface Dataset {
  id: string;
  name: string;
  data: DataFile[];
  status: string;
}

function useDatasets(useCloud = false) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  const addDataset = useCallback((datasetName: string) => {
    return createDataset({ name: datasetName  }, useCloud)
      .then((dataset) => {
        setDatasets((datasets) => [
          ...datasets,
          dataset,
        ]);
      });
  }, [useCloud]);

  const removeDataset = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}`, {
      method: 'DELETE',
    }, useCloud)
      .then(() => {
        setDatasets((datasets) =>
          datasets.filter((dataset) => dataset.id !== datasetId)
        );
      });
  }, [useCloud]);

  const fetchDatasets = useCallback(() => {
    return fetch('/v1/datasets', {
        headers: {
          "Content-Type": "application/json",
        },
      }, useCloud)
      .then((response) => response.json())
      .then((datasets) => {
        setDatasets(datasets);
        return datasets;
      })
      .catch((error) => {
        console.error('Error fetching datasets:', error);
        throw error;
      });
  }, [useCloud]);

  const getDatasetData = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}/data`, {}, useCloud)
      .then((response) => response.json())
      .then((data) => {
        const datasetIndex = datasets.findIndex((dataset) => dataset.id === datasetId);

        if (datasetIndex >= 0) {
          setDatasets((datasets) => [
           ...datasets.slice(0, datasetIndex),
            {
             ...datasets[datasetIndex],
              data,
            },
           ...datasets.slice(datasetIndex + 1),
          ]);
        }

        return data;
      });
  }, [datasets, useCloud]);

  const removeDatasetData = useCallback((datasetId: string, dataId: string) => {
    return fetch(`/v1/datasets/${datasetId}/data/${dataId}`, {
      method: 'DELETE',
    }, useCloud);
  }, [useCloud]);

  return {
    datasets,
    addDataset,
    removeDataset,
    getDatasetData,
    removeDatasetData,
    refreshDatasets: fetchDatasets,
  };
};

export default useDatasets;
