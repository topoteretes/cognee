import { useCallback, useEffect, useRef, useState } from 'react';
import { v4 } from 'uuid';

import { fetch } from '@/utils';
import { DataFile } from './useData';
import createDataset from "../datasets/createDataset";

export interface Dataset {
  id: string;
  name: string;
  data: DataFile[];
  status: string;
}

function useDatasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const statusTimeout = useRef<any>(null);

  const fetchDatasetStatuses = useCallback((datasets: Dataset[]) => {
    fetch(
      `/v1/datasets/status?dataset=${datasets.map(d => d.id).join('&dataset=')}`,
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    )
      .then((response) => response.json())
      .then((statuses) => setDatasets(
        (datasets) => (
          datasets.map((dataset) => ({
            ...dataset,
            status: statuses[dataset.id]
          }))
      )));
  }, []);

  const checkDatasetStatuses = useCallback((datasets: Dataset[]) => {
    fetchDatasetStatuses(datasets);

    if (statusTimeout.current !== null) {
      clearTimeout(statusTimeout.current);
    }

    statusTimeout.current = setTimeout(() => {
      checkDatasetStatuses(datasets);
    }, 50000);
  }, [fetchDatasetStatuses]);
  
  useEffect(() => {
    return () => {
      if (statusTimeout.current !== null) {
        clearTimeout(statusTimeout.current);
        statusTimeout.current = null;
      }
    };
  }, []);

  const addDataset = useCallback((datasetName: string) => {
    return createDataset({ name: datasetName  })
      .then((dataset) => {
        setDatasets((datasets) => [
          ...datasets,
          dataset,
        ]);
      });
  }, []);

  const removeDataset = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}`, {
      method: 'DELETE',
    })
      .then(() => {
        setDatasets((datasets) =>
          datasets.filter((dataset) => dataset.id !== datasetId)
        );
      });
  }, []);

  const fetchDatasets = useCallback(() => {
    return fetch('/v1/datasets', {
        headers: {
          "Content-Type": "application/json",
        },
      })
      .then((response) => response.json())
      .then((datasets) => {
        setDatasets(datasets);

        if (datasets.length > 0) {
          checkDatasetStatuses(datasets);
        }

        return datasets;
      })
      .catch((error) => {
        console.error('Error fetching datasets:', error);
      });
  }, [checkDatasetStatuses]);

  const getDatasetData = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}/data`)
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
  }, [datasets]);

  const removeDatasetData = useCallback((datasetId: string, dataId: string) => {
    return fetch(`/v1/datasets/${datasetId}/data/${dataId}`, {
      method: 'DELETE',
    });
  }, []);

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
