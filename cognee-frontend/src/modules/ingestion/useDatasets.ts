import { useCallback, useEffect, useRef, useState } from 'react';
import { v4 } from 'uuid';
import { DataFile } from './useData';
import { fetch } from '@/utils';

export interface Dataset {
  id: string;
  name: string;
  data: DataFile[];
  status: string;
}

function useDatasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const statusTimeout = useRef<any>(null);

  const fetchDatasetStatuses = useCallback((datasets: Dataset[]) => {
    fetch(
      `/v1/datasets/status?dataset=${datasets.map(d => d.id).join('&dataset=')}`,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`,
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
    }, 5000);
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
    setDatasets((datasets) => [
      ...datasets,
      {
        id: v4(),
        name: datasetName,
        data: [],
        status: 'DATASET_INITIALIZED',
      }
    ]);
  }, []);

  const removeDataset = useCallback((datasetId: string) => {
    setDatasets((datasets) =>
      datasets.filter((dataset) => dataset.id !== datasetId)
    );
  }, []);

  const fetchDatasets = useCallback(() => {
    fetch('/v1/datasets', {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('access_token')}`,
      },
    })
      .then((response) => response.json())
      .then((datasets) => {
        setDatasets(datasets);

        if (datasets.length > 0) {
          checkDatasetStatuses(datasets);
        } else {
          window.location.href = '/wizard';
        }
      })
      .catch((error) => {
        console.error('Error fetching datasets:', error);
      });
  }, [checkDatasetStatuses]);

  return { datasets, addDataset, removeDataset, refreshDatasets: fetchDatasets };
};

export default useDatasets;
