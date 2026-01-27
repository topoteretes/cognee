import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { fetch } from '@/utils';
import { DataFile } from './useData';
import createDataset from "../datasets/createDataset";

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

function useDatasets(useCloud = false, searchValue: string = "") {
  const allDatasets = useRef<Dataset[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  // const statusTimeout = useRef<any>(null);

  // const fetchDatasetStatuses = useCallback((datasets: Dataset[]) => {
  //   fetch(
  //     `/v1/datasets/status?dataset=${datasets.map(d => d.id).join('&dataset=')}`,
  //     {
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //     },
  //     useCloud,
  //   )
  //     .then((response) => response.json())
  //     .then((statuses) => setDatasets(
  //       (datasets) => (
  //         datasets.map((dataset) => ({
  //           ...dataset,
  //           status: statuses[dataset.id]
  //         }))
  //     )));
  // }, [useCloud]);

  // const checkDatasetStatuses = useCallback((datasets: Dataset[]) => {
  //   fetchDatasetStatuses(datasets);

  //   if (statusTimeout.current !== null) {
  //     clearTimeout(statusTimeout.current);
  //   }

  //   statusTimeout.current = setTimeout(() => {
  //     checkDatasetStatuses(datasets);
  //   }, 50000);
  // }, [fetchDatasetStatuses]);

  // useEffect(() => {
  //   return () => {
  //     if (statusTimeout.current !== null) {
  //       clearTimeout(statusTimeout.current);
  //       statusTimeout.current = null;
  //     }
  //   };
  // }, []);

  useLayoutEffect(() => {
    setDatasets(filterDatasets(allDatasets.current, searchValue));
  }, [searchValue]);

  const addDataset = useCallback((datasetName: string) => {
    return createDataset({ name: datasetName  }, useCloud)
      .then((dataset) => {
        const newDatasets = [
          ...allDatasets.current,
          dataset,
        ];
        allDatasets.current = newDatasets;
        setDatasets(filterDatasets(newDatasets, searchValue));
      });
  }, [searchValue, useCloud]);

  const removeDataset = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}`, {
      method: 'DELETE',
    }, useCloud)
      .then(() => {
        const newDatasets = allDatasets.current.filter((dataset) => dataset.id !== datasetId)
        allDatasets.current = newDatasets;
        setDatasets(filterDatasets(newDatasets, searchValue));
      });
  }, [searchValue, useCloud]);

  const fetchDatasets = useCallback(() => {
    return fetch('/v1/datasets', {
        headers: {
          "Content-Type": "application/json",
        },
      }, useCloud)
      .then((response) => response.json())
      .then((datasets) => {
        allDatasets.current = datasets;
        setDatasets(filterDatasets(datasets, searchValue));

        // if (datasets.length > 0) {
        //   checkDatasetStatuses(datasets);
        // }

        return datasets;
      })
      .catch((error) => {
        console.error('Error fetching datasets:', error);
        throw error;
      });
  }, [searchValue, useCloud]);

  useEffect(() => {
    if (allDatasets.current.length === 0) {
      fetchDatasets();
    }
  }, [fetchDatasets]);

  const getDatasetData = useCallback((datasetId: string) => {
    return fetch(`/v1/datasets/${datasetId}/data`, {}, useCloud)
      .then((response) => response.json())
      .then((data) => {
        const datasetIndex = allDatasets.current.findIndex((dataset) => dataset.id === datasetId);

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
  }, [searchValue, useCloud]);

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
