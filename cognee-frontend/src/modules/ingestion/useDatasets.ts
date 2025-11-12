import { useCallback, useState } from 'react';

import { fetch } from '@/utils';
import { DataFile } from './useData';
import createDataset from "../datasets/createDataset";

export interface Dataset {
  id: string;
  name: string;
  data?: DataFile[];
  status?: string;
}

function useDatasets(useCloud = false) {
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

  const buildDeleteQuery = useCallback((datasetId: string, dataId: string) => {
    const params = new URLSearchParams({
      data_id: dataId,
      dataset_id: datasetId,
      mode: "hard",
    });
    return `/v1/delete?${params.toString()}`;
  }, []);

  const addDataset = useCallback((datasetName: string) => {
    return createDataset({ name: datasetName  }, useCloud)
      .then((dataset) => {
        setDatasets((datasets) => [
          ...datasets,
          dataset,
        ]);
      });
  }, [useCloud]);

  const removeDataset = useCallback(async (datasetId: string) => {
    // Always fetch latest data entries to make sure all documents are removed everywhere
    const datasetData: DataFile[] = await fetch(`/v1/datasets/${datasetId}/data`, {}, useCloud)
      .then((response) => response.json())
      .catch(() => []);

    if (Array.isArray(datasetData) && datasetData.length > 0) {
      await Promise.all(
        datasetData.map((dataItem) =>
          fetch(buildDeleteQuery(datasetId, dataItem.id), { method: 'DELETE' }, useCloud)
        ),
      );
    }

    return fetch(`/v1/datasets/${datasetId}`, {
      method: 'DELETE',
    }, useCloud)
      .then(() => {
        setDatasets((datasets) =>
          datasets.filter((dataset) => dataset.id !== datasetId)
        );
      });
  }, [buildDeleteQuery, useCloud]);

  const fetchDatasets = useCallback(() => {
    return fetch('/v1/datasets', {
        headers: {
          "Content-Type": "application/json",
        },
      }, useCloud)
      .then((response) => response.json())
      .then((datasets) => {
        setDatasets(datasets);

        // if (datasets.length > 0) {
        //   checkDatasetStatuses(datasets);
        // }

        return datasets;
      })
      .catch((error) => {
        console.error('Error fetching datasets:', error);
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
    return fetch(buildDeleteQuery(datasetId, dataId), {
      method: 'DELETE',
    }, useCloud)
      .then(() => {
        setDatasets((current) => current.map((dataset) => {
          if (dataset.id !== datasetId) return dataset;
          return {
            ...dataset,
            data: dataset.data?.filter((dataItem) => dataItem.id !== dataId) || [],
          };
        }));
      });
  }, [buildDeleteQuery, setDatasets, useCloud]);

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
