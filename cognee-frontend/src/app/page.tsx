'use client';

import { useCallback, useEffect, useState } from 'react';
import styles from "./page.module.css";
import { Notification, NotificationContainer, Text, useNotifications } from 'ohmy-ui';
import useDatasets from '@/modules/ingestion/useDatasets';
import DataView, { Data } from '@/modules/ingestion/DataView';
import DatasetsView from '@/modules/ingestion/DatasetsView';
import classNames from 'classnames';
import addData from '@/modules/ingestion/addData';
import cognifyDataset from '@/modules/datasets/cognifyDataset';
import deleteDataset from '@/modules/datasets/deleteDataset';
import getDatasetData from '@/modules/datasets/getDatasetData';
import getExplorationGraphUrl from '@/modules/exploration/getExplorationGraphUrl';
import { Footer } from '@/ui/Partials';

export default function Home() {
  const {
    datasets,
    refreshDatasets,
  } = useDatasets();

  const [datasetData, setDatasetData] = useState<Data[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  const openDatasetData = (dataset: { id: string }) => {
    getDatasetData(dataset)
      .then(setDatasetData)
      .then(() => setSelectedDataset(dataset.id));
  };

  const closeDatasetData = () => {
    setDatasetData([]);
    setSelectedDataset(null);
  };

  const { notifications, showNotification } = useNotifications();

  const onDataAdd = useCallback((dataset: { id: string }, files: File[]) => {
    return addData(dataset, files)
      .then(() => {
        showNotification("Data added successfully.", 5000);
        openDatasetData(dataset);
      });
  }, [showNotification])

  const onDatasetCognify = useCallback((dataset: { id: string }) => {
    showNotification(`Cognification started for dataset "${dataset.id}".`, 5000);

    return cognifyDataset(dataset)
      .then(() => {
        showNotification(`Dataset "${dataset.id}" cognified.`, 5000);
      })
      .catch(() => {
        showNotification(`Dataset "${dataset.id}" cognification failed. Please try again.`, 5000);
      });
  }, [showNotification]);

  const onDatasetDelete = useCallback((dataset: { id: string }) => {
    deleteDataset(dataset)
     .then(() => {
        showNotification(`Dataset "${dataset.id}" deleted.`, 5000);
        refreshDatasets();
      })
  }, [refreshDatasets, showNotification]);

  const onDatasetExplore = useCallback((dataset: { id: string }) => {
    return getExplorationGraphUrl(dataset);
  }, []);

  return (
    <main className={styles.main}>
      <div className={styles.data}>
        <div className={classNames(styles.datasetsView, {
          [styles.openDatasetData]: datasetData.length > 0,
        })}>
          <DatasetsView
            datasets={datasets}
            onDataAdd={onDataAdd}
            onDatasetClick={openDatasetData}
            onDatasetCognify={onDatasetCognify}
            onDatasetDelete={onDatasetDelete}
            onDatasetExplore={onDatasetExplore}
          />
        </div>
        {datasetData.length > 0 && selectedDataset && (
          <div className={styles.dataView}>
            <DataView
              data={datasetData}
              datasetId={selectedDataset}
              onClose={closeDatasetData}
              onDataAdd={onDataAdd}
            />
          </div>
        )}
      </div>
      <Footer />
      <NotificationContainer gap="1" bottom right>
        {notifications.map((notification, index: number) => (
          <Notification
            key={notification.id}
            isOpen={notification.isOpen}
            style={{ top: `${index * 60}px` }}
            expireIn={notification.expireIn}
            onClose={notification.delete}
          >
            <Text>{notification.message}</Text>
          </Notification>
        ))}
      </NotificationContainer>
    </main>
  );
}
