'use client';

import { useCallback, useEffect, useState } from 'react';
import styles from "./page.module.css";
import { GhostButton, Notification, NotificationContainer, Spacer, Stack, Text, useBoolean, useNotifications } from 'ohmy-ui';
import useDatasets from '@/modules/ingestion/useDatasets';
import DataView, { Data } from '@/modules/ingestion/DataView';
import DatasetsView from '@/modules/ingestion/DatasetsView';
import classNames from 'classnames';
import addData from '@/modules/ingestion/addData';
import cognifyDataset from '@/modules/datasets/cognifyDataset';
import getDatasetData from '@/modules/datasets/getDatasetData';
import { Footer, SettingsModal } from '@/ui/Partials';
import { TextLogo } from '@/ui/App';
import { SettingsIcon } from '@/ui/Icons';

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

  const onDatasetCognify = useCallback((dataset: { id: string, name: string }) => {
    showNotification(`Cognification started for dataset "${dataset.name}".`, 5000);

    return cognifyDataset(dataset)
      .then(() => {
        showNotification(`Dataset "${dataset.name}" cognified.`, 5000);
      })
      .catch(() => {
        showNotification(`Dataset "${dataset.name}" cognification failed. Please try again.`, 5000);
      });
  }, [showNotification]);

  const {
    value: isSettingsModalOpen,
    setTrue: openSettingsModal,
    setFalse: closeSettingsModal,
  } = useBoolean(false);

  return (
    <main className={styles.main}>
      <Spacer inset vertical="2" horizontal="2">
        <Stack orientation="horizontal" gap="between" align="center">
          <TextLogo width={158} height={44} color="white" />
          <GhostButton hugContent onClick={openSettingsModal}>
            <SettingsIcon />
          </GhostButton>
        </Stack>
      </Spacer>
      <SettingsModal isOpen={isSettingsModalOpen} onClose={closeSettingsModal} />
      <Spacer inset vertical="1" horizontal="3">
        <div className={styles.data}>
          <div className={classNames(styles.datasetsView, {
            [styles.openDatasetData]: datasetData.length > 0,
          })}>
            <DatasetsView
              datasets={datasets}
              onDatasetClick={openDatasetData}
              onDatasetCognify={onDatasetCognify}
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
      </Spacer>
      <Spacer inset horizontal="3" wrap>
        <Footer />
      </Spacer>
      <NotificationContainer gap="1" bottom right>
        {notifications.map((notification, index: number) => (
          <Notification
            key={notification.id}
            isOpen={notification.isOpen}
            style={{ top: `${index * 60}px` }}
            expireIn={notification.expireIn}
            onClose={notification.delete}
          >
            <Text nowrap>{notification.message}</Text>
          </Notification>
        ))}
      </NotificationContainer>
    </main>
  );
}
