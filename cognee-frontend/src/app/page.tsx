'use client';

import { Fragment, useCallback, useEffect, useState } from 'react';
import styles from "./page.module.css";
import { CTAButton, H1, Notification, NotificationContainer, Stack, Text, UploadInput, useBoolean, useNotifications } from 'ohmy-ui';
import useDatasets from '@/modules/ingestion/useDatasets';
import DataView, { Data } from '@/modules/ingestion/DataView';
import DatasetsView from '@/modules/ingestion/DatasetsView';
import classNames from 'classnames';
import { TextLogo, LoadingIndicator } from '@/modules/app';
import { IFrameView } from '@/ui';

export default function Home() {
  const {
    datasets,
    refreshDatasets,
  } = useDatasets();

  const [datasetData, setDatasetData] = useState<Data[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

  const {
    value: isWizardShown,
    setFalse: hideWizard,
  } = useBoolean(true);
  const [wizardStep, setWizardStep] = useState<'add' | 'upload' | 'cognify' | 'explore'>('add');
  const [wizardData, setWizardData] = useState<File[] | null>(null);

  // useEffect(() => {
  //   if (datasets.length > 0) {
  //     hideWizard();
  //   }
  // }, [datasets, hideWizard]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  const openDatasetData = (dataset: { id: string }) => {
    fetch(`http://localhost:8000/datasets/${dataset.id}/data`)
      .then((response) => response.json())
      .then(setDatasetData)
      .then(() => setSelectedDataset(dataset.id));
  };

  const closeDatasetData = () => {
    setDatasetData([]);
    setSelectedDataset(null);
  };

  const { notifications, showNotification } = useNotifications();

  const handleDataAdd = useCallback((dataset: { id: string }, files: File[]) => {
    const formData = new FormData();
    formData.append('datasetId', dataset.id);
    const file = files[0];
    formData.append('data', file, file.name);

    return fetch('http://localhost:8000/add', {
      method: 'POST',
      body: formData,
    })
      .then(() => {
        showNotification("Data added successfully.", 5000);
        openDatasetData(dataset);
      });
  }, [showNotification])

  const addWizardData = useCallback((files: File[]) => {
    setWizardData(files);
    setWizardStep('upload');
  }, []);

  const {
    value: isUploadRunning,
    setTrue: disableUploadRun,
    setFalse: enableUploadRun,
  } = useBoolean(false);
  const uploadWizardData = useCallback(() => {
    disableUploadRun()
    handleDataAdd({ id: 'main' }, wizardData!)
      .then(() => {
        setWizardStep('cognify')
      })
      .finally(() => enableUploadRun());
  }, [disableUploadRun, enableUploadRun, handleDataAdd, wizardData]);

  const cognifyDataset = useCallback((dataset: { id: string }) => {
    showNotification(`Cognification started for dataset "${dataset.id}".`, 5000);

    return fetch('http://localhost:8000/cognify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        datasets: [dataset.id],
      }),
    })
      .then(() => {
        showNotification(`Dataset "${dataset.id}" cognified.`, 5000);
      })
      .catch((error) => {
        console.error(error);
      });
  }, [showNotification]);

  const {
    value: isCognifyRunning,
    setTrue: disableCognifyRun,
    setFalse: enableCognifyRun,
  } = useBoolean(false);
  const cognifyWizardData = useCallback(() => {
    disableCognifyRun();
    cognifyDataset({ id: 'main' })
      .then(() => {
        setWizardStep('explore');
      })
      .finally(() => enableCognifyRun());
  }, [cognifyDataset, disableCognifyRun, enableCognifyRun]);

  const deleteDataset = useCallback((dataset: { id: string }) => {
    fetch(`http://localhost:8000/datasets/${dataset.id}`, {
      method: 'DELETE',
    })
     .then(() => {
        showNotification(`Dataset "${dataset.id}" deleted.`, 5000);
        refreshDatasets();
      })
  }, [refreshDatasets, showNotification]);

  interface ExplorationWindowProps {
    url: string;
    title: string;
  }
  const [explorationWindowProps, setExplorationWindowProps] = useState<ExplorationWindowProps | null>(null);
  const {
    value: isExplorationWindowShown,
    setTrue: showExplorationWindow,
    setFalse: hideExplorationWindow,
  } = useBoolean(false);

  const openExplorationWindow = useCallback((explorationWindowProps: ExplorationWindowProps) => {
    setExplorationWindowProps(explorationWindowProps);
    showExplorationWindow();
  }, [showExplorationWindow]);
  
  const exploreDataset = useCallback((dataset: { id: string }) => {
    fetch(`http://localhost:8000/datasets/${dataset.id}/graph`)
      .then((response) => response.text())
      .then((text) => text.replace('"', ''))
      .then((graphUrl: string) => {
        openExplorationWindow({
          url: graphUrl,
          title: dataset.id,
        });
      });
  }, [openExplorationWindow]);

  const exploreWizardData = useCallback(() => {
    exploreDataset({ id: 'main' });
  }, [exploreDataset]);

  const closeWizard = useCallback(() => {
    hideExplorationWindow();
    hideWizard();
  }, [hideExplorationWindow, hideWizard]);

  if (isWizardShown) {
    return (
      <main className={classNames(styles.main, styles.noData)}>
        <TextLogo />
        <Stack gap="4" orientation="vertical" align="center/center" className={styles.noDataWizardContainer}>
          <H1>Add Knowledge</H1>
          <Stack gap="4" orientation="vertical" align="center/center">
            {wizardStep === 'upload' && wizardData && (
              <div className={styles.wizardDataset}>
                {wizardData.map((file, index) => (
                  <Fragment key={index}>
                    <Text bold>{file.name}</Text>
                    <Text className={styles.fileSize} size="small">
                      {getBiggestUnitSize(file.size)}
                    </Text>
                  </Fragment>
                ))}
              </div>
            )}
            {(wizardStep === 'add' || wizardStep === 'upload') && (
              <Text>No data in the system. Let&apos;s add your data.</Text>
            )}
            {wizardStep === 'cognify' && (
              <Text>Process data and make it explorable.</Text>
            )}
            {wizardStep === 'add' && (
              <UploadInput onChange={addWizardData}>
                <Text>Add data</Text>
              </UploadInput>
            )}
            {wizardStep === 'upload' && (
              <CTAButton disabled={isUploadRunning} onClick={uploadWizardData}>
                <Stack gap="2" orientation="horizontal" align="center/center">
                  <Text>Upload</Text>
                  {isUploadRunning && (
                    <LoadingIndicator />
                  )}
                </Stack>
              </CTAButton>
            )}
            {wizardStep === 'cognify' && (
              <>
                {isCognifyRunning && (
                  <Text>Processing may take a minute, depending on data size.</Text>
                )}
                <CTAButton disabled={isCognifyRunning} onClick={cognifyWizardData}>
                  <Stack gap="2" orientation="horizontal" align="center/center">
                    <Text>Cognify</Text>
                    {isCognifyRunning && (
                      <LoadingIndicator />
                    )}
                  </Stack>
                </CTAButton>
              </>
            )}
            {wizardStep === 'explore' && (
              <>
                {!isExplorationWindowShown && (
                  <CTAButton onClick={exploreWizardData}>
                    <Text>Start exploring the data</Text>
                  </CTAButton>
                )}
                {isExplorationWindowShown && (
                  <IFrameView
                    src={explorationWindowProps!.url}
                    title={explorationWindowProps!.title}
                    onClose={closeWizard}
                  />
                )}
              </>
            )}
          </Stack>
        </Stack>
      </main>
    );
  }

  return (
    <main className={styles.main}>
      <div className={classNames(styles.datasetsView, {
        [styles.openDatasetData]: datasetData.length > 0,
      })}>
        <DatasetsView
          datasets={datasets}
          onDataAdd={handleDataAdd}
          onDatasetClick={openDatasetData}
          onDatasetCognify={cognifyDataset}
          onDatasetDelete={deleteDataset}
          onDatasetExplore={exploreDataset}
        />
        {isExplorationWindowShown && (
          <IFrameView
            src={explorationWindowProps!.url}
            title={explorationWindowProps!.title}
            onClose={hideExplorationWindow}
          />
        )}
      </div>
      {datasetData.length > 0 && selectedDataset && (
        <div className={styles.dataView}>
          <DataView
            data={datasetData}
            datasetId={selectedDataset}
            onClose={closeDatasetData}
            onDataAdd={handleDataAdd}
          />
        </div>
      )}
      <NotificationContainer gap="1" bottom right>
        {notifications.map((notification, index) => (
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

function getBiggestUnitSize(sizeInBytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];

  let i = 0;
  while (sizeInBytes >= 1024 && i < units.length - 1) {
    sizeInBytes /= 1024;
    i++;
  }
  return `${sizeInBytes.toFixed(2)} ${units[i]}`;
}
