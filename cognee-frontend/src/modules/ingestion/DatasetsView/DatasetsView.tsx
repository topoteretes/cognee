import { useCallback, useState } from 'react';
import { IFrameView } from '@/ui';
import StatusIcon from './StatusIcon';
import { LoadingIndicator } from '@/modules/app';
import { DropdownMenu, GhostButton, Stack, Text, UploadInput, CTAButton, useBoolean, NeutralButton } from "ohmy-ui";
import styles from "./DatasetsView.module.css";
import { SearchView } from '@/ui/Partials';

interface Dataset {
  id: string;
  name: string;
  status: string;
}

const DatasetItem = GhostButton.mixClassName()("div")

interface ExplorationWindowConfig {
  url: string;
  title: string;
}

interface DatasetsViewProps {
  datasets: Dataset[];
  onDataAdd: (dataset: Dataset, files: File[]) => void;
  onDatasetClick: (dataset: Dataset) => void;
  onDatasetDelete: (dataset: Dataset) => void;
  onDatasetCognify: (dataset: Dataset) => Promise<void>;
  onDatasetExplore: (dataset: Dataset) => Promise<string>;
}

export default function DatasetsView({
  datasets,
  onDatasetClick,
  onDataAdd,
  onDatasetCognify,
  onDatasetDelete,
  onDatasetExplore,
}: DatasetsViewProps) {
  const {
    value: isCognifyRunning,
    setTrue: disableCognifyRun,
    setFalse: enableCognifyRun,
  } = useBoolean(false);

  const handleCognifyDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();

    disableCognifyRun();

    onDatasetCognify(dataset)
      .finally(() => enableCognifyRun());
  }

  // const handleDatasetDelete = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
  //   event.stopPropagation();
  //   onDatasetDelete(dataset);
  // }

  const [explorationWindowProps, setExplorationWindowProps] = useState<ExplorationWindowConfig | null>(null);
  const {
    value: isExplorationWindowShown,
    setTrue: showExplorationWindow,
    setFalse: hideExplorationWindow,
  } = useBoolean(false);

  const openExplorationWindow = useCallback((explorationWindowProps: ExplorationWindowConfig) => {
    setExplorationWindowProps(explorationWindowProps);
    showExplorationWindow();
  }, [showExplorationWindow]);

  const {
    value: isExploreLoading,
    setTrue: startLoadingExplore,
    setFalse: finishLoadingExplore,
  } = useBoolean(false);
  const handleExploreDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();

    startLoadingExplore();
    onDatasetExplore(dataset)
      .then((explorationWindowUrl) => {
        openExplorationWindow({
          url: explorationWindowUrl,
          title: dataset.id,
        });
      })
      .finally(() => finishLoadingExplore());
  }
  
  const handleDataAdd = (dataset: Dataset, files: File[]) => {
    onDataAdd(dataset, files);
  }

  const {
    value: isSearchWindowOpen,
    setTrue: openSearchWindow,
    setFalse: closeSearchWindow,
  } = useBoolean(false);
  
  const handleSearchDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();
    openSearchWindow();
  }
  
  return (
    <>
      <Stack orientation="vertical" gap="4">
        {datasets.map((dataset) => (
          <DatasetItem key={dataset.id} onClick={() => onDatasetClick(dataset)}>
            <Stack orientation="horizontal" gap="between" align="start/center">
              <Text>{dataset.name}</Text>
              <Stack orientation="horizontal" gap="2" align="center">
                <StatusIcon status={dataset.status} />
                <DropdownMenu>
                  <Stack gap="1" className={styles.datasetMenu} orientation="vertical">
                    {dataset.status === 'DATASET_PROCESSING_FINISHED' ? (
                      <>
                        <CTAButton
                          onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleExploreDataset(event, dataset)}
                        >
                          <Stack gap="2" orientation="horizontal" align="center/center">
                            <Text>Explore</Text>
                            {isExploreLoading && (
                              <LoadingIndicator />
                            )}
                          </Stack>
                        </CTAButton>
                        <NeutralButton
                          onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleSearchDataset(event, dataset)}
                        >
                          <Text>Search</Text>
                        </NeutralButton>
                      </>
                    ) : (
                      <CTAButton
                        onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleCognifyDataset(event, dataset)}
                      >
                        <Stack gap="2" orientation="horizontal" align="center/center">
                          <Text>Cognify</Text>
                          {isCognifyRunning && (
                            <LoadingIndicator />
                          )}
                        </Stack>
                      </CTAButton>
                    )}
                    <UploadInput as={GhostButton} onChange={(files: File[]) => handleDataAdd(dataset, files)}>
                      <Text>Add data</Text>
                    </UploadInput>
                    {/* <NegativeButton
                      onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleDatasetDelete(event, dataset)}
                    >
                      <Text>Delete</Text>
                    </NegativeButton> */}
                  </Stack>
                </DropdownMenu>
              </Stack>
            </Stack>
          </DatasetItem>
        ))}
      </Stack>
      {isSearchWindowOpen && (
        <SearchView onClose={closeSearchWindow} />
      )}
      {isExplorationWindowShown && (
        <IFrameView
          src={explorationWindowProps!.url}
          title={explorationWindowProps!.title}
          onClose={hideExplorationWindow}
        />
      )}
    </>
  );
}
