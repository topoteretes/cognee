import { useState } from 'react';
import Link from 'next/link';
import { Explorer } from '@/ui/Partials';
import StatusIcon from './StatusIcon';
import { LoadingIndicator } from '@/ui/App';
import { DropdownMenu, GhostButton, Stack, Text, CTAButton, useBoolean, Modal, Spacer } from "ohmy-ui";
import styles from "./DatasetsView.module.css";

interface Dataset {
  id: string;
  name: string;
  status: string;
}

const DatasetItem = GhostButton.remix({ Component: 'div' });

interface DatasetsViewProps {
  datasets: Dataset[];
  onDatasetClick: (dataset: Dataset) => void;
  onDatasetCognify: (dataset: Dataset) => Promise<void>;
}

export default function DatasetsView({
  datasets,
  onDatasetClick,
  onDatasetCognify,
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

  const [dataset, setExplorationDataset] = useState<{ id: string, name: string } | null>(null);
  const {
    value: isExplorationWindowShown,
    setTrue: showExplorationWindow,
    setFalse: hideExplorationWindow,
  } = useBoolean(false);

  const handleExploreDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();

    setExplorationDataset(dataset);
    showExplorationWindow();
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
                    {dataset.status === 'DATASET_PROCESSING_COMPLETED' ? (
                      <CTAButton
                        onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleExploreDataset(event, dataset)}
                      >
                        <Text>Explore</Text>
                      </CTAButton>
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
                    <Link href="/wizard?step=add">
                      <GhostButton>
                        <Text>Add data</Text>
                      </GhostButton>
                    </Link>
                  </Stack>
                </DropdownMenu>
              </Stack>
            </Stack>
          </DatasetItem>
        ))}
      </Stack>
      <Modal onClose={hideExplorationWindow} isOpen={isExplorationWindowShown} className={styles.explorerModal}>
        <Spacer horizontal="2" vertical="3" wrap>
          <Text>{dataset?.name}</Text>
        </Spacer> 
        <Explorer dataset={dataset!} />
      </Modal>
    </>
  );
}
