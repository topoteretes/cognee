import { DropdownMenu, GhostButton, Stack, Text, UploadInput, CTAButton } from "ohmy-ui";
import styles from "./DatasetsView.module.css";
import StatusIcon from './StatusIcon';

interface Dataset {
  id: string;
  name: string;
  status: string;
}

const DatasetItem = GhostButton.mixClassName()("div")

interface DatasetsViewProps {
  datasets: Dataset[];
  onDataAdd: (dataset: Dataset, files: File[]) => void;
  onDatasetClick: (dataset: Dataset) => void;
  onDatasetDelete: (dataset: Dataset) => void;
  onDatasetCognify: (dataset: Dataset) => void;
  onDatasetExplore: (dataset: Dataset) => void;
}

export default function DatasetsView({
  datasets,
  onDatasetClick,
  onDataAdd,
  onDatasetCognify,
  onDatasetDelete,
  onDatasetExplore,
}: DatasetsViewProps) {
  const handleCognifyDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();
    onDatasetCognify(dataset);
  }

  // const handleDatasetDelete = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
  //   event.stopPropagation();
  //   onDatasetDelete(dataset);
  // }

  const handleExploreDataset = (event: React.MouseEvent<HTMLButtonElement>, dataset: Dataset) => {
    event.stopPropagation();
    onDatasetExplore(dataset);
  }
  
  const handleDataAdd = (dataset: Dataset, files: File[]) => {
    onDataAdd(dataset, files);
  }
  
  return (
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
                    <CTAButton
                      onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleExploreDataset(event, dataset)}
                    >
                      <Text>Explore</Text>
                    </CTAButton>
                  ) : (
                    <CTAButton
                      onClick={(event: React.MouseEvent<HTMLButtonElement>) => handleCognifyDataset(event, dataset)}
                    >
                      <Text>Cognify</Text>
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
  );
}
