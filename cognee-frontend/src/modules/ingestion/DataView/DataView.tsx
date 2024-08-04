import { useCallback, useState } from 'react';
import {
  DropdownMenu,
  GhostButton,
  Stack,
  Text,
  UploadInput,
  CloseIcon,
} from "ohmy-ui";
import styles from "./DataView.module.css";
import RawDataPreview from './RawDataPreview';

export interface Data {
  id: string;
  name: string;
  filePath: string;
  mimeType: string;
}

interface DatasetLike {
  id: string;
}

interface DataViewProps {
  data: Data[];
  datasetId: string;
  onClose: () => void;
  onDataAdd: (dataset: DatasetLike, files: File[]) => void;
}

export default function DataView({ datasetId, data, onClose, onDataAdd }: DataViewProps) {
  // const handleDataDelete = () => {};
  const [rawData, setRawData] = useState<ArrayBuffer | null>(null);
  const [selectedData, setSelectedData] = useState<Data | null>(null);

  const showRawData = useCallback((dataItem: Data) => {
    setSelectedData(dataItem);

    fetch(`http://127.0.0.1:8000/datasets/${datasetId}/data/${dataItem.id}/raw`)
      .then((response) => response.arrayBuffer())
      .then(setRawData);

    document.body.click(); // Close the dropdown menu.
  }, [datasetId]);

  const resetDataPreview = useCallback(() => {
    setSelectedData(null);
    setRawData(null);
  }, []);

  const handleDataAdd = (files: File[]) => {
    onDataAdd({ id: datasetId }, files);
  }

  return (
    <Stack orientation="vertical" gap="4">
      <Stack gap="2" orientation="horizontal" align="/end">
        <div>
          <UploadInput onChange={handleDataAdd}>
            <Text>Add data</Text>
          </UploadInput>
        </div>
        <GhostButton hugContent onClick={onClose}>
          <CloseIcon />
        </GhostButton>
      </Stack>
      {rawData && selectedData && (
        <RawDataPreview
          fileName={selectedData.name}
          rawData={rawData}
          onClose={resetDataPreview}
        />
      )}
      <div className={styles.tableContainer}>
        <table className={styles.dataTable}>
          <thead>
            <tr>
              <th>Actions</th>
              <th>ID</th>
              <th>Name</th>
              <th>File path</th>
              <th>MIME type</th>
              <th>Keywords</th>
            </tr>
          </thead>
          <tbody>
            {data.map((dataItem) => (
              <tr key={dataItem.id}>
                <td>
                  <Stack orientation="horizontal" gap="2" align="center">
                    <DropdownMenu position="right">
                      <Stack gap="1" className={styles.datasetMenu} orientation="vertical">
                        <GhostButton onClick={() => showRawData(dataItem)}>
                          <Text>View raw data</Text>
                        </GhostButton>
                        {/* <NegativeButton onClick={handleDataDelete}>
                          <Text>Delete</Text>
                        </NegativeButton> */}
                      </Stack>
                    </DropdownMenu>
                  </Stack>
                </td>
                <td>
                  <Text>{dataItem.id}</Text>
                </td>
                <td>
                  <Text>{dataItem.name}</Text>
                </td>
                <td>
                  <Text>{dataItem.filePath}</Text>
                </td>
                <td>
                  <Text>{dataItem.mimeType}</Text>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Stack>
  );
}
