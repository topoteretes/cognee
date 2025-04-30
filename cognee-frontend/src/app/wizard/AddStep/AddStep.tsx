import { useCallback, useState } from 'react';
import { CTAButton, GhostButton, Stack, Text, TrashIcon, UploadIcon, UploadInput, useBoolean } from 'ohmy-ui';
import { Divider } from '@/ui/Layout';
import addData from '@/modules/ingestion/addData';
import { LoadingIndicator } from '@/ui/App';
import styles from './AddStep.module.css';
import { WizardHeading } from '@/ui/Partials/Wizard';

interface ConfigStepProps {
  onNext: () => void;
}

export default function AddStep({ onNext }: ConfigStepProps) {
  const [files, setFiles] = useState<File[]>([]);

  const {
    value: isUploading,
    setTrue: disableUploading,
    setFalse: enableUploading,
  } = useBoolean(false);

  const uploadFiles = useCallback(() => {
    disableUploading()
    addData({ name: 'main' }, files)
      .then(() => {
        onNext();
      })
      .finally(() => enableUploading());
  }, [disableUploading, enableUploading, files, onNext]);

  const addFiles = useCallback((files: File[]) => {
    setFiles((existingFiles) => {
      const newFiles = files.filter((file) => !existingFiles.some((existingFile) => existingFile.name === file.name));

      return [...existingFiles, ...newFiles]
    });
  }, []);

  const removeFile = useCallback((file: File) => {
    setFiles((files) => files.filter((f) => f !== file));
  }, []);

  return (
    <Stack orientation="vertical" gap="6">
      <WizardHeading><Text light size="large">Step 2/3</Text> Add knowledge</WizardHeading>
      <Divider />
      <Text align="center">
        Cognee lets you process your personal data, books, articles or company data.
        Simply add datasets to get started.
      </Text>
      <Stack gap="1">
        <UploadInput onChange={addFiles}>
          <Stack gap="2" orientation="horizontal" align="center/center">
            <UploadIcon key={files.length} />
            <Text>Upload your data</Text>
          </Stack>
        </UploadInput>
        <Stack gap="3" className={styles.files}>
          {files.map((file, index) => (
            <Stack gap="between" orientation="horizontal" align="center/" key={index}>
              <div key={index}>
                <Text bold>{file.name}</Text>
                <Text className={styles.fileSize} size="small">
                  {getBiggestUnitSize(file.size)}
                </Text>
              </div>
              <GhostButton hugContent onClick={() => removeFile(file)}>
                <TrashIcon />
              </GhostButton>
            </Stack>
          ))}
        </Stack>
      </Stack>
      <Stack align="/end">
        <CTAButton disabled={isUploading || files.length === 0} onClick={uploadFiles}>
          <Stack gap="2" orientation="horizontal" align="center/center">
            <Text>Next</Text>
            {isUploading && (
              <LoadingIndicator />
            )}
          </Stack>
        </CTAButton>
      </Stack>
    </Stack>
  )
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
