import { useCallback, useState } from 'react';
import { IFrameView } from '@/ui';
import { SettingsIcon } from '@/ui/Icons';
import { LoadingIndicator, TextLogo } from '@/modules/app';
import { CTAButton, GhostButton, H1, Stack, Text, UploadInput, useBoolean } from 'ohmy-ui';
import { Footer, SettingsModal } from '@/ui/Partials';

import styles from './WizardPage.module.css';

interface ExplorationWindowConfig {
  url: string;
  title: string;
}

interface WizardPageProps {
  onDataAdd: (dataset: { id: string }, files: File[]) => Promise<void>;
  onDataCognify: (dataset: { id: string }) => Promise<void>;
  onDataExplore: (dataset: { id: string }) => Promise<ExplorationWindowConfig>;
  onFinish: () => void;
}

export default function WizardPage({
  onDataAdd,
  onDataCognify,
  onDataExplore,
  onFinish,
}: WizardPageProps) {
  const [wizardStep, setWizardStep] = useState<'add' | 'upload' | 'cognify' | 'explore'>('add');
  const [wizardData, setWizardData] = useState<File[] | null>(null);
  
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
    onDataAdd({ id: 'main' }, wizardData!)
      .then(() => {
        setWizardStep('cognify')
      })
      .finally(() => enableUploadRun());
  }, [disableUploadRun, enableUploadRun, onDataAdd, wizardData]);

  const {
    value: isCognifyRunning,
    setTrue: disableCognifyRun,
    setFalse: enableCognifyRun,
  } = useBoolean(false);
  const cognifyWizardData = useCallback(() => {
    disableCognifyRun();
    onDataCognify({ id: 'main' })
      .then(() => {
        setWizardStep('explore');
      })
      .finally(() => enableCognifyRun());
  }, [onDataCognify, disableCognifyRun, enableCognifyRun]);

  const {
    value: isExploreLoading,
    setTrue: startLoadingExplore,
    setFalse: finishLoadingExplore,
  } = useBoolean(false);

  const [explorationWindowProps, setExplorationWindowProps] = useState<ExplorationWindowConfig | null>(null);

  const {
    value: isExplorationWindowShown,
    setTrue: showExplorationWindow,
  } = useBoolean(false);

  const openExplorationWindow = useCallback((explorationWindowProps: ExplorationWindowConfig) => {
    setExplorationWindowProps(explorationWindowProps);
    showExplorationWindow();
  }, [showExplorationWindow]);

  const exploreWizardData = useCallback(() => {
    startLoadingExplore();
    onDataExplore({ id: 'main' })
      .then((exploreWindowConfig) => {
        openExplorationWindow(exploreWindowConfig);
      })
      .finally(() => {
        finishLoadingExplore();
      });
  }, [finishLoadingExplore, onDataExplore, openExplorationWindow, startLoadingExplore]);
  
  const {
    value: isSettingsModalOpen,
    setTrue: openSettingsModal,
    setFalse: closeSettingsModal,
  } = useBoolean(false);

  return (
    <main className={styles.main}>
      <Stack orientation="horizontal" gap="between" align="center">
        <TextLogo />
        <GhostButton onClick={openSettingsModal}>
          <SettingsIcon />
        </GhostButton>
      </Stack>
      <SettingsModal isOpen={isSettingsModalOpen} onClose={closeSettingsModal} />
      <Stack gap="4" orientation="vertical" align="center/center" className={styles.wizardContainer}>
        {wizardStep === 'explore'
          ? (<H1>Explore the Knowledge</H1>)
          : (<H1>Add Knowledge</H1>)}
        <Stack gap="4" orientation="vertical" align="center/center">
          {wizardStep === 'upload' && wizardData && (
            <Stack gap="4" className={styles.wizardDataset}>
              {wizardData.map((file, index) => (
                <div key={index}>
                  <Text bold>{file.name}</Text>
                  <Text className={styles.fileSize} size="small">
                    {getBiggestUnitSize(file.size)}
                  </Text>
                </div>
              ))}
            </Stack>
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
                  <Stack gap="2" orientation="horizontal" align="center/center">
                    <Text>Start exploring</Text>
                    {isExploreLoading && (
                      <LoadingIndicator />
                    )}
                  </Stack>
                </CTAButton>
              )}
              {isExplorationWindowShown && (
                <IFrameView
                  src={explorationWindowProps!.url}
                  title={explorationWindowProps!.title}
                  onClose={onFinish}
                />
              )}
            </>
          )}
        </Stack>
      </Stack>
      <Footer />
    </main>
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
