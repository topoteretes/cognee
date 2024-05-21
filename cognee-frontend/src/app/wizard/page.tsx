'use client';

import { useCallback } from 'react';
import WizardPage from './WizardPage';
import addData from '@/modules/ingestion/addData';
import cognifyDataset from '@/modules/datasets/cognifyDataset';
import getExplorationGraphUrl from '@/modules/exploration/getExplorationGraphUrl';

export default function Page() {
  const onDataExplore = useCallback((dataset: { id: string }) => {
    return getExplorationGraphUrl(dataset)
        .then((explorationWindowUrl) => {
          return {
            url: explorationWindowUrl,
            title: dataset.id,
          };
        });
  }, []);

  const finishWizard = useCallback(() => {
    window.location.href = '/';
  }, []);
  
  return (
    <WizardPage
      onDataAdd={addData}
      onDataCognify={cognifyDataset}
      onDataExplore={onDataExplore}
      onFinish={finishWizard}
    />
  );
}
