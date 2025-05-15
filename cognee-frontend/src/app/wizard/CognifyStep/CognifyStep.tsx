import { useEffect, useRef } from 'react';
import { CTAButton, Stack, Text, useBoolean } from 'ohmy-ui';
import { Divider } from '@/ui/Layout';
import { CognifyLoadingIndicator } from '@/ui/App';
import { WizardHeading } from '@/ui/Partials/Wizard';
import cognifyDataset from '@/modules/datasets/cognifyDataset';

interface ConfigStepProps {
  onNext: () => void;
  dataset: { name: string }
}

export default function CognifyStep({ onNext, dataset }: ConfigStepProps) {
  const {
    value: isCognifyRunning,
    setFalse: stopCognifyIndicator,
  } = useBoolean(true);
  const cognifyPromise = useRef<Promise<void>>()

  useEffect(() => {
    if (cognifyPromise.current) {
      return;
    }
    
    cognifyPromise.current = cognifyDataset(dataset)
      .then(() => {
        stopCognifyIndicator();        
      });
  }, [stopCognifyIndicator, dataset]);

  return (
    <Stack orientation="vertical" gap="6">
      <WizardHeading><Text light size="large">Step 3/3</Text> Cognify</WizardHeading>
      <Divider />

      <Stack align="/center">
        <CognifyLoadingIndicator isLoading={isCognifyRunning} />
      </Stack>

      <Text align="center">
        Cognee decomposes your data into facts and connects them in relevant clusters,
        so that you can navigate your knowledge better.
      </Text>
      <CTAButton disabled={isCognifyRunning} onClick={onNext}>
        <Stack gap="2" orientation="horizontal" align="center/center">
          <Text>Explore data</Text>
        </Stack>
      </CTAButton>
    </Stack>
  )
}
