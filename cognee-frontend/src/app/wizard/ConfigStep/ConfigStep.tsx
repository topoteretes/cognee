import { Stack, Text } from 'ohmy-ui';
import { Divider } from '@/ui/Layout';
import Settings from '@/ui/Partials/SettingsModal/Settings';
import { WizardContent, WizardHeading } from '@/ui/Partials/Wizard';

interface ConfigStepProps {
  onNext: () => void;
}

export default function ConfigStep({ onNext }: ConfigStepProps) {
  return (
    <Stack orientation="vertical" gap="6">
      <WizardHeading><Text light size="large">Step 1/3</Text> Basic configuration</WizardHeading>
      <Divider />
      <Text align="center">
        Cognee helps you process your data and create a mind-like structure you can explore.
        To get started you need an OpenAI API key.
      </Text>
      <Settings onDone={onNext} submitButtonText="Next" />
    </Stack>
  )
}
