import { useState } from 'react';
import { CloseIcon, GhostButton, Spacer, Stack, useBoolean } from 'ohmy-ui';
import { TextLogo } from '@/ui/App';
import { SettingsIcon } from '@/ui/Icons';
import { Footer, SettingsModal } from '@/ui/Partials';
import ConfigStep from './ConfigStep';
import AddStep from './AddStep';
import CognifyStep from './CognifyStep';
import ExploreStep from './ExploreStep';
import { WizardContent } from '@/ui/Partials/Wizard';

import styles from './WizardPage.module.css';
import { Divider } from '@/ui/Layout';
import { useSearchParams } from 'next/navigation';

interface WizardPageProps {
  onFinish: () => void;
}

export default function WizardPage({
  onFinish,
}: WizardPageProps) {
  const searchParams = useSearchParams()
  const presetWizardStep = searchParams.get('step') as 'config';
  const [wizardStep, setWizardStep] = useState<'config' | 'add' | 'cognify' | 'explore'>(presetWizardStep || 'config');
  const {
    value: isSettingsModalOpen,
    setTrue: openSettingsModal,
    setFalse: closeSettingsModal,
  } = useBoolean(false);

  const dataset = { name: 'main' };

  return (
    <main className={styles.main}>
      <Spacer inset vertical="2" horizontal="2">
        <Stack orientation="horizontal" gap="between" align="center">
          <TextLogo width={158} height={44} color="white" />
          {wizardStep === 'explore' && (
            <GhostButton hugContent onClick={onFinish}>
              <CloseIcon />
            </GhostButton>
          )}
          {wizardStep === 'add' && (
            <GhostButton hugContent onClick={openSettingsModal}>
              <SettingsIcon />
            </GhostButton>
          )}
        </Stack>
      </Spacer>
      <Divider />
      <SettingsModal isOpen={isSettingsModalOpen} onClose={closeSettingsModal} />
      <div className={styles.wizardContainer}>
        {wizardStep === 'config' && (
          <WizardContent>
            <ConfigStep onNext={() => setWizardStep('add')} />
          </WizardContent>
        )}

        {wizardStep === 'add' && (
          <WizardContent>
            <AddStep onNext={() => setWizardStep('cognify')} />
          </WizardContent>
        )}

        {wizardStep === 'cognify' && (
          <WizardContent>
            <CognifyStep dataset={dataset} onNext={() => setWizardStep('explore')} />
          </WizardContent>
        )}

        {wizardStep === 'explore' && (
          <Spacer inset top="4" bottom="1" horizontal="4">
            <ExploreStep dataset={dataset} />
          </Spacer>
        )}
      </div>
      <Spacer inset horizontal="3" wrap>
        <Footer />
      </Spacer>
    </main>
  )
}
