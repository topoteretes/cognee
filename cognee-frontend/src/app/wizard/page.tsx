'use client';

import { Suspense, useCallback } from 'react';
import WizardPage from './WizardPage';

export default function Page() {
  const finishWizard = useCallback(() => {
    window.location.href = '/';
  }, []);
  
  return (
    <Suspense>
      <WizardPage
        onFinish={finishWizard}
      />
    </Suspense>
  );
}
