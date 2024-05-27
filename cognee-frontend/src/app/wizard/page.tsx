'use client';

import { useCallback } from 'react';
import WizardPage from './WizardPage';

export default function Page() {
  const finishWizard = useCallback(() => {
    window.location.href = '/';
  }, []);
  
  return (
    <WizardPage
      onFinish={finishWizard}
    />
  );
}
