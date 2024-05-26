import { H1 } from 'ohmy-ui';

interface WizardHeadingProps {
  children: React.ReactNode;
}

export default function WizardHeading({ children, ...props }: WizardHeadingProps) {
  return (
    <H1 {...props} align="center" size="small" style={{ color: '#40A9FF' }}>{children}</H1>
  );
}