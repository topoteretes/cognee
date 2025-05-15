import { Explorer } from '@/ui/Partials';
import { Spacer } from 'ohmy-ui';

interface ExploreStepProps {
  dataset: { name: string };
}

export default function ExploreStep({ dataset }: ExploreStepProps) {
  return (
    <Spacer horizontal="3">
      <Explorer dataset={dataset} />
    </Spacer>
  )
}
