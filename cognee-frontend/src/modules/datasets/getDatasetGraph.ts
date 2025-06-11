import { fetch } from '@/utils';

export default function getDatasetGraph(dataset: { id: string }) {
  return fetch(`/v1/datasets/${dataset.id}/graph`)
      .then((response) => response.json());
}
