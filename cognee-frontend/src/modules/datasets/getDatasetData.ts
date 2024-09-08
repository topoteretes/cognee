import { fetch } from '@/utils';

export default function getDatasetData(dataset: { id: string }) {
  return fetch(`/v1/datasets/${dataset.id}/data`)
      .then((response) => response.json());
}
