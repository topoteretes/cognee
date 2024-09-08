import { fetch } from '@/utils';

export default function deleteDataset(dataset: { id: string }) {
  return fetch(`/v1/datasets/${dataset.id}`, {
    method: 'DELETE',
  })
}
