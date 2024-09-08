import { fetch } from '@/utils';

export default function cognifyDataset(dataset: { id: string, name: string }) {
  return fetch('/v1/cognify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      datasets: [dataset.id],
    }),
  }).then((response) => response.json());
}
