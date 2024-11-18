import { fetch } from '@/utils';

export default function getHistory() {
  return fetch(
    '/v1/search',
  )
    .then((response) => response.json());
}
