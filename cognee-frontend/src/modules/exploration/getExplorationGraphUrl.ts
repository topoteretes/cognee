import { fetch } from '@/utils';

export default function getExplorationGraphUrl(dataset: { name: string }) {
  return fetch('/v1/visualize')
      .then(async (response) => {
        if (response.status !== 200) {
          throw new Error((await response.text()).replaceAll("\"", ""));
        }
        return response;
      })
      .then((response) => response.text());
}
