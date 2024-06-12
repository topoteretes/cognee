export default function getExplorationGraphUrl(dataset: { id: string }) {
  return fetch(`http://0.0.0.0:8000/datasets/${dataset.id}/graph`)
      .then(async (response) => {
        if (response.status !== 200) {
          throw new Error((await response.text()).replaceAll("\"", ""));
        }
        return response;
      })
      .then((response) => response.text())
      .then((text) => text.replace('"', ''));
}
