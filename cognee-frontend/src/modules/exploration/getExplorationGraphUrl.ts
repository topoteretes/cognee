export default function getExplorationGraphUrl(dataset: { id: string }) {
  return fetch(`http://0.0.0.0:8000/datasets/${dataset.id}/graph`)
      .then((response) => response.text())
      .then((text) => text.replace('"', ''));
}
