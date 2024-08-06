export default function cognifyDataset(dataset: { id: string, name: string }) {
  return fetch('http://127.0.0.1:8000/cognify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      datasets: [dataset.id],
    }),
  }).then((response) => response.json());
}
