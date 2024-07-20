export default function addData(dataset: { id: string }, files: File[]) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('data', file, file.name);
  })
  formData.append('datasetId', dataset.id);

  return fetch('http://0.0.0.0:8000/add', {
    method: 'POST',
    body: formData,
  }).then((response) => response.json());
}
