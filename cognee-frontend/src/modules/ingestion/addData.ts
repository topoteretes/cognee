import { fetch } from '@/utils';

export default function addData(dataset: { id: string }, files: File[]) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('data', file, file.name);
  })
  formData.append('datasetId', dataset.id);

  return fetch('/v1/add', {
    method: 'POST',
    body: formData,
  }).then((response) => response.json());
}
