import { fetch } from '@/utils';

export default function cognifyDataset(dataset: { id?: string, name?: string }, onUpdate = (data: []) => {}) {
  return fetch('/v1/cognify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      datasets: [dataset.id || dataset.name],
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      const websocket = new WebSocket(`ws://localhost:8000/api/v1/cognify/subscribe/${data.pipeline_run_id}`);

      websocket.onopen = () => {
        websocket.send(JSON.stringify({
          "Authorization": `Bearer ${localStorage.getItem("access_token")}`,
        }));
      };

      let isCognifyDone = false;
      
      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onUpdate(data);

        if (data.status === "PipelineRunCompleted") {
          isCognifyDone = true;
          websocket.close();
        }
      };

      return new Promise(async (resolve) => {
        while (!isCognifyDone) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

        resolve(true);
      });
    });
}
