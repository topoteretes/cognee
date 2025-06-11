import { fetch } from "@/utils";
import getDatasetGraph from "./getDatasetGraph";
import { Dataset } from "../ingestion/useDatasets";

interface GraphData {
  nodes: { id: string | number; label: string; properties?: {} }[];
  edges: { source: string; target: string; label: string }[];
}

export default function cognifyDataset(dataset: Dataset, onUpdate = (data: GraphData) => {}) {
  return fetch("/v1/cognify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset_ids: [dataset.id],
    }),
  })
    .then((response) => response.json())
    .then(() => {
      return getDatasetGraph(dataset)
        .then((data) => {
          onUpdate({
            nodes: data.nodes,
            edges: data.edges,
          });
        });
    });
    // Uncomment after the websockets are merged
    // .then((data) => {
    //   const websocket = new WebSocket(`ws://localhost:8000/api/v1/cognify/subscribe/${data.pipeline_run_id}`);

    //   websocket.onopen = () => {
    //     websocket.send(JSON.stringify({
    //       "Authorization": `Bearer ${localStorage.getItem("access_token")}`,
    //     }));
    //   };

    //   let isCognifyDone = false;
      
    //   websocket.onmessage = (event) => {
    //     const data = JSON.parse(event.data);
    //     onUpdate({
    //       nodes: data.payload.nodes,
    //       edges: data.payload.edges,
    //     });

    //     if (data.status === "PipelineRunCompleted") {
    //       isCognifyDone = true;
    //       websocket.close();
    //     }
    //   };

    //   return new Promise(async (resolve) => {
    //     while (!isCognifyDone) {
    //       await new Promise(resolve => setTimeout(resolve, 1000));
    //     }

    //     resolve(true);
    //   });
    // });
}
