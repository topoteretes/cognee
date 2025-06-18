import { fetch } from "@/utils";
// import getDatasetGraph from "./getDatasetGraph";
import { Dataset } from "../ingestion/useDatasets";

interface GraphData {
  nodes: { id: string; label: string; properties?: object }[];
  edges: { source: string; target: string; label: string }[];
}

export default async function cognifyDataset(dataset: Dataset, onUpdate: (data: GraphData) => void) {
  const data = await (
    fetch("/v1/cognify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        datasetIds: [dataset.id],
        runInBackground: true,
      }),
    })
    .then((response) => response.json())
  );
    // .then(() => {
    //   return getDatasetGraph(dataset)
    //     .then((data) => {
    //       onUpdate({
    //         nodes: data.nodes,
    //         edges: data.edges,
    //       });
    //     });
    // });

    const websocket = new WebSocket(`ws://localhost:8000/api/v1/cognify/subscribe/${data.pipeline_run_id}`);

    // websocket.onopen = () => {
    //   websocket.send(JSON.stringify({
    //     "Authorization": `Bearer ${localStorage.getItem("access_token")}`,
    //   }));
    // };

    let isCognifyDone = false;

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onUpdate?.({
        nodes: data.payload.nodes,
        edges: data.payload.edges,
      });

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
}
