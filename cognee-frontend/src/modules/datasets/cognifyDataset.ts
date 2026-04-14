// import getDatasetGraph from "./getDatasetGraph";
import { Dataset } from "../ingestion/useDatasets";
import { CogneeInstance } from "../instances/types";

// interface GraphData {
//   nodes: { id: string; label: string; properties?: object }[];
//   edges: { source: string; target: string; label: string }[];
// }

interface CognifyOptions {
  graphModel?: object;
  customPrompt?: string;
  llmModel?: string;
}

export default async function cognifyDataset(
  dataset: Dataset,
  instance: CogneeInstance,
  options?: CognifyOptions,
) {
  // const data = await (
  return instance.fetch("/v1/cognify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      datasets: [dataset.id],
      datasetIds: [dataset.id],
      runInBackground: false,
      ...(options?.graphModel ? { graphModel: options.graphModel } : {}),
      customPrompt: options?.customPrompt ?? "",
      ontologyKey: [],
      chunksPerBatch: 10,
      ...(options?.llmModel && { llmModel: options.llmModel }),
    }),
  })
  .then((response) => response.json());
  // .then(() => {
  //   return getDatasetGraph(dataset, instance)
  //     .then((data) => {
  //       onUpdate({
  //         nodes: data.nodes,
  //         edges: data.edges,
  //       });
  //     });
  // });
  // )

    // const websocket = new WebSocket(`ws://localhost:8000/api/v1/cognify/subscribe/${data.pipeline_run_id}`);

    // let isCognifyDone = false;

    // websocket.onmessage = (event) => {
    //   const data = JSON.parse(event.data);
    //   onUpdate?.({
    //     nodes: data.payload.nodes,
    //     edges: data.payload.edges,
    //   });

    //   if (data.status === "PipelineRunCompleted") {
    //     isCognifyDone = true;
    //     websocket.close();
    //   }
    // };

    // return new Promise(async (resolve) => {
    //   while (!isCognifyDone) {
    //     await new Promise(resolve => setTimeout(resolve, 1000));
    //   }

    //   resolve(true);
    // });
}
