// import getDatasetGraph from "./getDatasetGraph";
import { Dataset } from "../ingestion/useDatasets";
import { CogneeInstance } from "../instances/types";
import { getPipelineSettingsFromStorage } from "../configuration/pipelineSettings";

// interface GraphData {
//   nodes: { id: string; label: string; properties?: object }[];
//   edges: { source: string; target: string; label: string }[];
// }

// runInBackground=true means the server returns immediately — this only
// needs to cover a cold pod's startup, not the actual cognify run. See CLO-333.
const COGNIFY_TIMEOUT_MS = 60_000;

interface CognifyOptions {
  graphModel?: object;
  customPrompt?: string;
  ontologyKey?: string[];
  llmModel?: string;
  chunkSize?: number;
  chunksPerBatch?: number;
}

export default async function cognifyDataset(
  dataset: Dataset,
  instance: CogneeInstance,
  options?: CognifyOptions,
) {
  const pipelineSettings = getPipelineSettingsFromStorage();
  // const data = await (
  return instance.fetch("/v1/cognify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      // datasetIds resolves the dataset unambiguously; only send `datasets`
      // (names) when a name is actually present. Sending an empty or unmatched
      // name makes cognify create a new empty dataset named after that string
      // (a UUID-in-`datasets` does the same) — so a name-less caller must send
      // ids alone.
      ...(dataset.name ? { datasets: [dataset.name] } : {}),
      datasetIds: [dataset.id],
      runInBackground: true,
      ...(options?.graphModel ? { graphModel: options.graphModel } : {}),
      customPrompt: options?.customPrompt ?? "",
      ontologyKey: options?.ontologyKey ?? [],
      chunksPerBatch: options?.chunksPerBatch ?? pipelineSettings.chunksPerBatch,
      chunkSize: options?.chunkSize ?? pipelineSettings.chunkSize,
      ...(options?.llmModel && { llmModel: options.llmModel }),
    }),
    timeoutMs: COGNIFY_TIMEOUT_MS,
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
