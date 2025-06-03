import { useState } from "react";
import { fetch } from "@/utils";
import { v4 as uuid4 } from "uuid";
import { LoadingIndicator } from "@/ui/App";
import { CTAButton, Input } from "@/ui/elements";

interface CrewAIFormPayload extends HTMLFormElement {
  username1: HTMLInputElement;
  username2: HTMLInputElement;
}

interface CrewAITriggerProps {
  onData: (data: any) => void;
  onActivity: (activities: any) => void;
}

export default function CrewAITrigger({ onData, onActivity }: CrewAITriggerProps) {
  const [isCrewAIRunning, setIsCrewAIRunning] = useState(false);
  
  const handleRunCrewAI = (event: React.FormEvent<CrewAIFormPayload>) => {
    event.preventDefault();
    const formElements = event.currentTarget;

    const crewAIConfig = {
      username1: formElements.username1.value,
      username2: formElements.username2.value,
    };

    const websocket = new WebSocket("ws://localhost:8000/api/v1/crewai/subscribe");

    let isCrewAIDone = false;
    onActivity([{ id: uuid4(), timestamp: Date.now(), activity: "Running CrewAI" }]);

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.status === "PipelineRunActivity") {
        onActivity([data.payload]);
        return;
      }

      onData({
        nodes: data.payload.nodes,
        links: data.payload.edges,
      });

      const nodes_type_map: { [key: string]: number } = {};

      for (let i = 0; i < data.payload.nodes.length; i++) {
        const node = data.payload.nodes[i];
        if (!nodes_type_map[node.type]) {
          nodes_type_map[node.type] = 0;
        }
        nodes_type_map[node.type] += 1;
      }

      const activityMessage = Object.entries(nodes_type_map).reduce((message, [type, count]) => {
        return `${message}\n | ${type}: ${count}`;
      }, "Graph updated:");

      onActivity([{
        id: uuid4(),
        timestamp: Date.now(),
        activity: activityMessage,
      }]);

      if (data.status === "PipelineRunCompleted") {
        isCrewAIDone = true;
        websocket.close();
      }
    };

    setIsCrewAIRunning(true);

    return fetch("/v1/crewai/run", {
      method: "POST",
      body: JSON.stringify(crewAIConfig),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then(response => response.json())
      .finally(() => {
        websocket.close();
        setIsCrewAIRunning(false);
        onActivity([{ id: uuid4(), timestamp: Date.now(), activity: "CrewAI run done" }]);
      });
  };

  return (
    <form className="w-full flex flex-col gap-4" onSubmit={handleRunCrewAI}>
      <h1 className="text-2xl text-white">Cognee HR Crew Demo</h1>
      <p className="text-white">
        Run Agents and let them help you make a hiring decision.
        Analyze two candidates side by side, and decide based on their GitHub profiles.
      </p>
      <div className="flex flex-col w-full">
        <label className="block mb-1 text-white" htmlFor="username1">First Github username</label>
        <Input name="username1" type="text" placeholder="Github Username" required defaultValue="hajdul88" />
      </div>
      <div className="flex flex-col w-full">
        <label className="block mb-1 text-white" htmlFor="username2">Second Github username</label>
        <Input name="username2" type="text" placeholder="Github Username" required defaultValue="lxobr" />
      </div>
      <CTAButton type="submit" className="whitespace-nowrap">
        Start HR Crew Research
        {isCrewAIRunning && <LoadingIndicator />}
      </CTAButton>
    </form>
  );
}
