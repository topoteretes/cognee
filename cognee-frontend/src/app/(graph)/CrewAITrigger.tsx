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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onData: (data: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

    onActivity([{ id: uuid4(), timestamp: Date.now(), activity: "Dispatching hiring crew agents" }]);

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
        websocket.close();
      }
    };

    onData(null);
    setIsCrewAIRunning(true);

    return fetch("/v1/crewai/run", {
      method: "POST",
      body: JSON.stringify(crewAIConfig),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then(response => response.json())
      .then(() => {
        onActivity([{ id: uuid4(), timestamp: Date.now(), activity: "Hiring crew agents made a decision" }]);
      })
      .catch(() => {
        onActivity([{ id: uuid4(), timestamp: Date.now(), activity: "Hiring crew agents had problems while executing" }]);
      })
      .finally(() => {
        websocket.close();
        setIsCrewAIRunning(false);
      });
  };

  return (
    <form className="w-full flex flex-col gap-2" onSubmit={handleRunCrewAI}>
      <h1 className="text-2xl text-white">Cognee Dev Mexican Standoff</h1>
      <span className="text-white">Agents compare GitHub profiles, and make a decision who is a better developer</span>
      <div className="flex flex-row gap-2">
        <div className="flex flex-col w-full flex-1/2">
          <label className="block mb-1 text-white" htmlFor="username1">GitHub username</label>
          <Input name="username1" type="text" placeholder="Github Username" required defaultValue="hajdul88" />
        </div>
        <div className="flex flex-col w-full flex-1/2">
          <label className="block mb-1 text-white" htmlFor="username2">GitHub username</label>
          <Input name="username2" type="text" placeholder="Github Username" required defaultValue="lxobr" />
        </div>
      </div>
      <CTAButton type="submit" disabled={isCrewAIRunning} className="whitespace-nowrap">
        Start Mexican Standoff
        {isCrewAIRunning && <LoadingIndicator />}
      </CTAButton>
    </form>
  );
}
