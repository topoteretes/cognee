"use client";

import { v4 as uuid4 } from "uuid";
import classNames from "classnames";
import { NodeObject } from "react-force-graph-2d";
import { ChangeEvent, useImperativeHandle, useState } from "react";

import { DeleteIcon } from "@/ui/Icons";
import { FeedbackForm } from "@/ui/Partials";
import { CTAButton, Input, NeutralButton, Select } from "@/ui/elements";

interface GraphControlsProps {
  isAddNodeFormOpen: boolean;
  ref: React.RefObject<GraphControlsAPI>;
  onFitIntoView: () => void;
  onGraphShapeChange: (shape: string) => void;
}

export interface GraphControlsAPI {
  setSelectedNode: (node: NodeObject | null) => void;
  getSelectedNode: () => NodeObject | null;
}

type ActivityLog = {
  id: string;
  timestamp: number;
  activity: string;
}[];

type NodeProperties = {
  id: string;
  name: string;
  value: string;
}[];

export default function GraphControls({ isAddNodeFormOpen, onGraphShapeChange, onFitIntoView, ref }: GraphControlsProps) {
  const [selectedNode, setSelectedNode] = useState<NodeObject | null>(null);
  const [activityLog, setActivityLog] = useState<ActivityLog>([]);
  const [nodeProperties, setNodeProperties] = useState<NodeProperties>([]);
  const [newProperty, setNewProperty] = useState<NodeProperties[0]>({
    id: uuid4(),
    name: "",
    value: "",
  });

  const handlePropertyChange = (property: NodeProperties[0], property_key: string, event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;

    setNodeProperties(nodeProperties.map((nodeProperty) => (nodeProperty.id === property.id ? {...nodeProperty, [property_key]: value } : nodeProperty)));
  };

  const handlePropertyAdd = () => {
    if (newProperty.name && newProperty.value) {
      setNodeProperties([...nodeProperties, newProperty]);
      setNewProperty({ id: uuid4(), name: "", value: "" });
    } else {
      alert("Please fill in both name and value fields for the new property.");
    }
  };

  const handlePropertyDelete = (property: NodeProperties[0]) => {
    setNodeProperties(nodeProperties.filter((nodeProperty) => nodeProperty.id !== property.id));
  };

  const handleNewPropertyChange = (property: NodeProperties[0], property_key: string, event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;

    setNewProperty({...property, [property_key]: value });
  };

  useImperativeHandle(ref, () => ({
    setSelectedNode,
    getSelectedNode: () => selectedNode,
  }));

  const [selectedTab, setSelectedTab] = useState("nodeDetails");

  const handleGraphShapeControl = (event: ChangeEvent<HTMLSelectElement>) => {
    onGraphShapeChange(event.target.value);
  };

  return (
    <>
      <div className="flex">
        <button onClick={() => setSelectedTab("nodeDetails")} className={classNames("cursor-pointer pt-4 pb-4 align-center text-gray-300 border-b-2 w-30", { "border-b-indigo-600 text-white": selectedTab === "nodeDetails" })}>
          <span className="whitespace-nowrap">Node Details</span>
        </button>
        <button onClick={() => setSelectedTab("activityLog")} className={classNames("cursor-pointer pt-4 pb-4 align-center text-gray-300 border-b-2 w-30", { "border-b-indigo-600 text-white": selectedTab === "activityLog" })}>
          <span className="whitespace-nowrap">Activity Log</span>
        </button>
        <button onClick={() => setSelectedTab("feedback")} className={classNames("cursor-pointer pt-4 pb-4 align-center text-gray-300 border-b-2 w-30", { "border-b-indigo-600 text-white": selectedTab === "feedback" })}>
          <span className="whitespace-nowrap">Feedback</span>
        </button>
      </div>

      <div className="pt-4">
        {selectedTab === "nodeDetails" && (
          <>
            <div className="w-full flex flex-row gap-2 items-center mb-4">
              <label className="text-gray-300 whitespace-nowrap">Graph Shape:</label>
              <Select onChange={handleGraphShapeControl}>
                <option selected value="none">None</option>
                <option value="td">Top-down</option>
                <option value="bu">Bottom-up</option>
                <option value="lr">Left-right</option>
                <option value="rl">Right-left</option>
                <option value="radialin">Radial-in</option>
                <option value="radialout">Radial-out</option>
              </Select>
            </div>

            <NeutralButton onClick={onFitIntoView} className="mb-4">Fit Graph into View</NeutralButton>

            {isAddNodeFormOpen ? (
              <form className="flex flex-col gap-4" onSubmit={() => {}}>
                <div className="flex flex-row gap-4 items-center">
                  <span className="text-gray-300 whitespace-nowrap">Source Node ID:</span>
                  <Input readOnly type="text" defaultValue={selectedNode!.id} />
                </div>
                <div className="flex flex-col gap-4 items-end">
                  {nodeProperties.map((property) => (
                    <div key={property.id} className="w-full flex flex-row gap-2 items-center">
                      <Input className="flex-1/3" type="text" placeholder="Property name" required value={property.name} onChange={handlePropertyChange.bind(null, property, "name")} />
                      <Input className="flex-2/3" type="text" placeholder="Property value" required value={property.value} onChange={handlePropertyChange.bind(null, property, "value")} />
                      <button className="border-1 border-white p-2 rounded-sm" onClick={handlePropertyDelete.bind(null, property)}>
                        <DeleteIcon width={16} height={18} color="white" />
                      </button>
                    </div>
                  ))}
                  <div className="w-full flex flex-row gap-2 items-center">
                    <Input className="flex-1/3" type="text" placeholder="Property name" required value={newProperty.name} onChange={handleNewPropertyChange.bind(null, newProperty, "name")} />
                    <Input className="flex-2/3" type="text" placeholder="Property value" required value={newProperty.value} onChange={handleNewPropertyChange.bind(null, newProperty, "value")} />
                    <NeutralButton type="button" className="" onClick={handlePropertyAdd}>Add</NeutralButton>
                  </div>
                </div>
                <CTAButton type="submit">Add Node</CTAButton>
              </form>
            ) : (
              selectedNode ? (
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2 items-center">
                      <span className="text-gray-300">ID:</span>
                      <span className="text-white">{selectedNode.id}</span>
                    </div>

                    {Object.entries(selectedNode.properties).map(([key, value]) => (
                      <div key={key} className="flex gap-2 items-center">
                        <span className="text-gray-300">{key}:</span>
                        <span className="text-white">{typeof value === "object" ? JSON.stringify(value) : value as string}</span>
                      </div>
                    ))}
                  </div>

                  <CTAButton type="button" onClick={() => {}}>Edit Node</CTAButton>
                </div>
              ) : (
                <span className="text-white">No node selected.</span>
              )
            )}
          </>
        )}

        {selectedTab === "activityLog" && (
          <div className="flex flex-col gap-2">
            {activityLog.map((activity) => (
              <div key={activity.id} className="flex gap-2 items-center">
                <span className="text-gray-300">{activity.timestamp}</span>
                <span className="text-white">{activity.activity}</span>
              </div>
            ))}
            {!activityLog.length && <span className="text-white">No activity logged.</span>}
          </div>
        )}

        {selectedTab === "feedback" && (
          <div className="flex flex-col gap-2">
            <FeedbackForm onSuccess={() => {}} />
          </div>
        )}
      </div>
    </>
  );
}
