"use client";

import { v4 as uuid4 } from "uuid";
// import classNames from "classnames";
import { NodeObject, LinkObject } from "react-force-graph-2d";
import { ChangeEvent, useEffect, useImperativeHandle, useRef, useState } from "react";

import { DeleteIcon } from "@/ui/Icons";
import { CTAButton, Input, NeutralButton, Select } from "@/ui/elements";

interface GraphControlsProps {
  data?: {
    nodes: NodeObject[];
    links: LinkObject[];
  };
  isAddNodeFormOpen: boolean;
  ref: React.RefObject<GraphControlsAPI>;
  onFitIntoView: () => void;
  onGraphShapeChange: (shape: string) => void;
}

export interface GraphControlsAPI {
  setSelectedNode: (node: NodeObject | null) => void;
  getSelectedNode: () => NodeObject | null;
}

// type ActivityLog = {
//   id: string;
//   timestamp: number;
//   activity: string;
// };

type NodeProperty = {
  id: string;
  name: string;
  value: string;
};

// const formatter = new Intl.DateTimeFormat("en-GB", { dateStyle: "short", timeStyle: "medium" });

const DEFAULT_GRAPH_SHAPE = "lr";

const GRAPH_SHAPES = [{
  value: "none",
  label: "None",
}, {
  value: "td",
  label: "Top-down",
}, {
  value: "bu",
  label: "Bottom-up",
}, {
  value: "lr",
  label: "Left-right",
}, {
  value: "rl",
  label: "Right-left",
}, {
  value: "radialin",
  label: "Radial-in",
}, {
  value: "radialout",
  label: "Radial-out",
}];

export default function GraphControls({ data, isAddNodeFormOpen, onGraphShapeChange, onFitIntoView, ref }: GraphControlsProps) {
  const [selectedNode, setSelectedNode] = useState<NodeObject | null>(null);
  const [nodeProperties, setNodeProperties] = useState<NodeProperty[]>([]);
  const [newProperty, setNewProperty] = useState<NodeProperty>({
    id: uuid4(),
    name: "",
    value: "",
  });

  const handlePropertyChange = (property: NodeProperty, property_key: string, event: ChangeEvent<HTMLInputElement>) => {
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

  const handlePropertyDelete = (property: NodeProperty) => {
    setNodeProperties(nodeProperties.filter((nodeProperty) => nodeProperty.id !== property.id));
  };

  const handleNewPropertyChange = (property: NodeProperty, property_key: string, event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;

    setNewProperty({...property, [property_key]: value });
  };

  useImperativeHandle(ref, () => ({
    setSelectedNode,
    getSelectedNode: () => selectedNode,
  }));

  // const [selectedTab, setSelectedTab] = useState("nodeDetails");

  const handleGraphShapeControl = (event: ChangeEvent<HTMLSelectElement>) => {
    setIsAuthShapeChangeEnabled(false);
    onGraphShapeChange(event.target.value);
  };

  const [isAuthShapeChangeEnabled, setIsAuthShapeChangeEnabled] = useState(true);
  const shapeChangeTimeout = useRef<number | null>(null);
  
  useEffect(() => {
    onGraphShapeChange(DEFAULT_GRAPH_SHAPE);

    const graphShapesNum = GRAPH_SHAPES.length;

    function switchShape(shapeIndex: number) {
      if (!isAuthShapeChangeEnabled || !data) {
        if (shapeChangeTimeout.current) {
          clearTimeout(shapeChangeTimeout.current);
          shapeChangeTimeout.current = null;
        }

        return;
      }

      shapeChangeTimeout.current = setTimeout(() => {
        const newValue = GRAPH_SHAPES[shapeIndex].value;
        onGraphShapeChange(newValue);
        const graphShapeSelectElement = document.getElementById("graph-shape-select") as HTMLSelectElement;
        graphShapeSelectElement.value = newValue;

        switchShape((shapeIndex + 1) % graphShapesNum);
      }, 5000) as unknown as number;
    };

    switchShape(0);

    setTimeout(() => {
      onFitIntoView();
    }, 500);

    return () => {
      if (shapeChangeTimeout.current) {
        clearTimeout(shapeChangeTimeout.current);
        shapeChangeTimeout.current = null;
      }
    };
  }, [data, isAuthShapeChangeEnabled, onFitIntoView, onGraphShapeChange]);

  return (
    <>
      <div className="flex w-full">
        {/* <button onClick={() => setSelectedTab("nodeDetails")} className={classNames("cursor-pointer pt-4 pb-4 align-center text-gray-300 border-b-2 w-30 flex-1/3", { "border-b-indigo-600 text-white": selectedTab === "nodeDetails" })}> */}
        <span className="whitespace-nowrap text-white">Node Details</span>
        {/* </button> */}
        {/* <button onClick={() => setSelectedTab("feedback")} className={classNames("cursor-pointer pt-4 pb-4 align-center text-gray-300 border-b-2 w-30 flex-1/3", { "border-b-indigo-600 text-white": selectedTab === "feedback" })}>
          <span className="whitespace-nowrap">Feedback</span>
        </button> */}
      </div>

      <div className="pt-4">
        {/* {selectedTab === "nodeDetails" && ( */}
        <>
          <div className="w-full flex flex-row gap-2 items-center mb-4">
            <label className="text-gray-300 whitespace-nowrap flex-1/5">Graph Shape:</label>
            <Select defaultValue={DEFAULT_GRAPH_SHAPE} onChange={handleGraphShapeControl} id="graph-shape-select" className="flex-2/5">
              {GRAPH_SHAPES.map((shape) => (
                <option key={shape.value} value={shape.value}>{shape.label}</option>
              ))}
            </Select>
            <NeutralButton onClick={onFitIntoView} className="flex-2/5 whitespace-nowrap">Fit Graph into View</NeutralButton>
          </div>


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
                <div className="flex flex-col gap-2 overflow-y-auto max-h-96 pr-2">
                  <div className="flex gap-2 items-top">
                    <span className="text-gray-300">ID:</span>
                    <span className="text-white">{selectedNode.id}</span>
                  </div>
                  <div className="flex gap-2 items-top">
                    <span className="text-gray-300">Label:</span>
                    <span className="text-white">{selectedNode.label}</span>
                  </div>

                  {Object.entries(selectedNode.properties).map(([key, value]) => (
                    <div key={key} className="flex gap-2 items-top">
                      <span className="text-gray-300">{key.charAt(0).toUpperCase() + key.slice(1)}:</span>
                      <span className="text-white">{typeof value === "object" ? JSON.stringify(value) : value as string}</span>
                    </div>
                  ))}
                </div>

                {/* <CTAButton type="button" onClick={() => {}}>Edit Node</CTAButton> */}
              </div>
            ) : (
              <span className="text-white">No node selected.</span>
            )
          )}
        </>
        {/* )} */}
      </div>
    </>
  );
}
