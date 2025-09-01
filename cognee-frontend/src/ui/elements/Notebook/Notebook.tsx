"use client";

import { v4 as uuid4 } from "uuid";
import { MutableRefObject, useCallback, useEffect, useRef } from "react";

import { PlusIcon } from "@/ui/Icons";
import { IconButton, TextArea } from '@/ui/elements';

import NotebookCellHeader from "./NotebookCellHeader";
import { Cell, Notebook as NotebookType } from "./types";
import GraphVisualization, { GraphVisualizationAPI } from "@/app/(graph)/GraphVisualization";
import { GraphControlsAPI } from '@/app/(graph)/GraphControls';

interface NotebookProps {
  notebook: NotebookType;
  runCell: (notebook: NotebookType, cell: Cell) => Promise<void>;
  updateNotebook: (updatedNotebook: NotebookType) => void;
  saveNotebook: (notebook: NotebookType) => void;
}

export default function Notebook({ notebook, updateNotebook, saveNotebook, runCell }: NotebookProps) {
  const saveCells = useCallback(() => {
    saveNotebook(notebook);
  }, [notebook, saveNotebook]);

  useEffect(() => {
    window.addEventListener("beforeunload", saveCells);

    return () => {
      window.removeEventListener("beforeunload", saveCells);
    };
  }, [saveCells]);

  useEffect(() => {
    if (notebook.cells.length === 0) {
      const newCell: Cell = {
        id: uuid4(),
        name: "first cell",
        type: "code",
        content: "",
      };
      updateNotebook({
       ...notebook,
        cells: [newCell],
      });
    }
  }, [notebook, saveNotebook, updateNotebook]);

  const handleCellRun = useCallback((cell: Cell) => {
    return runCell(notebook, cell);
  }, [notebook, runCell]);

  const handleCellAdd = useCallback(() => {
    const newCell: Cell = {
      id: uuid4(),
      name: "new cell",
      type: "code",
      content: "",
    };
    updateNotebook({
      ...notebook,
      cells: [...notebook.cells, newCell],
    });
  }, [notebook, updateNotebook]);
  
  const handleCellRemove = useCallback((cell: Cell) => {
    updateNotebook({
      ...notebook,
      cells: notebook.cells.filter((c: Cell) => c.id !== cell.id),
    });
  }, [notebook, updateNotebook]);

  const handleCellInputChange = useCallback((notebook: NotebookType, cell: Cell, value: string) => {
    const newCell = {...cell, content: value };

    updateNotebook({
      ...notebook,
      cells: notebook.cells.map((cell: Cell) => (cell.id === newCell.id ? newCell : cell)),
    });
  }, [updateNotebook]);

  const handleCellUp = useCallback((cell: Cell) => {
    const index = notebook.cells.indexOf(cell);

    if (index > 0) {
      const newCells = [...notebook.cells];
      newCells[index] = notebook.cells[index - 1];
      newCells[index - 1] = cell;

      updateNotebook({
        ...notebook,
        cells: newCells,
      });
    }
  }, [notebook, updateNotebook]);

  const handleCellDown = useCallback((cell: Cell) => {
    const index = notebook.cells.indexOf(cell);

    if (index < notebook.cells.length - 1) {
      const newCells = [...notebook.cells];
      newCells[index] = notebook.cells[index + 1];
      newCells[index + 1] = cell;

      updateNotebook({
        ...notebook,
        cells: newCells,
      });
    }
  }, [notebook, updateNotebook]);

  const handleCellRename = useCallback((cell: Cell) => {
    const newName = prompt("Enter a new name for the cell:");

    if (newName) {
      updateNotebook({
       ...notebook,
        cells: notebook.cells.map((c: Cell) => (c.id === cell.id ? {...c, name: newName } : c)),
      });
    }
  }, [notebook, updateNotebook]);
  
  return (
    <div className="bg-white rounded-xl flex flex-col gap-6 px-7 py-5">
      {notebook.cells.map((cell: Cell) => (
        <div key={cell.id} className="flex flex-row rounded-xl border-1 border-gray-100">
          <div className="flex flex-col flex-1 overflow-hidden">
            <NotebookCellHeader
              cell={cell}
              runCell={handleCellRun}
              renameCell={handleCellRename}
              removeCell={handleCellRemove}
              moveCellUp={handleCellUp}
              moveCellDown={handleCellDown}
              className="rounded-tl-xl rounded-tr-xl"
            />

            <TextArea
              value={cell.content}
              onChange={handleCellInputChange.bind(null, notebook, cell)}
              // onKeyUp={handleCellRunOnEnter}
              isAutoExpanding
              name="cellInput"
              placeholder="Type your code here..."
              contentEditable={true}
              className="resize-none min-h-36 max-h-96 overflow-y-auto rounded-tl-none rounded-tr-none rounded-bl-xl rounded-br-xl border-0 !outline-0"
            />

            <div className="flex flex-col bg-gray-100 overflow-x-auto max-w-full">
              {cell.result && (
                <div className="px-2 py-2">
                  output: <CellResult content={cell.result} />
                </div>
              )}
              {!cell.result && cell.error && (
                <div className="px-2 py-2">
                  error: {cell.error}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}

      <div>
        <IconButton onClick={handleCellAdd}><PlusIcon /></IconButton>
      </div>
    </div>
  );
}


function CellResult({ content = [] }) {
  const parsedContent = [];

  const graphRef = useRef<GraphVisualizationAPI>();
  const graphControls = useRef<GraphControlsAPI>({
    setSelectedNode: () => {},
    getSelectedNode: () => null,
  });

  for (const line of content) {
    try {
      if (Array.isArray(line)) {
        for (const item of line) {
          if (typeof item === "string") {
            parsedContent.push(
              <pre key={item.slice(0, -10)}>
                {item}
              </pre>
            );
          }
          if (typeof item === "object" && item["search_result"] && Array.isArray(item["search_result"])) {
            for (const result of item["search_result"]) {
              parsedContent.push(
                <div className="w-full h-full bg-white">
                  <span className="text-sm pl-2 mb-4">query response (dataset: {item["dataset_name"]})</span>
                  <span className="block px-2 py-2">{result}</span>
                </div>
              );
            }
          }
          if (typeof item === "object" && item["graph"] && typeof item["graph"] === "object") {
            parsedContent.push(
              <div className="w-full h-full bg-white">
                <span className="text-sm pl-2 mb-4">reasoning graph</span>
                <GraphVisualization
                  data={transformToVisualizationData(item["graph"])}
                  ref={graphRef as MutableRefObject<GraphVisualizationAPI>}
                  graphControls={graphControls}
                  className="min-h-48"
                />
              </div>
            );
          }
        }
      }
    } catch (error) {
      console.error(error);
      parsedContent.push(line);
    }
  }

  return parsedContent.map((item, index) => (
    <div key={index} className="px-2 py-1">
      {item}
      {/* {typeof item === "object" && item["search_result"] && Array.isArray(item["search_result"]) && (
        (item["search_result"] as []).map((result: string) => (<pre key={result.slice(0, -10)}>{result}</pre>))
      )}
      {typeof item === "object" && item["graph"] && typeof item["graph"] === "object" && (
        (item["graph"])
      )} */}
    </div>
  ));

};

function transformToVisualizationData(triplets) {
  // Implementation to transform triplet to visualization data

  const nodes = {};
  const links = {};

  for (const triplet of triplets) {
    nodes[triplet.source.id] = {
      id: triplet.source.id,
      label: triplet.source.attributes.name,
      type: triplet.source.attributes.type,
      attributes: triplet.source.attributes,
    };
    nodes[triplet.destination.id] = {
      id: triplet.destination.id,
      label: triplet.destination.attributes.name,
      type: triplet.destination.attributes.type,
      attributes: triplet.destination.attributes,
    };
    links[`${triplet.source.id}_${triplet.attributes.relationship_name}_${triplet.destination.id}`] = {
      source: triplet.source.id,
      target: triplet.destination.id,
      label: triplet.attributes.relationship_name,
    }
  }

  return {
    nodes: Object.values(nodes),
    links: Object.values(links),
  };
}
