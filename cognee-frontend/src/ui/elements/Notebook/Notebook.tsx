"use client";

import { v4 as uuid4 } from "uuid";
import classNames from "classnames";
import { Fragment, MutableRefObject, useCallback, useEffect, useRef, useState } from "react";

import { CaretIcon, PlusIcon } from "@/ui/Icons";
import { IconButton, PopupMenu, TextArea } from "@/ui/elements";
import { GraphControlsAPI } from "@/app/(graph)/GraphControls";
import GraphVisualization, { GraphVisualizationAPI } from "@/app/(graph)/GraphVisualization";

import NotebookCellHeader from "./NotebookCellHeader";
import { Cell, Notebook as NotebookType } from "./types";

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

  const handleCellAdd = useCallback((afterCellIndex: number, cellType: "markdown" | "code") => {
    const newCell: Cell = {
      id: uuid4(),
      name: "new cell",
      type: cellType,
      content: "",
    };

    const newNotebook = {
      ...notebook,
      cells: [
        ...notebook.cells.slice(0, afterCellIndex + 1),
        newCell,
        ...notebook.cells.slice(afterCellIndex + 1),
      ],
    };

    toggleCellOpen(newCell.id);
    updateNotebook(newNotebook);
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

  const [openCells, setOpenCells] = useState(new Set(notebook.cells.map((c: Cell) => c.id)));

  const toggleCellOpen = (id: string) => {
    setOpenCells((prev) => {
      const newState = new Set(prev);

      if (newState.has(id)) {
        newState.delete(id)
      } else {
        newState.add(id);
      }

      return newState;
    });
  };

  return (
    <div className="bg-white rounded-xl flex flex-col gap-0.5 px-7 py-5 flex-1">
      <div className="mb-5">{notebook.name}</div>

      {notebook.cells.map((cell: Cell, index) => (
        <Fragment key={cell.id}>
          <div key={cell.id} className="flex flex-row rounded-xl border-1 border-gray-100">
            <div className="flex flex-col flex-1 relative">
              {cell.type === "code" ? (
                <>
                  <div className="absolute left-[-1.35rem] top-2.5">
                    <IconButton className="p-[0.25rem] m-[-0.25rem]" onClick={toggleCellOpen.bind(null, cell.id)}>
                      <CaretIcon className={classNames("transition-transform", openCells.has(cell.id) ? "rotate-0" : "rotate-180")} />
                    </IconButton>
                  </div>

                  <NotebookCellHeader
                    cell={cell}
                    runCell={handleCellRun}
                    renameCell={handleCellRename}
                    removeCell={handleCellRemove}
                    moveCellUp={handleCellUp}
                    moveCellDown={handleCellDown}
                    className="rounded-tl-xl rounded-tr-xl"
                  />

                  {openCells.has(cell.id) && (
                    <>
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
                        {cell.error && (
                          <div className="px-2 py-2">
                            error: {cell.error}
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </>
              ) : (
                openCells.has(cell.id) && (
                  <TextArea
                    value={cell.content}
                    onChange={handleCellInputChange.bind(null, notebook, cell)}
                    // onKeyUp={handleCellRunOnEnter}
                    isAutoExpanding
                    name="cellInput"
                    placeholder="Type your text here..."
                    contentEditable={true}
                    className="resize-none min-h-24 max-h-96 overflow-y-auto rounded-tl-none rounded-tr-none rounded-bl-xl rounded-br-xl border-0 !outline-0"
                  />
                )
              )}
            </div>
          </div>
          <div className="ml-[-1.35rem]">
            <PopupMenu
              openToRight={true}
              triggerElement={<PlusIcon />}
              triggerClassName="p-[0.25rem] m-[-0.25rem]"
            >
              <div className="flex flex-col gap-0.5">
                <button
                  onClick={() => handleCellAdd(index, "markdown")}
                  className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer"
                >
                  <span>text</span>
                </button>
              </div>
              <div
                onClick={() => handleCellAdd(index, "code")}
                className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer"
              >
                <span>code</span>
              </div>
            </PopupMenu>
          </div>
        </Fragment>
      ))}
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
