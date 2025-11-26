"use client";

import { v4 as uuid4 } from "uuid";
import classNames from "classnames";
import { Fragment, MouseEvent, RefObject, useCallback, useEffect, useRef, useState } from "react";

import { useModal } from "@/ui/elements/Modal";
import { CaretIcon, CloseIcon, PlusIcon } from "@/ui/Icons";
import { IconButton, PopupMenu, TextArea, Modal, GhostButton, CTAButton } from "@/ui/elements";
import { GraphControlsAPI } from "@/app/(graph)/GraphControls";
import GraphVisualization, { GraphVisualizationAPI } from "@/app/(graph)/GraphVisualization";

import NotebookCellHeader from "./NotebookCellHeader";
import { Cell, Notebook as NotebookType } from "./types";

interface NotebookProps {
  notebook: NotebookType;
  runCell: (notebook: NotebookType, cell: Cell, cogneeInstance: string) => Promise<void>;
  updateNotebook: (updatedNotebook: NotebookType) => void;
}

export default function Notebook({ notebook, updateNotebook, runCell }: NotebookProps) {
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
      toggleCellOpen(newCell.id)
    }
  }, [notebook, updateNotebook]);

  const handleCellRun = useCallback((cell: Cell, cogneeInstance: string) => {
    return runCell(notebook, cell, cogneeInstance);
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

  const removeCell = useCallback((cell: Cell, event?: MouseEvent) => {
    event?.preventDefault();

    updateNotebook({
      ...notebook,
      cells: notebook.cells.filter((c: Cell) => c.id !== cell.id),
    });
  }, [notebook, updateNotebook]);

  const {
    isModalOpen: isRemoveCellConfirmModalOpen,
    openModal: openCellRemoveConfirmModal,
    closeModal: closeCellRemoveConfirmModal,
    confirmAction: handleCellRemoveConfirm,
  } = useModal<Cell, MouseEvent>(false, removeCell);

  const handleCellRemove = useCallback((cell: Cell) => {
    openCellRemoveConfirmModal(cell);
  }, [openCellRemoveConfirmModal]);

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
    <>
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
                          {!!cell.error?.length && (
                            <div className="px-2 py-2">
                              error: {cell.error}
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <div className="absolute left-[-1.35rem] top-2.5">
                      <IconButton className="p-[0.25rem] m-[-0.25rem]" onClick={toggleCellOpen.bind(null, cell.id)}>
                        <CaretIcon className={classNames("transition-transform", openCells.has(cell.id) ? "rotate-0" : "rotate-180")} />
                      </IconButton>
                    </div>

                    <NotebookCellHeader
                      cell={cell}
                      renameCell={handleCellRename}
                      removeCell={handleCellRemove}
                      moveCellUp={handleCellUp}
                      moveCellDown={handleCellDown}
                      className="rounded-tl-xl rounded-tr-xl"
                    />

                    {openCells.has(cell.id) && (
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
                    )}
                  </>
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

      <Modal isOpen={isRemoveCellConfirmModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete notebook cell?</span>
            <IconButton onClick={closeCellRemoveConfirmModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Are you sure you want to delete a notebook cell? This action cannot be undone.</div>
          <div className="flex flex-row gap-4 mt-4 justify-end">
            <GhostButton type="button" onClick={closeCellRemoveConfirmModal}>cancel</GhostButton>
            <CTAButton onClick={handleCellRemoveConfirm} type="submit">delete</CTAButton>
          </div>
        </div>
      </Modal>
    </>
  );
}


function CellResult({ content }: { content: [] }) {
  const parsedContent = [];

  const graphRef = useRef<GraphVisualizationAPI>(null);
  const graphControls = useRef<GraphControlsAPI>({
    setSelectedNode: () => {},
    getSelectedNode: () => null,
  });

  for (const line of content) {
    try {
      if (Array.isArray(line)) {
        // Insights search returns uncommon graph data structure
        if (Array.from(line).length > 0 && Array.isArray(line[0]) && line[0][1]["relationship_name"]) {
          parsedContent.push(
            <div key={line[0][1]["relationship_name"]} className="w-full h-full bg-white">
              <span className="text-sm pl-2 mb-4">reasoning graph</span>
              <GraphVisualization
                data={transformInsightsGraphData(line)}
                ref={graphRef as RefObject<GraphVisualizationAPI>}
                graphControls={graphControls}
                className="min-h-80"
              />
            </div>
          );
          continue;
        }

        // @ts-expect-error line can be Array or string
        for (const item of line) {
          if (
            typeof item === "object" && item["search_result"] && (typeof(item["search_result"]) === "string"
            || (Array.isArray(item["search_result"]) && typeof(item["search_result"][0]) === "string"))
          ) {
            parsedContent.push(
              <div key={String(item["search_result"])} className="w-full h-full bg-white">
                <span className="text-sm pl-2 mb-4">query response (dataset: {item["dataset_name"]})</span>
                <span className="block px-2 py-2 whitespace-normal">{item["search_result"]}</span>
              </div>
            );
          } else if (typeof(item) === "object" && item["search_result"] && typeof(item["search_result"]) === "object") {
            parsedContent.push(
              <pre className="px-2 w-full h-full bg-white text-sm" key={String(item).slice(0, -10)}>
                {JSON.stringify(item, null, 2)}
              </pre>
            )
          } else if (typeof(item) === "string") {
            parsedContent.push(
              <pre className="px-2 w-full h-full bg-white text-sm whitespace-normal" key={item.slice(0, -10)}>
                {item}
              </pre>
            );
          } else if (typeof(item) === "object" && !(item["search_result"] || item["graphs"])) {
            parsedContent.push(
              <pre className="px-2 w-full h-full bg-white text-sm" key={String(item).slice(0, -10)}>
                {JSON.stringify(item, null, 2)}
              </pre>
            )
          }

          if (typeof item === "object" && item["graphs"] && typeof item["graphs"] === "object") {
            Object.entries<{ nodes: []; edges: []; }>(item["graphs"]).forEach(([datasetName, graph]) => {
              parsedContent.push(
                <div key={datasetName} className="w-full h-full bg-white">
                  <span className="text-sm pl-2 mb-4">reasoning graph (datasets: {datasetName})</span>
                  <GraphVisualization
                    data={transformToVisualizationData(graph)}
                    ref={graphRef as RefObject<GraphVisualizationAPI>}
                    graphControls={graphControls}
                    className="min-h-80"
                  />
                </div>
              );
            });
          }
        }
      }

      if (typeof(line) === "object" && line["result"] && typeof(line["result"]) === "string") {
        const datasets = Array.from(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          new Set(Object.values(line["datasets"]).map((dataset: any) => dataset.name))
        ).join(", ");

        parsedContent.push(
          <div key={line["result"]} className="w-full h-full bg-white">
            <span className="text-sm pl-2 mb-4">query response (datasets: {datasets})</span>
            <span className="block px-2 py-2 whitespace-normal">{line["result"]}</span>
          </div>
        );
      }
      if (typeof(line) === "object" && line["graphs"]) {
        Object.entries<{ nodes: []; edges: []; }>(line["graphs"]).forEach(([datasetName, graph]) => {
          parsedContent.push(
            <div key={datasetName} className="w-full h-full bg-white">
              <span className="text-sm pl-2 mb-4">reasoning graph (datasets: {datasetName})</span>
              <GraphVisualization
                data={transformToVisualizationData(graph)}
                ref={graphRef as RefObject<GraphVisualizationAPI>}
                graphControls={graphControls}
                className="min-h-80"
              />
            </div>
          );
        });
      }

      if (typeof(line) === "object" && line["result"] && typeof(line["result"]) === "object") {
        parsedContent.push(
          <pre className="px-2 w-full h-full bg-white text-sm" key={String(line).slice(0, -10)}>
            {JSON.stringify(line["result"], null, 2)}
          </pre>
        )
      }
      if (typeof(line) === "string") {
        parsedContent.push(
          <pre className="px-2 w-full h-full bg-white text-sm whitespace-normal" key={String(line).slice(0, -10)}>
            {line}
          </pre>
        )
      }
    } catch (error) {
      console.error(error);
      parsedContent.push(
        <pre className="px-2 w-full h-full bg-white text-sm whitespace-normal" key={String(line).slice(0, -10)}>
          {line}
        </pre>
      );
    }
  }

  return parsedContent.map((item, index) => (
    <div key={index} className="px-2 py-1">
      {item}
    </div>
  ));

};

function transformToVisualizationData(graph: { nodes: [], edges: [] }) {
  return {
    nodes: graph.nodes,
    links: graph.edges,
  };
}

type Triplet = [{
  id: string,
  name: string,
  type: string,
}, {
  relationship_name: string,
}, {
  id: string,
  name: string,
  type: string,
}]

function transformInsightsGraphData(triplets: Triplet[]) {
  const nodes: {
    [key: string]: {
      id: string,
      label: string,
      type: string,
    }
  } = {};
  const links: {
    [key: string]: {
      source: string,
      target: string,
      label: string,
    }
  } = {};          

  for (const triplet of triplets) {
    nodes[triplet[0].id] = {
      id: triplet[0].id,
      label: triplet[0].name || triplet[0].id,
      type: triplet[0].type,
    };
    nodes[triplet[2].id] = {
      id: triplet[2].id,
      label: triplet[2].name || triplet[2].id,
      type: triplet[2].type,
    };
    const linkKey = `${triplet[0]["id"]}_${triplet[1]["relationship_name"]}_${triplet[2]["id"]}`;
    links[linkKey] = {
      source: triplet[0].id,
      target: triplet[2].id,
      label: triplet[1]["relationship_name"],
    };
  }
  
  return {
    nodes: Object.values(nodes),
    links: Object.values(links),
  };
}
