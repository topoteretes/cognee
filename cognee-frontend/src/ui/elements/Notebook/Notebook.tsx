"use client";

import { v4 as uuid4 } from "uuid";
import { useCallback, useEffect } from "react";

import { PlusIcon } from "@/ui/Icons";
import { IconButton, TextArea } from '@/ui/elements';

import NotebookCellHeader from "./NotebookCellHeader";
import { Cell, Notebook as NotebookType } from "./types";

interface NotebookProps {
  notebook: NotebookType;
  runCell: (notebook: NotebookType, cell: Cell) => void;
  updateNotebook: (updatedNotebook: NotebookType) => void;
  saveNotebook: (notebook: NotebookType) => void;
}

export default function Notebook({ notebook, updateNotebook, saveNotebook, runCell }: NotebookProps) {
  const saveCells = useCallback(() => {
    saveNotebook(notebook);
  }, [notebook, saveNotebook]);

  useEffect(() => {
    window.addEventListener("beforeunload", saveCells);

    // const saveCellsTimeout = setTimeout(() => {
    //   saveCells();
    // }, 5000);

    return () => {
      window.removeEventListener("beforeunload", saveCells);

      // clearTimeout(saveCellsTimeout);
    };
  }, [saveCells]);

  const handleCellRun = useCallback((cell: Cell) => {
    return runCell(notebook, cell);
  }, [notebook, runCell]);

  const handleCellAdd = useCallback(() => {
    const newCell: Cell = {
      id: uuid4(),
      name: "New Cell",
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

  return (
    <div className="flex flex-row">
      <div>

      </div>

      <div className="flex-1 bg-white rounded-xl overflow-hidden flex flex-col gap-6 px-7 py-5">
        {notebook.cells.map((cell: Cell) => (
          <div key={cell.id} className="flex flex-row rounded-xl border-1 border-gray-100">
            <div className="flex flex-col flex-1">
              <NotebookCellHeader
                cell={cell}
                runCell={handleCellRun}
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
                className="resize-none min-h-14 max-h-96 overflow-y-auto rounded-tl-none rounded-tr-none rounded-bl-xl rounded-br-xl border-0 !outline-0"
              />

              <div className="flex flex-col bg-gray-100">
                {cell.result && (
                  <div className="px-2 py-2">
                    Output: <CellResult content={cell.result} />
                  </div>
                )}
                {!cell.result && cell.error && (
                  <div className="px-2 py-2">
                    Error: {cell.error}
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
    </div>
  );
}


function CellResult({ content = [] }) {
  return content.map((item, index) => (
    <div key={index} className="px-2 py-1">
      {typeof item === "string" ? (
        item
      ) : (
        <pre>{JSON.stringify(item, null, 2)}</pre>
      )}
    </div>
  ));

};
