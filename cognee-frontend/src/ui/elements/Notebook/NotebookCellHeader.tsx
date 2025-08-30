import classNames from "classnames";

import { MenuIcon, PlayIcon } from "@/ui/Icons";
import IconButton from "../IconButton";
import { Cell } from "./types";

interface NotebookCellHeaderProps {
  cell: Cell;
  runCell: (cell: Cell) => void;
  removeCell: (cell: Cell) => void;
  moveCellUp: (cell: Cell) => void;
  moveCellDown: (cell: Cell) => void;
  className?: string;
}

export default function NotebookCellHeader({
  cell,
  runCell,
  removeCell,
  moveCellUp,
  moveCellDown,
  className,
}: NotebookCellHeaderProps) {

  return (
    <div className={classNames("flex flex-row justify-between items-center h-9 bg-gray-100", className)}>
      <div className="flex flex-row items-center px-3.5">
        <IconButton onClick={() => runCell(cell)}><PlayIcon /></IconButton>
        <span className="ml-4">{cell.name}</span>
      </div>
      <div className="pr-4">
        <details className="relative">
          <summary className="list-none">
            <div className="p-[0.5rem] m-[-0.5rem] cursor-pointer hover:bg-white rounded-xl">
              <MenuIcon />
            </div>
          </summary>

          <div className="absolute right-0 top-full flex flex-col gap-4 pl-1 py-3 pr-4 whitespace-nowrap bg-white border-1 border-gray-100 z-10">
            <div className="flex flex-col gap-0.5">
              <button onClick={() => moveCellUp(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">move cell up</button>
              <button onClick={() => moveCellDown(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">move cell down</button>
            </div>
            <div className="flex flex-col gap-0.5 items-start">
              <button onClick={() => removeCell(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">delete</button>
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}
