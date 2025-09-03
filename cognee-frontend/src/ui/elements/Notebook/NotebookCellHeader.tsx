import classNames from "classnames";

import { useBoolean } from "@/utils";
import { LocalCogneeIcon, PlayIcon } from "@/ui/Icons";
import { PopupMenu } from "@/ui/elements";
import { LoadingIndicator } from "@/ui/App";

import { Cell } from "./types";
import IconButton from "../IconButton";

interface NotebookCellHeaderProps {
  cell: Cell;
  runCell: (cell: Cell) => Promise<void>;
  renameCell: (cell: Cell) => void;
  removeCell: (cell: Cell) => void;
  moveCellUp: (cell: Cell) => void;
  moveCellDown: (cell: Cell) => void;
  className?: string;
}

export default function NotebookCellHeader({
  cell,
  runCell,
  renameCell,
  removeCell,
  moveCellUp,
  moveCellDown,
  className,
}: NotebookCellHeaderProps) {
  const {
    value: isRunningCell,
    setTrue: setIsRunningCell,
    setFalse: setIsNotRunningCell,
  } = useBoolean(false);

  const handleCellRun = () => {
    setIsRunningCell();
    runCell(cell)
      .then(() => {
        setIsNotRunningCell();
      });
  };

  return (
    <div className={classNames("flex flex-row justify-between items-center h-9 bg-gray-100", className)}>
      <div className="flex flex-row items-center px-3.5">
        {isRunningCell ? <LoadingIndicator /> : <IconButton onClick={handleCellRun}><PlayIcon /></IconButton>}
        <span className="ml-4">{cell.name}</span>
      </div>
      <div className="pr-4 flex flex-row items-center gap-8">
        <div className="flex flex-row items-center gap-2">
          <LocalCogneeIcon className="text-indigo-700" />
          <span className="text-xs">local cognee</span>
        </div>
        <PopupMenu>
          <div className="flex flex-col gap-0.5">
            <button onClick={() => moveCellUp(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">move cell up</button>
            <button onClick={() => moveCellDown(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">move cell down</button>
          </div>
          <div className="flex flex-col gap-0.5 items-start">
            <button onClick={() => renameCell(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">rename</button>
            <button onClick={() => removeCell(cell)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">delete</button>
          </div>
        </PopupMenu>
      </div>
    </div>
  );
}
