"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Header from "@/ui/Layout/Header";
import { Notebook } from "@/ui/elements";

import useNotebooks from "@/modules/notebooks/useNotebooks";
import { Notebook as NotebookType } from "@/ui/elements/Notebook/types";
import DatasetsAccordion from "./DatasetsAccordion";
import CogneeInstancesAccordion from "./CogneeInstancesAccordion";
import NotebooksAccordion from "./NotebooksAccordion";

export default function Dashboard() {
  const {
    notebooks,
    refreshNotebooks,
    runCell,
    addNotebook,
    updateNotebook,
    saveNotebook,
    removeNotebook,
  } = useNotebooks();

  useEffect(() => {
    if (!notebooks.length) {
      refreshNotebooks();
    }
  }, [notebooks.length, refreshNotebooks]);

  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);

  const handleNotebookRemove = useCallback((notebookId: string) => {
    setSelectedNotebookId((currentSelectedNotebookId) => (
      currentSelectedNotebookId === notebookId ? null : currentSelectedNotebookId
    ));
    return removeNotebook(notebookId);
  }, [removeNotebook]);

  const saveNotebookTimeoutRef = useRef<number | null>(null);
  const saveNotebookThrottled = useCallback((notebook: NotebookType) => {
    const throttleTime = 1000;

    if (saveNotebookTimeoutRef.current) {
      clearTimeout(saveNotebookTimeoutRef.current);
      saveNotebookTimeoutRef.current = null;
    }

    saveNotebookTimeoutRef.current = setTimeout(() => {
      saveNotebook(notebook);
    }, throttleTime) as unknown as number;
  }, [saveNotebook]);

  useEffect(() => {
    return () => {
      if (saveNotebookTimeoutRef.current) {
        clearTimeout(saveNotebookTimeoutRef.current);
        saveNotebookTimeoutRef.current = null;
      }
    };
  }, []);

  const handleNotebookUpdate = useCallback((notebook: NotebookType) => {
    updateNotebook(notebook);
    saveNotebookThrottled(notebook);
  }, [saveNotebookThrottled, updateNotebook]);

  return (
    <div className="h-full flex flex-col">
      <div className="absolute top-0 right-0 bottom-0 left-0 flex flex-row gap-2.5">
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-3/5 h-full flex flex-row gap-2.5">
          <div className="flex-1/3 bg-gray-100 h-full"></div>
          <div className="flex-1/3 bg-gray-100 h-full"></div>
          <div className="flex-1/3 bg-gray-100 h-full"></div>
        </div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
      </div>

      <Header />

      <div className="relative flex-1 flex flex-row gap-2.5 items-start w-full max-w-[1920px] mx-auto">
        <div className="px-5 py-4 lg:w-80">
          {/* <div className="relative mb-5">
            <label htmlFor="search-input"><SearchIcon className="absolute left-3 top-[10px] cursor-text" /></label>
            <input id="search-input" className="text-xs leading-3 w-full h-8 flex flex-row items-center gap-2.5 rounded-3xl pl-9 placeholder-gray-300 border-gray-300 border-[1px] focus:outline-indigo-600" placeholder="Search datasets..." />
          </div> */}

          <NotebooksAccordion
            notebooks={notebooks}
            addNotebook={addNotebook}
            removeNotebook={handleNotebookRemove}
            openNotebook={setSelectedNotebookId}
          />

          <div className="mt-7 mb-14">
            <CogneeInstancesAccordion />
          </div>

          <DatasetsAccordion />
        </div>

        <div className="flex-1 flex flex-col justify-between h-full overflow-hidden">
            {selectedNotebookId && (
              <Notebook
                notebook={notebooks.find((notebook) => notebook.id === selectedNotebookId)!}
                updateNotebook={handleNotebookUpdate}
                saveNotebook={saveNotebook}
                runCell={runCell}
              />
            )}
        </div>
      </div>

    </div>
  );
}
