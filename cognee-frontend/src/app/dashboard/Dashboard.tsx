"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Header } from "@/ui/Layout";
import { SearchIcon } from "@/ui/Icons";
import { Notebook } from "@/ui/elements";
import { Notebook as NotebookType } from "@/ui/elements/Notebook/types";
import { Dataset } from "@/modules/ingestion/useDatasets";
import useNotebooks from "@/modules/notebooks/useNotebooks";

import NotebooksAccordion from "./NotebooksAccordion";
import CogneeInstancesAccordion from "./CogneeInstancesAccordion";
import AddDataToCognee from "./AddDataToCognee";
import InstanceDatasetsAccordion from "./InstanceDatasetsAccordion";

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
      refreshNotebooks()
        .then((notebooks) => {
          if (notebooks[0]) {
            setSelectedNotebookId(notebooks[0].id);
          }
        });
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

  const selectedNotebook = notebooks.find((notebook) => notebook.id === selectedNotebookId);

  // ############################
  // Datasets logic

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const refreshDatasetsRef = useRef(() => {});

  const handleDatasetsChange = useCallback((payload: { datasets: Dataset[], refreshDatasets: () => void }) => {
    const {
      datasets,
      refreshDatasets,
    } = payload;

    refreshDatasetsRef.current = refreshDatasets;
    setDatasets(datasets);
  }, []);

  return (
    <div className="h-full flex flex-col bg-gray-200">
      <Header />

      <div className="relative flex-1 flex flex-row gap-2.5 items-start w-full max-w-[1920px] max-h-[calc(100% - 3.5rem)] overflow-hidden mx-auto px-2.5 py-2.5">
        <div className="px-5 py-4 lg:w-96 bg-white rounded-xl min-h-full">
          <div className="relative mb-2">
            <label htmlFor="search-input"><SearchIcon className="absolute left-3 top-[10px] cursor-text" /></label>
            <input id="search-input" className="text-xs leading-3 w-full h-8 flex flex-row items-center gap-2.5 rounded-3xl pl-9 placeholder-gray-300 border-gray-300 border-[1px] focus:outline-indigo-600" placeholder="Search datasets..." />
          </div>

          <AddDataToCognee
            datasets={datasets}
            refreshDatasets={refreshDatasetsRef.current}
          />

          <NotebooksAccordion
            notebooks={notebooks}
            addNotebook={addNotebook}
            removeNotebook={handleNotebookRemove}
            openNotebook={setSelectedNotebookId}
          />

          <div className="mt-7 mb-14">
            <CogneeInstancesAccordion>
              <InstanceDatasetsAccordion
                onDatasetsChange={handleDatasetsChange}
              />
            </CogneeInstancesAccordion>
          </div>
        </div>

        <div className="flex-1 flex flex-col justify-between h-full overflow-y-auto">
            {selectedNotebook && (
              <Notebook
                key={selectedNotebook.id}
                notebook={selectedNotebook}
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
