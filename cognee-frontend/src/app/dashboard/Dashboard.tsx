"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Header } from "@/ui/Layout";
import { SearchIcon } from "@/ui/Icons";
import { CTAButton, Notebook } from "@/ui/elements";
import { fetch, isCloudEnvironment } from "@/utils";
import { Notebook as NotebookType } from "@/ui/elements/Notebook/types";
import { useAuthenticatedUser } from "@/modules/auth";
import { Dataset } from "@/modules/ingestion/useDatasets";
import useNotebooks from "@/modules/notebooks/useNotebooks";
import { ENABLE_NOTEBOOKS } from "@/config/featureFlags";

import AddDataToCognee from "./AddDataToCognee";
import NotebooksAccordion from "./NotebooksAccordion";
import CogneeInstancesAccordion from "./CogneeInstancesAccordion";
import InstanceDatasetsAccordion from "./InstanceDatasetsAccordion";

interface DashboardProps {
  user?: {
    id: string;
    name: string;
    email: string;
    picture: string;
  };
  accessToken: string;
}

export default function Dashboard({ accessToken }: DashboardProps) {
  fetch.setAccessToken(accessToken);
  const { user } = useAuthenticatedUser();

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
    if (!ENABLE_NOTEBOOKS) {
      return;
    }

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
    return removeNotebook(notebookId)
      .then(() => {
        setSelectedNotebookId((currentSelectedNotebookId) => (
          currentSelectedNotebookId === notebookId ? null : currentSelectedNotebookId
        ));
      });
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

  const isCloudEnv = isCloudEnvironment();

  return (
    <div className="h-full flex flex-col">
      {/* <video
        autoPlay
        loop
        muted
        playsInline
        className="fixed inset-0 z-0 object-cover w-full h-full"
      >
        <source src="/videos/background-video-blur.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video> */}

      <Header user={user} />

      <div className="relative flex-1 flex flex-row gap-2.5 items-start w-full max-w-[1920px] max-h-[calc(100% - 3.5rem)] overflow-hidden mx-auto px-2.5 pb-2.5">
        <div className="px-5 py-4 lg:w-96 bg-white rounded-xl h-[calc(100%-2.75rem)]">
          <div className="relative mb-2">
            <label htmlFor="search-input"><SearchIcon className="absolute left-3 top-[10px] cursor-text" /></label>
            <input id="search-input" className="text-xs leading-3 w-full h-8 flex flex-row items-center gap-2.5 rounded-3xl pl-9 placeholder-gray-300 border-gray-300 border-[1px] focus:outline-indigo-600" placeholder="Search datasets..." />
          </div>

          <AddDataToCognee
            datasets={datasets}
            refreshDatasets={refreshDatasetsRef.current}
            useCloud={isCloudEnv}
          />

          {ENABLE_NOTEBOOKS && (
            <NotebooksAccordion
              notebooks={notebooks}
              addNotebook={addNotebook}
              removeNotebook={handleNotebookRemove}
              openNotebook={setSelectedNotebookId}
            />
          )}

          <div className="mt-7 mb-14">
            <CogneeInstancesAccordion>
              <InstanceDatasetsAccordion
                onDatasetsChange={handleDatasetsChange}
              />
            </CogneeInstancesAccordion>
          </div>

          <div className="fixed bottom-2.5 w-[calc(min(1920px,100%)/5)] lg:w-96 ml-[-1.25rem] mx-auto">
            <a href="/plan">
              <CTAButton className="w-full">Select a plan</CTAButton>
            </a>
          </div>
        </div>

        <div className="flex-1 flex flex-col justify-between h-full overflow-y-auto">
          {ENABLE_NOTEBOOKS ? (
            selectedNotebook ? (
              <Notebook
                key={selectedNotebook.id}
                notebook={selectedNotebook}
                updateNotebook={handleNotebookUpdate}
                runCell={runCell}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500">
                Select or create a notebook to start scripting ingest/search flows.
              </div>
            )
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-500 text-center px-10">
              Notebook automation is disabled for this deployment. Manage datasets on the left and
              use LibreChat/MCP for search.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
