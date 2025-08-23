"use client";

import { useState } from "react";
import { useBoolean } from "@/utils";
import Accordion from "@/ui/elements/Accordion";
import { CloudIcon, SearchIcon, NotebookIcon, PlusIcon, LocalCogneeIcon, DatasetIcon } from "@/ui/Icons";
import Header from "@/ui/Layout/Header";

interface NotebookCell {
  id: string;
  title: string;
  type: "code" | "text" | "markdown";
  content: string;
}

interface Notebook {
  id: string;
  name: string;
  cells: NotebookCell[];
}

interface Instance {
  id: string;
  name: string;
  type: "local" | "cloud";
}

interface Data {
  id: string;
  name: string;
  type: "csv" | "json" | "xml";
}

interface Dataset {
  id: string;
  name: string;
  data: Data[];
}

export default function Dashboard() {
  const notebooks: Notebook[] = [{
    id: "1",
    name: "My First Notebook",
    cells: [{
      id: "1",
      title: "Code Cell",
      type: "code",
      content: "print('Hello, World!')",
    }, {
      id: "2",
      title: "Text Cell",
      type: "text",
      content: "This is a text cell.",
    }, {
      id: "3",
      title: "Markdown Cell",
      type: "markdown",
      content: "# This is a Markdown Cell",
    }],
  }, {
    id: "2",
    name: "My Second Notebook",
    cells: [],
  }];

  const instances: Instance[] = [{
    id: "1",
    name: "Local Cognee",
    type: "local",
  }, {
    id: "2",
    name: "Cloud Cognee",
    type: "cloud",
  }];

  const datasets: Dataset[] = [{
    id: "1",
    name: "Dataset 1",
    data: [{
      id: "1",
      name: "Data 1",
      type: "csv",
    }],
  }, {
    id: "2",
    name: "Dataset 2",
    data: [{
      id: "2",
      name: "Data 2",
      type: "json",
    }, {
      id: "3",
      name: "Data 3",
      type: "xml",
    }],
  }];

  const {
    value: isNotebookPanelOpen,
    setTrue: openNotebookPanel,
    setFalse: closeNotebookPanel,
  } = useBoolean(true);

  const {
    value: isInstancesPanelOpen,
    setTrue: openInstancesPanel,
    setFalse: closeInstancesPanel,
  } = useBoolean(true);

  const {
    value: isDatasetsPanelOpen,
    setTrue: openDatasetsPanel,
    setFalse: closeDatasetsPanel,
  } = useBoolean(true);

  const [openDatasets, openDataset] = useState(new Set());

  const toggleDataset = (id: string) => {
    openDataset((prev) => {
      const newState = new Set(prev);

      if (newState.has(id)) {
        newState.delete(id)
      } else {
        newState.add(id);
      }

      return newState;
    });
  };

  const [selectedNotebook, setSelectedNotebook] = useState<Notebook | null>(null);

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

      <div className="relative flex-[1] flex flex-row gap-2.5 items-start w-full max-w-[1920px] mx-auto">
        <div className="flex-1/5 px-5 py-4">
          <div className="relative mb-5">
            <label htmlFor="search-input"><SearchIcon className="absolute left-3 top-[10px] cursor-text" /></label>
            <input id="search-input" className="text-xs leading-3 w-full h-8 flex flex-row items-center gap-2.5 rounded-3xl pl-9 placeholder-gray-300 border-gray-300 border-[1px] focus:outline-indigo-600" placeholder="Search datasets..." />
          </div>

          <Accordion
            title={<span>Notebooks</span>}
            isOpen={isNotebookPanelOpen}
            openAccordion={openNotebookPanel}
            closeAccordion={closeNotebookPanel}
            tools={<PlusIcon />}
          >
            {notebooks.map((notebook) => (
              <button key={notebook.id} onClick={() => setSelectedNotebook(notebook)} className="flex flex-row gap-2.5 items-center px-0.5 py-1.5">
                <NotebookIcon />
                <span className="text-xs">{notebook.name}</span>
              </button>
            ))}
          </Accordion>

          <div className="mt-7 mb-14">
            <Accordion
              title={<span>Cognee Instances</span>}
              isOpen={isInstancesPanelOpen}
              openAccordion={openInstancesPanel}
              closeAccordion={closeInstancesPanel}
              tools={<PlusIcon />}
            >
              {instances.map((instance) => (
                <div key={instance.id} className="flex flex-row gap-2.5 items-center px-0.5 py-1.5">
                  {instance.type === "local" ? <LocalCogneeIcon className="text-indigo-700" /> : <CloudIcon color="#000000" />}
                  <span className="text-xs">{instance.name}</span>
                </div>
              ))}
            </Accordion>
          </div>

          <Accordion
            title={<span>Datasets</span>}
            isOpen={isDatasetsPanelOpen}
            openAccordion={openDatasetsPanel}
            closeAccordion={closeDatasetsPanel}
            tools={<PlusIcon />}
          >
            <div className="flex flex-col">
              {datasets.map((dataset) => {
                return (
                  <Accordion
                    key={dataset.id}
                    title={(
                      <div key={dataset.id} className="flex flex-row gap-2.5 items-center px-0.5 py-1.5">
                        <DatasetIcon />
                        <span className="text-xs">{dataset.name}</span>
                      </div>
                    )}
                    isOpen={openDatasets.has(dataset.id)}
                    openAccordion={() => toggleDataset(dataset.id)}
                    closeAccordion={() => toggleDataset(dataset.id)}
                    tools={<PlusIcon />}
                  >
                    <div className="pl-4">
                      {dataset.data.map((data) => (
                        <div key={data.id} className="flex flex-row gap-2.5 items-center px-0.5 py-1.5">
                          {/* <DatasetIcon /> */}
                          <span className="text-xs">{data.name}</span>
                        </div>
                      ))}
                    </div>
                  </Accordion>
                );
              })}
            </div>
          </Accordion>

        </div>
        <div className="flex-4/5 flex flex-col justify-between h-full">
            <div className="">
              {selectedNotebook?.cells.map((cell) => (
                <div key={cell.id} className="flex flex-col px-4 py-4">
                  <div key={cell.id} className="flex flex-row justify-between items-center px-1 py-2">
                    <div className="text-xs text-gray-400 whitespace-nowrap">{cell.title}</div>
                    <div className="text-xs text-gray-400 whitespace-nowrap">{cell.type}</div>
                  </div>
                  <div className="">{cell.content}</div>
                </div>
              ))}
            </div>

            <div className="">
              <span>Graph Visualization</span>
            </div>
        </div>
      </div>

    </div>
  );
}
