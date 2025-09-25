import { useCallback, useState } from "react";
import { fetch, isCloudEnvironment } from "@/utils";
import { Cell, Notebook } from "@/ui/elements/Notebook/types";

function useNotebooks() {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);

  const addNotebook = useCallback((notebookName: string) => {
    return fetch("/v1/notebooks", {
        body: JSON.stringify({ name: notebookName }),
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }, isCloudEnvironment())
      .then((response) => response.json())
      .then((notebook) => {
        setNotebooks((notebooks) => [
          ...notebooks,
          notebook,
        ]);

        return notebook;
      });
  }, []);

  const removeNotebook = useCallback((notebookId: string) => {
    return fetch(`/v1/notebooks/${notebookId}`, {
      method: "DELETE",
    }, isCloudEnvironment())
    .then(() => {
      setNotebooks((notebooks) =>
        notebooks.filter((notebook) => notebook.id !== notebookId)
      );
    });
  }, []);

  const fetchNotebooks = useCallback(() => {
    return fetch("/v1/notebooks", {
      headers: {
        "Content-Type": "application/json",
      },
    }, isCloudEnvironment())
    .then((response) => response.json())
    .then((notebooks) => {
      setNotebooks(notebooks);

      return notebooks;
    })
    .catch((error) => {
      console.error("Error fetching notebooks:", error);
      throw error
    });
  }, []);

  const updateNotebook = useCallback((updatedNotebook: Notebook) => {
    setNotebooks((existingNotebooks) =>
      existingNotebooks.map((notebook) =>
        notebook.id === updatedNotebook.id
          ? updatedNotebook
          : notebook
      )
    );
  }, []);

  const saveNotebook = useCallback((notebook: Notebook) => {
    return fetch(`/v1/notebooks/${notebook.id}`, {
      body: JSON.stringify({
        name: notebook.name,
        cells: notebook.cells,
      }),
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
    }, isCloudEnvironment())
    .then((response) => response.json())
  }, []);

  const runCell = useCallback((notebook: Notebook, cell: Cell, cogneeInstance: string) => {
    setNotebooks((existingNotebooks) =>
      existingNotebooks.map((existingNotebook) =>
        existingNotebook.id === notebook.id ? {
          ...existingNotebook,
          cells: existingNotebook.cells.map((existingCell) =>
            existingCell.id === cell.id ? {
              ...existingCell,
              result: undefined,
              error: undefined,
            } : existingCell
          ),
        } : notebook
      )
    );

    return fetch(`/v1/notebooks/${notebook.id}/${cell.id}/run`, {
        body: JSON.stringify({
          content: cell.content,
        }),
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }, cogneeInstance === "cloud")
      .then((response) => response.json())
      .then((response) => {
        setNotebooks((existingNotebooks) =>
          existingNotebooks.map((existingNotebook) =>
            existingNotebook.id === notebook.id ? {
              ...existingNotebook,
              cells: existingNotebook.cells.map((existingCell) =>
                existingCell.id === cell.id ? {
                 ...existingCell,
                  result: response.result,
                  error: response.error,
                } : existingCell
              ),
            } : notebook
          )
        );
      });
  }, []);

  return {
    notebooks,
    addNotebook,
    saveNotebook,
    updateNotebook,
    removeNotebook,
    refreshNotebooks: fetchNotebooks,
    runCell,
  };
};

export default useNotebooks;
